"""Matrix NMS, box-adapted from SOLOv2.

Reference
---------
Wang et al., "SOLOv2: Dynamic and Fast Instance Segmentation", NeurIPS 2020.
The original Matrix NMS operates on instance masks; here it is adapted to
axis-aligned bounding boxes (xyxy) and used as the detection post-processing
step of the Improved Faster R-CNN (Wang et al., Metals 2021), replacing the
standard greedy NMS.

Journal component implemented
-----------------------------
Improvement #5: Matrix NMS replacing standard NMS in detection post-processing.

Design notes
------------
Matrix NMS removes the sequential, iterative suppression loop of greedy NMS and
instead estimates, in a single fully-vectorized pass, a *decay factor* for every
box based on how strongly it overlaps with higher-scoring boxes of the SAME
class. Each box's score is multiplied by this decay; boxes whose decayed score
drops below ``score_threshold`` are discarded. This avoids any Python-level loop
over boxes.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["box_iou", "matrix_nms"]


def box_iou(boxes: Tensor) -> Tensor:
    """Pairwise IoU between a set of axis-aligned boxes.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        Boxes in ``xyxy`` format (x1, y1, x2, y2).

    Returns
    -------
    Tensor[N, N]
        Symmetric matrix where entry (i, j) is the IoU of box ``i`` and box
        ``j``. The diagonal is the self-IoU (== 1 for non-degenerate boxes).
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 0))

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Per-box area; clamp to be non-negative for degenerate boxes.
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # [N]

    # Pairwise intersection coordinates via broadcasting -> [N, N].
    inter_x1 = torch.max(x1[:, None], x1[None, :])
    inter_y1 = torch.max(y1[:, None], y1[None, :])
    inter_x2 = torch.min(x2[:, None], x2[None, :])
    inter_y2 = torch.min(y2[:, None], y2[None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h  # [N, N]

    union = areas[:, None] + areas[None, :] - inter
    # Clamp denominator to avoid division by zero for empty/degenerate boxes.
    iou = inter / union.clamp(min=1e-6)
    return iou


def matrix_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    sigma: float = 0.5,
    method: str = "gaussian",
    score_threshold: float = 0.05,
    max_detections: int = 100,
) -> tuple[Tensor, Tensor]:
    """Box-adapted Matrix NMS.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        Candidate boxes in ``xyxy`` format.
    scores : Tensor[N]
        Confidence score per box.
    labels : Tensor[N]
        Integer class label per box; suppression is class-aware (only boxes
        sharing a label can suppress one another).
    sigma : float, default 0.5
        Width of the Gaussian decay kernel (only used when ``method='gaussian'``).
    method : {'gaussian', 'linear'}, default 'gaussian'
        Decay function applied to overlapping boxes.
    score_threshold : float, default 0.05
        Boxes whose *decayed* score is not strictly greater than this are
        dropped.
    max_detections : int, default 100
        Maximum number of boxes returned (top-scoring after decay).

    Returns
    -------
    (kept_indices, decayed_scores) : tuple[Tensor, Tensor]
        ``kept_indices`` is a ``LongTensor`` indexing into the ORIGINAL input
        ordering; ``decayed_scores`` are the corresponding post-decay scores,
        aligned with ``kept_indices``. Both are sorted by decayed score
        descending.

    Notes
    -----
    Fully vectorized; contains no Python loop over boxes.
    """
    device = boxes.device
    n = scores.numel()

    empty_idx = torch.empty((0,), dtype=torch.long, device=device)
    empty_score = torch.empty((0,), dtype=scores.dtype, device=device)
    if n == 0:
        return empty_idx, empty_score

    # ---- 1. Sort by score descending; remember the permutation so we can map
    #         results back to the original input ordering at the end. ----
    order = torch.argsort(scores, descending=True)  # [N], index into original
    s_scores = scores[order]
    s_boxes = boxes[order]
    s_labels = labels[order]

    # ---- 2. Pairwise IoU on the sorted boxes. ----
    iou = box_iou(s_boxes)  # [N, N]

    # ---- 3. Keep only the influence of *higher-scoring* boxes on each box.
    #         After sorting, row/col index == rank, so a higher-scoring box has a
    #         SMALLER index. Zero the lower triangle AND the diagonal so that
    #         entry (i, j) is non-zero only when i < j (box i outranks box j). ----
    iou = iou.triu(diagonal=1)  # strictly upper triangular

    # ---- 4. Class-aware masking: a box may only be suppressed by a higher-
    #         scoring box of the SAME label. Zero cross-class IoU. ----
    same_label = (s_labels[:, None] == s_labels[None, :]).to(iou.dtype)
    iou = iou * same_label

    # ---- 5. For each box j (column), the strongest overlap any *suppressor*
    #         (row i) itself has -> per-suppressor compensation term.
    #         ious_cmax[i] = max over j of iou[i, j], broadcast down each column. ----
    ious_cmax, _ = iou.max(dim=0)  # [N], max over rows i for each column j... see below

    # NOTE on the compensation term:
    # Following SOLOv2, the decay of box j caused by suppressor i is discounted
    # by how much suppressor i is *itself* overlapped by an even-higher box.
    # ious_cmax for suppressor i is the max IoU of i with any box it could
    # suppress (its strongest column entry across the matrix). We compute it per
    # row i and broadcast across columns.
    ious_cmax_row, _ = iou.max(dim=1)  # [N] = strongest overlap of suppressor i
    ious_cmax_i = ious_cmax_row[:, None].expand(n, n)  # [N(i), N(j)]

    # ---- 6. Compute the decay for every (suppressor i, box j) pair, then take
    #         the worst (minimum) decay over all suppressors i for each box j. ----
    if method == "gaussian":
        decay = torch.exp(-(iou ** 2 - ious_cmax_i ** 2) / sigma)
    elif method == "linear":
        # decay_ij = (1 - iou_ij) / (1 - ious_cmax_i); clamp denom for safety.
        denom = (1.0 - ious_cmax_i).clamp(min=1e-6)
        decay = (1.0 - iou) / denom
    else:
        raise ValueError(
            f"matrix_nms: unknown method {method!r}; expected 'gaussian' or 'linear'."
        )

    # Pairs with zero IoU (no overlap, cross-class, or non-suppressor i>=j) must
    # NOT decay box j. For gaussian, iou=0 & cmax=0 already gives exp(0)=1; but
    # cross-class / lower-triangle entries were zeroed in `iou` while their
    # ious_cmax_i may be > 0, which would spuriously inflate decay > 1 (gaussian)
    # or distort it (linear). Force those pairs to a neutral decay of 1.0.
    no_effect = iou == 0
    decay = torch.where(no_effect, torch.ones_like(decay), decay)

    # Worst-case (minimum) decay over suppressors i, per box j.
    decay_factor, _ = decay.min(dim=0)  # [N]
    decay_factor = decay_factor.clamp(min=0.0, max=1.0)

    # ---- 7. Apply decay; threshold; keep top-k. ----
    new_scores = s_scores * decay_factor  # aligned with sorted order

    keep_mask = new_scores > score_threshold
    kept_sorted = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)  # indices into sorted order
    if kept_sorted.numel() == 0:
        return empty_idx, empty_score

    kept_scores = new_scores[kept_sorted]

    # Top-k by decayed score (already roughly descending, but decay can reorder).
    if kept_scores.numel() > max_detections:
        topk_scores, topk_local = kept_scores.topk(max_detections)
        kept_sorted = kept_sorted[topk_local]
        kept_scores = topk_scores
    else:
        # Ensure descending order of the returned detections.
        sort_local = torch.argsort(kept_scores, descending=True)
        kept_sorted = kept_sorted[sort_local]
        kept_scores = kept_scores[sort_local]

    # ---- 8. Map sorted-order indices back to ORIGINAL input ordering. ----
    kept_original = order[kept_sorted]
    return kept_original.to(torch.long), kept_scores
