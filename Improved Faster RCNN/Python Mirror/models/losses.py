"""Class-imbalance loss functions for the Improved Faster R-CNN.

Journal component (6): "Class-imbalance losses available: Focal Loss
(gamma=2, alpha=0.75) and Weighted Cross-Entropy" from Wang et al.,
"Automatic Detection and Classification of Steel Surface Defect Using
Deep CNNs", Metals 2021.

Steel surface defect datasets are heavily class-imbalanced (some defect
types are far rarer than others, and the background/negative RoIs vastly
outnumber the positive ones). To counteract this the paper makes two
class-balancing classification objectives available:

  * Focal Loss      -- Lin et al., "Focal Loss for Dense Object Detection",
                       ICCV 2017 (arXiv:1708.02002). Down-weights the loss
                       contributed by easy, well-classified examples so that
                       training focuses on the hard / rare ones.
  * Weighted CE     -- ordinary cross-entropy with per-class weights, which
                       directly rescales the contribution of each defect
                       class according to its frequency.

This module intentionally does NOT import ``config`` -- the losses are fully
parameterized so they can be reused unchanged by train.py / a custom RoI head.
"""

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "FocalLoss",
    "WeightedCrossEntropyLoss",
    "fastrcnn_focal_loss",
]


class FocalLoss(nn.Module):
    r"""Multi-class Focal Loss.

    Implements the focal loss of Lin et al., "Focal Loss for Dense Object
    Detection" (ICCV 2017, arXiv:1708.02002), generalized to the multi-class
    softmax setting:

        ce  = CrossEntropy(logits, targets)          # per-sample, no reduction
        pt  = exp(-ce)                               # = softmax prob. of the
                                                     #   ground-truth class
        FL  = alpha * (1 - pt) ** gamma * ce

    The modulating factor ``(1 - pt) ** gamma`` shrinks the loss for confident
    (easy) examples and leaves hard examples almost untouched, focusing
    training on rare steel-defect classes. ``alpha`` is a global balancing
    scalar (the paper uses ``alpha = 0.75``, ``gamma = 2.0``).

    Args:
        alpha: Global balancing weight applied to every sample (default 0.75,
            as in the journal). Use ``alpha = 1.0`` to disable.
        gamma: Focusing parameter ``>= 0`` (default 2.0). ``gamma = 0`` reduces
            this to a plain (alpha-scaled) cross-entropy.
        reduction: One of ``'none'`` | ``'mean'`` | ``'sum'``.
        ignore_index: Target value that is ignored and does not contribute to
            the loss or to the mean denominator (default -100, matching
            ``torch.nn.functional.cross_entropy``).
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"reduction must be 'none', 'mean' or 'sum', got {reduction!r}"
            )
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = int(ignore_index)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the focal loss.

        Args:
            logits: Float tensor of shape ``[N, C]`` (raw, un-normalized class
                scores).
            targets: Long tensor of shape ``[N]`` with values in ``[0, C)`` (or
                equal to ``ignore_index``).

        Returns:
            Scalar tensor if ``reduction`` is ``'mean'`` / ``'sum'``; a tensor
            of shape ``[N]`` (with ignored entries set to 0) if ``'none'``.
        """
        # Per-sample cross-entropy; ce >= 0. ignore_index entries get ce = 0.
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        # pt is the softmax probability assigned to the ground-truth class.
        pt = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()

        # 'mean': average only over the non-ignored samples so that masking a
        # batch does not silently shrink the gradient magnitude.
        valid = targets != self.ignore_index
        num_valid = valid.sum()
        if num_valid == 0:
            # No valid targets -> return a differentiable zero.
            return loss.sum() * 0.0
        return loss.sum() / num_valid.to(loss.dtype)


class WeightedCrossEntropyLoss(nn.Module):
    r"""Cross-entropy with optional per-class weights.

    Journal component (6): the "Weighted Cross-Entropy" class-imbalance option.
    This is a thin, serialization-friendly wrapper around
    :func:`torch.nn.functional.cross_entropy` that lets you pass per-class
    weights (e.g. inverse class frequencies) so that rare steel-defect classes
    contribute proportionally more to the loss.

    Args:
        class_weights: Per-class weights of shape ``[C]`` as a list / tuple /
            1-D tensor, or ``None`` for uniform weighting. When provided it is
            registered as a buffer so it moves with ``.to(device)`` and is saved
            in the module's ``state_dict``.
        reduction: One of ``'none'`` | ``'mean'`` | ``'sum'``.
    """

    def __init__(
        self,
        class_weights: Optional[Union[Sequence[float], Tensor]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"reduction must be 'none', 'mean' or 'sum', got {reduction!r}"
            )
        self.reduction = reduction

        if class_weights is None:
            weight_tensor: Optional[Tensor] = None
        else:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            if weight_tensor.dim() != 1:
                raise ValueError(
                    "class_weights must be 1-D of shape [C], got shape "
                    f"{tuple(weight_tensor.shape)}"
                )
        # register_buffer accepts None and keeps the key in the state_dict.
        self.register_buffer("class_weights", weight_tensor)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute weighted cross-entropy.

        Args:
            logits: Float tensor of shape ``[N, C]``.
            targets: Long tensor of shape ``[N]`` with values in ``[0, C)``.

        Returns:
            Scalar (``'mean'`` / ``'sum'``) or ``[N]`` tensor (``'none'``).
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction=self.reduction,
        )


def fastrcnn_focal_loss(
    class_logits: Tensor,
    box_regression: Tensor,
    labels: Sequence[Tensor],
    regression_targets: Sequence[Tensor],
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> Tuple[Tensor, Tensor]:
    """Focal-loss drop-in for torchvision's ``fastrcnn_loss``.

    Mirrors ``torchvision.models.detection.roi_heads.fastrcnn_loss`` but swaps
    the classification cross-entropy for the multi-class Focal Loss of Lin et
    al. (ICCV 2017). The box-regression term is unchanged: a smooth-L1
    (Huber, beta=1) loss computed only over the positive (foreground) samples
    and normalized by the total number of sampled RoIs -- exactly as
    torchvision does. This is a utility for an optional custom RoI head.

    Args:
        class_logits: Tensor of shape ``[sum(R_i), C]`` -- per-RoI class scores
            for the whole batch (C includes the background class at index 0).
        box_regression: Tensor of shape ``[sum(R_i), C * 4]`` -- per-class box
            deltas for every RoI.
        labels: List (length = batch size) of long tensors; concatenated to
            shape ``[sum(R_i)]``. Class 0 is background.
        regression_targets: List of float tensors with the encoded regression
            targets; concatenated to shape ``[sum(R_i), 4]``.
        alpha: Focal-loss balancing weight (default 0.75, journal value).
        gamma: Focal-loss focusing parameter (default 2.0, journal value).

    Returns:
        Tuple ``(classification_loss, box_loss)`` of scalar tensors.
    """
    labels_cat = torch.cat(list(labels), dim=0)
    regression_targets_cat = torch.cat(list(regression_targets), dim=0)

    # --- Classification term: focal loss over ALL sampled RoIs -------------
    classification_loss = FocalLoss(
        alpha=alpha, gamma=gamma, reduction="mean"
    )(class_logits, labels_cat)

    # --- Box-regression term: smooth-L1 over positive RoIs only ------------
    # Positive (foreground) samples have label > 0 (0 == background).
    sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
    labels_pos = labels_cat[sampled_pos_inds_subset]

    N, num_classes_times_4 = box_regression.shape
    num_classes = num_classes_times_4 // 4
    box_regression = box_regression.reshape(N, num_classes, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets_cat[sampled_pos_inds_subset],
        beta=1.0 / 9.0,
        reduction="sum",
    )
    # Normalize by the total number of sampled RoIs (torchvision convention).
    box_loss = box_loss / labels_cat.numel()

    return classification_loss, box_loss
