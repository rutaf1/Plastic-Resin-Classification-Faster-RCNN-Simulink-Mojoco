"""Assembly of the Improved Faster R-CNN (Wang et al., *Metals* 2021).

This module wires every architectural improvement into a single
``torchvision.models.detection.FasterRCNN``-compatible ``nn.Module`` via the
:func:`build_model` factory:

  #1  Backbone = ResNet50-vd + DCNv2          -> ``models.resnet50_vd`` / ``models.backbone``
  #2  SPP on the deepest feature              -> applied inside ``models.fpn`` / backbone
  #3  Enhanced FPN with CoordConv laterals    -> ``models.fpn``
  #4  Custom anchors [0.25,0.5,1,2,5]         -> ``models.anchors``
  #5  Matrix NMS replacing standard NMS       -> ``models.matrix_nms`` (bound below)
  #6  Focal Loss / Weighted CE (optional)     -> ``models.losses``

Design contract
---------------
``build_model`` takes only explicit arguments (with defaults) and MUST NOT import
``config.py`` -- the training / inference entry points read the config and pass
values in. Sibling ``models`` modules are imported with relative imports.

The default path (``use_focal_loss=False``) yields a model whose ``forward``
works in BOTH modes:
  * train mode  : ``model(images, targets) -> dict`` of losses,
  * eval  mode  : ``model(images)         -> list[dict]`` of detections,
for 480x480x3 RGB inputs and 7 foreground classes (+1 background = 8 internally).
"""

import types

import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import roi_heads as _roi_heads
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from .backbone import ResNet50vdFPN
from .anchors import build_anchor_generator
from .matrix_nms import matrix_nms
from .losses import fastrcnn_focal_loss


# ImageNet RGB normalization statistics (3-channel input). The Improved Faster
# R-CNN consumes ordinary RGB steel-surface images, so the standard torchvision
# detection normalization applies.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def matrix_nms_postprocess(
    self,
    class_logits,
    box_regression,
    proposals,
    image_shapes,
):
    """Drop-in replacement for ``RoIHeads.postprocess_detections``.

    Implements *journal improvement #5*: the final greedy per-class
    ``batched_nms`` of torchvision's ``RoIHeads.postprocess_detections`` is
    replaced by :func:`models.matrix_nms.matrix_nms` (box-adapted SOLOv2 Matrix
    NMS). Every other step -- softmax over class logits, box decoding via
    ``self.box_coder.decode``, clipping to the image, dropping the background
    class (index 0), flattening per-class predictions into individual
    detections, the ``self.score_thresh`` score filter, and the
    ``self.detections_per_img`` cap -- mirrors the original method exactly so the
    method is a faithful behavioural substitute.

    This function is meant to be bound onto a live ``model.roi_heads`` instance
    with ``types.MethodType`` (so ``self`` is the RoIHeads module).

    Args:
        class_logits:   Tensor ``[sum(R_i), C]`` of per-RoI class scores.
        box_regression: Tensor ``[sum(R_i), C * 4]`` of per-class box deltas.
        proposals:      List (len = batch) of proposal boxes ``[R_i, 4]``.
        image_shapes:   List of ``(H, W)`` per image (post-resize).

    Returns:
        ``(all_boxes, all_scores, all_labels)`` -- three lists, one entry per
        image, exactly as the original ``postprocess_detections`` returns.
    """
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label (index 0)
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # ---- Matrix NMS (journal improvement #5) replacing batched_nms ----
        # Class-aware suppression is handled internally by matrix_nms via the
        # `labels` argument; the cap to `detections_per_img` is applied through
        # max_detections so the returned set is already top-k by decayed score.
        keep, decayed_scores = matrix_nms(
            boxes,
            scores,
            labels,
            sigma=getattr(self, "matrix_nms_sigma", 0.5),
            method="gaussian",
            score_threshold=self.score_thresh,
            max_detections=self.detections_per_img,
        )
        boxes, scores, labels = boxes[keep], decayed_scores, labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


def _focal_roi_heads_forward(
    self,
    features,
    proposals,
    image_shapes,
    targets=None,
):
    """RoIHeads.forward variant that uses Focal Loss for classification.

    Identical to ``torchvision.models.detection.roi_heads.RoIHeads.forward``
    except the training-time classification/box loss is computed with
    :func:`models.losses.fastrcnn_focal_loss` (*journal improvement #6*) instead
    of the default ``fastrcnn_loss`` cross-entropy. The eval-time path is the
    unchanged original (which routes through the bound Matrix-NMS
    ``postprocess_detections``).

    Bound onto a live RoIHeads instance with ``types.MethodType``.
    """
    if targets is not None:
        for t in targets:
            floating_point_types = (torch.float, torch.double, torch.half)
            if t["boxes"].dtype not in floating_point_types:
                raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
            if not t["labels"].dtype == torch.int64:
                raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

    if self.training:
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
    else:
        labels = None
        regression_targets = None
        matched_idxs = None

    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)

    result = []
    losses = {}
    if self.training:
        if labels is None or regression_targets is None:
            raise ValueError("labels and regression_targets cannot be None in training")
        # ---- Focal-loss classification head (journal improvement #6) ----
        loss_classifier, loss_box_reg = fastrcnn_focal_loss(
            class_logits,
            box_regression,
            labels,
            regression_targets,
            alpha=getattr(self, "focal_alpha", 0.75),
            gamma=getattr(self, "focal_gamma", 2.0),
        )
        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    else:
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

    return result, losses


def build_model(
    num_classes=7,
    input_size=480,
    anchor_sizes=((16,), (32,), (64,), (128,), (256,)),
    anchor_ratios=(0.25, 0.5, 1.0, 2.0, 5.0),
    score_thresh=0.05,
    matrix_nms_sigma=0.5,
    detections_per_img=100,
    pretrained_backbone=False,
    use_focal_loss=False,
):
    """Build the Improved Faster R-CNN model.

    Assembles the ResNet50-vd + DCNv2 + SPP + CoordConv-FPN backbone, the custom
    anchor generator, a multi-scale RoI-align head, and a torchvision
    ``FasterRCNN``; then binds Matrix NMS (improvement #5) and, optionally, the
    Focal-loss classification head (improvement #6).

    Args:
        num_classes: Number of FOREGROUND classes (default 7). The torchvision
            detector is built with ``num_classes + 1`` to reserve index 0 for
            background.
        input_size: Fixed square input side length in pixels (default 480). Used
            for both ``min_size`` and ``max_size`` so images are resized to a
            fixed 480x480 (per the journal spec).
        anchor_sizes: One tuple of anchor base sizes per FPN level. MUST contain
            exactly 5 entries to match the 5 pyramid maps ('0'..'4').
        anchor_ratios: Shared anchor aspect ratios (journal: 0.25,0.5,1,2,5).
        score_thresh: Minimum detection score kept during post-processing.
        matrix_nms_sigma: Gaussian decay width for Matrix NMS.
        detections_per_img: Max detections returned per image.
        pretrained_backbone: Reserved for API symmetry; this backbone trains
            from scratch (no ImageNet weights are loaded), so this flag is
            accepted but currently does not load pretrained weights.
        use_focal_loss: If ``True``, swap the RoIHeads classification loss for
            Focal Loss (improvement #6). Default ``False`` keeps the standard
            cross-entropy path fully working.

    Returns:
        An ``nn.Module`` (torchvision ``FasterRCNN``) ready for training and
        inference on 480x480x3 inputs.
    """
    # ---- 1. Backbone: ResNet50-vd + DCNv2 + SPP + CoordConv FPN (imp. #1-3) --
    backbone = ResNet50vdFPN(dcn_stages=(2, 3, 4), out_channels=256, extra_block=True)

    # ---- 2. Custom anchors (improvement #4) ---------------------------------
    # 5 sizes <-> 5 feature maps; the ratios are broadcast across levels.
    if len(anchor_sizes) != 5:
        raise ValueError(
            f"anchor_sizes must have exactly 5 entries (one per FPN level), "
            f"got {len(anchor_sizes)}."
        )
    anchor_generator = build_anchor_generator(sizes=anchor_sizes, aspect_ratios=anchor_ratios)

    # ---- 3. RoI pooling over the four finer pyramid levels ------------------
    # The coarsest extra level ('4') is used by the RPN for proposals but not
    # for RoI feature extraction (standard FPN Faster R-CNN convention).
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    # ---- 4. Torchvision FasterRCNN (background class added internally) -------
    model = FasterRCNN(
        backbone,
        num_classes=num_classes + 1,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        min_size=input_size,
        max_size=input_size,
        box_score_thresh=score_thresh,
        box_detections_per_img=detections_per_img,
        image_mean=_IMAGENET_MEAN,
        image_std=_IMAGENET_STD,
    )

    # ---- 5. Bind Matrix NMS post-processing (improvement #5) ----------------
    # Stash the sigma on the RoIHeads so the bound method can read it.
    model.roi_heads.matrix_nms_sigma = matrix_nms_sigma
    model.roi_heads.postprocess_detections = types.MethodType(
        matrix_nms_postprocess, model.roi_heads
    )

    # ---- 6. Optional Focal-loss classification head (improvement #6) --------
    if use_focal_loss:
        model.roi_heads.focal_alpha = 0.75
        model.roi_heads.focal_gamma = 2.0
        model.roi_heads.forward = types.MethodType(
            _focal_roi_heads_forward, model.roi_heads
        )

    return model


if __name__ == "__main__":
    # Smoke build: instantiate and report the trainable parameter count.
    m = build_model(num_classes=7, input_size=480)
    n_params = sum(p.numel() for p in m.parameters())
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Improved Faster R-CNN built successfully.")
    print(f"Total parameters    : {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")
