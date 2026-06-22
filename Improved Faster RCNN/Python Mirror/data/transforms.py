"""Letterbox resize for the Improved Faster R-CNN data pipeline.

The raw steel/plastic-defect frames are 956x720. The detector expects a fixed
480x480x3 input. A naive ``resize(480, 480)`` would *stretch* the image
(956:720 = 1.328 aspect ratio -> 1:1), distorting every object and its
bounding box. **Letterbox** resizing instead:

  1. scales the image by a single factor ``s = min(480/W, 480/H)`` so the whole
     image fits inside 480x480 while preserving aspect ratio,
  2. pastes the scaled image onto a 480x480 gray canvas, centering it,
  3. applies the *same* affine map ``(x, y) -> (x*s + pad_left, y*s + pad_top)``
     to every bounding box so the annotations stay pixel-aligned.

For 956x720 -> 480x480 the math is:
    s = min(480/956, 480/720) = 480/956 = 0.50209...
    new_w = 480,  new_h = round(720*0.50209) = 362
    pad_left = (480-480)//2 = 0,  pad_top = (480-362)//2 = 59
so the content occupies rows 59..421 with 59px gray bars top and bottom.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from PIL import Image

#: Neutral gray used to fill the letterbox bars (matches the common YOLO value).
DEFAULT_PAD_VALUE = 114


def letterbox_resize(
    image: Image.Image,
    boxes: Optional[torch.Tensor] = None,
    target_size: int = 480,
    pad_value: int = DEFAULT_PAD_VALUE,
    resample: int = Image.BILINEAR,
) -> Tuple[Image.Image, Optional[torch.Tensor], dict]:
    """Aspect-ratio-preserving resize-with-padding to ``target_size`` square.

    Args:
        image:        a PIL RGB image of arbitrary size.
        boxes:        optional ``FloatTensor[N, 4]`` in ``xyxy`` *pixel*
                      coordinates of the original image. Transformed in lockstep.
        target_size:  output side length in pixels (default 480).
        pad_value:    gray level (0-255) used for the padding bars.
        resample:     PIL resampling filter for the image scale step.

    Returns:
        ``(canvas, new_boxes, meta)`` where

          * ``canvas``    is a ``target_size x target_size`` RGB PIL image,
          * ``new_boxes`` is the transformed ``FloatTensor[N, 4]`` (or ``None``
            if ``boxes`` was ``None``), clamped to the canvas,
          * ``meta``      = ``{"scale", "pad_left", "pad_top", "new_w", "new_h",
            "orig_size"}`` — enough to invert the mapping back to original
            coordinates if needed (see :func:`undo_letterbox_boxes`).
    """
    orig_w, orig_h = image.size
    scale = min(target_size / orig_w, target_size / orig_h)

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    # Guard against rounding pushing a dimension over the target.
    new_w = min(new_w, target_size)
    new_h = min(new_h, target_size)

    resized = image.resize((new_w, new_h), resample)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2

    canvas = Image.new("RGB", (target_size, target_size), (pad_value, pad_value, pad_value))
    canvas.paste(resized, (pad_left, pad_top))

    new_boxes = None
    if boxes is not None and len(boxes) > 0:
        new_boxes = boxes.clone().to(dtype=torch.float32)
        # Same affine map for both corners: scale then shift by the pad offset.
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]] * scale + pad_left
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]] * scale + pad_top
        # Clamp to the visible canvas (boxes never run into negative / overflow).
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clamp(0.0, float(target_size))
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clamp(0.0, float(target_size))
    elif boxes is not None:
        new_boxes = boxes.clone().to(dtype=torch.float32)

    meta = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "new_w": new_w,
        "new_h": new_h,
        "orig_size": (orig_w, orig_h),
    }
    return canvas, new_boxes, meta


def undo_letterbox_boxes(boxes: torch.Tensor, meta: dict) -> torch.Tensor:
    """Map boxes from letterboxed (480x480) coords back to original pixels.

    Useful at inference time to report detections in the source image's frame.
    Inverse of the affine map applied by :func:`letterbox_resize`.
    """
    out = boxes.clone().to(dtype=torch.float32)
    out[:, [0, 2]] = (out[:, [0, 2]] - meta["pad_left"]) / meta["scale"]
    out[:, [1, 3]] = (out[:, [1, 3]] - meta["pad_top"]) / meta["scale"]
    orig_w, orig_h = meta["orig_size"]
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, float(orig_w))
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, float(orig_h))
    return out
