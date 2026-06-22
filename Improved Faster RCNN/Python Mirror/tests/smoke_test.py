"""Self-contained smoke test for the Improved Faster R-CNN (Wang et al., Metals 2021).

Runs end-to-end on CPU with tiny random data and a fixed seed. It exercises:

  1. ``build_model(num_classes=7)``                       -> constructs the detector.
  2. TRAIN mode : a forward pass with targets returns a dict of scalar loss
     tensors; the summed loss back-propagates and produces at least one gradient.
  3. EVAL  mode : a forward pass without targets returns a list of detection
     dicts with the expected ``boxes``/``labels``/``scores`` keys, shapes, and
     dtypes.

Run it from either the repository root or the ``python_alternatif`` directory::

    python tests/smoke_test.py

A ``sys.path`` shim makes the ``models`` / ``config`` / ``data`` packages
importable regardless of the launch directory.
"""

import os
import sys

# --- Import shim -----------------------------------------------------------
# Make the ``python_alternatif`` package directory importable whether this
# script is launched from the repo root or from inside ``python_alternatif``.
# This file lives at  <python_alternatif>/tests/smoke_test.py, so the package
# root is the parent of the directory that contains this file.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)  # .../python_alternatif
for _p in (_PKG_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from models.faster_rcnn import build_model


def _make_dummy_target(num_boxes, input_size, num_classes, generator):
    """Create one valid target dict (boxes [N,4] float32, labels [N] int64).

    Boxes are random but guaranteed to be well-formed (x1<x2, y1<y2) and to lie
    inside the ``input_size`` square. Labels are foreground classes in
    ``[1, num_classes]`` (index 0 is reserved for background).
    """
    # Random top-left corners with room for a positive-area box.
    x1 = torch.randint(0, input_size - 20, (num_boxes, 1), generator=generator).float()
    y1 = torch.randint(0, input_size - 20, (num_boxes, 1), generator=generator).float()
    # Random positive widths/heights that stay within the image bounds.
    w = torch.randint(10, 20, (num_boxes, 1), generator=generator).float()
    h = torch.randint(10, 20, (num_boxes, 1), generator=generator).float()
    x2 = torch.clamp(x1 + w, max=float(input_size))
    y2 = torch.clamp(y1 + h, max=float(input_size))
    boxes = torch.cat([x1, y1, x2, y2], dim=1).to(torch.float32)
    # Foreground labels in [1, num_classes].
    labels = torch.randint(1, num_classes + 1, (num_boxes,), generator=generator).to(torch.int64)
    return {"boxes": boxes, "labels": labels}


def main():
    # ---- Reproducibility & CPU ------------------------------------------
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    device = torch.device("cpu")

    num_classes = 7
    input_size = 480

    # ---- 1. Build the model ---------------------------------------------
    print("[1/3] Building model: build_model(num_classes=7) ...")
    model = build_model(num_classes=num_classes, input_size=input_size)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Model built. Total parameters: {n_params:,}")

    # ---- 2. TRAIN mode: loss dict + backward ----------------------------
    print("[2/3] TRAIN mode forward + backward ...")
    model.train()

    images = [
        torch.rand(3, input_size, input_size, generator=gen),
        torch.rand(3, input_size, input_size, generator=gen),
    ]
    targets = [
        _make_dummy_target(num_boxes=3, input_size=input_size, num_classes=num_classes, generator=gen),
        _make_dummy_target(num_boxes=2, input_size=input_size, num_classes=num_classes, generator=gen),
    ]

    loss_dict = model(images, targets)

    # Assert: a dict of scalar tensors.
    assert isinstance(loss_dict, dict), f"expected dict of losses, got {type(loss_dict)}"
    assert len(loss_dict) > 0, "loss dict is empty"
    for k, v in loss_dict.items():
        assert isinstance(v, torch.Tensor), f"loss '{k}' is not a Tensor (got {type(v)})"
        assert v.dim() == 0, f"loss '{k}' is not a scalar tensor (shape {tuple(v.shape)})"
        assert torch.isfinite(v).all(), f"loss '{k}' is not finite (value {v.item()})"
        print(f"      loss[{k}] = {v.item():.6f}")

    # Sum the losses and back-propagate.
    total_loss = sum(loss for loss in loss_dict.values())
    assert isinstance(total_loss, torch.Tensor) and total_loss.dim() == 0, "summed loss is not a scalar tensor"
    print(f"      total_loss = {total_loss.item():.6f}")

    model.zero_grad(set_to_none=True)
    total_loss.backward()

    # Assert: at least one parameter received a gradient.
    grad_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grad_params) > 0, "no parameter received a gradient after backward()"
    num_nonzero = sum(1 for p in grad_params if torch.any(p.grad != 0))
    print(f"      backward(): {len(grad_params)} params got grads, {num_nonzero} have non-zero grads.")
    assert num_nonzero > 0, "all gradients are zero after backward()"

    # ---- 3. EVAL mode: detections ---------------------------------------
    print("[3/3] EVAL mode inference ...")
    model.eval()
    dummy = torch.rand(3, input_size, input_size, generator=gen)
    with torch.no_grad():
        out = model([dummy])

    # Assert: list of dicts with the expected keys / shapes / dtypes.
    assert isinstance(out, list), f"eval output is not a list (got {type(out)})"
    assert len(out) == 1, f"expected 1 detection dict (one input image), got {len(out)}"
    det = out[0]
    assert isinstance(det, dict), f"detection entry is not a dict (got {type(det)})"
    for key in ("boxes", "labels", "scores"):
        assert key in det, f"detection dict missing key '{key}'"
        assert isinstance(det[key], torch.Tensor), f"detection['{key}'] is not a Tensor"

    boxes, labels, scores = det["boxes"], det["labels"], det["scores"]
    n_det = boxes.shape[0]

    # Shape relationships.
    assert boxes.dim() == 2 and boxes.shape[1] == 4, f"boxes shape must be [N,4], got {tuple(boxes.shape)}"
    assert labels.shape == (n_det,), f"labels shape must be [{n_det}], got {tuple(labels.shape)}"
    assert scores.shape == (n_det,), f"scores shape must be [{n_det}], got {tuple(scores.shape)}"

    # Dtypes.
    assert boxes.dtype == torch.float32, f"boxes dtype must be float32, got {boxes.dtype}"
    assert labels.dtype == torch.int64, f"labels dtype must be int64, got {labels.dtype}"
    assert scores.dtype == torch.float32, f"scores dtype must be float32, got {scores.dtype}"

    # Value sanity (only meaningful if any detections survived the score filter).
    if n_det > 0:
        assert torch.all((labels >= 1) & (labels <= num_classes)), "labels outside [1, num_classes]"
        assert torch.all((scores >= 0.0) & (scores <= 1.0)), "scores outside [0, 1]"

    print(f"      Detections returned: {n_det}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
