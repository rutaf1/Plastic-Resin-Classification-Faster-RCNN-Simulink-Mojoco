"""Inference entry point for the Improved Faster R-CNN (Wang et al., *Metals* 2021).

Loads a trained checkpoint (or a freshly-initialized model when none is given),
runs the detector on a single RGB image (a path supplied via ``--image`` or, if
omitted, a synthetic random 480x480 image), and reports the predicted boxes,
class labels and confidence scores. Detections below ``--score-thresh`` are
filtered out. Optionally, an annotated copy of the image is written to disk with
:func:`torchvision.utils.draw_bounding_boxes`.

This is one of the two scripts (alongside ``train.py``) allowed to read
``config.py``; the model-building code under ``models/`` stays config-agnostic
and receives every hyperparameter as an explicit argument. Class labels are
resolved through ``CLASS_NAMES`` from :mod:`config`.

Runs on CPU (no CUDA required). Example invocations::

    python inference.py                                  # synthetic random image
    python inference.py --image sample.jpg               # real image, fresh model
    python inference.py --image sample.jpg \\
        --checkpoint outputs/model.pth --output annotated.png --score-thresh 0.3
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

# ``config`` and ``models`` live in this same ``python_alternatif`` package.
# Support both ``python inference.py`` (run as a top-level script, so the
# package directory is on sys.path and absolute imports resolve) and
# ``python -m python_alternatif.inference`` (run as a module, relative imports).
try:  # pragma: no cover - import-style shim, exercised at runtime only
    from config import CONFIG, Config
    from models.faster_rcnn import build_model
except ImportError:  # pragma: no cover
    from .config import CONFIG, Config
    from .models.faster_rcnn import build_model


def load_image_tensor(image_path: Optional[str], input_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load (or synthesize) an image and return it in two complementary forms.

    The detector consumes a float tensor in ``[0, 1]`` of shape ``[3, H, W]``;
    annotation/drawing wants the same picture as a ``uint8`` tensor in
    ``[0, 255]``. We produce both from one source so the boxes the model predicts
    line up exactly with the canvas we draw on.

    Args:
        image_path: Path to an RGB image on disk. When ``None`` (or missing), a
            synthetic random ``input_size`` x ``input_size`` image is generated
            instead -- handy for a dependency-free smoke run.
        input_size: Fixed square side length (pixels) the image is resized to,
            matching the model's ``min_size == max_size`` resize (journal spec:
            480x480x3). Resizing here is cosmetic for drawing; the model resizes
            internally regardless.

    Returns:
        ``(float_tensor, uint8_tensor)`` where ``float_tensor`` is
        ``[3, H, W]`` in ``[0, 1]`` (model input) and ``uint8_tensor`` is
        ``[3, H, W]`` in ``[0, 255]`` (drawing canvas).
    """
    if image_path is not None and os.path.isfile(image_path):
        pil_img = Image.open(image_path).convert("RGB")
        # Letterbox (aspect-ratio-preserving + gray pad) to match training -- a
        # plain resize would STRETCH non-square frames (e.g. 956x720) and shift
        # every box. Detections come back in this 480x480 letterboxed frame, so
        # we draw on the same canvas and they line up exactly.
        canvas, _, _meta = _letterbox(pil_img, input_size)
        uint8 = torch.from_numpy(_pil_to_uint8_array(canvas))  # [H, W, 3]
        uint8 = uint8.permute(2, 0, 1).contiguous()  # [3, H, W]
    else:
        if image_path is not None:
            print(f"[inference] image '{image_path}' not found; using a synthetic random image.")
        else:
            print("[inference] no --image given; using a synthetic random image.")
        # Reproducible-ish random RGB canvas in [0, 255].
        uint8 = (torch.rand(3, input_size, input_size) * 255).to(torch.uint8)

    float_tensor = uint8.float() / 255.0
    return float_tensor, uint8


def _letterbox(pil_img: "Image.Image", input_size: int):
    """Robustly import and call the shared letterbox transform.

    Mirrors the top-of-file import shim so inference works whether run as a
    top-level script or as a package module.
    """
    try:
        from data.transforms import letterbox_resize
    except ImportError:  # pragma: no cover - package-module style
        from .data.transforms import letterbox_resize
    return letterbox_resize(pil_img, None, input_size)


def _pil_to_uint8_array(pil_img: "Image.Image"):
    """Convert a PIL RGB image to an ``[H, W, 3]`` uint8 numpy array.

    Isolated so the (only) numpy dependency is contained and the rest of the
    module stays free of numpy-specific calls.
    """
    import numpy as np

    # ``np.array`` (not ``asarray``) forces a writable copy, avoiding torch's
    # "non-writable tensor" warning when this is later wrapped via from_numpy.
    return np.array(pil_img, dtype="uint8")


@torch.no_grad()
def detect(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    score_thresh: float,
) -> Dict[str, torch.Tensor]:
    """Run the detector on one image and return score-filtered detections.

    Puts the model in eval mode, performs a single forward pass under
    ``torch.no_grad`` (the post-processing already runs Matrix NMS -- journal
    improvement #5 -- inside ``model.roi_heads``), then keeps only the
    detections whose score strictly exceeds ``score_thresh``.

    Args:
        model: A built Improved Faster R-CNN (``build_model(...)`` output).
        image_tensor: A single image as a float tensor ``[3, H, W]`` in
            ``[0, 1]``. (A leading batch dim is also accepted and squeezed.)
        score_thresh: Minimum confidence; detections at or below it are dropped.

    Returns:
        Dict with keys ``"boxes"`` (``[K, 4]`` xyxy, float), ``"labels"``
        (``[K]`` int64, 1-indexed foreground labels), and ``"scores"``
        (``[K]`` float), all on CPU.
    """
    model.eval()

    # Accept either [3, H, W] or [1, 3, H, W]; the detector wants a list of CHW.
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor.to(_model_device(model))

    outputs: List[Dict[str, torch.Tensor]] = model([image_tensor])
    out = outputs[0]

    boxes = out.get("boxes", torch.empty((0, 4)))
    labels = out.get("labels", torch.empty((0,), dtype=torch.int64))
    scores = out.get("scores", torch.empty((0,)))

    keep = scores > score_thresh
    return {
        "boxes": boxes[keep].cpu(),
        "labels": labels[keep].cpu(),
        "scores": scores[keep].cpu(),
    }


def _model_device(model: torch.nn.Module) -> torch.device:
    """Best-effort lookup of the device a model's parameters live on."""
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - models always have parameters here
        return torch.device("cpu")


def label_name(label_idx: int, class_names: List[str]) -> str:
    """Map a 1-indexed detector label to its human-readable class name.

    The torchvision head reserves index 0 for background, so foreground labels
    are 1..NUM_CLASSES and index into ``class_names`` as ``label_idx - 1``.
    Out-of-range labels degrade gracefully to a generic ``"cls<idx>"`` tag.
    """
    fg_index = int(label_idx) - 1
    if 0 <= fg_index < len(class_names):
        return class_names[fg_index]
    return f"cls{int(label_idx)}"


def print_detections(
    detections: Dict[str, torch.Tensor],
    class_names: List[str],
) -> None:
    """Pretty-print the detections to stdout, one row per box."""
    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]

    n = boxes.shape[0]
    print(f"[inference] {n} detection(s) above threshold:")
    if n == 0:
        return
    print(f"  {'#':>3} {'label':<18} {'score':>7}  {'x1':>7} {'y1':>7} {'x2':>7} {'y2':>7}")
    for i in range(n):
        name = label_name(int(labels[i]), class_names)
        x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
        print(
            f"  {i:>3} {name:<18} {float(scores[i]):>7.3f}  "
            f"{x1:>7.1f} {y1:>7.1f} {x2:>7.1f} {y2:>7.1f}"
        )


def annotate_and_save(
    image_uint8: torch.Tensor,
    detections: Dict[str, torch.Tensor],
    class_names: List[str],
    output_path: str,
) -> None:
    """Draw boxes + ``name score`` captions onto the image and save it.

    Uses :func:`torchvision.utils.draw_bounding_boxes` on the uint8 canvas, then
    writes the result via Pillow. Gracefully no-ops the drawing when there are no
    detections (still saves the bare image so the output always exists).
    """
    from torchvision.utils import draw_bounding_boxes

    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]

    if boxes.shape[0] > 0:
        captions = [
            f"{label_name(int(labels[i]), class_names)} {float(scores[i]):.2f}"
            for i in range(boxes.shape[0])
        ]
        drawn = draw_bounding_boxes(
            image_uint8,
            boxes,
            labels=captions,
            colors="red",
            width=2,
        )
    else:
        drawn = image_uint8

    # Convert [3, H, W] uint8 -> PIL and save.
    pil_out = Image.fromarray(drawn.permute(1, 2, 0).contiguous().numpy())
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pil_out.save(output_path)
    print(f"[inference] annotated image saved to: {output_path}")


def load_model(checkpoint: Optional[str], cfg: Config) -> torch.nn.Module:
    """Build the model from ``cfg`` and optionally load a checkpoint.

    Builds the Improved Faster R-CNN with the config's anchor/score/NMS settings
    (so inference matches training), then -- if ``checkpoint`` points to a real
    file -- restores weights. The checkpoint may be either a raw ``state_dict``
    or a training dict containing a ``"model"`` / ``"model_state_dict"`` /
    ``"state_dict"`` entry. With no checkpoint, a freshly-initialized (random)
    model is returned, which still runs end-to-end for a smoke test.

    Args:
        checkpoint: Path to a ``.pth`` checkpoint, or ``None``.
        cfg: Resolved :class:`config.Config` instance.

    Returns:
        The model on ``cfg.DEVICE``, ready for :func:`detect`.
    """
    model = build_model(
        num_classes=cfg.NUM_CLASSES,
        input_size=cfg.INPUT_SIZE,
        anchor_sizes=cfg.ANCHOR_SIZES,
        anchor_ratios=cfg.ANCHOR_RATIOS,
        score_thresh=cfg.SCORE_THRESH,
        matrix_nms_sigma=cfg.MATRIX_NMS_SIGMA,
        detections_per_img=cfg.DETECTIONS_PER_IMG,
        pretrained_backbone=False,
        use_focal_loss=False,
    )

    if checkpoint is not None and os.path.isfile(checkpoint):
        # ``weights_only=True`` is the safe default in modern torch; fall back for
        # older builds that don't support the kwarg.
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch without weights_only
            ckpt = torch.load(checkpoint, map_location="cpu")

        state_dict = _extract_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[inference] loaded checkpoint: {checkpoint}")
        if missing:
            print(f"[inference]   {len(missing)} missing key(s) (left at init).")
        if unexpected:
            print(f"[inference]   {len(unexpected)} unexpected key(s) ignored.")
    else:
        if checkpoint is not None:
            print(f"[inference] checkpoint '{checkpoint}' not found; using a fresh (random) model.")
        else:
            print("[inference] no --checkpoint given; using a fresh (random) model.")

    model.to(cfg.DEVICE)
    return model


def _extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    """Return the parameter ``state_dict`` from a checkpoint object.

    Accepts either a bare ``state_dict`` (mapping str -> tensor) or a wrapper
    dict that nests it under a conventional key.
    """
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            inner = ckpt.get(key)
            if isinstance(inner, dict):
                return inner
    return ckpt


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the inference script."""
    parser = argparse.ArgumentParser(
        description="Run the Improved Faster R-CNN on a single image (CPU-friendly).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an RGB input image. If omitted/missing, a synthetic random image is used.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pth checkpoint. If omitted/missing, a fresh (random) model is used.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save an annotated image. If omitted, no image is written.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=None,
        help="Minimum detection score to keep (default: Config.SCORE_THRESH).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Script entry point: load model + image, detect, print, optionally save."""
    args = parse_args(argv)

    cfg = CONFIG
    score_thresh = args.score_thresh if args.score_thresh is not None else cfg.SCORE_THRESH

    model = load_model(args.checkpoint, cfg)
    float_tensor, uint8_tensor = load_image_tensor(args.image, cfg.INPUT_SIZE)

    detections = detect(model, float_tensor, score_thresh)
    print_detections(detections, cfg.CLASS_NAMES)

    if args.output is not None:
        annotate_and_save(uint8_tensor, detections, cfg.CLASS_NAMES, args.output)


if __name__ == "__main__":
    main()
