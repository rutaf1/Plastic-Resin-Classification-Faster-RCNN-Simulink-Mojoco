"""ONNX-Runtime detection helper, callable from MATLAB (via ``pyrunfile``/``py.``).

The exported ``model_best.onnx`` (Improved Faster R-CNN) cannot be imported into
MATLAB as a network -- it contains GridSample/RoiAlign/NonMaxSuppression ops that
MATLAB's ONNX importer turns into placeholder layers. The robust route is to run
the graph with ONNX Runtime and call THAT from MATLAB. This module does exactly
that, reusing the project's *exact* letterbox preprocessing so results match the
PyTorch model.

From MATLAB (after `pyenv` points at this Python):
    res = py.onnx_detect.detect("frame.png", "model_best.onnx", 0.5);
    boxes  = double(res{'boxes'});    % [N x 4]  xyxy in ORIGINAL image pixels
    labels = double(res{'labels'});   % [N x 1]  class id 1..7
    scores = double(res{'scores'});   % [N x 1]
    names  = cell(res{'names'});      % {N}      class name strings

Standalone test:
    python onnx_detect.py --image ../../dataset/images_aug/frame_000081.png \
        --onnx ../outputs/model_best.onnx --score-thresh 0.5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

# Make the project package importable (this file lives in python_alternatif/matlab/).
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from data.transforms import letterbox_resize  # noqa: E402  (path set above)

# Class names (index = id-1). Kept inline so MATLAB needs no extra import.
CLASS_NAMES = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]

# Cache InferenceSession per ONNX path so repeated MATLAB calls (e.g. one per
# frame) don't reload the ~170 MB model every time.
_SESSIONS = {}


def _get_session(onnx_path):
    import onnxruntime as ort
    key = os.path.abspath(str(onnx_path))
    if key not in _SESSIONS:
        _SESSIONS[key] = ort.InferenceSession(key, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def _preprocess(image_path, input_size=480):
    """Letterbox to SxS, return (input[1,3,S,S] float32 RGB [0,1], meta, orig_size)."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    canvas, _, meta = letterbox_resize(img, None, input_size)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0      # HWC RGB [0,1]
    arr = np.transpose(arr, (2, 0, 1))[None]                # [1,3,S,S]
    return np.ascontiguousarray(arr), meta, (orig_w, orig_h)


def _undo_letterbox(boxes, meta, orig_size):
    """Map xyxy boxes from letterboxed SxS coords back to original pixels (numpy)."""
    if boxes.size == 0:
        return boxes
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - meta["pad_left"]) / meta["scale"]
    out[:, [1, 3]] = (out[:, [1, 3]] - meta["pad_top"]) / meta["scale"]
    w, h = orig_size
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out


def detect(image_path, onnx_path, score_thresh=0.5, input_size=480):
    """Run the ONNX detector on one image; return detections in ORIGINAL pixels.

    Returns a dict with numpy arrays: 'boxes' [N,4] xyxy float32, 'labels' [N]
    int64 (1..7), 'scores' [N] float32, and 'names' (list[str]). MATLAB reads
    these via double(res{'boxes'}) etc.
    """
    score_thresh = float(score_thresh)
    inp, meta, orig_size = _preprocess(image_path, int(input_size))

    sess = _get_session(onnx_path)   # cached across calls
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    feed = inp if len(ishape) == 4 else inp[0]   # graph may want [3,S,S] or [1,3,S,S]
    outs = sess.run(None, {iname: feed})
    out = {o.name: outs[i] for i, o in enumerate(sess.get_outputs())}

    boxes = np.asarray(out.get("boxes", outs[0])).reshape(-1, 4)
    labels = np.asarray(out.get("labels", outs[1])).reshape(-1).astype(np.int64)
    scores = np.asarray(out.get("scores", outs[2])).reshape(-1).astype(np.float32)

    keep = scores >= score_thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    boxes = _undo_letterbox(boxes, meta, orig_size)
    names = [CLASS_NAMES[int(l) - 1] if 1 <= int(l) <= len(CLASS_NAMES) else f"cls{int(l)}"
             for l in labels]

    return {"boxes": boxes.astype(np.float32), "labels": labels,
            "scores": scores, "names": names}


def main():
    p = argparse.ArgumentParser(description="ONNX-Runtime detection (MATLAB helper / standalone test).")
    p.add_argument("--image", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--input-size", type=int, default=480)
    args = p.parse_args()

    res = detect(args.image, args.onnx, args.score_thresh, args.input_size)
    n = len(res["scores"])
    print(f"{n} detection(s) >= {args.score_thresh}:")
    for i in range(n):
        x1, y1, x2, y2 = res["boxes"][i]
        print(f"  {res['names'][i]:<8} id={int(res['labels'][i])} score={res['scores'][i]:.3f} "
              f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")


if __name__ == "__main__":
    main()
