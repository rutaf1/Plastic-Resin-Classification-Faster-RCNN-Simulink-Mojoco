"""Metrics, logging, and live plotting for Improved Faster R-CNN training.

Object detection has no single native "accuracy"/"RMSE", so we define them
sensibly and document them (the user asked for accuracy / RMSE / loss on both
train and validation):

* **Loss**            -- the torchvision detection loss (sum of
                         loss_classifier + loss_box_reg + loss_objectness +
                         loss_rpn_box_reg). Reported for train (running mean)
                         and validation (computed with BN frozen).
* **Accuracy**        -- detection accuracy at IoU >= 0.5 with score >= a
                         threshold: ``TP / (TP + FP + FN)`` after greedy,
                         class-aware matching of predictions to ground-truth.
                         (Precision, Recall and F1 are reported alongside.)
* **RMSE**            -- localization RMSE in pixels: root-mean-square error of
                         the 4 box coordinates [x1,y1,x2,y2] over all matched
                         true-positive pairs. Measures box-regression quality.
* **mAP**             -- COCO mean Average Precision (torchmetrics), the metric
                         the journal headlines. mAP@[.50:.95] and mAP@.50.

All box coords are in the 480x480 letterboxed space.
"""

from __future__ import annotations

import csv
import json
import os
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
from torchvision.ops import box_iou


# --------------------------------------------------------------------------- #
# Detection matching -> TP / FP / FN + squared localization error
# --------------------------------------------------------------------------- #
class DetStatAccumulator:
    """Accumulates TP/FP/FN and squared box error across images.

    Greedy, class-aware matching at a fixed IoU threshold; a prediction matches
    the highest-IoU unused GT of the same class (IoU >= ``iou_thresh``).
    """

    def __init__(self, iou_thresh: float = 0.5, score_thresh: float = 0.5):
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.sq_err = 0.0   # sum of squared coordinate errors over TP pairs
        self.n_coord = 0    # number of coordinates summed (= 4 * #TP)

    @torch.no_grad()
    def update(self, preds: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]):
        for pred, tgt in zip(preds, targets):
            self._update_one(pred, tgt)

    def _update_one(self, pred, tgt):
        p_boxes = pred["boxes"].detach().cpu()
        p_scores = pred["scores"].detach().cpu()
        p_labels = pred["labels"].detach().cpu()
        keep = p_scores >= self.score_thresh
        p_boxes, p_scores, p_labels = p_boxes[keep], p_scores[keep], p_labels[keep]

        g_boxes = tgt["boxes"].detach().cpu()
        g_labels = tgt["labels"].detach().cpu()

        n_g = g_boxes.shape[0]
        if p_boxes.shape[0] == 0:
            self.fn += n_g
            return
        if n_g == 0:
            self.fp += p_boxes.shape[0]
            return

        # Process predictions in descending score order (standard detection eval).
        order = torch.argsort(p_scores, descending=True)
        ious = box_iou(p_boxes, g_boxes)  # [P, G]
        gt_used = torch.zeros(n_g, dtype=torch.bool)

        for pi in order.tolist():
            same = (g_labels == p_labels[pi]) & (~gt_used)
            if same.any():
                iou_row = ious[pi].clone()
                iou_row[~same] = -1.0
                best_iou, best_g = torch.max(iou_row, dim=0)
                if best_iou.item() >= self.iou_thresh:
                    self.tp += 1
                    gt_used[best_g] = True
                    err = p_boxes[pi] - g_boxes[best_g]
                    self.sq_err += float((err ** 2).sum())
                    self.n_coord += 4
                    continue
            self.fp += 1
        self.fn += int((~gt_used).sum())

    def compute(self) -> Dict[str, float]:
        eps = 1e-9
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = tp / (tp + fp + fn + eps)   # detection accuracy (Jaccard-style)
        rmse = (self.sq_err / self.n_coord) ** 0.5 if self.n_coord else float("nan")
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "rmse": rmse,
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }


# --------------------------------------------------------------------------- #
# Validation loss with BatchNorm frozen (so val forward doesn't pollute BN)
# --------------------------------------------------------------------------- #
@contextmanager
def _train_mode_frozen_bn(model):
    """Put model in train() (so it returns the loss dict) but freeze all BN.

    BatchNorm running stats update during the forward pass regardless of
    ``torch.no_grad``; setting BN modules to eval() makes them use (and not
    update) their running statistics, so computing a validation loss leaves the
    model's BN state untouched.
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    was_training = model.training
    model.train()
    bn_prev = []
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            bn_prev.append((m, m.training))
            m.eval()
    try:
        yield
    finally:
        for m, prev in bn_prev:
            m.train(prev)
        model.train(was_training)


@torch.no_grad()
def compute_loss_over_loader(model, loader, device, max_batches: int = 0, progress=None) -> Dict[str, float]:
    """Mean detection loss (+ components) over a loader, with BN frozen."""
    totals: Dict[str, float] = {}
    n = 0
    with _train_mode_frozen_bn(model):
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())
            totals["loss"] = totals.get("loss", 0.0) + float(sum(loss_dict.values()).detach())
            n += 1
            if progress is not None:
                progress.update(1)
            if max_batches and n >= max_batches:
                break
    return {k: v / max(n, 1) for k, v in totals.items()}


# --------------------------------------------------------------------------- #
# Full evaluation: detection metrics + (optional) torchmetrics mAP
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_detection(model, loader, device, iou_thresh=0.5, score_thresh=0.5,
                       max_batches: int = 0, with_map: bool = True, progress=None) -> Dict[str, float]:
    """Run eval-mode inference and compute accuracy/RMSE/P/R/F1 (+ optional mAP)."""
    model.eval()
    acc = DetStatAccumulator(iou_thresh=iou_thresh, score_thresh=score_thresh)

    metric_map = None
    if with_map:
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            metric_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        except Exception:
            metric_map = None

    n = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds = model(images)
        preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        tgts_cpu = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        acc.update(preds_cpu, tgts_cpu)
        if metric_map is not None:
            metric_map.update(preds_cpu, [{"boxes": t["boxes"], "labels": t["labels"]} for t in tgts_cpu])
        n += 1
        if progress is not None:
            progress.update(1)
        if max_batches and n >= max_batches:
            break

    out = acc.compute()
    if metric_map is not None:
        res = metric_map.compute()
        out["map"] = float(res.get("map", float("nan")))
        out["map_50"] = float(res.get("map_50", float("nan")))
        out["map_75"] = float(res.get("map_75", float("nan")))
    else:
        out["map"] = out["map_50"] = out["map_75"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# CSV/JSON logger + live matplotlib plots
# --------------------------------------------------------------------------- #
class TrainingLogger:
    """Append per-epoch metrics to CSV + JSON and (re)draw training curves PNG."""

    FIELDS = [
        "epoch", "lr", "time_s",
        "train_loss", "val_loss",
        "train_acc", "val_acc",
        "train_rmse", "val_rmse",
        "train_map", "val_map", "val_map50",
        "val_precision", "val_recall", "val_f1",
    ]

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "metrics_log.csv")
        self.json_path = os.path.join(out_dir, "metrics_log.json")
        self.png_path = os.path.join(out_dir, "training_curves.png")
        self.history: List[Dict[str, float]] = []
        # Fresh CSV header.
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, row: Dict[str, float]):
        clean = {k: row.get(k, "") for k in self.FIELDS}
        self.history.append({k: row.get(k, float("nan")) for k in self.FIELDS})
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(clean)
        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def plot(self):
        """Render train/val curves to a PNG (safe to call every epoch)."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless backend
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"   [plot] matplotlib unavailable ({e}); skipping curves.")
            return
        if not self.history:
            return

        ep = [h["epoch"] for h in self.history]

        def col(name):
            return [h.get(name, float("nan")) for h in self.history]

        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Improved Faster R-CNN -- Training Curves", fontsize=14, fontweight="bold")

        ax[0, 0].plot(ep, col("train_loss"), "-o", ms=3, label="train")
        ax[0, 0].plot(ep, col("val_loss"), "-o", ms=3, label="val")
        ax[0, 0].set_title("Loss"); ax[0, 0].set_xlabel("epoch"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

        ax[0, 1].plot(ep, col("train_acc"), "-o", ms=3, label="train")
        ax[0, 1].plot(ep, col("val_acc"), "-o", ms=3, label="val")
        ax[0, 1].set_title("Accuracy (IoU>=0.5)"); ax[0, 1].set_xlabel("epoch"); ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

        ax[0, 2].plot(ep, col("train_rmse"), "-o", ms=3, label="train")
        ax[0, 2].plot(ep, col("val_rmse"), "-o", ms=3, label="val")
        ax[0, 2].set_title("Localization RMSE (px)"); ax[0, 2].set_xlabel("epoch"); ax[0, 2].legend(); ax[0, 2].grid(alpha=0.3)

        ax[1, 0].plot(ep, col("val_map"), "-o", ms=3, label="mAP@[.5:.95]")
        ax[1, 0].plot(ep, col("val_map50"), "-o", ms=3, label="mAP@.5")
        ax[1, 0].set_title("Validation mAP"); ax[1, 0].set_xlabel("epoch"); ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

        ax[1, 1].plot(ep, col("val_precision"), "-o", ms=3, label="precision")
        ax[1, 1].plot(ep, col("val_recall"), "-o", ms=3, label="recall")
        ax[1, 1].plot(ep, col("val_f1"), "-o", ms=3, label="F1")
        ax[1, 1].set_title("Validation P / R / F1"); ax[1, 1].set_xlabel("epoch"); ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

        ax[1, 2].plot(ep, col("lr"), "-o", ms=3, color="tab:purple")
        ax[1, 2].set_title("Learning Rate"); ax[1, 2].set_xlabel("epoch"); ax[1, 2].grid(alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(self.png_path, dpi=110)
        plt.close(fig)
