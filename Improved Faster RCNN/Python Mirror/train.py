"""Training + validation entry point for the Improved Faster R-CNN.

Reimplementation of Wang et al., *Metals* 2021, applied to the plastic-defect
dataset (CVAT ``annotations_aug.xml`` + ``images_aug/``, 956x720 frames, 7
classes). Frames are **letterbox-resized to 480x480** with boxes transformed in
lockstep (see :mod:`data.cvat_dataset` / :mod:`data.transforms`).

Pipeline wired here:
  * :mod:`config`                         -- hyperparameters,
  * :func:`models.faster_rcnn.build_model`-- the Improved Faster R-CNN,
  * :func:`data.cvat_dataset.build_train_val_datasets`
                                           -- leakage-free train/val split
                                              (augmentations grouped with their
                                              base frame).

Each epoch shows a tqdm progress bar and then logs, for BOTH train and
validation: loss, accuracy (IoU>=0.5), localization RMSE (px), mAP, and
precision/recall/F1 (see :mod:`metrics`). Metrics are appended to
``outputs/metrics_log.csv`` + ``.json`` and re-plotted to
``outputs/training_curves.png`` every epoch. The optimizer/schedule follow the
journal: SGD(lr, momentum, weight_decay), linear warm-up, multi-step LR drops.

Run (from the python_alternatif dir):
    python train.py                                  # CVAT data, config defaults (batch 16)
    python train.py --batch-size 16 --eval-every 5   # faster: validate every 5 epochs
    python train.py --dataset synthetic --epochs 1   # no data needed (pipeline test)
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from config import Config
from models.faster_rcnn import build_model
from metrics import TrainingLogger, evaluate_detection, compute_loss_over_loader

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - graceful fallback if tqdm missing
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _NullBar()

    class _NullBar:
        def update(self, *a, **k): pass
        def set_postfix(self, *a, **k): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _str2bool(value):
    """Parse a permissive boolean for argparse (e.g. ``--use-focal-loss true``)."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args():
    """Command-line overrides for the most commonly tuned training knobs."""
    p = argparse.ArgumentParser(
        description="Train the Improved Faster R-CNN on the plastic-defect dataset."
    )
    p.add_argument("--dataset", choices=["cvat", "synthetic"], default="cvat",
                   help="'cvat' uses the real letterboxed dataset (default); "
                        "'synthetic' uses random data to smoke-test the loop.")
    p.add_argument("--epochs", type=int, default=None, help="epochs (default: Config.EPOCHS).")
    p.add_argument("--batch-size", type=int, default=None, help="images/batch (default: Config.BATCH_SIZE).")
    p.add_argument("--data-root", type=str, default=None,
                   help="dataset root holding images_aug/ + annotations_aug.xml (default: Config.DATA_ROOT).")
    p.add_argument("--val-fraction", type=float, default=0.2, help="fraction of base frames for validation.")
    p.add_argument("--seed", type=int, default=42, help="split + init seed.")
    p.add_argument("--use-focal-loss", type=_str2bool, nargs="?", const=True, default=False,
                   help="swap the RoI classification loss for Focal Loss (journal imp. #6).")
    p.add_argument("--eval-every", type=int, default=1, help="run validation every N epochs (0 disables).")
    p.add_argument("--eval-max-batches", type=int, default=0,
                   help="cap validation batches for a quick estimate (0 = full val set).")
    p.add_argument("--max-train-batches", type=int, default=0,
                   help="cap training iterations per epoch (0 = full epoch). Useful for quick runs.")
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: Config.NUM_WORKERS).")
    p.add_argument("--no-export-onnx", action="store_true",
                   help="skip exporting the best checkpoint to ONNX when training finishes.")
    p.add_argument("--onnx-nms", choices=["standard", "matrix"], default="standard",
                   help="NMS baked into the exported ONNX graph (default: standard = deployment-safe).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_loaders(cfg, args):
    """Build (train_loader, val_loader, collate) for the chosen dataset."""
    batch_size = args.batch_size if args.batch_size is not None else cfg.BATCH_SIZE
    workers = args.workers if args.workers is not None else cfg.NUM_WORKERS

    if args.dataset == "synthetic":
        from data.dataset import SyntheticDefectDataset, collate_fn
        train_ds = SyntheticDefectDataset(length=32, image_size=cfg.INPUT_SIZE, num_classes=cfg.NUM_CLASSES)
        val_ds = SyntheticDefectDataset(length=8, image_size=cfg.INPUT_SIZE, num_classes=cfg.NUM_CLASSES)
    else:
        from data.cvat_dataset import build_train_val_datasets, collate_fn
        root = args.data_root or cfg.DATA_ROOT
        image_dir = os.path.join(root, "images_aug")
        xml_path = os.path.join(root, "annotations_aug.xml")
        if not os.path.isdir(image_dir) or not os.path.isfile(xml_path):
            raise FileNotFoundError(
                f"Expected '{image_dir}' and '{xml_path}'. Pass --data-root to point at your dataset, "
                f"or use --dataset synthetic to test the pipeline without data."
            )
        train_ds, val_ds = build_train_val_datasets(
            image_dir, xml_path, input_size=cfg.INPUT_SIZE,
            val_fraction=args.val_fraction, seed=args.seed,
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, collate_fn=collate_fn, drop_last=False)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# LR schedule (journal: linear warm-up + step drop)
# --------------------------------------------------------------------------- #
def make_warmup_step_scheduler(optimizer, warmup_iters, warmup_factor,
                               drop_epochs, iters_per_epoch, gamma=0.1):
    """Per-iteration LambdaLR: linear warm-up then multi-step x``gamma`` drops.

    ``drop_epochs`` is an iterable of (0-based) epoch boundaries at which the LR
    is multiplied by ``gamma``. The drops compound (e.g. two drops -> x0.01).
    """
    drop_iters = sorted(max(int(e), 0) * max(iters_per_epoch, 1) for e in drop_epochs)

    def lr_lambda(current_iter):
        if warmup_iters > 0 and current_iter < warmup_iters:
            alpha = float(current_iter) / float(warmup_iters)
            return warmup_factor * (1.0 - alpha) + alpha
        # Compound one gamma factor per drop boundary already passed.
        n_passed = sum(1 for di in drop_iters if current_iter >= di)
        return gamma ** n_passed

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Train  (validation metrics live in metrics.py: loss, accuracy, RMSE, mAP, P/R/F1)
# --------------------------------------------------------------------------- #
def main():
    """Train the Improved Faster R-CNN end-to-end with per-epoch validation."""
    args = parse_args()
    cfg = Config()
    torch.manual_seed(args.seed)

    epochs = args.epochs if args.epochs is not None else cfg.EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else cfg.BATCH_SIZE
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ---- Data ---------------------------------------------------------------
    train_loader, val_loader = build_loaders(cfg, args)
    iters_per_epoch = max(len(train_loader), 1)

    print("=" * 72)
    print("Improved Faster R-CNN -- training")
    print(f"  device            : {device}")
    print(f"  dataset           : {args.dataset}")
    print(f"  train / val images: {len(train_loader.dataset)} / {len(val_loader.dataset)}")
    print(f"  epochs            : {epochs}   batch size: {batch_size}")
    print(f"  classes (fg)      : {cfg.NUM_CLASSES}  {cfg.CLASS_NAMES}")
    print(f"  input size        : {cfg.INPUT_SIZE}x{cfg.INPUT_SIZE} (letterboxed)")
    print(f"  focal loss        : {args.use_focal_loss}")
    print("=" * 72)

    # ---- Model --------------------------------------------------------------
    model = build_model(
        num_classes=cfg.NUM_CLASSES,
        input_size=cfg.INPUT_SIZE,
        anchor_sizes=cfg.ANCHOR_SIZES,
        anchor_ratios=cfg.ANCHOR_RATIOS,
        score_thresh=cfg.SCORE_THRESH,
        matrix_nms_sigma=cfg.MATRIX_NMS_SIGMA,
        detections_per_img=cfg.DETECTIONS_PER_IMG,
        pretrained_backbone=False,
        use_focal_loss=args.use_focal_loss,
    )
    model.to(device)

    # ---- Optimizer + schedule (journal) ------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    warmup_iters = max(min(cfg.WARMUP_ITERS, epochs * iters_per_epoch - 1), 0)
    lr_scheduler = make_warmup_step_scheduler(
        optimizer, warmup_iters=warmup_iters, warmup_factor=1.0 / 1000.0,
        drop_epochs=cfg.LR_DROP_EPOCHS, iters_per_epoch=iters_per_epoch, gamma=0.1,
    )

    # ---- Logger (CSV + JSON + live PNG curves) ------------------------------
    logger = TrainingLogger(cfg.OUTPUT_DIR)
    print(f"  logging metrics -> {os.path.join(cfg.OUTPUT_DIR, 'metrics_log.csv')}")
    print(f"  live curves     -> {os.path.join(cfg.OUTPUT_DIR, 'training_curves.png')}")
    print("=" * 72)

    # ---- Training loop ------------------------------------------------------
    best_map = -1.0
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        counted = 0
        n_iters = args.max_train_batches if args.max_train_batches else iters_per_epoch

        # tqdm progress bar over the epoch's iterations.
        bar = tqdm(total=n_iters, desc=f"Epoch {epoch+1}/{epochs} [train]", ncols=110, leave=True)
        for it, (images, targets) in enumerate(train_loader):
            if args.max_train_batches and it >= args.max_train_batches:
                break
            images = [img.to(device) for img in images]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Non-finite-loss guard: skip the step so divergent weights never
            # feed inf boxes into the native NMS (which can hard-crash).
            if not torch.isfinite(losses):
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                bar.update(1)
                bar.set_postfix_str("WARNING: non-finite loss, step skipped")
                continue

            optimizer.zero_grad()
            losses.backward()
            if cfg.GRAD_CLIP_NORM and cfg.GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.GRAD_CLIP_NORM)
            optimizer.step()
            lr_scheduler.step()

            loss_value = float(losses.detach().cpu())
            running_loss += loss_value
            counted += 1
            bar.update(1)
            bar.set_postfix(loss=f"{loss_value:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        bar.close()

        train_loss = running_loss / max(counted, 1)

        # ---- Metrics: train subset + full validation ------------------------
        do_eval = bool(args.eval_every) and (epoch + 1) % args.eval_every == 0
        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss}

        if do_eval:
            # Train-set metrics on a small sample (tracks the train/val gap).
            tb = args.max_train_batches or cfg.TRAIN_EVAL_BATCHES
            tbar = tqdm(total=tb, desc=f"Epoch {epoch+1}/{epochs} [train-eval]", ncols=110, leave=False)
            tr = evaluate_detection(model, train_loader, device,
                                    iou_thresh=cfg.EVAL_IOU_THRESH, score_thresh=cfg.EVAL_SCORE_THRESH,
                                    max_batches=tb, with_map=True, progress=tbar)
            tbar.close()

            # Full validation: detection metrics (+mAP) and a separate val loss.
            n_val = args.eval_max_batches or len(val_loader)
            vbar = tqdm(total=n_val, desc=f"Epoch {epoch+1}/{epochs} [val]", ncols=110, leave=False)
            va = evaluate_detection(model, val_loader, device,
                                    iou_thresh=cfg.EVAL_IOU_THRESH, score_thresh=cfg.EVAL_SCORE_THRESH,
                                    max_batches=args.eval_max_batches, with_map=True, progress=vbar)
            vbar.close()
            vl = compute_loss_over_loader(model, val_loader, device, max_batches=args.eval_max_batches)

            row.update({
                "train_acc": tr["accuracy"], "train_rmse": tr["rmse"], "train_map": tr["map"],
                "val_loss": vl.get("loss", float("nan")),
                "val_acc": va["accuracy"], "val_rmse": va["rmse"],
                "val_map": va["map"], "val_map50": va["map_50"],
                "val_precision": va["precision"], "val_recall": va["recall"], "val_f1": va["f1"],
            })

        row["time_s"] = round(time.time() - epoch_start, 1)
        logger.log(row)
        logger.plot()

        # ---- Per-epoch summary line ----------------------------------------
        if do_eval:
            print(f"  epoch {epoch+1:>3d}/{epochs} | {row['time_s']:.0f}s | lr {row['lr']:.2e}\n"
                  f"     LOSS  train={train_loss:.4f}  val={row['val_loss']:.4f}\n"
                  f"     ACC   train={row['train_acc']:.4f}  val={row['val_acc']:.4f}   (IoU>=0.5, score>={cfg.EVAL_SCORE_THRESH})\n"
                  f"     RMSE  train={row['train_rmse']:.2f}px  val={row['val_rmse']:.2f}px\n"
                  f"     mAP   val@[.5:.95]={row['val_map']:.4f}  val@.5={row['val_map50']:.4f}  "
                  f"| P={row['val_precision']:.3f} R={row['val_recall']:.3f} F1={row['val_f1']:.3f}")
        else:
            print(f"  epoch {epoch+1:>3d}/{epochs} | {row['time_s']:.0f}s | train_loss={train_loss:.4f} (no eval this epoch)")

        # ---- Checkpoints ----------------------------------------------------
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "metrics": row,
            "class_names": cfg.CLASS_NAMES,
        }, os.path.join(cfg.OUTPUT_DIR, "model_latest.pth"))
        cur_map = row.get("val_map", float("nan"))
        if cur_map == cur_map and cur_map > best_map:  # not-NaN and improved
            best_map = cur_map
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "model_best.pth"))
            print(f"     -> new best val mAP={best_map:.4f} (saved model_best.pth)")

    print(f"Training complete. Best val mAP={best_map:.4f}. "
          f"Logs + curves in {cfg.OUTPUT_DIR}/")

    # ---- Export the BEST checkpoint to ONNX --------------------------------
    if not args.no_export_onnx:
        try:
            from export_onnx import export_to_onnx
            best_ckpt = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
            onnx_path = os.path.join(cfg.OUTPUT_DIR, "model_best.onnx")
            # Load best weights into the in-memory model (falls back to current
            # weights if no best checkpoint was ever saved, e.g. eval disabled).
            if os.path.isfile(best_ckpt):
                model.load_state_dict(torch.load(best_ckpt, map_location=device))
                print(f"[onnx] exporting BEST checkpoint (val mAP={best_map:.4f}) -> {onnx_path}")
            else:
                print(f"[onnx] no model_best.pth; exporting latest weights -> {onnx_path}")
            export_to_onnx(model, onnx_path, input_size=cfg.INPUT_SIZE,
                           nms=args.onnx_nms, verify=True)
        except Exception as e:
            print(f"[onnx] export failed ({type(e).__name__}: {e}). "
                  f"You can retry manually: python export_onnx.py")


if __name__ == "__main__":
    main()
