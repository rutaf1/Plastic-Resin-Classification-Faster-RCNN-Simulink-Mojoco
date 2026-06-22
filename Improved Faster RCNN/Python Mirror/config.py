"""Configuration for the Improved Faster R-CNN.

Centralizes all hyperparameters for the re-implementation of:
    Wang et al., "Automatic Detection and Classification of Steel Surface
    Defect Using Deep CNNs", Metals 2021.

This is the ONLY module (besides train.py / inference.py) that holds concrete
hyperparameter values. The model-building code under ``models/`` is kept
config-agnostic: it receives every value as an explicit argument so it can be
reused outside this project. ``train.py`` and ``inference.py`` read ``Config``
and forward the values into ``build_model(...)`` and the training loop.

Import-light: the only third-party import is ``torch``, and it is used solely
(and defensively) to auto-detect CUDA. If torch is unavailable for any reason,
the device gracefully falls back to ``"cpu"``.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


def _auto_device() -> str:
    """Return ``"cuda"`` when a CUDA device is available, else ``"cpu"``.

    Guarded so that importing :mod:`config` never fails even if torch is not
    installed or CUDA initialization raises.
    """
    try:
        import torch  # local, guarded import to keep this module import-light

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# The 7 foreground classes of the plastic-defect dataset. The CVAT labels are
# "NAME (idx)" with idx the 1-based class id; index here is idx-1 (label-1).
#   1 PET  2 HDPE  3 PVC  4 LDPE  5 PP  6 PS  7 Other
_DEFAULT_CLASS_NAMES: List[str] = [
    "PET",
    "HDPE",
    "PVC",
    "LDPE",
    "PP",
    "PS",
    "Other",
]


@dataclass
class Config:
    """Hyperparameters for training / inference of the Improved Faster R-CNN.

    All counts of object classes refer to the FOREGROUND classes only. The
    torchvision detector adds the implicit background class internally, so the
    detector is constructed with ``num_classes = NUM_CLASSES + 1``.
    """

    # ----- Dataset / I/O -----------------------------------------------------
    #: Number of FOREGROUND classes (background is added internally by the head).
    NUM_CLASSES: int = 7
    #: Human-readable names for the 7 foreground classes (index == label-1).
    CLASS_NAMES: List[str] = field(default_factory=lambda: list(_DEFAULT_CLASS_NAMES))
    #: Fixed square input resolution (pixels). Images are RGB 480x480x3.
    INPUT_SIZE: int = 480
    #: Root directory holding the dataset. Contains ``images_aug/`` and
    #: ``annotations_aug.xml`` (CVAT format). Relative to the python_alternatif
    #: working dir, the dataset sits one level up in ``../dataset``.
    DATA_ROOT: str = "../dataset"
    #: Directory for checkpoints, logs and exported predictions.
    OUTPUT_DIR: str = "./outputs"
    #: DataLoader worker processes. 0 is the safe default on Windows.
    NUM_WORKERS: int = 0

    # ----- Anchors (journal component #4: custom anchors) --------------------
    # Tuned to THIS dataset via box-statistics analysis (k-means + anchor-recall)
    # in the 480x480 letterboxed space. Objects are small/medium: sqrt-area spans
    # ~11..153 px (median ~47), so the vanilla top level (256) was wasted. These
    # 5 scales give anchor-recall@IoU0.5 = 99.9% and @IoU0.6 = 98.8% over all
    # 10,929 boxes (vs 74% @0.6 for (16,32,64,128,256)).
    #: One anchor size tuple per FPN level (P2..P6). Matches 5 pyramid levels.
    ANCHOR_SIZES: Tuple[Tuple[int, ...], ...] = (
        (12,),
        (24,),
        (40,),
        (64,),
        (104,),
    )
    #: Aspect ratios (h/w). 0.2 covers very wide boxes (w/h~5, the dataset's p99),
    #: 5.0 covers very tall ones; bulk (h/w 0.4..2.1) sits in between. Extended
    #: from the vanilla [0.5,1,2] like the journal, but data-fitted at the tails.
    ANCHOR_RATIOS: Tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0)

    # ----- RPN / detection post-processing -----------------------------------
    #: IoU threshold used by the RPN's proposal NMS.
    RPN_NMS_THRESH: float = 0.7
    #: Sigma for the Gaussian decay kernel of Matrix NMS (journal component #5).
    MATRIX_NMS_SIGMA: float = 0.5
    #: Minimum score to keep a detection before NMS.
    SCORE_THRESH: float = 0.05
    #: Maximum detections returned per image after post-processing.
    DETECTIONS_PER_IMG: int = 100

    # ----- Evaluation / metrics ----------------------------------------------
    #: IoU threshold for matching predictions to GT in accuracy/P/R/F1/RMSE.
    EVAL_IOU_THRESH: float = 0.5
    #: Min detection score counted as a positive prediction in those metrics.
    EVAL_SCORE_THRESH: float = 0.5
    #: How many train batches to sample for train-set metrics each epoch
    #: (full train eval would be slow; a sample tracks the train/val gap).
    TRAIN_EVAL_BATCHES: int = 20

    # ----- Optimization ------------------------------------------------------
    # Defaults tuned for TRAINING FROM SCRATCH (no ImageNet weights) on a single
    # RTX 4080 Super (16 GB) at 480x480. The backbone has no pretraining, so the
    # schedule is long with a generous warm-up; gradient clipping is the safety
    # net against early divergence.
    #: Base learning rate (SGD). Linear-scaling rule from the journal's 0.0025 @
    #: batch 4 -> 0.005 @ batch 8. Lower to 0.0025 if you see early instability.
    LR: float = 0.005
    #: Images per training batch. Measured: batch 8 used ~6.7 GB on the RTX 4080
    #: Super, so 16 (~13 GB) fits and trains ~2x faster. Drop to 8 if you hit OOM.
    BATCH_SIZE: int = 16
    #: Total training epochs. From-scratch detection needs a long schedule;
    #: validation mAP is tracked each epoch and the best is saved (model_best.pth),
    #: so you can stop early once mAP plateaus.
    EPOCHS: int = 200
    #: Linear LR warm-up iterations (~12 epochs at batch 8) -- important when
    #: training the backbone from random init.
    WARMUP_ITERS: int = 2000
    #: SGD momentum.
    MOMENTUM: float = 0.9
    #: L2 weight decay.
    WEIGHT_DECAY: float = 1e-4
    #: Epochs at which the LR is multiplied by 0.1 (multi-step schedule).
    LR_DROP_EPOCHS: Tuple[int, ...] = (130, 175)
    #: Max global gradient L2-norm for clipping (stabilizes early training and
    #: prevents loss divergence -> NaN when training from scratch). <=0 disables.
    GRAD_CLIP_NORM: float = 10.0

    # ----- Class-imbalance losses (journal component #6) ---------------------
    #: Focal Loss alpha balancing factor.
    FOCAL_ALPHA: float = 0.75
    #: Focal Loss focusing parameter gamma.
    FOCAL_GAMMA: float = 2.0
    #: Per-class weights (length == NUM_CLASSES) for Weighted Cross-Entropy.
    #: Default = uniform; tune to up-weight rare defect classes.
    CLASS_WEIGHTS: List[float] = field(
        default_factory=lambda: [1.0] * 7
    )

    # ----- Runtime -----------------------------------------------------------
    #: Compute device, auto-detected at construction time.
    DEVICE: str = field(default_factory=_auto_device)

    def __post_init__(self) -> None:
        """Validate cross-field invariants the journal architecture relies on."""
        if len(self.CLASS_NAMES) != self.NUM_CLASSES:
            raise ValueError(
                f"CLASS_NAMES has {len(self.CLASS_NAMES)} entries but "
                f"NUM_CLASSES={self.NUM_CLASSES}."
            )
        if len(self.CLASS_WEIGHTS) != self.NUM_CLASSES:
            raise ValueError(
                f"CLASS_WEIGHTS has {len(self.CLASS_WEIGHTS)} entries but "
                f"NUM_CLASSES={self.NUM_CLASSES}."
            )
        if len(self.ANCHOR_SIZES) != len(self.ANCHOR_RATIOS_PER_LEVEL):
            # Sanity link between pyramid levels and anchor configuration.
            raise ValueError(
                "ANCHOR_SIZES must provide one size tuple per FPN level."
            )

    @property
    def ANCHOR_RATIOS_PER_LEVEL(self) -> Tuple[Tuple[float, ...], ...]:
        """Replicate the shared aspect-ratio tuple across every FPN level.

        torchvision's ``AnchorGenerator`` expects one ratios tuple per feature
        map level; the journal uses the same set of ratios at all levels.
        """
        return tuple(tuple(self.ANCHOR_RATIOS) for _ in self.ANCHOR_SIZES)


#: A ready-to-use default instance for convenience in scripts.
CONFIG = Config()


if __name__ == "__main__":
    # Quick self-check when run directly: print the resolved configuration.
    cfg = Config()
    for k, v in cfg.__dict__.items():
        print(f"{k:>20s} = {v}")
    print(f"{'ANCHOR_RATIOS_PER_LEVEL':>20s} = {cfg.ANCHOR_RATIOS_PER_LEVEL}")
