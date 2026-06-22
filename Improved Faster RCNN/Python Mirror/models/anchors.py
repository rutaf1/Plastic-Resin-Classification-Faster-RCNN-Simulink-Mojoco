"""Custom anchor generator.

Implements the journal component:
    "Custom anchors: aspect ratios changed from [0.5, 1, 2] to
     [0.25, 0.5, 1, 2, 5]."
    (Wang et al., "Automatic Detection and Classification of Steel Surface
     Defect Using Deep CNNs", Metals 2021 -- Improved Faster R-CNN.)

The Enhanced FPN produces 5 feature-map levels (P2, P3, P4, P5, P6), so the
anchor generator is configured with exactly 5 sizes (``len(sizes) == 5``), one
tuple of anchor sizes per pyramid level.

The extended aspect-ratio set ``[0.25, 0.5, 1, 2, 5]`` adds very narrow/elongated
ratios (0.25 and 5) on top of the vanilla Faster R-CNN ratios. These long, thin
anchors are deliberately chosen to better cover the long/narrow "crazing" and
"scratch" steel-surface defects highlighted in the journal, which the default
[0.5, 1, 2] ratios fit poorly.

This module is intentionally free of any dependency on ``config.py`` -- the
caller (``train.py`` / ``inference.py``) supplies values explicitly.
"""

from torchvision.models.detection.anchor_utils import AnchorGenerator


def build_anchor_generator(
    sizes=((16,), (32,), (64,), (128,), (256,)),
    aspect_ratios=(0.25, 0.5, 1.0, 2.0, 5.0),
) -> AnchorGenerator:
    """Build the torchvision ``AnchorGenerator`` for the Improved Faster R-CNN.

    Args:
        sizes: One tuple of anchor base sizes per FPN level. There are 5 levels
            (P2..P6) in the Enhanced FPN, so ``len(sizes)`` must be 5.
        aspect_ratios: The shared set of anchor aspect ratios applied at every
            level. Defaults to the journal's extended set
            ``(0.25, 0.5, 1.0, 2.0, 5.0)``, which targets long/narrow crazing &
            scratch defects.

    Returns:
        A configured ``AnchorGenerator`` instance. The same ``aspect_ratios``
        tuple is broadcast across all ``len(sizes)`` feature-map levels.
    """
    return AnchorGenerator(
        sizes=sizes,
        aspect_ratios=(tuple(aspect_ratios),) * len(sizes),
    )
