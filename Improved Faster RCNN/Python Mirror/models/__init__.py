"""Models package for the Improved Faster R-CNN.

Re-implements components from Wang et al., "Automatic Detection and
Classification of Steel Surface Defect Using Deep CNNs", Metals 2021.

Public symbols are re-exported here for convenience. Each import is guarded
by try/except so that a partially-built package (e.g. during incremental
development) still imports without raising; missing symbols simply remain
undefined and absent from ``__all__``.
"""

__all__ = []


def _export(name, value):
    """Register a successfully imported symbol into the module namespace."""
    globals()[name] = value
    if name not in __all__:
        __all__.append(name)


try:
    from .faster_rcnn import build_model
    _export("build_model", build_model)
except Exception:  # pragma: no cover - partial package tolerance
    pass

try:
    from .deform_conv import DCNv2
    _export("DCNv2", DCNv2)
except Exception:  # pragma: no cover
    pass

try:
    from .coordconv import CoordConv
    _export("CoordConv", CoordConv)
except Exception:  # pragma: no cover
    pass

try:
    from .spp import SPP
    _export("SPP", SPP)
except Exception:  # pragma: no cover
    pass

try:
    from .matrix_nms import matrix_nms
    _export("matrix_nms", matrix_nms)
except Exception:  # pragma: no cover
    pass

try:
    from .fpn import EnhancedFPN
    _export("EnhancedFPN", EnhancedFPN)
except Exception:  # pragma: no cover
    pass

try:
    from .resnet50_vd import ResNet50vd
    _export("ResNet50vd", ResNet50vd)
except Exception:  # pragma: no cover
    pass

try:
    from .backbone import ResNet50vdFPN
    _export("ResNet50vdFPN", ResNet50vdFPN)
except Exception:  # pragma: no cover
    pass

try:
    from .losses import FocalLoss, WeightedCrossEntropyLoss
    _export("FocalLoss", FocalLoss)
    _export("WeightedCrossEntropyLoss", WeightedCrossEntropyLoss)
except Exception:  # pragma: no cover
    pass

try:
    from .anchors import build_anchor_generator
    _export("build_anchor_generator", build_anchor_generator)
except Exception:  # pragma: no cover
    pass
