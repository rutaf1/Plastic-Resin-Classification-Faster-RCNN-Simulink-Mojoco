"""Detection backbone: ResNet50-vd + Enhanced FPN.

This module assembles the two main feature-extraction improvements proposed in
Wang et al., "Automatic Detection and Classification of Steel Surface Defect
Using Deep CNNs", *Metals* 2021, into a single backbone object that
torchvision's ``FasterRCNN`` can consume directly.

Components combined here
------------------------
* :class:`~models.resnet50_vd.ResNet50vd`
    The ResNet50-vd backbone (three-conv stem, AvgPool+1x1 downsample shortcut)
    with DCNv2 (modulated deformable convolution) on the 3x3 convs of the
    chosen stages. Implements *journal improvement #1* (and SPP, #2, internally
    on its deepest feature, per the ResNet50vd implementation).
* :class:`~models.fpn.EnhancedFPN`
    A Feature Pyramid Network whose lateral 1x1 convolutions are replaced by
    CoordConv. Implements *journal improvement #3*.

torchvision's ``FasterRCNN`` requires a backbone ``nn.Module`` that:
  1. exposes an integer attribute ``out_channels`` (the channel count shared by
     every pyramid level fed to the RPN / RoI heads), and
  2. returns an ``OrderedDict[str, Tensor]`` of feature maps from ``forward``.

Both contracts are satisfied by :class:`ResNet50vdFPN`.
"""

from collections import OrderedDict

import torch.nn as nn

from .resnet50_vd import ResNet50vd
from .fpn import EnhancedFPN


class ResNet50vdFPN(nn.Module):
    """ResNet50-vd + Enhanced (CoordConv) FPN detection backbone.

    The body (ResNet50-vd, optionally with DCNv2 and SPP) produces a dict of
    bottom-up feature maps; the Enhanced FPN fuses them top-down into a feature
    pyramid where every level has the same number of channels (``out_channels``)
    so the downstream RPN and RoI heads can be shared across levels.

    Number of output feature maps
    -----------------------------
    The body contributes the standard four residual-stage feature maps
    (C2..C5). With ``extra_block=True`` the FPN appends one extra coarser level
    on top (a max-pool / strided level above P5), yielding **5** pyramid feature
    maps with keys ``'0'``, ``'1'``, ``'2'``, ``'3'``, ``'4'``. This count is
    deliberate: it matches the **5 anchor sizes** used by the custom anchor
    generator (one anchor size per pyramid level; see
    :func:`models.anchors.build_anchor_generator`). With ``extra_block=False``
    the backbone yields 4 feature maps (keys ``'0'``..``'3'``).

    Parameters
    ----------
    dcn_stages : tuple of int, default ``(2, 3, 4)``
        Which residual stages of the ResNet50-vd body apply DCNv2 to their 3x3
        convs. Passed straight through to :class:`ResNet50vd`. Stage numbering
        follows the paper (stage 2 == ``conv3_x`` == C3, etc., per the
        ResNet50vd implementation).
    out_channels : int, default ``256``
        Channel count of every output pyramid level. Exposed as the required
        ``self.out_channels`` integer attribute for ``FasterRCNN``.
    extra_block : bool, default ``True``
        If ``True`` the FPN adds one extra coarse level on top of the pyramid,
        producing 5 feature maps total (see above).
    """

    def __init__(self, dcn_stages=(2, 3, 4), out_channels=256, extra_block=True):
        super().__init__()

        # Bottom-up feature extractor (improvements #1 + #2).
        self.body = ResNet50vd(dcn_stages=dcn_stages)

        # Per-stage channel counts of the body's bottom-up outputs (C2..C5),
        # used to size the FPN's lateral CoordConv projections.
        in_channels_list = self.body.out_channels_list

        # Top-down feature pyramid with CoordConv laterals (improvement #3).
        self.fpn = EnhancedFPN(in_channels_list, out_channels, extra_block)

        # REQUIRED by torchvision FasterRCNN: a single integer giving the
        # channel depth of every pyramid level handed to the RPN / RoI heads.
        self.out_channels = out_channels

    def forward(self, x):
        """Run the backbone.

        Parameters
        ----------
        x : Tensor
            Image batch of shape ``(N, 3, H, W)`` (the project uses
            480x480 RGB inputs).

        Returns
        -------
        OrderedDict[str, Tensor]
            Feature pyramid keyed ``'0'``..``'4'`` (or ``'0'``..``'3'`` when
            ``extra_block=False``). Each value has ``out_channels`` channels and
            progressively coarser spatial resolution.
        """
        # Bottom-up pass -> OrderedDict of C2..C5 feature maps.
        body_features = self.body(x)

        # Top-down FPN fusion -> OrderedDict of P-level feature maps.
        features = self.fpn(body_features)

        # Guarantee the contracted return type even if the FPN hands back a
        # plain dict.
        if not isinstance(features, OrderedDict):
            features = OrderedDict(features)
        return features
