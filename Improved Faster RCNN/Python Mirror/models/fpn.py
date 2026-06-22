"""Enhanced Feature Pyramid Network (FPN).

This module implements the FPN used by the "Improved Faster R-CNN" of
Wang et al., "Automatic Detection and Classification of Steel Surface Defect
Using Deep CNNs", Metals 2021.

It is built on the classic top-down FPN of Lin et al., "Feature Pyramid
Networks for Object Detection" (CVPR 2017), with two journal-specific
enhancements:

  * **CoordConv laterals (journal enhancement #3).** Each standard 1x1 lateral
    convolution that projects a backbone feature map (C2..C5) down to the common
    pyramid channel width is REPLACED by a :class:`CoordConv` (Liu et al.,
    "An Intriguing Failing of Convolutional Neural Networks and the CoordConv
    Solution", NeurIPS 2018). The appended coordinate channels give the lateral
    projection explicit access to spatial position, which the journal reports
    improves localization of steel-surface defects.

  * **SPP on the top map (journal enhancement #2).** A YOLO-style Spatial
    Pyramid Pooling block (:class:`SPP`, which preserves spatial H x W) is
    applied to the deepest lateral feature (the C5 projection) before the
    top-down pathway begins, enriching the coarsest level with multi-scale
    context.

The output keys ('0','1','2','3' and optionally '4') follow torchvision's
convention so the module can feed a torchvision ``FasterRCNN`` / ``RPN``
directly.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coordconv import CoordConv
from .spp import SPP


class EnhancedFPN(nn.Module):
    """Top-down FPN with CoordConv lateral connections and an SPP top block.

    Args:
        in_channels_list (list[int]): Number of channels of each input feature
            map, ordered from the finest/shallowest level to the
            coarsest/deepest level. For a ResNet50-vd backbone returning
            C2..C5 this is ``[256, 512, 1024, 2048]``.
        out_channels (int): Common channel width of every pyramid level.
            Default ``256`` (torchvision default).
        extra_block (bool): If ``True`` append an extra coarsest level ``'4'``
            (commonly called P6) produced by a stride-2 max pool of the deepest
            output map. torchvision's ``RPN`` expects this extra level.
            Default ``True``.

    Forward input/output:
        Both are ``OrderedDict[str, Tensor]``. Any input keys are accepted; the
        values are consumed in insertion order and must match the order of
        ``in_channels_list``. The output uses the torchvision-style integer
        string keys ``'0','1','2','3'`` for P2..P5 (and ``'4'`` for P6 when
        ``extra_block`` is set). Every output tensor has ``out_channels``
        channels.
    """

    def __init__(self, in_channels_list, out_channels=256, extra_block=True):
        super().__init__()

        if len(in_channels_list) == 0:
            raise ValueError("in_channels_list must contain at least one entry.")

        self.out_channels = out_channels
        self.extra_block = extra_block
        self.num_levels = len(in_channels_list)

        # --- CoordConv lateral connections (journal enhancement #3) ----------
        # One CoordConv per input level. CoordConv REPLACES the standard 1x1
        # lateral conv of the original FPN: it appends normalized (x, y)
        # coordinate channels to the input before a 1x1 projection to
        # ``out_channels``.
        self.lateral_convs = nn.ModuleList(
            [
                CoordConv(in_ch, out_channels, kernel_size=1)
                for in_ch in in_channels_list
            ]
        )

        # --- 3x3 output (anti-aliasing / smoothing) convs --------------------
        # Applied to every merged level to reduce the aliasing introduced by the
        # nearest-neighbor up-sampling in the top-down pathway (Lin et al.).
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )

        # --- SPP on the deepest lateral (journal enhancement #2) -------------
        # Operates on the top-level lateral (the C5 projection) and preserves
        # its spatial resolution, so the top-down pathway is unchanged shape-wise.
        self.spp = SPP(out_channels, out_channels)

        self._init_weights()

    def _init_weights(self):
        """Kaiming-uniform init for the smoothing convs (FPN convention).

        CoordConv and SPP own and initialize their internal convolutions.
        """
        for m in self.output_convs:
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Run the top-down pathway.

        Args:
            x (OrderedDict[str, Tensor]): Backbone feature maps ordered from
                finest (P2/C2) to coarsest (P5/C5).

        Returns:
            OrderedDict[str, Tensor]: Pyramid feature maps with keys
            ``'0'..'3'`` (and ``'4'`` if ``extra_block``), each with
            ``out_channels`` channels.
        """
        feats = list(x.values())
        if len(feats) != self.num_levels:
            raise ValueError(
                "EnhancedFPN expected {} feature maps but received {}.".format(
                    self.num_levels, len(feats)
                )
            )

        # 1) Lateral projections via CoordConv for every level.
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, feats)
        ]

        # 2) Enrich the deepest lateral with SPP multi-scale context, keeping
        #    its spatial size unchanged (journal enhancement #2).
        laterals[-1] = self.spp(laterals[-1])

        # 3) Top-down pathway: from coarsest to finest, up-sample the coarser
        #    map (nearest neighbor) and add it into the next-finer lateral.
        for i in range(self.num_levels - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 4) Per-level 3x3 smoothing conv.
        outputs = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

        # 5) Assemble torchvision-style OrderedDict keyed by level index.
        result = OrderedDict()
        for level, out in enumerate(outputs):
            result[str(level)] = out

        # 6) Optional extra coarsest level (P6) for the RPN.
        if self.extra_block:
            result[str(self.num_levels)] = F.max_pool2d(
                outputs[-1], kernel_size=1, stride=2, padding=0
            )

        return result
