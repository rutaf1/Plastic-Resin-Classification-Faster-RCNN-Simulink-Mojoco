"""ResNet50-vd backbone with optional DCNv2 (modulated deformable convolutions).

This module implements the backbone described in Wang et al., "Automatic
Detection and Classification of Steel Surface Defect Using Deep CNNs"
(Metals 2021). The paper's first improvement over the vanilla Faster R-CNN
backbone is to replace the standard ResNet50 with the **ResNet50-vd** variant
and to insert **DCNv2** (modulated deformable convolutions) into the deeper
stages.

The "vd" variant follows He et al., "Bag of Tricks for Image Classification
with Convolutional Neural Networks" (CVPR 2019):

  * **Deep stem (ResNet-C / -B trick):** the single 7x7 stride-2 stem
    convolution is replaced by three stacked 3x3 convolutions with channel
    counts (32, 32, 64). The first 3x3 carries the stride 2. A 3x3 max-pool
    with stride 2 follows, so the stem produces a stride-4 feature map.

  * **ResNet-D downsampling:** in every down-sampling residual block the
    shortcut branch no longer uses a stride-2 1x1 convolution (which throws
    away 3/4 of the activations). Instead it applies ``AvgPool2d(2, stride=2,
    ceil_mode=True)`` followed by a *stride-1* 1x1 convolution. The spatial
    stride-2 is moved onto the 3x3 convolution inside the bottleneck.

Bottleneck block (the standard ResNet50 bottleneck):

    1x1 reduce  ->  3x3 (stride + DCN live here)  ->  1x1 expand

each convolution followed by BatchNorm; ReLU after the first two and after the
residual add. Stage layer counts for ResNet50 are ``[3, 4, 6, 3]``.

When a stage index (1..4) is listed in ``dcn_stages`` the 3x3 convolution of
**every** bottleneck in that stage is a :class:`DCNv2` instead of a plain
``nn.Conv2d``. The paper applies DCNv2 to stages 2, 3 and 4.
"""

from collections import OrderedDict

import torch.nn as nn

from .deform_conv import DCNv2


def _conv3x3(in_planes, out_planes, stride=1, dilation=1, use_dcn=False):
    """Build the 3x3 convolution of a bottleneck.

    When ``use_dcn`` is True a modulated deformable convolution (DCNv2) is
    returned in place of ``nn.Conv2d``. DCNv2 is expected to expose the same
    constructor signature as ``nn.Conv2d`` so the two are interchangeable.
    """
    if use_dcn:
        return DCNv2(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def _conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution (no bias, BN follows)."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BottleneckVd(nn.Module):
    """ResNet50-vd bottleneck block.

    Layout: 1x1 reduce -> 3x3 (stride + optional DCNv2) -> 1x1 expand.

    The down-sampling shortcut uses the ResNet-D trick: ``AvgPool2d`` followed
    by a stride-1 1x1 convolution, instead of a strided 1x1 convolution. The
    spatial stride is carried by the inner 3x3 convolution.
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        use_dcn=False,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        # 1x1 reduce
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)

        # 3x3 -- this is where the spatial stride and (optionally) DCNv2 live.
        self.conv2 = _conv3x3(planes, planes, stride=stride, use_dcn=use_dcn)
        self.bn2 = norm_layer(planes)

        # 1x1 expand
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet50vd(nn.Module):
    """ResNet50-vd feature extractor with optional DCNv2 in selected stages.

    Args:
        dcn_stages: tuple of stage indices (numbered 1..4, i.e. layer1..layer4)
            in which the 3x3 conv of every bottleneck is replaced by DCNv2.
            Defaults to ``(2, 3, 4)`` as in the paper.
        norm_layer: normalization layer constructor (default ``nn.BatchNorm2d``).

    Forward returns an ``OrderedDict`` with keys ``'0'..'3'`` mapping to the
    outputs of stage1..stage4 at strides 4, 8, 16, 32 relative to the input.

    Attributes:
        out_channels_list: ``[256, 512, 1024, 2048]`` -- channel counts of the
            feature maps returned under keys ``'0'``..``'3'``.
    """

    # ResNet50 stage layer counts.
    _layers = (3, 4, 6, 3)

    def __init__(self, dcn_stages=(2, 3, 4), norm_layer=nn.BatchNorm2d):
        super().__init__()
        self._norm_layer = norm_layer
        self.dcn_stages = tuple(dcn_stages)

        # ------------------------------------------------------------------
        # Deep stem (ResNet-C/-D trick): three 3x3 convs replacing the 7x7.
        # Channels 32 -> 32 -> 64; first conv carries stride 2.
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
        )
        # 3x3 max-pool, stride 2 -> stem output is stride 4.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = 64

        # Stages (layer1..layer4). Stage 1 keeps stride 1 (after the stem +
        # pool already give stride 4); stages 2..4 each down-sample by 2.
        self.layer1 = self._make_layer(64, self._layers[0], stride=1, stage_idx=1)
        self.layer2 = self._make_layer(128, self._layers[1], stride=2, stage_idx=2)
        self.layer3 = self._make_layer(256, self._layers[2], stride=2, stage_idx=3)
        self.layer4 = self._make_layer(512, self._layers[3], stride=2, stage_idx=4)

        # Channel counts of layer1..layer4 outputs (planes * expansion).
        self.out_channels_list = [256, 512, 1024, 2048]

        self._init_weights()

    # ------------------------------------------------------------------
    def _make_layer(self, planes, blocks, stride, stage_idx):
        norm_layer = self._norm_layer
        use_dcn = stage_idx in self.dcn_stages
        downsample = None

        # A downsample branch is needed when the spatial size changes (stride
        # != 1) or when the channel count changes (inplanes != planes*exp).
        if stride != 1 or self.inplanes != planes * BottleneckVd.expansion:
            if stride != 1:
                # ResNet-D: AvgPool2d(2) then stride-1 1x1 conv on the shortcut.
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                    _conv1x1(self.inplanes, planes * BottleneckVd.expansion, stride=1),
                    norm_layer(planes * BottleneckVd.expansion),
                )
            else:
                # Channel-only projection (first block of stage1): plain 1x1.
                downsample = nn.Sequential(
                    _conv1x1(self.inplanes, planes * BottleneckVd.expansion, stride=1),
                    norm_layer(planes * BottleneckVd.expansion),
                )

        layers = []
        layers.append(
            BottleneckVd(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                use_dcn=use_dcn,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes * BottleneckVd.expansion
        for _ in range(1, blocks):
            layers.append(
                BottleneckVd(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    use_dcn=use_dcn,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)  # stride 4

        c2 = self.layer1(x)   # stride 4   -> key '0', 256 ch
        c3 = self.layer2(c2)  # stride 8   -> key '1', 512 ch
        c4 = self.layer3(c3)  # stride 16  -> key '2', 1024 ch
        c5 = self.layer4(c4)  # stride 32  -> key '3', 2048 ch

        out = OrderedDict()
        out["0"] = c2
        out["1"] = c3
        out["2"] = c4
        out["3"] = c5
        return out
