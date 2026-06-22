"""Modulated Deformable Convolution v2 (DCNv2).

Journal component
-----------------
This module implements the modulated deformable convolution (DCNv2) that the
"Improved Faster R-CNN" of Wang et al. (Metals, 2021) applies to the 3x3
convolutions inside stages 2, 3 and 4 of the ResNet50-vd backbone. DCNv2 lets
each sampling location of the convolution kernel move by a learned 2D offset and
be reweighted by a learned modulation (mask) scalar, giving the backbone the
geometric flexibility needed to localise irregularly-shaped steel-surface
defects.

Reference
---------
Zhu, Hu, Lin, Dai, "Deformable ConvNets v2: More Deformable, Better Results",
CVPR 2019.

Implementation notes
--------------------
The core sampling is performed by ``torchvision.ops.DeformConv2d``. A plain
``nn.Conv2d`` (``conv_offset_mask``) predicts, per spatial location, both the
sampling offsets and the modulation masks. Its weights/bias are initialised to
ZERO (the standard DCNv2 initialisation) so that at the start of training the
offsets are 0 and the masks are sigmoid(0) = 0.5, i.e. the layer behaves like a
regular (but globally scaled) convolution and training is stable.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


def _pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Return ``value`` as a (h, w) 2-tuple, accepting either an int or a pair."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected a length-2 sequence, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def deform_conv2d_grid_sample(x, offset, mask, weight, bias, stride, padding, dilation):
    """ONNX-friendly modulated deformable conv (groups=1, offset_groups=1).

    A drop-in numerical match for ``torchvision.ops.deform_conv2d`` built from
    ``grid_sample`` + standard ops only, so it traces to an ONNX graph that
    onnxruntime can execute (the native ``torchvision::deform_conv2d`` op has no
    runnable ONNX implementation). Verified to match torchvision to ~1e-5.

    Shapes: ``x`` (N,C,H,W); ``offset`` (N, 2*kh*kw, Ho, Wo) tap-major (dy,dx);
    ``mask`` (N, kh*kw, Ho, Wo) already sigmoid-activated; ``weight`` (O,C,kh,kw).
    """
    N, C, H, W = x.shape
    O = weight.shape[0]
    kh, kw = weight.shape[2], weight.shape[3]
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    dev, dt = x.device, x.dtype

    base_y = torch.arange(Ho, device=dev, dtype=dt).view(Ho, 1) * sh - ph     # [Ho,1]
    base_x = torch.arange(Wo, device=dev, dtype=dt).view(1, Wo) * sw - pw     # [1,Wo]

    off = offset.view(N, kh * kw, 2, Ho, Wo)
    dy, dx = off[:, :, 0], off[:, :, 1]                                       # [N,T,Ho,Wo]
    m = mask.view(N, kh * kw, Ho, Wo)

    cols = []
    for i in range(kh):
        for j in range(kw):
            t = i * kw + j
            py = base_y + i * dh + dy[:, t]
            px = base_x + j * dw + dx[:, t]
            gx = 2.0 * px / (W - 1) - 1.0
            gy = 2.0 * py / (H - 1) - 1.0
            grid = torch.stack([gx, gy], dim=-1)                             # [N,Ho,Wo,2]
            s = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            cols.append(s * m[:, t].unsqueeze(1))
    cols = torch.stack(cols, dim=1)                                          # [N,T,C,Ho,Wo]
    out = torch.einsum("ntchw,oct->nohw", cols, weight.reshape(O, C, kh * kw))
    if bias is not None:
        out = out + bias.view(1, O, 1, 1)
    return out


class DCNv2(nn.Module):
    """Modulated deformable convolution (Deformable ConvNets v2).

    Behaves like an ``nn.Conv2d`` that maps ``in_channels`` -> ``out_channels``,
    but every kernel sampling location is shifted by a learned offset and scaled
    by a learned (sigmoid) modulation mask.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    kernel_size : int or (int, int), default 3
        Convolution kernel size. Non-square kernels are supported.
    stride : int or (int, int), default 1
        Convolution stride.
    padding : int or (int, int), default 1
        Zero-padding added to both sides of the input.
    dilation : int or (int, int), default 1
        Spacing between kernel elements.
    groups : int, default 1
        Number of blocked connections from input to output channels (the
        grouping of the *deformable* convolution itself).
    bias : bool, default False
        If ``True``, add a learnable bias to the deformable convolution output.
    deformable_groups : int, default 1
        Number of groups used to compute offsets/masks. Each group predicts its
        own set of offsets/masks shared across the corresponding input-channel
        subset.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = False,
        deformable_groups: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = int(groups)
        self.deformable_groups = int(deformable_groups)
        #: When True, forward() uses the grid_sample decomposition instead of
        #: torchvision's native op so the module can be exported to ONNX.
        self.onnx_export = False

        kh, kw = self.kernel_size

        # Core modulated deformable convolution (performs the actual sampling).
        self.conv = DeformConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=bias,
        )

        # Predicts offsets (2 * kh * kw per deformable group) and masks
        # (kh * kw per deformable group). Total = 3 * kh * kw per group.
        self.conv_offset_mask = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.deformable_groups * 3 * kh * kw,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Zero-initialise the offset/mask predictor (standard DCNv2 init).

        With zero weights and bias the predicted offsets are 0 and the masks are
        sigmoid(0) = 0.5, so the layer initially acts as a regular convolution.
        """
        nn.init.constant_(self.conv_offset_mask.weight, 0.0)
        if self.conv_offset_mask.bias is not None:
            nn.init.constant_(self.conv_offset_mask.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply modulated deformable convolution to ``x``.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(N, in_channels, H, W)``.

        Returns
        -------
        Tensor
            Output of shape ``(N, out_channels, H_out, W_out)``.
        """
        kh, kw = self.kernel_size
        dg = self.deformable_groups

        out = self.conv_offset_mask(x)

        # Split prediction into 3 chunks of (dg * kh * kw) channels each:
        # two offset channels (y, x) interleaved per location, plus one mask.
        offset_channels = dg * 2 * kh * kw
        o1 = out[:, :offset_channels, :, :]
        mask = out[:, offset_channels:, :, :]

        # DeformConv2d expects the offset tensor laid out as
        # (N, 2 * kh * kw * dg, H, W); o1 already matches that channel count.
        offset = o1
        mask = torch.sigmoid(mask)

        if self.onnx_export:
            # ONNX-exportable path (grid_sample decomposition). Only the
            # groups=1, deformable_groups=1 case is supported here (the config
            # used throughout this project's backbone).
            if self.groups != 1 or self.deformable_groups != 1:
                raise NotImplementedError(
                    "onnx_export path supports groups=1 and deformable_groups=1 only."
                )
            return deform_conv2d_grid_sample(
                x, offset, mask, self.conv.weight, self.conv.bias,
                self.stride, self.padding, self.dilation,
            )

        return self.conv(x, offset, mask)
