"""CoordConv layer.

Journal component (Improved Faster R-CNN, Wang et al., Metals 2021):
This module implements the *Enhanced FPN* improvement in which the lateral
1x1 convolutions of the Feature Pyramid Network are replaced by CoordConv.
CoordConv augments a convolution's input with hard-coded coordinate channels
so the filters can learn spatially-varying behaviour, which the paper uses to
improve localisation of steel-surface defects across the pyramid levels.

Reference:
    R. Liu et al., "An Intriguing Failing of Convolutional Neural Networks
    and the CoordConv Solution", NeurIPS 2018.

The core idea: before the convolution, concatenate to the input feature map
two extra channels holding the normalized x- and y-coordinates of every pixel
(optionally a third radial-distance channel), then run a standard Conv2d over
the augmented input.
"""

import torch
import torch.nn as nn


class CoordConv(nn.Module):
    """Convolution augmented with pixel-coordinate channels.

    The forward pass builds an x-coordinate map and a y-coordinate map, each
    normalized to [-1, 1] and broadcast to the batch size and spatial extent of
    the input. When ``with_r`` is True an additional radius channel
    ``sqrt(x^2 + y^2)`` is appended. These channels are concatenated to the
    input along the channel dimension and fed to an ordinary ``Conv2d`` whose
    input width is ``in_channels + 2`` (or ``+ 3`` when ``with_r`` is set).

    Args:
        in_channels: Number of channels in the input feature map.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Convolution kernel size (default 1, as used for the FPN
            lateral connections).
        stride: Convolution stride.
        padding: Convolution zero-padding.
        with_r: If True, add a third radial-distance coordinate channel.
        bias: If True, the wrapped Conv2d learns an additive bias.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, with_r=False, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_r = with_r

        # Two extra coordinate channels (x, y); plus one radius channel if asked.
        extra = 3 if with_r else 2
        self.conv = nn.Conv2d(
            in_channels + extra,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        """Augment ``x`` with coordinate channels and convolve.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Tensor of shape (N, out_channels, H', W').
        """
        n, _, h, w = x.shape

        # Build normalized coordinate ramps on the same device/dtype as x.
        # x-coordinate: varies along the width (columns), constant down rows.
        # y-coordinate: varies along the height (rows), constant across cols.
        if w > 1:
            xs = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype)
        else:
            xs = torch.zeros(1, device=x.device, dtype=x.dtype)
        if h > 1:
            ys = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype)
        else:
            ys = torch.zeros(1, device=x.device, dtype=x.dtype)

        # Expand ramps to full (1, 1, H, W) coordinate maps.
        x_coord = xs.view(1, 1, 1, w).expand(n, 1, h, w)
        y_coord = ys.view(1, 1, h, 1).expand(n, 1, h, w)

        coords = [x_coord, y_coord]

        if self.with_r:
            # Radial distance from the center of the (normalized) feature map.
            r = torch.sqrt(x_coord * x_coord + y_coord * y_coord)
            coords.append(r)

        out = torch.cat([x] + coords, dim=1)
        return self.conv(out)
