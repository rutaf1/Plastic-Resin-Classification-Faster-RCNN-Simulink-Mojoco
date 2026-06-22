"""Spatial Pyramid Pooling (SPP) block.

Journal component (Wang et al., Metals 2021, "Automatic Detection and
Classification of Steel Surface Defect Using Deep CNNs"):
    Improvement #2 -- Spatial Pyramid Pooling applied on the deepest feature
    map before/within the FPN.

This is the YOLO-style SPP block (as opposed to the original SPP-net layer of
He et al.). The key property is that it PRESERVES the spatial resolution
(H, W) of the input feature map: instead of pooling to fixed-size bins and
flattening, it uses several stride-1 max-pool branches with size-preserving
padding and concatenates the multi-scale context along the channel dimension.
This keeps SPP compatible with downstream convolutional / FPN processing.

References:
    - He et al., "Spatial Pyramid Pooling in Deep Convolutional Networks for
      Visual Recognition", 2015 (the original SPP idea).
    - Redmon & Farhadi, YOLOv3 / YOLOv3-SPP (the stride-1, size-preserving
      multi-scale max-pool block re-used here).
"""

import torch
import torch.nn as nn


class SPP(nn.Module):
    """YOLO-style Spatial Pyramid Pooling that preserves spatial size.

    The block reduces channels with a 1x1 Conv-BN-ReLU, runs several stride-1
    max-pool branches (each padded so that H, W are preserved), concatenates
    the original projected feature with every pooled branch along the channel
    axis, and finally fuses everything back to ``out_channels`` with a second
    1x1 Conv-BN-ReLU.

    Args:
        in_channels (int): Number of channels of the input feature map.
        out_channels (int): Number of channels of the output feature map.
        pool_sizes (tuple[int, ...]): Max-pool kernel sizes. Each kernel uses
            ``stride=1`` and ``padding=k // 2`` so the spatial size is
            preserved. Defaults to ``(5, 9, 13)`` (the YOLOv3-SPP setting).

    Shape:
        - Input:  ``[N, in_channels, H, W]``
        - Output: ``[N, out_channels, H, W]`` (same H, W as the input).
    """

    def __init__(self, in_channels, out_channels, pool_sizes=(5, 9, 13)):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)

        # Hidden width: halve the input channels before the pyramid (YOLO style).
        hidden = in_channels // 2

        # cv1: 1x1 Conv-BN-ReLU projecting in_channels -> hidden.
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        # Size-preserving stride-1 max-pool branches.
        self.pools = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
                for k in self.pool_sizes
            ]
        )

        # After concatenation: cv1 output + one branch per pool size.
        concat_channels = hidden * (len(self.pool_sizes) + 1)

        # cv2: 1x1 Conv-BN-ReLU fusing concatenated features -> out_channels.
        self.cv2 = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply the SPP block.

        Args:
            x (Tensor): Input feature map of shape ``[N, in_channels, H, W]``.

        Returns:
            Tensor: Output feature map of shape ``[N, out_channels, H, W]``.
        """
        x = self.cv1(x)
        # [cv1_out, pool5(x), pool9(x), pool13(x)] concatenated on channels.
        features = [x] + [pool(x) for pool in self.pools]
        x = torch.cat(features, dim=1)
        x = self.cv2(x)
        return x
