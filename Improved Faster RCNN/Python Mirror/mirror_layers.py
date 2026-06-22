"""PyTorch custom layers that MIRROR the MATLAB custom layers EXACTLY, so weights
transfer 1:1 and forward outputs match numerically.

MATLAB layers mirrored:
  - DeformableConvolution2DLayer  -> MirrorDCN
  - CoordConv2DLayer              -> MirrorCoordConv

Conventions matched to MATLAB's deformableSample / CoordConv predict:
  * DCN offset channels: first K = dy, next K = dx (K=kH*kW). Kernel tap order
    t = i + j*kH  (row/i fastest), matching MATLAB meshgrid(0:kW-1,0:kH-1)(:).
  * 1-based sampling coords with validity mask on [1,H]x[1,W] (zero outside),
    bilinear; main conv via im2col GEMM with the same tap ordering.
  * CoordConv: append i,j coord channels in linspace(-1,1) (i along H rows, j
    along W cols), order [X, i, j], then convolve.

Each layer has .to_matlab() returning {paramName: ndarray} in MATLAB layout
(conv weight [kH,kW,inC,outC]; bias [1,1,outC]).
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def matlab_roi_pool(feats, rois, image_size=480, pool=7):
    """RoI pooling mirroring MATLAB roiPooling.m EXACTLY (differentiable).
    feats: list [P2,P3,P4,P5] each [1,C,H,W]. rois: [M,4] xyxy in image coords.
    Returns [M,C,pool,pool] (H=row, W=col -> matches MirrorFC flatten).
    Level by canonicalSize=sqrt(imH*imW)/4; per-bin center sample, clamp [1,cur], bilinear."""
    dev, dt = rois.device, feats[0].dtype
    C = feats[0].shape[1]; M = rois.shape[0]
    out = torch.zeros(M, C, pool, pool, device=dev, dtype=dt)
    if M == 0:
        return out
    w = (rois[:, 2] - rois[:, 0]).clamp(min=1); h = (rois[:, 3] - rois[:, 1]).clamp(min=1)
    canon = math.sqrt(image_size * image_size) / 4
    lvl = torch.floor(4 + torch.log2(torch.sqrt(w * h) / canon + 1e-6)).clamp(1, 4).long() - 1
    t = (torch.arange(pool, device=dev, dtype=dt) + 0.5) / pool          # bin-center fractions
    for L in range(4):
        idx = torch.where(lvl == L)[0]
        if idx.numel() == 0:
            continue
        fl = feats[L]; curH, curW = fl.shape[2], fl.shape[3]
        sX = curW / image_size; sY = curH / image_size
        n = idx.numel()
        xs1 = rois[idx, 0] * sX; xs2 = rois[idx, 2] * sX
        ys1 = rois[idx, 1] * sY; ys2 = rois[idx, 3] * sY
        xc = (xs1[:, None] + t[None, :] * (xs2 - xs1)[:, None]).clamp(1, curW)   # [n,pool] (cols)
        yc = (ys1[:, None] + t[None, :] * (ys2 - ys1)[:, None]).clamp(1, curH)   # [n,pool] (rows)
        gx = 2 * (xc - 1) / max(curW - 1, 1) - 1
        gy = 2 * (yc - 1) / max(curH - 1, 1) - 1
        gxx = gx[:, None, :].expand(n, pool, pool)   # [n, bi(row), bj(col)] -> x by col
        gyy = gy[:, :, None].expand(n, pool, pool)   # y by row
        grid = torch.stack([gxx, gyy], -1).reshape(1, n * pool * pool, 1, 2)
        s = F.grid_sample(fl.float(), grid.float(), mode='bilinear', padding_mode='border', align_corners=True)
        out[idx] = s.reshape(C, n, pool, pool).permute(1, 0, 2, 3).to(dt)
    return out


def _conv_w_to_matlab(w):   # [outC,inC,kH,kW] -> [kH,kW,inC,outC]
    return np.ascontiguousarray(np.transpose(w.detach().cpu().numpy(), (2, 3, 1, 0)))

def _bias_to_matlab(b):     # [C] -> [1,1,C]
    return np.ascontiguousarray(b.detach().cpu().numpy().reshape(1, 1, -1))


class MirrorCoordConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, stride=1, padding=0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.k, self.stride, self.padding = k, stride, padding
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch + 2, k, k) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        N, C, H, W = x.shape
        i = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(N, 1, H, W)
        j = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(N, 1, H, W)
        xc = torch.cat([x, i, j], dim=1)                      # [X, i, j]
        return F.conv2d(xc, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def to_matlab(self):
        return {"Weights": _conv_w_to_matlab(self.weight), "Bias": _bias_to_matlab(self.bias)}


class MirrorDCN(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kH = self.kW = k
        self.sH = self.sW = stride
        self.pH = self.pW = padding
        self.dH = self.dW = dilation
        K = k * k
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        # offset/mask predictors init to ZERO (MATLAB convention): offset=0, mask=0.5
        self.offset_weight = nn.Parameter(torch.zeros(2 * K, in_ch, k, k))
        self.offset_bias = nn.Parameter(torch.zeros(2 * K))
        self.mask_weight = nn.Parameter(torch.zeros(K, in_ch, k, k))
        self.mask_bias = nn.Parameter(torch.zeros(K))

    def forward(self, x):
        # GPU-efficient: vectorized over N/C with F.grid_sample (CUDA bilinear),
        # K grid_samples + K 1x1 convs. Same MATLAB convention (1-based coords,
        # tap t=i+j*kH, offset channels first-K dy / next-K dx, zero outside image).
        N, C, H, W = x.shape
        kH, kW, K = self.kH, self.kW, self.kH * self.kW
        dev, dt = x.device, x.dtype
        offsets = F.conv2d(x, self.offset_weight, self.offset_bias,
                           stride=self.sH, padding=self.pH, dilation=self.dH)
        masks = torch.sigmoid(F.conv2d(x, self.mask_weight, self.mask_bias,
                              stride=self.sH, padding=self.pH, dilation=self.dH))
        Ho, Wo = offsets.shape[2], offsets.shape[3]
        dy = offsets[:, :K]; dx = offsets[:, K:2 * K]
        oy = torch.arange(Ho, device=dev, dtype=dt).view(1, Ho, 1)
        ox = torch.arange(Wo, device=dev, dtype=dt).view(1, 1, Wo)
        out = None
        for t in range(K):
            i = t % kH; j = t // kH
            sampY = oy * self.sH + 1.0 + i * self.dH - self.pH + dy[:, t]   # [N,Ho,Wo] 1-based
            sampX = ox * self.sW + 1.0 + j * self.dW - self.pW + dx[:, t]
            gy = 2.0 * (sampY - 1.0) / max(H - 1, 1) - 1.0
            gx = 2.0 * (sampX - 1.0) / max(W - 1, 1) - 1.0
            grid = torch.stack([gx, gy], dim=-1)                            # [N,Ho,Wo,2]
            # 'border' clamp matches MATLAB's corner-clamp; the center-validity mask
            # zeros samples whose center falls outside the image (MATLAB convention).
            sampled = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
            valid = ((sampY >= 1) & (sampY <= H) & (sampX >= 1) & (sampX <= W)).to(dt).unsqueeze(1)
            sampled = sampled * masks[:, t:t + 1] * valid                    # modulation + validity
            wt = self.weight[:, :, i, j].reshape(self.out_ch, self.in_ch, 1, 1)
            contrib = F.conv2d(sampled, wt)
            out = contrib if out is None else out + contrib
        return out + self.bias.view(1, self.out_ch, 1, 1)

    def to_matlab(self):
        return {
            "Weights":       _conv_w_to_matlab(self.weight),
            "Bias":          _bias_to_matlab(self.bias),
            "OffsetWeights": _conv_w_to_matlab(self.offset_weight),
            "OffsetBias":    _bias_to_matlab(self.offset_bias),
            "MaskWeights":   _conv_w_to_matlab(self.mask_weight),
            "MaskBias":      _bias_to_matlab(self.mask_bias),
        }
