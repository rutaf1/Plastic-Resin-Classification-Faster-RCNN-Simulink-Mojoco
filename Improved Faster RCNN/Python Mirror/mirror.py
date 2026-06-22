"""Full PyTorch mirror of the MATLAB net (Faster_RCNN_Modified.m). This file
holds the BACKBONE (ResNet50-vd + DCNv2) mirror; modules are named EXACTLY like
the MATLAB layers so weights transfer 1:1. FPN/SPP/heads are added next.

Backbone layout @ input 480x480: stem -> C2(120) -> C3(60) -> C4(30) -> C5(15).
Conv layers carry bias (MATLAB convs have Bias). BN eps=1e-5 (MATLAB default).
DCN uses MirrorDCN (verified to match MATLAB to ~1e-7).
"""

from __future__ import annotations
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _matlab_same_pad(x, k, s, value=0.0):
    """Replicate MATLAB 'same' padding for stride>1 (asymmetric: extra pad at
    bottom/right). For stride 1 + odd k this equals symmetric (k-1)/2 padding."""
    def p(n):
        o = math.ceil(n / s)
        tot = max((o - 1) * s + k - n, 0)
        return tot // 2, tot - tot // 2     # (before, after) — MATLAB floors the 'before' pad
    pt, pb = p(x.shape[-2]); pl, pr = p(x.shape[-1])
    return F.pad(x, (pl, pr, pt, pb), value=value)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirror_layers import MirrorDCN, MirrorCoordConv, _conv_w_to_matlab, _bias_to_matlab


class BackboneMirror(nn.Module):
    def __init__(self):
        super().__init__()
        L = nn.ModuleDict()
        def C(name, i, o, k, s):
            L[name] = nn.Conv2d(i, o, k, stride=s, padding=(0 if k == 1 else 1), bias=True)
        def B(name, c):
            L[name] = nn.BatchNorm2d(c, eps=1e-5)
        def D(name, i, o, s):
            L[name] = MirrorDCN(i, o, 3, s, 1)

        # ---- stem (no BN/relu, like MATLAB) ----
        # stem 'conv' is stride-2 'same': use pad=0 here + MATLAB-same pad in forward.
        L['conv'] = nn.Conv2d(3, 32, 3, stride=2, padding=0, bias=True)
        C('conv_2', 32, 32, 3, 1); C('conv_1', 32, 64, 3, 1)
        # ---- stage1 (256 out, 64 inner, no DCN, stride 1) ----
        C('conv_3',64,64,1,1);  B('batchnorm',64);   C('conv_4',64,64,3,1);  B('batchnorm_1',64);  C('conv_5',64,256,1,1);  B('batchnorm_2',256)
        C('conv_6',64,256,1,1); B('batchnorm_3',256)                                   # shortcut
        C('conv_7',256,64,1,1); B('batchnorm_4',64);  C('conv_8',64,64,3,1);  B('batchnorm_5',64);  C('conv_9',64,256,1,1);  B('batchnorm_6',256)
        C('conv_10',256,64,1,1);B('batchnorm_7',64);  C('conv_11',64,64,3,1); B('batchnorm_8',64);  C('conv_12',64,256,1,1); B('batchnorm_9',256)
        # ---- stage2 (512 out, 128 inner, DCN, stride 2) ----
        C('conv_13',256,128,1,1);B('batchnorm_10',128); D('deformConv',128,128,2);   B('batchnorm_12',128); C('conv_15',128,512,1,1); B('batchnorm_13',512)
        C('conv_14',256,512,1,1);B('batchnorm_11',512)                                 # shortcut (after avgpool)
        C('conv_17',512,128,1,1);B('batchnorm_14',128); D('deformConv_1',128,128,1); B('batchnorm_15',128); C('conv_16',128,512,1,1); B('batchnorm_16',512)
        C('conv_19',512,128,1,1);B('batchnorm_17',128); D('deformConv_2',128,128,1); B('batchnorm_18',128); C('conv_18',128,512,1,1); B('batchnorm_19',512)
        C('conv_21',512,128,1,1);B('batchnorm_20',128); D('deformConv_3',128,128,1); B('batchnorm_21',128); C('conv_20',128,512,1,1); B('batchnorm_22',512)
        # ---- stage3 (1024 out, 256 inner, DCN, stride 2) ----
        C('conv_22',512,256,1,1);B('batchnorm_23',256); D('deformConv_4',256,256,2);  B('batchnorm_25',256); C('conv_23',256,1024,1,1); B('batchnorm_26',1024)
        C('conv_41',512,1024,1,1);B('batchnorm_24',1024)                               # shortcut
        C('conv_25',1024,256,1,1);B('batchnorm_27',256); D('deformConv_5',256,256,1); B('batchnorm_28',256); C('conv_24',256,1024,1,1); B('batchnorm_29',1024)
        C('conv_27',1024,256,1,1);B('batchnorm_30',256); D('deformConv_6',256,256,1); B('batchnorm_31',256); C('conv_26',256,1024,1,1); B('batchnorm_32',1024)
        C('conv_29',1024,256,1,1);B('batchnorm_33',256); D('deformConv_7',256,256,1); B('batchnorm_34',256); C('conv_28',256,1024,1,1); B('batchnorm_35',1024)
        C('conv_31',1024,256,1,1);B('batchnorm_36',256); D('deformConv_8',256,256,1); B('batchnorm_37',256); C('conv_30',256,1024,1,1); B('batchnorm_38',1024)
        C('conv_33',1024,256,1,1);B('batchnorm_39',256); D('deformConv_9',256,256,1); B('batchnorm_40',256); C('conv_32',256,1024,1,1); B('batchnorm_41',1024)
        # ---- stage4 (2048 out, 512 inner, DCN, stride 2) ----
        C('conv_34',1024,512,1,1);B('batchnorm_42',512); D('deformConv_10',512,512,2);B('batchnorm_44',512); C('conv_36',512,2048,1,1); B('batchnorm_45',2048)
        C('conv_35',1024,2048,1,1);B('batchnorm_43',2048)                              # shortcut
        C('conv_38',2048,512,1,1);B('batchnorm_46',512); D('deformConv_11',512,512,1);B('batchnorm_47',512); C('conv_37',512,2048,1,1); B('batchnorm_48',2048)
        C('conv_40',2048,512,1,1);B('batchnorm_49',512); D('deformConv_12',512,512,1);B('batchnorm_50',512); C('conv_39',512,2048,1,1); B('batchnorm_51',2048)
        self.L = L

    def _bottleneck(self, x, c1, b1, mid, bmid, c3, b3):
        L = self.L
        y = F.relu(L[b1](L[c1](x)))
        y = F.relu(L[bmid](L[mid](y)))
        y = L[b3](L[c3](y))
        return y

    def forward(self, x):
        return self._backbone(x)

    def _backbone(self, x):
        L = self.L
        # stem (stride-2 conv + maxpool use MATLAB 'same' asymmetric padding)
        x = L['conv'](_matlab_same_pad(x, 3, 2))
        x = L['conv_2'](x); x = L['conv_1'](x)
        x = F.max_pool2d(_matlab_same_pad(x, 3, 2, value=float('-inf')), 3, stride=2, padding=0)
        # stage1
        y = self._bottleneck(x,'conv_3','batchnorm','conv_4','batchnorm_1','conv_5','batchnorm_2')
        sc = L['batchnorm_3'](L['conv_6'](x));               r = F.relu(y + sc)
        y = self._bottleneck(r,'conv_7','batchnorm_4','conv_8','batchnorm_5','conv_9','batchnorm_6');   r = F.relu(y + r)
        y = self._bottleneck(r,'conv_10','batchnorm_7','conv_11','batchnorm_8','conv_12','batchnorm_9'); C2 = F.relu(y + r)
        # stage2
        y = self._bottleneck(C2,'conv_13','batchnorm_10','deformConv','batchnorm_12','conv_15','batchnorm_13')
        sc = L['batchnorm_11'](L['conv_14'](F.avg_pool2d(C2,2,2)));  r = F.relu(y + sc)
        y = self._bottleneck(r,'conv_17','batchnorm_14','deformConv_1','batchnorm_15','conv_16','batchnorm_16'); r = F.relu(y + r)
        y = self._bottleneck(r,'conv_19','batchnorm_17','deformConv_2','batchnorm_18','conv_18','batchnorm_19'); r = F.relu(y + r)
        y = self._bottleneck(r,'conv_21','batchnorm_20','deformConv_3','batchnorm_21','conv_20','batchnorm_22'); C3 = F.relu(y + r)
        # stage3
        y = self._bottleneck(C3,'conv_22','batchnorm_23','deformConv_4','batchnorm_25','conv_23','batchnorm_26')
        sc = L['batchnorm_24'](L['conv_41'](F.avg_pool2d(C3,2,2)));  r = F.relu(y + sc)
        y = self._bottleneck(r,'conv_25','batchnorm_27','deformConv_5','batchnorm_28','conv_24','batchnorm_29'); r = F.relu(y + r)
        y = self._bottleneck(r,'conv_27','batchnorm_30','deformConv_6','batchnorm_31','conv_26','batchnorm_32'); r = F.relu(y + r)
        y = self._bottleneck(r,'conv_29','batchnorm_33','deformConv_7','batchnorm_34','conv_28','batchnorm_35'); r = F.relu(y + r)
        y = self._bottleneck(r,'conv_31','batchnorm_36','deformConv_8','batchnorm_37','conv_30','batchnorm_38'); r = F.relu(y + r)
        y = self._bottleneck(r,'conv_33','batchnorm_39','deformConv_9','batchnorm_40','conv_32','batchnorm_41'); C4 = F.relu(y + r)
        # stage4
        y = self._bottleneck(C4,'conv_34','batchnorm_42','deformConv_10','batchnorm_44','conv_36','batchnorm_45')
        sc = L['batchnorm_43'](L['conv_35'](F.avg_pool2d(C4,2,2)));  r = F.relu(y + sc)
        y = self._bottleneck(r,'conv_38','batchnorm_46','deformConv_11','batchnorm_47','conv_37','batchnorm_48'); r = F.relu(y + r)
        y = self._bottleneck(r,'conv_40','batchnorm_49','deformConv_12','batchnorm_50','conv_39','batchnorm_51'); C5 = F.relu(y + r)
        return C2, C3, C4, C5

    def export_weights(self):
        """Return {matlab_layer_name: {paramName: ndarray (MATLAB layout)}}."""
        out = {}
        for name, m in self.L.items():
            if isinstance(m, nn.Conv2d):
                out[name] = {"Weights": _conv_w_to_matlab(m.weight), "Bias": _bias_to_matlab(m.bias)}
            elif isinstance(m, nn.BatchNorm2d):
                r = lambda v: np.ascontiguousarray(v.detach().cpu().numpy().reshape(1, 1, -1))
                out[name] = {"Scale": r(m.weight), "Offset": r(m.bias),
                             "TrainedMean": r(m.running_mean), "TrainedVariance": r(m.running_var)}
            elif isinstance(m, (MirrorDCN, MirrorCoordConv)):
                out[name] = m.to_matlab()
        return out


class FullMirror(BackboneMirror):
    """Backbone + Enhanced-FPN (CoordConv laterals + SPP) + RPN heads, mirroring
    the full MATLAB net. forward() returns the 12 outputs used for the round-trip
    and detection: scoresP2-5, boxDeltasP2-5, and the RoI feature maps
    conv_54/conv_55/conv_56/coordConv_11."""
    def __init__(self):
        super().__init__()
        L = self.L
        def C(name, i, o, k):
            L[name] = nn.Conv2d(i, o, k, stride=1, padding=(0 if k == 1 else 1), bias=True)
        def CC(name, i, o):                      # 1x1 CoordConv (+2 coord channels)
            L[name] = MirrorCoordConv(i, o, 1, 1, 0)
        def B(name, c):
            L[name] = nn.BatchNorm2d(c, eps=1e-5)
        # P5 lateral chain (from C5=2048)
        CC('coordConv_9', 2048, 256); C('conv_51', 256, 512, 3); CC('coordConv_10', 512, 256)
        # P2 lateral chain (from C2=256)
        CC('coordConv', 256, 256);    C('conv_42', 256, 512, 3); CC('coordConv_1', 512, 256)
        # P3 lateral chain (from C3=512)
        CC('coordConv_3', 512, 256);  C('conv_45', 256, 512, 3); CC('coordConv_4', 512, 256)
        # P4 lateral chain (from C4=1024)
        CC('coordConv_6', 1024, 256); C('conv_48', 256, 512, 3); CC('coordConv_7', 512, 256)
        # SPP blocks (concat 1024 -> conv1x1 512 -> bn -> conv3x3 512 -> CoordConv 256)
        C('conv_43', 1024, 512, 1); B('batchnorm_52', 512); C('conv_44', 512, 512, 3); CC('coordConv_2', 512, 256)
        C('conv_46', 1024, 512, 1); B('batchnorm_53', 512); C('conv_47', 512, 512, 3); CC('coordConv_5', 512, 256)
        C('conv_49', 1024, 512, 1); B('batchnorm_54', 512); C('conv_50', 512, 512, 3); CC('coordConv_8', 512, 256)
        C('conv_52', 1024, 512, 1); B('batchnorm_55', 512); C('conv_53', 512, 512, 3); CC('coordConv_11', 512, 256)
        # top-down merge convs (concat 512 -> 256) + P5 conv
        C('conv_54', 512, 256, 3); C('conv_55', 512, 256, 3); C('conv_56', 512, 256, 3); C('conv_57', 256, 256, 3)
        # RPN heads (5 anchors): scores 10ch, deltas 20ch
        for s in ['P2', 'P3', 'P4', 'P5']:
            C(f'scores{s}', 256, 10, 1); C(f'boxDeltas{s}', 256, 20, 1)

    def _spp_concat(self, x):
        # MATLAB SPP concat order = [pool9, pool5, pool13, skip]  (stride1, 'same')
        p5  = F.max_pool2d(x, 5,  stride=1, padding=2)
        p9  = F.max_pool2d(x, 9,  stride=1, padding=4)
        p13 = F.max_pool2d(x, 13, stride=1, padding=6)
        return torch.cat([p9, p5, p13, x], dim=1)

    def _fpn_level(self, x, cc0, conv_a, cc1, conv_b, bn, conv_c, cc_out):
        L = self.L
        x = L[cc0](x); x = L[conv_a](x); x = L[cc1](x)
        x = self._spp_concat(x)
        x = L[conv_b](x); x = L[bn](x); x = L[conv_c](x); x = L[cc_out](x)
        return x

    def forward(self, x):
        L = self.L
        C2, C3, C4, C5 = self._backbone(x)
        up = lambda t: F.interpolate(t, scale_factor=2, mode='nearest')
        sppP2 = self._fpn_level(C2, 'coordConv',   'conv_42', 'coordConv_1', 'conv_43', 'batchnorm_52', 'conv_44', 'coordConv_2')
        sppP3 = self._fpn_level(C3, 'coordConv_3', 'conv_45', 'coordConv_4', 'conv_46', 'batchnorm_53', 'conv_47', 'coordConv_5')
        sppP4 = self._fpn_level(C4, 'coordConv_6', 'conv_48', 'coordConv_7', 'conv_49', 'batchnorm_54', 'conv_50', 'coordConv_8')
        sppP5 = self._fpn_level(C5, 'coordConv_9', 'conv_51', 'coordConv_10', 'conv_52', 'batchnorm_55', 'conv_53', 'coordConv_11')
        P2 = L['conv_54'](torch.cat([sppP2, up(sppP3)], dim=1))   # concat_1
        P3 = L['conv_55'](torch.cat([sppP3, up(sppP4)], dim=1))   # concat_3
        P4 = L['conv_56'](torch.cat([sppP4, up(sppP5)], dim=1))   # concat_5
        P5 = L['conv_57'](sppP5)
        feats = {'P2': P2, 'P3': P3, 'P4': P4, 'P5': P5}
        out = {}
        for s, f in feats.items():
            out['scores' + s] = L['scores' + s](f)
            out['boxDeltas' + s] = L['boxDeltas' + s](f)
        out['conv_54'] = P2; out['conv_55'] = P3; out['conv_56'] = P4
        out['coordConv_11'] = sppP5
        return out


class MirrorFC(nn.Module):
    """Mirror of the MATLAB fcNet (detection head): RoI [7,7,256] -> fc1(1024) ->
    relu -> fc2(1024) -> relu -> fc3_cls(numClasses+1)+softmax, fc3_bbox(numClasses*4)."""
    def __init__(self, num_fm=256, pool=7, num_classes=7):
        super().__init__()
        self.pool, self.num_fm = pool, num_fm
        self.fc1 = nn.Linear(pool * pool * num_fm, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3_cls = nn.Linear(1024, num_classes + 1)
        self.fc3_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, roi):                       # roi [N,C,H,W]
        # MATLAB fullyConnectedLayer flattens [H,W,C] COLUMN-MAJOR (h fastest, w, c).
        # permute to [N,C,W,H] then row-major reshape reproduces that order.
        x = roi.permute(0, 1, 3, 2).reshape(roi.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls = F.softmax(self.fc3_cls(x), dim=1)
        bbox = self.fc3_bbox(x)
        return cls, bbox

    def to_matlab(self):
        def fc(lin):  # MATLAB FC: Weights [out,in], Bias [out,1] (same 2-D as PyTorch)
            return {"Weights": np.ascontiguousarray(lin.weight.detach().cpu().numpy()),
                    "Bias": np.ascontiguousarray(lin.bias.detach().cpu().numpy().reshape(-1, 1))}
        return {"fc1": fc(self.fc1), "fc2": fc(self.fc2),
                "fc3_cls": fc(self.fc3_cls), "fc3_bbox": fc(self.fc3_bbox)}
