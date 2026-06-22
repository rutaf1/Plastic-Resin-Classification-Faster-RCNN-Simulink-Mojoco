"""Train the FullMirror + MirrorFC (the PyTorch model that EXACTLY mirrors the
MATLAB net) on the CVAT plastic-defect dataset, maximizing the RTX 4080 Super
(16 GB). After training, weights export to .mat and load straight into the MATLAB
net (load_weights_from_mat.m) for inference.

Two-stage logic ported from Training.m (anchors [0.25,0.5,1,2,5] base
[20,40,72,120]; RPN objectness BCE on the fg score channel + bbox smooth-L1;
Matrix NMS proposals; RoI sampling + det cls/bbox). GPU-max: AMP autocast +
GradScaler, channels_last, cudnn.benchmark, pinned loaders.

Run:
  python train_mirror.py --epochs 200 --batch 20 --lr 1e-4
  python train_mirror.py --smoke           # 2-iter GPU sanity + VRAM report
"""
import os, sys, argparse, math, time
# Reduce CUDA fragmentation so reserved memory tracks allocated more tightly — gives
# effective headroom at batch 20 on the 16 GB card (must be set before importing torch).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'python_alternatif'))
from mirror import FullMirror, MirrorFC
from mirror_layers import matlab_roi_pool
from data.cvat_dataset import build_train_val_datasets, collate_fn
from metrics import DetStatAccumulator, TrainingLogger
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x=None, **k): return x if x is not None else None

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True   # faster fp32 matmul on Ada (RTX 4080)
torch.backends.cudnn.allow_tf32 = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STRIDES = [4, 8, 16, 32]          # P2..P5 (feature 120/60/30/15 @ 480)
BASES   = [20, 40, 72, 120]
RATIOS  = [0.25, 0.5, 1.0, 2.0, 5.0]
NA = len(RATIOS)


# --------------------------- anchors ---------------------------
def level_anchors(fh, fw, stride, base, device):
    ws = torch.tensor([base * math.sqrt(r) for r in RATIOS], device=device)
    hs = torch.tensor([base / math.sqrt(r) for r in RATIOS], device=device)
    ys = (torch.arange(fh, device=device) + 0.5) * stride
    xs = (torch.arange(fw, device=device) + 0.5) * stride
    cy, cx = torch.meshgrid(ys, xs, indexing='ij')         # [fh,fw] (h-major)
    cx = cx.reshape(-1); cy = cy.reshape(-1)               # location-major
    # anchor-major: for each anchor a, all locations
    A = []
    for a in range(NA):
        A.append(torch.stack([cx - ws[a] / 2, cy - hs[a] / 2,
                              cx + ws[a] / 2, cy + hs[a] / 2], 1))
    return torch.cat(A, 0)                                  # [NA*fh*fw, 4] xyxy


def box_iou(a, b):
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def encode(anchors, gt):
    aw = anchors[:, 2] - anchors[:, 0]; ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + aw / 2; acy = anchors[:, 1] + ah / 2
    gw = gt[:, 2] - gt[:, 0]; gh = gt[:, 3] - gt[:, 1]
    gcx = gt[:, 0] + gw / 2; gcy = gt[:, 1] + gh / 2
    return torch.stack([(gcx - acx) / aw, (gcy - acy) / ah,
                        torch.log(gw / aw), torch.log(gh / ah)], 1)


def decode(anchors, d):
    aw = anchors[:, 2] - anchors[:, 0]; ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + aw / 2; acy = anchors[:, 1] + ah / 2
    cx = d[:, 0] * aw + acx; cy = d[:, 1] * ah + acy
    w = torch.exp(d[:, 2].clamp(max=4)) * aw; h = torch.exp(d[:, 3].clamp(max=4)) * ah
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def matrix_nms(boxes, scores, sigma=0.2, score_thr=0.05, topk=300):
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    s, order = scores.sort(descending=True)
    b = boxes[order]
    iou = box_iou(b, b).triu(1)
    iou_cmax = iou.max(0).values
    decay = torch.exp(-(iou ** 2 - iou_cmax[:, None] ** 2) / sigma).min(0).values
    decay[0] = 1.0
    ns = s * decay
    keep = ns >= score_thr
    idx = order[keep]; ns = ns[keep]
    o2 = ns.sort(descending=True).indices
    return idx[o2][:topk]


# --------------------------- RPN reshape ---------------------------
def rpn_flatten(scores_list, deltas_list):
    """Concatenate RPN outputs over levels -> [N, total_anchors, 2], [..,4],
    anchor-major within each level (matching level_anchors)."""
    N = scores_list[0].shape[0]
    sc, dl = [], []
    for s, d in zip(scores_list, deltas_list):
        H, W = s.shape[2], s.shape[3]
        s = s.view(N, NA, 2, H, W).permute(0, 1, 3, 4, 2).reshape(N, NA * H * W, 2)
        d = d.view(N, NA, 4, H, W).permute(0, 1, 3, 4, 2).reshape(N, NA * H * W, 4)
        sc.append(s); dl.append(d)
    return torch.cat(sc, 1), torch.cat(dl, 1)


def rpn_loss(scoresF, deltasF, anchors, gts, n_samp=256, pos_frac=0.2, pos_iou=0.7, neg_iou=0.3):
    dev = scoresF.device
    cls_losses, reg_losses = [], []
    for n in range(scoresF.shape[0]):
        gt = gts[n]
        if gt.numel() == 0:
            cls_losses.append(scoresF[n][:, 1].sum() * 0); continue
        iou = box_iou(anchors, gt)
        maxiou, arg = iou.max(1)
        labels = torch.full((anchors.shape[0],), -1, device=dev)
        labels[maxiou < neg_iou] = 0
        labels[maxiou >= pos_iou] = 1
        labels[iou.max(0).indices] = 1            # best anchor per GT
        pos = torch.where(labels == 1)[0]; neg = torch.where(labels == 0)[0]
        npos = min(int(n_samp * pos_frac), pos.numel())
        nneg = min(n_samp - npos, neg.numel())
        if pos.numel() > npos: pos = pos[torch.randperm(pos.numel(), device=dev)[:npos]]
        if neg.numel() > nneg: neg = neg[torch.randperm(neg.numel(), device=dev)[:nneg]]
        samp = torch.cat([pos, neg])
        obj = scoresF[n][samp, 1]
        tgt = torch.cat([torch.ones(pos.numel(), device=dev), torch.zeros(neg.numel(), device=dev)])
        cls_losses.append(F.binary_cross_entropy_with_logits(obj, tgt))
        if pos.numel():
            td = encode(anchors[pos], gt[arg[pos]])
            reg_losses.append(F.smooth_l1_loss(deltasF[n][pos], td))
    cls = torch.stack(cls_losses).mean() if cls_losses else scoresF.sum() * 0
    reg = torch.stack(reg_losses).mean() if reg_losses else scoresF.sum() * 0
    return cls, reg


def get_proposals(scoresF, deltasF, anchors, n, max_pre=2000, max_post=300):
    obj = torch.sigmoid(scoresF[n][:, 1])
    d = deltasF[n]
    if obj.numel() > max_pre:
        top = obj.topk(max_pre).indices; obj = obj[top]; d = d[top]; anc = anchors[top]
    else:
        anc = anchors
    boxes = decode(anc, d).detach()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, 479); boxes[:, 1::2] = boxes[:, 1::2].clamp(0, 479)
    keep = matrix_nms(boxes, obj.detach(), topk=max_post)
    return boxes[keep]


# --------------------------- RoI + det ---------------------------
# RoI pooling = matlab_roi_pool (ported from MATLAB roiPooling.m, verified 2.6e-6).

def sample_rois(props, gt, gt_lab, n_samp=128, pos_frac=0.35, pos_iou=0.5, neg_iou=0.5, ncls=7):
    dev = props.device
    if props.numel() == 0 or gt.numel() == 0:
        return props[:0], torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, ncls * 4, device=dev)
    iou = box_iou(props, gt); maxiou, arg = iou.max(1)
    pos = torch.where(maxiou >= pos_iou)[0]; neg = torch.where(maxiou < neg_iou)[0]
    npos = min(max(int(n_samp * pos_frac), 1), pos.numel())
    nneg = min(n_samp - npos, neg.numel())
    if pos.numel() > npos: pos = pos[torch.randperm(pos.numel(), device=dev)[:npos]]
    if neg.numel() > nneg: neg = neg[torch.randperm(neg.numel(), device=dev)[:nneg]]
    samp = torch.cat([pos, neg])
    labels = torch.full((samp.numel(),), ncls, dtype=torch.long, device=dev)  # bg = ncls (0-based)
    labels[:pos.numel()] = (gt_lab[arg[pos]] - 1).long()                       # fg classes 0..ncls-1
    tgt = torch.zeros(samp.numel(), ncls * 4, device=dev)
    if pos.numel():
        td = encode(props[pos], gt[arg[pos]])
        for k in range(pos.numel()):
            c = int(gt_lab[arg[pos][k]] - 1)
            tgt[k, c * 4:c * 4 + 4] = td[k]
    return props[samp], labels, tgt


def det_loss(cls_probs, bbox_pred, labels, targets, ncls=7):
    # fc returns softmax probs (to match MATLAB classScores); CE via NLL on log-probs.
    clsl = F.nll_loss(torch.log(cls_probs.clamp_min(1e-8)), labels)
    fg = labels < ncls
    if fg.any():
        regl = F.smooth_l1_loss(bbox_pred[fg], targets[fg])
    else:
        regl = bbox_pred.sum() * 0
    return clsl, regl


# --------------------------- train ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    # batch 20 ~= 14.6 GB (91% of 16 GB) — set for training from scratch.
    ap.add_argument('--batch', type=int, default=20)
    ap.add_argument('--lr', type=float, default=2e-4, help='AdamW base LR (peak after warmup)')
    ap.add_argument('--warmup-iters', type=int, default=500, help='linear LR warmup iterations')
    ap.add_argument('--grad-clip', type=float, default=10.0, help='max grad norm (0 to disable)')
    ap.add_argument('--data-root', default=os.path.join(os.path.dirname(HERE), 'dataset'))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--train-eval-batches', type=int, default=10, help='batches for train-metric eval (0=all)')
    ap.add_argument('--val-batches', type=int, default=0, help='batches for val eval (0=all)')
    ap.add_argument('--out', default=os.path.join(HERE, 'trained_mirror.pt'))
    ap.add_argument('--resume', default='', help='path to checkpoint to resume from (leave empty to auto-detect latest)')
    ap.add_argument('--workers', type=int, default=6, help='DataLoader workers (overlap PNG decode/resize with GPU; 0 = main thread)')
    args = ap.parse_args()

    net = FullMirror().to(DEVICE, memory_format=torch.channels_last)
    fc = MirrorFC(256, 7, 7).to(DEVICE)
    train_ds, val_ds = build_train_val_datasets(
        os.path.join(args.data_root, 'images_aug'),
        os.path.join(args.data_root, 'annotations_aug.xml'), input_size=480)
    _dl_extra = dict(persistent_workers=True, prefetch_factor=4) if args.workers > 0 else {}
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_fn, pin_memory=True, drop_last=True, **_dl_extra)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            collate_fn=collate_fn, pin_memory=True, **_dl_extra)
    # AdamW: weight decay only on >=2D weights (conv/linear), NOT on BN/bias/1D —
    # standard practice for AdamW (decaying norm/bias params hurts).
    decay, no_decay = [], []
    for mod in (net, fc):
        for _pn, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
    params = decay + no_decay
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.08},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler()
    levels = ['P2', 'P3', 'P4', 'P5']

    # LR schedule = linear warmup (warmup_iters) then cosine to eta_min, computed
    # purely from the global iter count -> no scheduler state, resume-safe by design.
    iters_per_epoch = max(1, len(loader))
    total_iters = args.epochs * iters_per_epoch
    base_lr, eta_min, warmup_iters = args.lr, 1e-6, args.warmup_iters

    def lr_at(git):
        if git < warmup_iters:
            return base_lr * (git + 1) / warmup_iters
        prog = (git - warmup_iters) / max(1, total_iters - warmup_iters)
        return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def _anchors_for(out):
        return torch.cat([level_anchors(out['scores' + s].shape[2], out['scores' + s].shape[3],
                          STRIDES[i], BASES[i], DEVICE) for i, s in enumerate(levels)], 0)

    ar4 = torch.arange(4, device=DEVICE)

    def _forward(imgs):
        """Shared backbone + RPN forward -> (anchors, scoresF, deltasF, fmaps).
        The expensive part; reused by the loss path AND the detection path so eval
        runs ONE forward per batch instead of two."""
        out = net(imgs)
        anchors = _anchors_for(out)
        scoresF, deltasF = rpn_flatten([out['scores' + s] for s in levels],
                                       [out['boxDeltas' + s] for s in levels])
        fmaps = [out['conv_54'], out['conv_55'], out['conv_56'], out['coordConv_11']]
        return anchors, scoresF, deltasF, fmaps

    def _loss_from(anchors, scoresF, deltasF, fmaps, gts, labs, nimg):
        rc, rr = rpn_loss(scoresF.float(), deltasF.float(), anchors, gts)
        dc_list, dr_list = [], []
        for n in range(nimg):
            props = get_proposals(scoresF.float(), deltasF.float(), anchors, n)
            rois, rlab, rtgt = sample_rois(props, gts[n], labs[n])
            if rois.shape[0] == 0:
                continue
            rf = matlab_roi_pool([f[n:n + 1] for f in fmaps], rois, 480, 7).to(memory_format=torch.channels_last)
            cl, bb = fc(rf)
            dc, dr = det_loss(cl.float(), bb.float(), rlab, rtgt)
            dc_list.append(dc); dr_list.append(dr)
        if dc_list:
            dcls = torch.stack(dc_list).mean(); dreg = torch.stack(dr_list).mean()
        else:
            dcls = dreg = scoresF.sum() * 0
        return rc + rr + dcls + dreg, (rc, rr, dcls, dreg)

    def _detect_from(anchors, scoresF, deltasF, fmaps, nimg, score_thr):
        """Decode detections from a forward result. Low score_thr -> mAP sees the
        full ranked list; DetStatAccumulator applies its own threshold for accuracy."""
        scoresF = scoresF.float(); deltasF = deltasF.float()
        preds = []
        for n in range(nimg):
            props = get_proposals(scoresF, deltasF, anchors, n, max_post=30)
            if props.shape[0] == 0:
                preds.append({'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long),
                              'scores': torch.zeros(0)}); continue
            rf = matlab_roi_pool([f[n:n + 1] for f in fmaps], props, 480, 7).to(memory_format=torch.channels_last)
            cl, bb = fc(rf); cl = cl.float(); bb = bb.float()
            sc, cls = cl[:, :7].max(1)                          # fg class (exclude bg col 7)
            deltas = bb.gather(1, cls[:, None] * 4 + ar4[None, :])
            boxes = decode(props, deltas)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, 479); boxes[:, 1::2] = boxes[:, 1::2].clamp(0, 479)
            keep0 = sc >= score_thr
            boxes, lab, scr = boxes[keep0], cls[keep0] + 1, sc[keep0]
            if boxes.shape[0]:
                k = matrix_nms(boxes, scr, topk=100)
                boxes, lab, scr = boxes[k], lab[k], scr[k]
            preds.append({'boxes': boxes.cpu(), 'labels': lab.cpu(), 'scores': scr.cpu()})
        return preds

    def step(images, targets):
        imgs = torch.stack([im.to(DEVICE, non_blocking=True) for im in images]).to(memory_format=torch.channels_last)
        gts = [t['boxes'].to(DEVICE) for t in targets]
        labs = [t['labels'].to(DEVICE) for t in targets]
        with torch.cuda.amp.autocast():
            anchors, scoresF, deltasF, fmaps = _forward(imgs)
            loss, comps = _loss_from(anchors, scoresF, deltasF, fmaps, gts, labs, imgs.shape[0])
        return loss, comps

    @torch.no_grad()
    def detect(imgs_t, score_thr=0.05):
        """Standalone inference -> per-image {boxes,labels(1-based),scores}."""
        anchors, scoresF, deltasF, fmaps = _forward(imgs_t)
        return _detect_from(anchors, scoresF, deltasF, fmaps, imgs_t.shape[0], score_thr)

    @torch.no_grad()
    def evaluate(eval_loader, max_batches=0, acc_score_thr=0.3, det_score_thr=0.05, tag='val'):
        was_train = net.training
        net.eval(); fc.eval()
        # accuracy/P/R/F1 measured at acc_score_thr (0.5); detect returns low-threshold
        # preds (det_score_thr=0.05) so mAP gets the full ranked list (proper COCO mAP).
        accD = DetStatAccumulator(iou_thresh=0.5, score_thresh=acc_score_thr)
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            mapm = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
        except Exception:
            mapm = None
        lsum = 0.0; nb = 0
        bar = tqdm(eval_loader, desc=f'eval[{tag}]', ncols=90, leave=False)
        for images, targets in (bar if bar is not None else eval_loader):
            imgs = torch.stack([im.to(DEVICE, non_blocking=True) for im in images]).to(memory_format=torch.channels_last)
            gts = [t['boxes'].to(DEVICE) for t in targets]
            labs = [t['labels'].to(DEVICE) for t in targets]
            with torch.cuda.amp.autocast():
                anchors, scoresF, deltasF, fmaps = _forward(imgs)          # ONE forward for both
                loss, _ = _loss_from(anchors, scoresF, deltasF, fmaps, gts, labs, imgs.shape[0])
                preds = _detect_from(anchors, scoresF, deltasF, fmaps, imgs.shape[0], det_score_thr)
            lsum += float(loss)
            tg = [{'boxes': t['boxes'], 'labels': t['labels']} for t in targets]
            accD.update(preds, tg)
            if mapm is not None:
                mapm.update(preds, tg)
            nb += 1
            if max_batches and nb >= max_batches:
                break
        m = accD.compute(); m['loss'] = lsum / max(nb, 1)
        if mapm is not None:
            r = mapm.compute(); m['map'] = float(r['map']); m['map50'] = float(r['map_50'])
        else:
            m['map'] = m['map50'] = float('nan')
        if was_train: net.train(); fc.train()
        return m

    net.train(); fc.train()
    if args.smoke:
        it = iter(loader)
        for i in range(2):
            images, targets = next(it)
            opt.zero_grad(set_to_none=True)
            loss, comps = step(images, targets)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f'iter {i}: loss={float(loss):.4f} rpn_cls={float(comps[0]):.3f} '
                  f'rpn_reg={float(comps[1]):.3f} det_cls={float(comps[2]):.3f} '
                  f'det_reg={float(comps[3]):.3f} | peakVRAM={mem:.2f} GB')
        vm = evaluate(val_loader, max_batches=2, tag='val')
        print(f'eval smoke: val_loss={vm["loss"]:.4f} acc={vm["accuracy"]:.4f} '
              f'rmse={vm["rmse"]:.3f} map={vm["map"]:.4f}')
        print('=== SMOKE OK ===')
        return

    out_dir = os.path.dirname(os.path.abspath(args.out))
    logger = TrainingLogger(os.path.join(out_dir, 'train_logs'))
    ckpt_path = os.path.join(out_dir, 'checkpoint_latest.pt')
    best_path = os.path.join(out_dir, 'best_model.pt')
    best_map = -1.0; giter = 0; start_epoch = 0

    # ---- resume logic ----
    resume_path = args.resume if args.resume else ckpt_path
    if os.path.isfile(resume_path):
        print(f'[resume] loading checkpoint: {resume_path}')
        ckpt = torch.load(resume_path, map_location=DEVICE)
        net.load_state_dict(ckpt['net'])
        fc.load_state_dict(ckpt['fc'])
        try:
            opt.load_state_dict(ckpt['opt'])
        except (ValueError, KeyError) as e:
            print(f'[resume] optimizer state incompatible ({e}); starting optimizer fresh.')
        start_epoch = ckpt.get('epoch', 0) + 1
        giter      = ckpt.get('iter', 0)
        best_map   = ckpt.get('best_map', -1.0)
        # LR is derived from giter via lr_at() — no scheduler state to restore.
        print(f'[resume] resumed from epoch {start_epoch}, iter {giter}, best_map={best_map:.4f}')
        torch.cuda.empty_cache()                                 # free fragmented VRAM after resume
    else:
        print('[resume] no checkpoint found, training from scratch')

    print(f'train={len(train_ds)} val={len(val_ds)} | logs -> {logger.out_dir}')

    for ep in range(start_epoch, args.epochs):
        net.train(); fc.train()
        t0 = time.time(); tot = 0.0; nb = 0
        bar = tqdm(loader, desc=f'Epoch {ep + 1}/{args.epochs}', ncols=110, dynamic_ncols=True)
        for images, targets in (bar if bar is not None else loader):
            lr_now = lr_at(giter)
            for g in opt.param_groups:
                g['lr'] = lr_now
            opt.zero_grad(set_to_none=True)
            loss, comps = step(images, targets)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(opt); scaler.update()
            tot += loss.detach().item(); nb += 1; giter += 1
            if giter % 20 == 0:                                 # checkpoint every 20 iterations
                torch.save({'net': net.state_dict(), 'fc': fc.state_dict(),
                            'opt': opt.state_dict(), 'epoch': ep, 'iter': giter,
                            'best_map': best_map}, ckpt_path)
            if bar is not None:
                bar.set_postfix(loss=f'{float(loss):.3f}', vram=f'{torch.cuda.max_memory_allocated()/1e9:.1f}G')
        train_loss = tot / max(nb, 1)

        trm = evaluate(loader, max_batches=args.train_eval_batches, tag='train')
        vam = evaluate(val_loader, max_batches=args.val_batches, tag='val')
        row = {'epoch': ep + 1, 'lr': opt.param_groups[0]['lr'], 'time_s': time.time() - t0,
               'train_loss': train_loss, 'val_loss': vam['loss'],
               'train_acc': trm['accuracy'], 'val_acc': vam['accuracy'],
               'train_rmse': trm['rmse'], 'val_rmse': vam['rmse'],
               'train_map': trm['map'], 'val_map': vam['map'], 'val_map50': vam['map50'],
               'val_precision': vam['precision'], 'val_recall': vam['recall'], 'val_f1': vam['f1']}
        logger.log(row); logger.plot()
        
        print(f'epoch {ep + 1}/{args.epochs}: '
              f'train[loss={train_loss:.3f} acc={trm["accuracy"]:.3f} rmse={trm["rmse"]:.2f}] '
              f'val[loss={vam["loss"]:.3f} acc={vam["accuracy"]:.3f} rmse={vam["rmse"]:.2f} '
              f'mAP={vam["map"]:.3f} mAP50={vam["map50"]:.3f} '
              f'P={vam["precision"]:.3f} R={vam["recall"]:.3f} F1={vam["f1"]:.3f}] '
              f'lr={row["lr"]:.2e} {row["time_s"]:.0f}s '
              f'peakVRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB')

        # checkpoint every 5 epochs
        if (ep + 1) % 5 == 0:
            ep_ckpt = os.path.join(out_dir, f'checkpoint_epoch_{ep + 1:04d}.pt')
            torch.save({'net': net.state_dict(), 'fc': fc.state_dict(),
                        'opt': opt.state_dict(), 'epoch': ep, 'iter': giter,
                        'best_map': best_map}, ep_ckpt)
            # also update latest so resume always points to most recent epoch boundary
            torch.save({'net': net.state_dict(), 'fc': fc.state_dict(),
                        'opt': opt.state_dict(), 'epoch': ep, 'iter': giter,
                        'best_map': best_map}, ckpt_path)
            print(f'   [ckpt] saved epoch checkpoint -> {ep_ckpt}')

        if vam['map'] > best_map:                               # save BEST model only
            best_map = vam['map']
            torch.save({'net': net.state_dict(), 'fc': fc.state_dict(),
                        'epoch': ep + 1, 'val_map': best_map}, best_path)
            print(f'   * new best (val mAP={best_map:.4f}) -> {best_path}')
    print(f'done. best val mAP={best_map:.4f} -> {best_path}')


if __name__ == '__main__':
    main()
