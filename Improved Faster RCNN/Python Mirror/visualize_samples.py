"""Render N letterboxed samples with their transformed boxes for verification.

This is the visual acceptance test for the 956x720 -> 480x480 letterbox
pipeline: it loads samples through :class:`data.cvat_dataset.CvatDefectDataset`
(so the EXACT same transform used in training is exercised), draws the
transformed bounding boxes + class labels on the letterboxed canvas, and tiles
them into a single grid image. If the boxes hug the objects and the gray
letterbox bars are visible, the resize + box transform are correct.

Run (from the python_alternatif dir):
    python visualize_samples.py --num 6 --out samples_letterbox.png
"""

from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

from config import Config
from data.cvat_dataset import CLASS_NAMES, CvatDefectDataset

# One distinct color per foreground class id (1..7).
_CLASS_COLORS = {
    1: (231, 76, 60),    # PET   - red
    2: (46, 204, 113),   # HDPE  - green
    3: (52, 152, 219),   # PVC   - blue
    4: (241, 196, 15),   # LDPE  - yellow
    5: (155, 89, 182),   # PP    - purple
    6: (230, 126, 34),   # PS    - orange
    7: (26, 188, 156),   # Other - teal
}


def _font(size: int = 13):
    """Best-effort truetype font, falling back to PIL's bitmap default."""
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_sample(image_tensor, target, title: str) -> Image.Image:
    """Draw boxes+labels on one letterboxed sample, return an annotated image."""
    # [3,S,S] in [0,1] -> PIL RGB.
    arr = (image_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy())
    img = Image.fromarray(arr, mode="RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _font(13)

    boxes = target["boxes"].tolist()
    labels = target["labels"].tolist()
    for (x1, y1, x2, y2), lab in zip(boxes, labels):
        color = _CLASS_COLORS.get(int(lab), (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = CLASS_NAMES[int(lab) - 1] if 1 <= int(lab) <= len(CLASS_NAMES) else str(lab)
        tag = f"{name}({int(lab)})"
        tw = draw.textlength(tag, font=font)
        ty = max(0, y1 - 14)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + 14], fill=color)
        draw.text((x1 + 2, ty + 1), tag, fill=(0, 0, 0), font=font)

    # Title strip on top.
    strip_h = 18
    out = Image.new("RGB", (img.width, img.height + strip_h), (30, 30, 40))
    out.paste(img, (0, strip_h))
    d2 = ImageDraw.Draw(out)
    d2.text((4, 3), title, fill=(255, 255, 255), font=_font(12))
    return out


def make_grid(samples, cols: int = 3, pad: int = 6, bg=(20, 20, 28)) -> Image.Image:
    """Tile annotated sample images into a grid."""
    rows = (len(samples) + cols - 1) // cols
    cw = max(s.width for s in samples)
    ch = max(s.height for s in samples)
    grid = Image.new("RGB", (cols * cw + pad * (cols + 1), rows * ch + pad * (rows + 1)), bg)
    for i, s in enumerate(samples):
        r, c = divmod(i, cols)
        grid.paste(s, (pad + c * (cw + pad), pad + r * (ch + pad)))
    return grid


def pick_indices(dataset, num: int, random: bool = False, seed: int = 0):
    """Pick `num` indices, one per distinct base frame for variety.

    By default orders candidates by box count (most boxes first -> clearest
    visual test). With ``random=True`` the candidate order is a deterministic
    shuffle (seeded), so re-running yields a *different* set of samples.
    """
    n = len(dataset)
    if random:
        import torch
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(n, generator=g).tolist()
    else:
        order = sorted(range(n), key=lambda i: -len(dataset.records[i]["boxes"]))

    chosen, seen_bases = [], set()
    # Prefer variety: spread across distinct base frames.
    for i in order:
        base = dataset.records[i]["name"].split("_aug")[0]
        if base in seen_bases:
            continue
        seen_bases.add(base)
        chosen.append(i)
        if len(chosen) == num:
            break
    # Top up if we ran out of distinct bases.
    for i in order:
        if len(chosen) == num:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:num]


def main():
    p = argparse.ArgumentParser(description="Visualize letterboxed dataset samples.")
    p.add_argument("--num", type=int, default=6, help="number of samples (default 6)")
    p.add_argument("--image-dir", default=None, help="images folder (default: <DATA_ROOT>/images_aug)")
    p.add_argument("--xml", default=None, help="CVAT xml (default: <DATA_ROOT>/annotations_aug.xml)")
    p.add_argument("--out", default="samples_letterbox.png", help="output grid PNG")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--random", action="store_true", help="pick a random (seeded) set instead of the busiest frames")
    p.add_argument("--seed", type=int, default=0, help="seed for --random selection (vary for different samples)")
    args = p.parse_args()

    cfg = Config()
    data_root = cfg.DATA_ROOT
    image_dir = args.image_dir or os.path.join(data_root, "images_aug")
    xml_path = args.xml or os.path.join(data_root, "annotations_aug.xml")

    ds = CvatDefectDataset(image_dir, xml_path=xml_path, input_size=cfg.INPUT_SIZE, split="all")
    print(f"[visualize] dataset images: {len(ds)} | input size: {cfg.INPUT_SIZE}x{cfg.INPUT_SIZE}")

    idxs = pick_indices(ds, args.num, random=args.random, seed=args.seed)
    panels = []
    for i in idxs:
        img_t, tgt = ds[i]
        name = ds.records[i]["name"]
        meta = tgt["letterbox_meta"]
        title = (f"{name}  s={meta['scale']:.3f} pad=({meta['pad_left']},{meta['pad_top']}) "
                 f"boxes={tgt['boxes'].shape[0]}")
        panels.append(_draw_sample(img_t, tgt, title))
        print(f"  [{i}] {name}: {tgt['boxes'].shape[0]} box(es), "
              f"scale={meta['scale']:.4f}, pad_top={meta['pad_top']}")

    grid = make_grid(panels, cols=args.cols)
    grid.save(args.out)
    print(f"[visualize] saved grid -> {os.path.abspath(args.out)}  ({grid.width}x{grid.height})")


if __name__ == "__main__":
    main()
