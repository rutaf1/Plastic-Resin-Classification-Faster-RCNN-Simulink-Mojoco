"""CVAT-XML dataset for the Improved Faster R-CNN (plastic-defect frames).

The annotations ship as a single CVAT-for-images ``annotations_aug.xml`` (v1.1)::

    <annotations>
      <image id="0" name="frame_000011.png" width="956" height="720">
        <box label="PET (1)" xtl="156.42" ytl="93.87" xbr="318.60" ybr="186.46" .../>
        ...
      </image>
      ...
    </annotations>

with 7 foreground classes encoded in the label text as ``"NAME (idx)"`` where
``idx`` is the 1-based class id (torchvision reserves 0 for background):

    1 PET   2 HDPE   3 PVC   4 LDPE   5 PP   6 PS   7 Other

Every frame is 956x720 and is **letterbox-resized to 480x480** (see
:mod:`data.transforms`) with bounding boxes transformed in lockstep, so the
images are network-ready and undistorted.

Train/val splitting groups augmented variants (``frame_XXXX_aug1`` /
``_aug2``) with their base frame (``frame_XXXX``) so that no augmentation of a
validation frame leaks into the training set.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .transforms import letterbox_resize

import random
import torchvision.transforms.functional as TF

#: Canonical class order; index i (0-based) corresponds to label id i+1.
CLASS_NAMES: List[str] = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]

#: Parses the 1-based class id out of a CVAT label like ``"HDPE (2)"``.
_LABEL_ID_RE = re.compile(r"\((\d+)\)")
#: Strips the augmentation suffix to recover the base frame name.
_AUG_SUFFIX_RE = re.compile(r"_aug\d+$", re.IGNORECASE)


def _label_to_id(label: str) -> int:
    """Map a CVAT label string to its 1-based foreground class id.

    Prefers the explicit ``(idx)`` in the label; falls back to a name lookup.
    """
    m = _LABEL_ID_RE.search(label)
    if m:
        return int(m.group(1))
    name = label.split("(")[0].strip()
    if name in CLASS_NAMES:
        return CLASS_NAMES.index(name) + 1
    raise ValueError(f"Cannot resolve class id from label {label!r}.")


def _base_frame(image_name: str) -> str:
    """``frame_000011_aug2.png`` -> ``frame_000011`` (groups augmentations)."""
    stem = os.path.splitext(image_name)[0]
    return _AUG_SUFFIX_RE.sub("", stem)


def parse_cvat_xml(xml_path: str) -> List[Dict]:
    """Parse a CVAT-for-images XML into a list of per-image annotation records.

    Returns a list of ``{"name", "width", "height", "boxes": [[x1,y1,x2,y2],...],
    "labels": [int,...]}`` dicts, one per ``<image>`` element (in file order).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records: List[Dict] = []
    for img_el in root.findall("image"):
        name = img_el.get("name")
        width = float(img_el.get("width", 0))
        height = float(img_el.get("height", 0))

        boxes: List[List[float]] = []
        labels: List[int] = []
        for box_el in img_el.findall("box"):
            x1 = float(box_el.get("xtl"))
            y1 = float(box_el.get("ytl"))
            x2 = float(box_el.get("xbr"))
            y2 = float(box_el.get("ybr"))
            # Normalize possibly-swapped corners and drop degenerate boxes.
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(_label_to_id(box_el.get("label")))

        records.append(
            {"name": name, "width": width, "height": height, "boxes": boxes, "labels": labels}
        )
    return records


def split_records(
    records: List[Dict], val_fraction: float = 0.2, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split records into (train, val) grouping augmentations with their base.

    Deterministic given ``seed``. All variants of a base frame land in the same
    split, preventing train/val leakage through augmentation.
    """
    # Collect unique base frames in stable (sorted) order.
    bases = sorted({_base_frame(r["name"]) for r in records})

    # Deterministic shuffle via a seeded generator (no global RNG dependency).
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(bases), generator=g).tolist()
    shuffled = [bases[i] for i in perm]

    n_val = int(round(len(shuffled) * val_fraction))
    val_bases = set(shuffled[:n_val])

    train, val = [], []
    for r in records:
        (val if _base_frame(r["name"]) in val_bases else train).append(r)
    return train, val


class CvatDefectDataset(Dataset):
    """Detection dataset over CVAT XML + an image folder, with letterbox resize.

    Args:
        image_dir:    folder containing the frame images (e.g. ``images_aug``).
        xml_path:     path to ``annotations_aug.xml``.
        input_size:   square network input side (default 480).
        split:        one of ``"train"`` / ``"val"`` / ``"all"``.
        val_fraction: fraction of *base frames* held out for validation.
        seed:         split RNG seed (deterministic).
        records:      optionally pass pre-parsed/pre-split records to avoid
                      re-parsing the XML for the second split (see
                      :func:`build_train_val_datasets`).

    Each item is ``(image_tensor[3,S,S] in [0,1], target)`` where ``target`` is
    the torchvision detection dict: ``boxes FloatTensor[N,4] xyxy`` (already in
    480x480 letterboxed coords), ``labels Int64Tensor[N]`` in ``1..7``,
    ``image_id``, ``area``, ``iscrowd``, plus ``orig_size`` / ``letterbox_meta``.
    """

    def __init__(
        self,
        image_dir: str,
        xml_path: Optional[str] = None,
        input_size: int = 480,
        split: str = "all",
        val_fraction: float = 0.2,
        seed: int = 42,
        records: Optional[List[Dict]] = None,
    ):
        self.image_dir = image_dir
        self.input_size = input_size
        self.split = split

        if records is None:
            if xml_path is None:
                raise ValueError("Provide either `xml_path` or pre-parsed `records`.")
            all_records = parse_cvat_xml(xml_path)
            if split == "all":
                records = all_records
            else:
                train, val = split_records(all_records, val_fraction, seed)
                records = train if split == "train" else val

        # Keep only records whose image file actually exists on disk.
        self.records = [r for r in records if os.path.exists(os.path.join(image_dir, r["name"]))]
        missing = len(records) - len(self.records)
        if missing:
            print(f"[CvatDefectDataset:{split}] WARNING: {missing} annotated image(s) not found on disk.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_path = os.path.join(self.image_dir, rec["name"])
        image = Image.open(img_path).convert("RGB")

        boxes = torch.as_tensor(rec["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(rec["labels"], dtype=torch.int64)

        canvas, new_boxes, meta = letterbox_resize(image, boxes, self.input_size)
        if new_boxes is None:
            new_boxes = torch.zeros((0, 4), dtype=torch.float32)

        # --- augmentasi acak HANYA untuk split train ---
        if self.split == "train":
            # 1. Random horizontal flip (p=0.5)
            if random.random() < 0.5:
                canvas = TF.hflip(canvas)
                if new_boxes.numel():
                    w = self.input_size
                    x1 = w - new_boxes[:, 2]
                    x2 = w - new_boxes[:, 0]
                    new_boxes[:, 0], new_boxes[:, 2] = x1, x2

            # 2. Random vertical flip (p=0.3) - sesuaikan kalau orientasi objek
            #    di line produksi memang bisa terbalik
            if random.random() < 0.3:
                canvas = TF.vflip(canvas)
                if new_boxes.numel():
                    h = self.input_size
                    y1 = h - new_boxes[:, 3]
                    y2 = h - new_boxes[:, 1]
                    new_boxes[:, 1], new_boxes[:, 3] = y1, y2

            # 3. Color jitter (tidak mengubah koordinat box, aman)
            canvas = TF.adjust_brightness(canvas, random.uniform(0.8, 1.2))
            canvas = TF.adjust_contrast(canvas, random.uniform(0.8, 1.2))
            canvas = TF.adjust_saturation(canvas, random.uniform(0.8, 1.2))

        keep = (new_boxes[:, 2] - new_boxes[:, 0] >= 1.0) & (new_boxes[:, 3] - new_boxes[:, 1] >= 1.0)
        new_boxes = new_boxes[keep]
        labels = labels[keep] if labels.numel() else labels




        image_tensor = to_tensor(canvas)  # [3, S, S] float in [0, 1]

        area = (
            (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
            if new_boxes.numel()
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": new_boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((new_boxes.shape[0],), dtype=torch.int64),
            "orig_size": torch.tensor(meta["orig_size"], dtype=torch.int64),
            "letterbox_meta": meta,
        }
        return image_tensor, target


def build_train_val_datasets(
    image_dir: str,
    xml_path: str,
    input_size: int = 480,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[CvatDefectDataset, CvatDefectDataset]:
    """Parse the XML once and return ``(train_ds, val_ds)`` with a clean split."""
    all_records = parse_cvat_xml(xml_path)
    train_rec, val_rec = split_records(all_records, val_fraction, seed)
    train_ds = CvatDefectDataset(image_dir, input_size=input_size, split="train", records=train_rec)
    val_ds = CvatDefectDataset(image_dir, input_size=input_size, split="val", records=val_rec)
    return train_ds, val_ds


def collate_fn(batch):
    """Standard detection collate: tuple of images, tuple of targets."""
    return tuple(zip(*batch))
