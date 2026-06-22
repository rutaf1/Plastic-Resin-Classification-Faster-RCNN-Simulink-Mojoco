"""Datasets for the Improved Faster R-CNN re-implementation.

This module supplies the data plumbing for the steel-surface-defect detector of
Wang et al., "Automatic Detection and Classification of Steel Surface Defect
Using Deep CNNs", Metals 2021. The paper targets RGB inputs of fixed size
480 x 480 x 3 with 7 foreground defect classes (labels 1..7; label 0 is reserved
for the background class used internally by the torchvision detector).

Two datasets are provided:

  * :class:`SyntheticDefectDataset` -- procedurally generates random images and
    boxes so the full training / inference pipeline can be exercised end-to-end
    without any real data (useful for smoke tests and CI). It is fully
    deterministic per index via an index-seeded ``torch.Generator`` and never
    touches the global RNG.
  * :class:`CocoStyleDefectDataset` -- a documented stub that loads real images
    from a folder plus a COCO-style annotation JSON (via ``pycocotools``),
    resizing every image to 480 x 480 and scaling the boxes accordingly.

Both return targets in the exact dict format consumed by
``torchvision.models.detection.FasterRCNN``::

    target = {
        'boxes':    FloatTensor[N, 4],   # xyxy, absolute pixels, within image
        'labels':   LongTensor[N],       # values in 1..7 (foreground classes)
        'image_id': LongTensor[1],
        'area':     FloatTensor[N],      # box areas (COCO 'area' convention)
        'iscrowd':  LongTensor[N],       # zeros (no crowd annotations)
    }

This module is intentionally free of any ``config`` import; everything is
parameterized so train.py / inference.py remain the only config readers.
"""

import os
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = [
    "SyntheticDefectDataset",
    "CocoStyleDefectDataset",
    "collate_fn",
]

# Fixed input geometry from the journal (480 x 480 x 3 RGB).
IMAGE_SIZE: int = 480
NUM_FOREGROUND_CLASSES: int = 7


class SyntheticDefectDataset(Dataset):
    """Deterministic synthetic dataset of random images + boxes.

    Generates ``[3, H, W]`` float images in ``[0, 1]`` together with 1..3 valid
    random boxes and foreground labels in ``1..num_classes``. Intended purely as
    a stand-in for the real steel-defect data so that the model, training loop
    and inference code can be run end-to-end (e.g. in ``tests/smoke_test.py``).

    Determinism: every sample is produced from a per-index
    :class:`torch.Generator` seeded with ``base_seed + index``. The same index
    therefore always yields the same image/target, and the global PyTorch RNG is
    never read or mutated -- so dataloader worker shuffling and model init are
    unaffected.

    Args:
        length: Number of samples reported by ``__len__`` (fixed dataset size).
        image_size: Square spatial size H == W (default 480, journal value).
        num_classes: Number of foreground classes; labels are drawn from
            ``1..num_classes`` (default 7, journal value). Label 0 is reserved
            for background and is never emitted.
        min_boxes: Minimum number of boxes per image (default 1).
        max_boxes: Maximum number of boxes per image (default 3).
        base_seed: Offset added to ``index`` to seed the per-sample generator.
        channels: Number of image channels (default 3 for RGB).
    """

    def __init__(
        self,
        length: int = 16,
        image_size: int = IMAGE_SIZE,
        num_classes: int = NUM_FOREGROUND_CLASSES,
        min_boxes: int = 1,
        max_boxes: int = 3,
        base_seed: int = 0,
        channels: int = 3,
    ) -> None:
        super().__init__()
        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        if image_size < 2:
            raise ValueError(f"image_size must be >= 2, got {image_size}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if not (1 <= min_boxes <= max_boxes):
            raise ValueError(
                "require 1 <= min_boxes <= max_boxes, got "
                f"min_boxes={min_boxes}, max_boxes={max_boxes}"
            )
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        self.length = int(length)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.min_boxes = int(min_boxes)
        self.max_boxes = int(max_boxes)
        self.base_seed = int(base_seed)
        self.channels = int(channels)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[Tensor, dict]:
        """Generate the deterministic ``(image, target)`` pair for ``index``."""
        if index < 0:
            index += self.length
        if not (0 <= index < self.length):
            raise IndexError(
                f"index {index} out of range for dataset of length {self.length}"
            )

        # Per-sample generator -> deterministic AND independent of global RNG.
        gen = torch.Generator()
        gen.manual_seed(self.base_seed + index)

        size = self.image_size

        # ---- Image: random RGB in [0, 1], shape [C, H, W] -----------------
        image = torch.rand(
            (self.channels, size, size), generator=gen, dtype=torch.float32
        )

        # ---- Number of boxes in [min_boxes, max_boxes] --------------------
        span = self.max_boxes - self.min_boxes + 1
        num_boxes = int(
            torch.randint(0, span, (1,), generator=gen).item()
        ) + self.min_boxes

        boxes = torch.empty((num_boxes, 4), dtype=torch.float32)
        for i in range(num_boxes):
            boxes[i] = self._random_box(gen, size)

        labels = torch.randint(
            1, self.num_classes + 1, (num_boxes,), generator=gen, dtype=torch.int64
        )

        # COCO 'area' convention: width * height of the box.
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        area = widths * heights

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((num_boxes,), dtype=torch.int64),
        }
        return image, target

    @staticmethod
    def _random_box(gen: torch.Generator, size: int) -> Tensor:
        """Sample one valid xyxy box strictly inside ``[0, size]``.

        Guarantees ``0 <= x1 < x2 <= size`` and ``0 <= y1 < y2 <= size`` with at
        least 1px width/height, so the box is always well-formed for the
        detector (no degenerate / out-of-bounds boxes).
        """
        # Draw two distinct x coordinates and two distinct y coordinates.
        # randint high is exclusive; using `size` keeps coords in [0, size-1],
        # then we add +1 to the max to guarantee x2 > x1 (width >= 1) and to
        # allow the far edge `size` to be reachable.
        x = torch.randint(0, size, (2,), generator=gen).tolist()
        y = torch.randint(0, size, (2,), generator=gen).tolist()

        x1, x2 = min(x), max(x)
        y1, y2 = min(y), max(y)

        # Ensure non-zero extent; clamp the upper edge to `size`.
        if x2 <= x1:
            x2 = min(x1 + 1, size)
        if y2 <= y1:
            y2 = min(y1 + 1, size)
        # If x1 was already at the last index, shift the lower edge instead.
        if x2 <= x1:
            x1 = x2 - 1
        if y2 <= y1:
            y1 = y2 - 1

        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


class CocoStyleDefectDataset(Dataset):
    """COCO-style dataset stub for the real steel-defect data (documented).

    Loads images from ``image_dir`` and annotations from a COCO-format JSON file
    using ``pycocotools``. Each image is resized to ``image_size x image_size``
    (the journal's fixed 480 x 480 input) and the boxes are scaled by the same
    per-axis factors so they stay aligned with the resized image. The returned
    target uses the identical schema as :class:`SyntheticDefectDataset`.

    The COCO ``category_id`` values are expected to map to the 7 foreground
    defect classes. By default they are passed through unchanged (assuming the
    JSON already uses ids 1..7); supply ``category_id_to_label`` to remap
    arbitrary category ids onto the contiguous ``1..num_classes`` range.

    .. note::
        This is a *stub*: it implements the full loading / resizing / box-scaling
        logic but has not been exercised against a concrete dataset here. Adjust
        ``category_id_to_label`` and (optionally) the transform to match your
        annotation file before training on real data.

    Args:
        image_dir: Directory containing the image files referenced by the JSON.
        annotation_file: Path to the COCO-style annotation JSON.
        image_size: Square output size H == W (default 480).
        num_classes: Number of foreground classes (default 7), used only to
            validate remapped labels.
        category_id_to_label: Optional dict mapping raw COCO ``category_id`` ->
            contiguous label in ``1..num_classes``. If ``None``, category ids are
            used verbatim as labels.
        transforms: Optional callable ``(image_tensor, target) -> (image, target)``
            applied after resizing/box-scaling (e.g. augmentation/normalization).

    Raises:
        ImportError: If ``pycocotools`` is not installed (with an informative
            message telling the user how to install it).
        FileNotFoundError: If ``image_dir`` or ``annotation_file`` is missing.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        image_size: int = IMAGE_SIZE,
        num_classes: int = NUM_FOREGROUND_CLASSES,
        category_id_to_label: Optional[dict] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        try:
            from pycocotools.coco import COCO  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "CocoStyleDefectDataset requires 'pycocotools', which is not "
                "installed. Install it with `pip install pycocotools` (on "
                "Windows: `pip install pycocotools-windows`) to load COCO-style "
                "steel-defect annotations."
            ) from exc

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"image_dir not found: {image_dir!r}")
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(
                f"annotation_file not found: {annotation_file!r}"
            )

        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.category_id_to_label = category_id_to_label
        self.transforms = transforms

        self.coco = COCO(annotation_file)
        # Sorted, stable ordering of image ids for reproducible indexing.
        self.ids: List[int] = sorted(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[Tensor, dict]:
        """Load, resize and box-scale the COCO sample at position ``index``."""
        # Imported lazily so the module imports without Pillow when only the
        # synthetic dataset is used.
        from PIL import Image
        import torchvision.transforms.functional as TF

        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        path = os.path.join(self.image_dir, file_name)

        # Load as RGB and remember the original size for box scaling.
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
            orig_w, orig_h = pil_img.size  # PIL reports (width, height)
            resized = pil_img.resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )

        # [3, H, W] float tensor in [0, 1].
        image = TF.to_tensor(resized)

        # Per-axis scale factors mapping original pixels -> resized pixels.
        scale_x = self.image_size / float(orig_w)
        scale_y = self.image_size / float(orig_h)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes_list: List[List[float]] = []
        labels_list: List[int] = []
        iscrowd_list: List[int] = []
        for ann in anns:
            # COCO bbox is [x, y, width, height] in original-image pixels.
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue  # skip degenerate annotations
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y

            # Clip to the resized image bounds and re-check validity.
            x1 = min(max(x1, 0.0), self.image_size)
            y1 = min(max(y1, 0.0), self.image_size)
            x2 = min(max(x2, 0.0), self.image_size)
            y2 = min(max(y2, 0.0), self.image_size)
            if x2 <= x1 or y2 <= y1:
                continue

            raw_cat = ann["category_id"]
            if self.category_id_to_label is not None:
                if raw_cat not in self.category_id_to_label:
                    raise KeyError(
                        f"category_id {raw_cat} missing from "
                        "category_id_to_label mapping"
                    )
                label = self.category_id_to_label[raw_cat]
            else:
                label = raw_cat
            if not (1 <= label <= self.num_classes):
                raise ValueError(
                    f"label {label} out of foreground range "
                    f"[1, {self.num_classes}] for annotation {ann.get('id')}"
                )

            boxes_list.append([x1, y1, x2, y2])
            labels_list.append(int(label))
            iscrowd_list.append(int(ann.get("iscrowd", 0)))

        if boxes_list:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            iscrowd = torch.as_tensor(iscrowd_list, dtype=torch.int64)
        else:
            # Empty-annotation image: return correctly-shaped empty tensors.
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    """Standard detection collate: transpose a list of samples into tuples.

    Object-detection targets have a variable number of boxes per image and so
    cannot be stacked into a single tensor. torchvision detectors instead
    consume a *list* of images and a parallel *list* of target dicts, which is
    exactly what this collate produces.

    Args:
        batch: List of ``(image, target)`` tuples from the dataset.

    Returns:
        Tuple ``(images, targets)`` where each element is a tuple of length
        ``len(batch)``.
    """
    return tuple(zip(*batch))
