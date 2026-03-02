"""
COCO Detection dataset for use with FMTK detection decoders (e.g. FasterRCNNDecoder).

Expects the standard COCO directory layout:
    dataset_path/
        train2017/          (or val2017/)
        annotations/
            instances_train2017.json
            instances_val2017.json

Returns dicts compatible with pipeline.train/train_eval (batch["y"] = targets):
    {
        "x":  Tensor [C, H, W],
        "y":  {"boxes": Tensor [N,4], "labels": Tensor [N], "image_id", "area"},
        "idx": int,
    }
"""

import os
import torch
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
from fmtk.datasets.base import VisionDataset


# Standard COCO category IDs are not contiguous (1-90 with gaps).
# We build a mapping from COCO cat_id -> contiguous 1..80 at runtime.
# Background = 0 is reserved by torchvision convention.


class COCODetectionDataset(VisionDataset):
    """
    COCO object detection dataset.

    Parameters (via dataset_cfg dict)
    ----------------------------------
    dataset_path : str
        Root directory containing image folders and annotations/.
    target_size : int or tuple
        Resize images to this size (default 1024, matching ViTDet).
    mean / std : list[float]
        Normalization parameters (default ImageNet).
    model_id : str, optional
        If provided, uses HuggingFace AutoImageProcessor instead of
        torchvision transforms.
    min_area : float
        Skip annotations with area < min_area (default 0).
    """

    ANNOTATION_FILE = {
        "train": "instances_train2017.json",
        "val": "instances_val2017.json",
        "test": "instances_val2017.json",
    }

    IMAGE_DIR = {
        "train": "train2017",
        "val": "val2017",
        "test": "val2017",
    }

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.dataset_path = dataset_cfg.get("dataset_path", "./data/coco")
        self.target_size = dataset_cfg.get("target_size", 1024)
        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        self.min_area = dataset_cfg.get("min_area", 0)
        self.transform = dataset_cfg.get("transform", None)

        # Paths
        ann_file = os.path.join(
            self.dataset_path, "annotations", self.ANNOTATION_FILE[split]
        )
        self._image_dir = os.path.join(self.dataset_path, self.IMAGE_DIR[split])

        # Load COCO API
        print(f"[COCO] Loading annotations from {ann_file} ...")
        self.coco = COCO(ann_file)

        # Build contiguous category mapping: coco_cat_id -> 0..num_classes-1
        # RF-DETR/DETR-family uses 0-indexed labels; no-object is handled
        # separately by the criterion (it fills unmatched queries with num_classes).
        cat_ids = sorted(self.coco.getCatIds())
        self._cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}
        self.class_names = [
            self.coco.loadCats([cid])[0]["name"] for cid in cat_ids
        ]
        self.num_classes = len(self.class_names)

        # Filter image IDs: keep only images that have at least one annotation
        all_img_ids = sorted(self.coco.getImgIds())
        self._image_ids = []
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            if len(ann_ids) > 0:
                self._image_ids.append(img_id)

        print(
            f"[COCO] split={split}  images={len(self._image_ids)}  "
            f"classes={self.num_classes}"
        )

        if self.transform is None and preprocess:
            self.preprocess()

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, idx):
        img_id = self._image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self._image_dir, img_info["file_name"])

        # Load image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        orig_size = torch.as_tensor([orig_h, orig_w], dtype=torch.int64)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        for ann in anns:
            if ann.get("area", 0) < self.min_area:
                continue
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, width, height]
            if w <= 0 or h <= 0:
                continue
            # Convert to (x1, y1, x2, y2)
            boxes.append([x, y, x + w, y + h])
            labels.append(self._cat_id_to_label[ann["category_id"]])
            areas.append(ann.get("area", w * h))

        # Apply image transform
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Scale boxes if image was resized
        if image.shape[1] != orig_h or image.shape[2] != orig_w:
            scale_y = image.shape[1] / orig_h
            scale_x = image.shape[2] / orig_w
            boxes = [
                [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                for b in boxes
            ]

        # Build target dict
        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            areas_t = torch.as_tensor(areas, dtype=torch.float32)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)

        targets = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id]),
            "area": areas_t,
            "orig_size": orig_size,
        }

        return {"x": image, "y": targets, "idx": idx}

    @property
    def labels(self):
        """Not meaningful for detection (multi-label per image); returns image IDs."""
        return self._image_ids

    @property
    def indices(self):
        return torch.arange(len(self))

    def preprocess(self):
        """Build default transform: resize + normalize."""
        if isinstance(self.target_size, int):
            size = (self.target_size, self.target_size)
        else:
            size = self.target_size

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def get_class_names(self):
        return self.class_names

    def get_coco_api(self):
        """Return the underlying pycocotools COCO object for evaluation."""
        return self.coco

    def get_image_id(self, idx):
        """Return the COCO image ID for a given dataset index."""
        return self._image_ids[idx]


def coco_collate_fn(batch):
    """
    Custom collate function for detection DataLoader.

    Standard default_collate cannot handle variable-length target dicts.
    This stacks images and keeps targets as a list of dicts.
    Uses batch["y"] for pipeline compatibility.

    Usage:
        DataLoader(dataset, collate_fn=coco_collate_fn, ...)
    """
    images = torch.stack([item["x"] for item in batch], dim=0)
    targets = [item["y"] for item in batch]
    indices = [item["idx"] for item in batch]
    return {"x": images, "y": targets, "idx": indices}
