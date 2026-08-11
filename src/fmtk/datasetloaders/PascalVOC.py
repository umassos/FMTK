import base64
import io
import json
import os
import zlib

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fmtk.datasetloaders.base import VisionDataset

# Canonical VOC class order; index 0 is background (any pixel not covered
# by an annotated object bitmap).
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# The Supervisely export marks the boundary/ambiguous region around objects
# with a "neutral" class, which is the standard VOC ignore label.
IGNORE_LABEL = "neutral"
IGNORE_INDEX = 255


class PascalVOCDataset(VisionDataset):
    """
    PASCAL VOC semantic segmentation, read from a Supervisely-format export:
    `{split}/img/*.jpg` + `{split}/ann/*.jpg.json`, where each annotation JSON
    lists per-object bitmap masks (base64+zlib-compressed indexed PNGs, pasted
    at a pixel offset) rather than one flat segmentation PNG per image.
    """

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        dataset_path = dataset_cfg.get("dataset_path", "./data")
        self.img_dir = os.path.join(dataset_path, split, "img")
        self.ann_dir = os.path.join(dataset_path, split, "ann")
        if not os.path.isdir(self.img_dir):
            raise ValueError(f"No such split '{split}' under {dataset_path}")

        self.filenames = sorted(
            f for f in os.listdir(self.img_dir) if f.lower().endswith(".jpg")
        )

        self.class_names = VOC_CLASSES
        self.num_classes = len(VOC_CLASSES)
        self.ignore_index = IGNORE_INDEX
        self.class_to_idx = {name: i for i, name in enumerate(VOC_CLASSES)}

        self.target_size = dataset_cfg.get("target_size", 224)
        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        self.transform = None

        if preprocess:
            self.preprocess()

    def __len__(self):
        return len(self.filenames)

    def preprocess(self):
        self.transform = transforms.Compose([
            transforms.Resize(
                (self.target_size, self.target_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def _decode_bitmap(self, bitmap):
        """Supervisely bitmaps are base64+zlib-compressed indexed PNGs;
        convert('L') resolves the palette so any nonzero pixel is 'on'."""
        raw = base64.b64decode(bitmap["data"])
        png_bytes = zlib.decompress(raw)
        img = Image.open(io.BytesIO(png_bytes))
        return np.array(img.convert("L")) > 0

    def _build_mask(self, ann_path, height, width):
        with open(ann_path) as f:
            ann = json.load(f)

        mask = np.zeros((height, width), dtype=np.uint8)
        for obj in ann["objects"]:
            if obj.get("geometryType") != "bitmap":
                continue

            title = obj["classTitle"]
            if title == IGNORE_LABEL:
                label = self.ignore_index
            elif title in self.class_to_idx:
                label = self.class_to_idx[title]
            else:
                continue

            obj_mask = self._decode_bitmap(obj["bitmap"])
            ox, oy = obj["bitmap"]["origin"]
            h, w = obj_mask.shape
            region = mask[oy:oy + h, ox:ox + w]
            region[obj_mask] = label

        return mask

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = os.path.join(self.img_dir, filename)
        ann_path = os.path.join(self.ann_dir, f"{filename}.json")

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        mask = self._build_mask(ann_path, height, width)
        mask_img = Image.fromarray(mask, mode="L").resize(
            (self.target_size, self.target_size), resample=Image.NEAREST
        )
        mask_tensor = torch.from_numpy(np.array(mask_img)).long()

        if self.transform is not None:
            image = self.transform(image)

        return {"x": image, "y": mask_tensor, "idx": index}

    def get_class_names(self):
        return self.class_names
