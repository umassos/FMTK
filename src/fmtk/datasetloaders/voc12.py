import os
import json
import base64
import zlib
import io
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from fmtk.datasetloaders.base import VisionDataset


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES)  # 21 (including background)
IGNORE_INDEX = 255


class VOC12Dataset(VisionDataset):
    """
    Pascal VOC 2012 semantic segmentation dataset.

    Expects a Supervisely-format layout on disk:

        <dataset_path>/
        ├── train/
        │   ├── img/          (*.jpg)
        │   └── ann/          (*.jpg.json – Supervisely bitmap annotations)
        ├── val/
        │   ├── img/
        │   └── ann/
        └── ...

    Each ``__getitem__`` returns::

        {
            "x": image_tensor,          # [3, target_size, target_size]
            "y": segmentation_tensor,   # [target_size, target_size]  int64, class ids 0-20, 255=ignore
            "idx": int,
        }

    Parameters
    ----------
    dataset_cfg : dict
        Must contain ``dataset_path``.  Optional keys:
        ``target_size`` (default 512), ``mean``, ``std``.
    task_cfg : dict
        Task configuration (e.g. ``{"task_type": "segmentation"}``).
    split : str
        One of ``"train"``, ``"val"``, or ``"test"`` (test maps to val).
    preprocess : bool
        Whether to build the transform pipeline on init.
    """

    SPLIT_MAP = {
        "train": "train",
        "val": "val",
        "test": "val",
        "trainval": "trainval",
    }

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        self.target_size = dataset_cfg.get("target_size", 512)
        self.num_channels = 3
        self.num_classes = NUM_CLASSES
        self.class_names = list(VOC_CLASSES)
        self.ignore_index = IGNORE_INDEX

        self.transform = None

        dataset_path = dataset_cfg.get("dataset_path", "./data")
        self._dataset_path = dataset_path

        disk_split = self.SPLIT_MAP.get(split)
        if disk_split is None:
            raise ValueError(
                f"Unsupported split '{split}'. Use 'train', 'val', or 'test'."
            )

        img_dir = os.path.join(dataset_path, disk_split, "img")
        ann_dir = os.path.join(dataset_path, disk_split, "ann")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.isdir(ann_dir):
            raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

        self._image_paths = []
        self._ann_paths = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ann_name = fname + ".json"
            ann_path = os.path.join(ann_dir, ann_name)
            if not os.path.isfile(ann_path):
                continue
            self._image_paths.append(os.path.join(img_dir, fname))
            self._ann_paths.append(ann_path)

        print(
            f"[VOC12] split={split} ({disk_split}), "
            f"images={len(self._image_paths)}, classes={self.num_classes}"
        )

        if preprocess:
            self.preprocess()

    # ------------------------------------------------------------------
    # Annotation decoding
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_bitmap(bitmap_data):
        """Decode a Supervisely bitmap: base64 -> zlib -> PNG -> numpy bool mask."""
        raw = base64.b64decode(bitmap_data)
        raw = zlib.decompress(raw)
        mask = np.array(Image.open(io.BytesIO(raw)))
        return mask.astype(bool)

    @staticmethod
    def _build_segmentation_mask(ann, height, width):
        """Convert Supervisely JSON annotation to a (H, W) class-index mask."""
        seg = np.zeros((height, width), dtype=np.int64)

        for obj in ann.get("objects", []):
            class_title = obj.get("classTitle", "")
            if class_title == "neutral":
                class_idx = IGNORE_INDEX
            elif class_title in CLASS_TO_IDX:
                class_idx = CLASS_TO_IDX[class_title]
            else:
                continue

            geom = obj.get("geometryType", "")
            if geom != "bitmap":
                continue

            bitmap_info = obj["bitmap"]
            origin_x, origin_y = bitmap_info["origin"]
            mask = VOC12Dataset._decode_bitmap(bitmap_info["data"])

            mh, mw = mask.shape
            y1, y2 = origin_y, min(origin_y + mh, height)
            x1, x2 = origin_x, min(origin_x + mw, width)
            seg[y1:y2, x1:x2][mask[: y2 - y1, : x2 - x1]] = class_idx

        return seg

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image = Image.open(self._image_paths[idx]).convert("RGB")

        with open(self._ann_paths[idx], "r") as f:
            ann = json.load(f)

        h, w = ann["size"]["height"], ann["size"]["width"]
        seg = self._build_segmentation_mask(ann, h, w)
        seg = Image.fromarray(seg.astype(np.uint8), mode="L")

        if self.transform is not None:
            image = self.transform(image)
            seg = TF.resize(
                seg,
                [self.target_size, self.target_size],
                interpolation=TF.InterpolationMode.NEAREST,
            )

        seg = torch.as_tensor(np.array(seg), dtype=torch.long)

        return {"x": image, "y": seg, "idx": idx}

    @property
    def labels(self):
        return None

    @property
    def indices(self):
        return torch.arange(len(self))

    def preprocess(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.target_size, self.target_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def get_class_names(self):
        return self.class_names