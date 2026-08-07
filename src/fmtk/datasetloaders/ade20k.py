import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from fmtk.datasetloaders.base import VisionDataset


ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book",
    "hill", "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel",
    "bus", "towel", "light", "truck", "tower", "chandelier", "awning",
    "streetlight", "booth", "television receiver", "airplane", "dirt track",
    "apparel", "pole", "land", "bannister", "escalator", "ottoman",
    "bottle", "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike",
    "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase",
    "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
    "plate", "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag",
]

NUM_CLASSES = len(ADE20K_CLASSES)   # 150
IGNORE_INDEX = 255


class ADE20KDataset(VisionDataset):
    """
    ADE20K semantic segmentation dataset.

    Expects the standard ADEChallengeData2016 layout on disk::

        <dataset_path>/
        ├── images/
        │   ├── training/    *.jpg
        │   └── validation/  *.jpg
        └── annotations/
            ├── training/    *.png   (uint8, 0=unlabeled, 1-150=class index)
            └── validation/  *.png

    Raw annotation values:
    - ``0``     → unlabeled / ignore (mapped to ``255``)
    - ``1-150`` → class indices (mapped to ``0-149``)

    Each ``__getitem__`` returns::

        {
            "x":   image_tensor,          # [3, target_size, target_size]
            "y":   segmentation_tensor,   # [target_size, target_size]  int64, 0-149 / 255=ignore
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
        One of ``"train"`` or ``"val"`` (``"test"`` maps to ``"validation"``).
    preprocess : bool
        Whether to build the transform pipeline on init.
    """

    # ADE20K uses "training" / "validation" as directory names
    SPLIT_MAP = {"train": "training", "val": "validation", "test": "validation"}

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        self.target_size = dataset_cfg.get("target_size", 512)
        self.num_channels = 3
        self.num_classes = NUM_CLASSES
        self.class_names = list(ADE20K_CLASSES)
        self.ignore_index = IGNORE_INDEX

        self.transform = None

        dataset_path = dataset_cfg.get("dataset_path", "./data")
        self._dataset_path = dataset_path

        disk_split = self.SPLIT_MAP.get(split)
        if disk_split is None:
            raise ValueError(
                f"Unsupported split '{split}'. Use 'train', 'val', or 'test'."
            )

        img_dir = os.path.join(dataset_path, "images", disk_split)
        ann_dir = os.path.join(dataset_path, "annotations", disk_split)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.isdir(ann_dir):
            raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

        self._image_paths = []
        self._ann_paths = []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fn)[0]
            ann_path = os.path.join(ann_dir, f"{stem}.png")
            if os.path.isfile(ann_path):
                self._image_paths.append(os.path.join(img_dir, fn))
                self._ann_paths.append(ann_path)

        print(
            f"[ADE20KDataset] split={split} ({disk_split}), "
            f"images={len(self._image_paths)}, classes={self.num_classes}"
        )

        if preprocess:
            self.preprocess()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image = Image.open(self._image_paths[idx]).convert("RGB")

        ann = np.array(Image.open(self._ann_paths[idx]), dtype=np.int64)
        # 0 = unlabeled → ignore (255); 1-150 → 0-149
        seg_ids = ann - 1                          # 0-149, unlabeled becomes -1
        seg_ids[ann == 0] = IGNORE_INDEX           # fix unlabeled → 255

        seg = Image.fromarray(seg_ids.astype(np.uint8), mode="L")

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
