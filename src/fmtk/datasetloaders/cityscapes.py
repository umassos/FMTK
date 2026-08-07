import os
import warnings
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from fmtk.datasetloaders.base import VisionDataset


CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

NUM_CLASSES = len(CITYSCAPES_CLASSES)   # 19
IGNORE_INDEX = 255


def _build_labelid_to_trainid_lut() -> np.ndarray:
    """Build a LUT[label_id] -> train_id using the official cityscapesscripts."""
    try:
        from cityscapesscripts.helpers import labels as cs_labels
    except ImportError as e:
        raise ImportError(
            "cityscapesscripts is required for CityscapesDataset. "
            "Install with: pip install cityscapesscripts"
        ) from e
    max_id = max(lab.id for lab in cs_labels.labels if lab.id >= 0)
    lut = np.full(max_id + 1, IGNORE_INDEX, dtype=np.int32)
    for lab in cs_labels.labels:
        if lab.id >= 0:
            lut[lab.id] = int(lab.trainId)
    return lut


class CityscapesDataset(VisionDataset):
    """
    Cityscapes semantic segmentation dataset.

    Expects the standard Cityscapes layout on disk::

        <dataset_path>/
        ├── leftImg8bit/
        │   ├── train/  <city>/  *_leftImg8bit.png
        │   └── val/    <city>/  *_leftImg8bit.png
        └── gtFine/
            ├── train/  <city>/  *_gtFine_labelIds.png
            └── val/    <city>/  *_gtFine_labelIds.png

    Label IDs are mapped to the 19 train-IDs via ``cityscapesscripts``.
    Unlabeled / ignored pixels receive ``255``.

    Each ``__getitem__`` returns::

        {
            "x":   image_tensor,          # [3, target_size, target_size]
            "y":   segmentation_tensor,   # [target_size, target_size]  int64, 0-18 / 255=ignore
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
        One of ``"train"`` or ``"val"`` (``"test"`` maps to ``"val"``).
    preprocess : bool
        Whether to build the transform pipeline on init.
    """

    SPLIT_MAP = {"train": "train", "val": "val", "test": "val"}

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        self.target_size = dataset_cfg.get("target_size", 512)
        self.num_channels = 3
        self.num_classes = NUM_CLASSES
        self.class_names = list(CITYSCAPES_CLASSES)
        self.ignore_index = IGNORE_INDEX

        self.transform = None
        self._lut = _build_labelid_to_trainid_lut()

        dataset_path = dataset_cfg.get("dataset_path", "./data")
        self._dataset_path = dataset_path

        disk_split = self.SPLIT_MAP.get(split)
        if disk_split is None:
            raise ValueError(
                f"Unsupported split '{split}'. Use 'train', 'val', or 'test'."
            )

        img_root = os.path.join(dataset_path, "leftImg8bit", disk_split)
        gt_root = os.path.join(dataset_path, "gtFine", disk_split)

        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"Image root not found: {img_root}")
        if not os.path.isdir(gt_root):
            raise FileNotFoundError(f"gtFine root not found: {gt_root}")

        self._image_paths = []
        self._gt_paths = []
        missing = 0
        for city in sorted(os.listdir(img_root)):
            city_img_dir = os.path.join(img_root, city)
            if not os.path.isdir(city_img_dir):
                continue
            gt_city_dir = os.path.join(gt_root, city)
            for fn in sorted(os.listdir(city_img_dir)):
                if not fn.endswith("_leftImg8bit.png"):
                    continue
                stem = fn.replace("_leftImg8bit.png", "")
                gt_path = os.path.join(gt_city_dir, f"{stem}_gtFine_labelIds.png")
                if os.path.isfile(gt_path):
                    self._image_paths.append(os.path.join(city_img_dir, fn))
                    self._gt_paths.append(gt_path)
                else:
                    missing += 1

        if missing:
            warnings.warn(f"[CityscapesDataset] {missing} images skipped (missing *_gtFine_labelIds.png).")

        print(
            f"[CityscapesDataset] split={split} ({disk_split}), "
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

        gt = np.array(Image.open(self._gt_paths[idx]), dtype=np.int64)
        # map label IDs → train IDs (out-of-range indices → 255)
        clipped = np.clip(gt, 0, self._lut.size - 1)
        train_ids = self._lut[clipped].astype(np.int64)

        seg = Image.fromarray(train_ids.astype(np.uint8), mode="L")

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
