import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from fmtk.datasets.base import VisionDataset


SPLIT_MAP = {
    "train": "training",
    "val": "validation",
    "test": "validation",
}


class ADE20KDataset(VisionDataset):
    """
    ADE20K dataset for scene classification (and optional segmentation).

    Expects the Kaggle folder layout:

        <dataset_path>/
        ├── index_ade20k.pkl
        ├── objects.txt
        └── images/
            └── ADE/
                ├── training/
                │   ├── <category>/
                │   │   ├── <scene>/
                │   │   │   ├── ADE_train_XXXXXXXX.jpg
                │   │   │   ├── ADE_train_XXXXXXXX_seg.png
                │   │   │   └── ...
                │   │   └── ...
                │   └── ...
                └── validation/
                    └── ...

    Parameters
    ----------
    dataset_cfg : dict
        Must contain ``dataset_path``.  Optional keys:
        ``target_size`` (default 224), ``mean``, ``std``,
        ``load_seg`` (bool, default False – also load segmentation masks).
    task_cfg : dict
        Task configuration (e.g. ``{"task_type": "classification"}``).
    split : str
        One of ``"train"``, ``"val"``, or ``"test"``.
    preprocess : bool
        Whether to build the transform pipeline on init.
    """

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        self.target_size = dataset_cfg.get("target_size", 224)
        self.num_channels = 3
        self.load_seg = dataset_cfg.get("load_seg", False)

        self.transform = None
        self.seg_transform = None

        dataset_path = dataset_cfg.get("dataset_path", "./data")
        self._dataset_path = dataset_path

        ade_split = SPLIT_MAP.get(split)
        if ade_split is None:
            raise ValueError(
                f"Unsupported split '{split}'. Use 'train', 'val', or 'test'."
            )

        split_dir = os.path.join(dataset_path, "images", "ADE", ade_split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}"
            )

        self._image_paths, self._labels, self.class_names = self._scan_images(
            split_dir
        )
        self.num_classes = len(self.class_names)

        print(
            f"[ADE20K] split={split} ({ade_split}), "
            f"images={len(self._image_paths)}, classes={self.num_classes}"
        )

        if preprocess:
            self.preprocess()

    # ------------------------------------------------------------------
    # Scanning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_images(split_dir):
        """Walk the split directory and collect (image_path, scene_label) pairs.

        Scene label is the *leaf* folder name (e.g. ``bedroom``,
        ``airport_terminal``).  Only ``*.jpg`` files whose name does NOT
        contain ``_seg`` or ``_parts`` are collected.
        """
        image_paths = []
        scene_labels = []
        scene_set = set()

        for root, _dirs, files in os.walk(split_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(".jpg"):
                    continue
                if "_seg" in fname or "_parts" in fname:
                    continue
                scene_name = os.path.basename(root)
                image_paths.append(os.path.join(root, fname))
                scene_labels.append(scene_name)
                scene_set.add(scene_name)

        class_names = sorted(scene_set)
        label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        labels = [label_to_idx[s] for s in scene_labels]

        return image_paths, labels, class_names

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image = Image.open(self._image_paths[idx]).convert("RGB")
        label = self._labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        sample = {
            "x": image,
            "y": torch.tensor(label, dtype=torch.long),
            "idx": idx,
        }

        if self.load_seg:
            seg_path = self._image_paths[idx].replace(".jpg", "_seg.png")
            if os.path.exists(seg_path):
                seg = Image.open(seg_path)
                if self.seg_transform is not None:
                    seg = self.seg_transform(seg)
                else:
                    seg = transforms.functional.resize(
                        seg,
                        [self.target_size, self.target_size],
                        interpolation=transforms.InterpolationMode.NEAREST,
                    )
                    seg = torch.as_tensor(np.array(seg), dtype=torch.long)
                sample["seg"] = seg

        return sample

    @property
    def labels(self):
        return torch.tensor(self._labels, dtype=torch.long)

    @property
    def indices(self):
        return torch.arange(len(self))

    def preprocess(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.target_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def get_class_names(self):
        return self.class_names
