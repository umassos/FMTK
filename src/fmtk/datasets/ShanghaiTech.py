import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import scipy.io as sio
from fmtk.datasets.base import VisionDataset


class ShanghaiTechDataset(VisionDataset):
    """
    ShanghaiTech Crowd Counting Dataset.

    Parameters
    ----------
    dataset_cfg : dict
        Configuration containing dataset_path (root of ShanghaiTech)
        and optional 'part' ('A' or 'B').
    task_cfg : dict
        Configuration containing task_type and other task parameters.
    split : str
        'train' or 'test' (val handled as train subset if needed).
    preprocess : bool
        Whether to apply normalization/resizing.
    """

    def __init__(self, dataset_cfg, task_cfg, split="train", preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.dataset_path = dataset_cfg.get("dataset_path", "./data/ShanghaiTech")
        self.part = dataset_cfg.get("part", "A").upper()  # 'A' or 'B'
        assert self.part in ["A", "B"], "part must be 'A' or 'B'"
        self.target_size = dataset_cfg.get("target_size", 224)

        subdir = f"part_{self.part}/{split}_data"
        self.image_dir = os.path.join(self.dataset_path, subdir, "images")
        self.gt_dir = os.path.join(self.dataset_path, subdir, "ground-truth")

        # List images
        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        )

        # Load ground-truth counts for all images
        self._counts = []
        for img_name in self.image_files:
            base_name = os.path.splitext(img_name)[0]
            if base_name.startswith("processed_"):
                base_name = base_name.replace("processed_", "")
            gt_name = f"GT_{base_name}.mat"
            gt_path = os.path.join(self.gt_dir, gt_name)
            mat = sio.loadmat(gt_path)
            points = mat["image_info"][0, 0][0, 0][0]
            self._counts.append(len(points))

        # Mean/std for normalization (ImageNet / DINO-friendly)
        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        self.num_channels = 3
        self.transform = None

        if preprocess:
            self.preprocess()

    def preprocess(self):
        """Build the image transform pipeline (resize + normalize)."""
        pipeline = [
            transforms.Resize(
                (self.target_size, self.target_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ]
        self.transform = transforms.Compose(pipeline)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns
        -------
        dict with keys:
            x : torch.FloatTensor [3, H, W]
            y : torch.FloatTensor scalar (crowd count)
            idx : int
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        count = torch.tensor(self._counts[idx], dtype=torch.float32)

        return {"x": image, "y": count, "idx": idx}

    @property
    def labels(self):
        return np.array(self._counts)

    @property
    def indices(self):
        return torch.arange(len(self))
