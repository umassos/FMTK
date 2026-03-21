import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from fmtk.datasetloaders.base import VisionDataset


class NYUDepthV2Dataset(VisionDataset):
    """
    NYU Depth v2 dataset for monocular depth estimation.

    Expects the dataset downloaded to disk with the structure:
        dataset_path/
            train/
                rgb/    000000.png  000001.png  ...
                depth/  000000.npy  000001.npy  ...  (float32, metres)
            test/
                rgb/    000000.png  ...
                depth/  000000.npy  ...

    Use download-nyu-depth.py to populate this directory from HuggingFace.

    Parameters
    ----------
    dataset_cfg : dict
        dataset_path    : str   path to root directory (required)
        target_size     : int   resize H and W         (default: 224)
        max_depth       : float clip depth to this max (default: 10.0 m)
        normalize_depth : bool  divide by max_depth -> [0, 1] (default: True)
        mean            : list  ImageNet mean          (default: [0.485, 0.456, 0.406])
        std             : list  ImageNet std           (default: [0.229, 0.224, 0.225])
    task_cfg : dict
        task_type : "regression"
    split : str
        "train" or "test"
    """

    def __init__(self, dataset_cfg, task_cfg, split="train", preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.dataset_path = dataset_cfg.get("dataset_path", "")
        assert self.dataset_path, "dataset_cfg['dataset_path'] must be set."

        self.target_size = dataset_cfg.get("target_size", 224)
        self.max_depth = dataset_cfg.get("max_depth", 10.0)
        self.normalize_depth = dataset_cfg.get("normalize_depth", True)
        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])

        self.img_transform = None
        self.depth_transform = None

        rgb_dir = os.path.join(self.dataset_path, split, "rgb")
        depth_dir = os.path.join(self.dataset_path, split, "depth")
        assert os.path.isdir(rgb_dir), f"RGB directory not found: {rgb_dir}"
        assert os.path.isdir(depth_dir), f"Depth directory not found: {depth_dir}"

        self._rgb_files = sorted(
            [os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)
             if f.lower().endswith(".png")]
        )
        self._depth_files = sorted(
            [os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
             if f.lower().endswith((".npy", ".png"))]
        )
        assert len(self._rgb_files) == len(self._depth_files), (
            f"RGB ({len(self._rgb_files)}) and depth ({len(self._depth_files)}) "
            "file counts do not match."
        )

        if preprocess:
            self.preprocess()

    def preprocess(self):
        self.img_transform = transforms.Compose([
            transforms.Resize(
                (self.target_size, self.target_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        # Nearest-neighbour for depth to avoid blending across boundaries
        self.depth_transform = transforms.Resize(
            (self.target_size, self.target_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def __len__(self):
        return len(self._rgb_files)

    def __getitem__(self, idx):
        image = Image.open(self._rgb_files[idx]).convert("RGB")

        depth_path = self._depth_files[idx]
        if depth_path.endswith(".npy"):
            depth_arr = np.load(depth_path).astype(np.float32)
        else:
            # uint16 PNG where values are in millimetres -> convert to metres
            depth_arr = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0

        depth_pil = Image.fromarray(depth_arr, mode="F")

        if self.img_transform is not None:
            image = self.img_transform(image)

        depth = self._process_depth(depth_pil)

        return {"x": image, "y": depth, "idx": idx}

    def _process_depth(self, depth_pil):
        """Resize, clip, and optionally normalise a float PIL depth image -> [H, W] tensor."""
        if self.depth_transform is not None:
            depth_pil = self.depth_transform(depth_pil)

        depth = torch.from_numpy(np.array(depth_pil, dtype=np.float32))  # [H, W]
        depth = depth.clamp(0.0, self.max_depth)

        if self.normalize_depth:
            depth = depth / self.max_depth  # [0, 1]

        return depth

    @property
    def indices(self):
        return torch.arange(len(self))
