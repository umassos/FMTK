import numpy as np
import torch
from torchvision import transforms
from datasets import load_from_disk
from fmtk.datasetloaders.base import VisionDataset


class ImageNet1kDataset(VisionDataset):
    """
    ImageNet-1K dataset loaded from a Hugging Face DatasetDict on disk.

    Parameters
    ----------
    dataset_cfg : dict
        dataset_path : str
            Root directory passed to load_from_disk (the directory that
            contains the 'train' and 'val' shards).
        target_size : int, optional  (default 224)
        mean : list[float], optional (default ImageNet stats)
        std  : list[float], optional (default ImageNet stats)
    task_cfg : dict
        task_type : str  (only "classification" is relevant here)
    split : str
        "train" or "val"
    preprocess : bool
        Whether to build and attach the transform pipeline at init.
    """

    NUM_CLASSES = 1000

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std  = dataset_cfg.get("std",  [0.229, 0.224, 0.225])
        self.target_size = dataset_cfg.get("target_size", 224)
        self.num_channels = 3
        self.num_classes  = self.NUM_CLASSES
        self.transform = None

        dataset_path = dataset_cfg["dataset_path"]
        full_ds = load_from_disk(dataset_path)

        # HF DatasetDict key is 'val', not 'validation'
        hf_split = split  # "train" or "val"
        self.dataset = full_ds[hf_split]

        if preprocess:
            self.preprocess()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image  = sample["image"].convert("RGB")   # PIL Image
        label  = sample["label"]

        if self.transform is not None:
            image = self.transform(image)

        return {
            "x":   image,
            "y":   torch.tensor(label, dtype=torch.long),
            "idx": idx,
        }

    def preprocess(self):
        self.transform = transforms.Compose([
            transforms.Resize(
                self.target_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    @property
    def labels(self):
        return np.array(self.dataset["label"])

    @property
    def indices(self):
        return torch.arange(len(self))