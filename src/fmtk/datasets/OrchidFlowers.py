import os
import numpy as np
import torch
from torchvision import datasets, transforms
from fmtk.datasets.base import VisionDataset


class OrchidFlowersDataset(VisionDataset):
    """
    Parameters
    ----------
    dataset_cfg : dict
        Configuration containing dataset_path and other parameters.
    task_cfg : dict
        Configuration containing task_type and other task parameters.
    split : str
        'train' or 'test'.
    preprocess : bool
        Whether to apply normalization/resizing.
    """

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        self.target_size = dataset_cfg.get("target_size", 224)
        self.num_channels = 3
        self.transform = None

        dataset_path = dataset_cfg.get("dataset_path", "./data/OrchidFlowers")
        split_dir = os.path.join(dataset_path, split)

        # ImageFolder reads one subfolder per class
        self.dataset = datasets.ImageFolder(root=split_dir)

        self.class_names = self.dataset.classes
        self.num_classes = len(self.class_names)

        if preprocess:
            self.preprocess()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {"x": image, "y": torch.tensor(label, dtype=torch.long), "idx": idx}

    @property
    def labels(self):
        return np.array([s[1] for s in self.dataset.samples])

    @property
    def indices(self):
        return torch.arange(len(self))

    def preprocess(self):
        pipeline = transforms.Compose(
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
        self.transform = pipeline
        self.dataset.transform = pipeline

    def get_class_names(self):
        return self.class_names
