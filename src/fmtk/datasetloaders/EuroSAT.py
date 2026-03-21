import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from PIL import Image
from fmtk.datasetloaders.base import VisionDataset
import json
from transformers import AutoImageProcessor


class EuroSATDataset(VisionDataset):
    """
    EuroSAT dataset (10 classes). Loads from CSV with columns: Filename, Label, ClassName,
    or falls back to ImageFolder if no CSV is found.
    """

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        self.target_size = dataset_cfg.get("target_size", 224)
        self.image_size = 64  # EuroSAT patch size
        self.num_channels = 3
        self.transform = None

        dataset_path = dataset_cfg.get("dataset_path", "./data")
        csv_path = os.path.join(dataset_path, f"{split}.csv")
        if not os.path.exists(csv_path) and split == "val":
            csv_path = os.path.join(dataset_path, "validation.csv")

        self._dataset_path = dataset_path
        self._df = pd.read_csv(csv_path)

        self._filename_col = "Filename"
        self._label_col = "Label"

        with open(os.path.join(dataset_path, "label_map.json"), "r") as f:
            self.class_names = list(json.load(f).keys())
        self.num_classes = len(self.class_names)

        if preprocess:
            self.preprocess()

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self._df.iloc[idx]
        filename = row[self._filename_col]
        label = int(row[self._label_col])
        image_path = os.path.join(self._dataset_path, filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"x": image, "y": torch.tensor(label, dtype=torch.long), "idx": idx}

    @property
    def labels(self):
        return self._df[self._label_col].values

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

    def get_class_names(self):
        return self.class_names
