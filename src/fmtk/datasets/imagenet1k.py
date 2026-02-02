import os
from typing import Optional
import torch
from torchvision import transforms
from PIL import Image
import io

from fmtk.datasets.base import VisionDataset

os.environ.setdefault("HF_HOME", "/scratch4/workspace/kgudipaty_umass_edu-workspace/classification")
from datasets import load_dataset, Image as HFImage

class ImageNet1kDataset(VisionDataset):
    """
    Hugging Face ImageNet-1k dataset wrapper that mirrors the CIFAR-10 loader.
    """

    def __init__(self, dataset_cfg, task_cfg, split, preprocess: bool = True):
        super().__init__(dataset_cfg, task_cfg, split)
        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        self.image_size = dataset_cfg.get("image_size", 224)

        self.transform: Optional[transforms.Compose] = None

        hf_split = self._map_split(split)
        load_kwargs = {"split": hf_split}

        self.dataset = load_dataset("ILSVRC/imagenet-1k", **load_kwargs)
        self.dataset = self.dataset.cast_column("image", HFImage(decode=False))
        self.dataset = self.dataset.with_format(type="numpy")

        self.class_names = self.dataset.features["label"].names
        self.num_classes = len(self.class_names)

        if preprocess:
            self.preprocess()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        byte_data = sample["image"]["bytes"]
        image = Image.open(io.BytesIO(byte_data)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return {"x": image, "y": label}

    def preprocess(self):
        resize = transforms.Resize((self.image_size, self.image_size))
        normalize = transforms.Normalize(self.mean, self.std)
        self.transform = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                normalize,
            ]
        )

    def get_class_names(self):
        return self.class_names

    @staticmethod
    def _map_split(split):
        mapping = {
            "train": "train",
            "val": "validation",
            "test": "validation",
        }
        if split not in mapping:
            raise ValueError(f"Unsupported split '{split}' for ImageNet")
        return mapping[split]