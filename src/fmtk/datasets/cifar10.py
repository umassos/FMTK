import torch
from torchvision import datasets, transforms
from fmtk.datasets.base import VisionDataset


class CIFAR10Dataset(VisionDataset):
    """
    CIFAR-10 dataset for vision tasks.

    Parameters
    ----------
    dataset_cfg : dict
        Configuration containing dataset_path and other parameters
    task_cfg : dict
        Configuration containing task_type and other task parameters
    split : str
        Split of the dataset, 'train', 'val' or 'test'
    """

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        # ImageNet normalization as default (e.g., DINOv2)
        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])

        # Image input dimensions as expected by the backbone/encoder
        self.target_size = dataset_cfg.get("target_size", 224)
        self.image_size = 32
        self.num_channels = 3

        dataset_path = dataset_cfg.get("dataset_path", "./data")
        download = dataset_cfg.get("download", True)

        print(f"Dataset path: {dataset_path}")

        self.dataset = datasets.CIFAR10(
            root=dataset_path,
            train=(split == "train"),
            download=download,
            transform=transforms.ToTensor(),
        )

        self.class_names = self.dataset.classes
        self.num_classes = len(self.class_names)

        if preprocess:
            self.preprocess()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        return {
            "x": image,
            "y": torch.tensor(label, dtype=torch.long),
        }

    def preprocess(self):
        
        self.dataset.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.target_size, self.target_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def get_class_names(self):
        return self.class_names
