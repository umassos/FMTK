import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from fmtk.datasets.base import VisionDataset

# Hardcoded RESISC45 class labels (45 classes) - ensures consistent mapping across all splits
RESISC45_CLASSES = [
    'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
    'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
    'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
    'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection',
    'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park',
    'mountain', 'overpass', 'palace', 'parking_lot', 'railway',
    'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway',
    'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium',
    'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland'
]

# Create fixed class-to-index mapping
RESISC45_CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(RESISC45_CLASSES)}


class CSVImageDataset(Dataset):
    """Custom dataset that loads images from CSV file with filename and label columns."""
    
    def __init__(self, csv_path, dataset_path, transform=None):
        """
        Parameters
        ----------
        csv_path : str
            Path to CSV file with 'filename' and 'label' columns
        dataset_path : str
            Root directory where images are stored (organized by class folders)
        transform : callable, optional
            Optional transform to be applied on a sample
        """
        self.df = pd.read_csv(csv_path)
        self.dataset_path = dataset_path
        self.transform = transform
        
        # Use hardcoded class mapping for consistency across all splits
        self.class_names = RESISC45_CLASSES
        self.class_to_idx = RESISC45_CLASS_TO_IDX
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        label_str = row['label']
        
        # Construct image path: dataset_path/label/filename
        image_path = os.path.join(self.dataset_path, label_str, filename)
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
        
        # Get label index
        if label_str not in self.class_to_idx:
            raise ValueError(
                f"Label '{label_str}' not found in RESISC45 class mapping. "
                f"Available classes: {self.class_names}"
            )
        label = self.class_to_idx[label_str]
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @property
    def classes(self):
        return self.class_names


class RESIC45Dataset(VisionDataset):
    """

    Parameters
    ----------
    dataset_cfg : dict
        Configuration containing dataset_path and other parameters
    task_cfg : dict
        Configuration containing task_type and other task parameters
    split : str
        Split of the dataset, 'train', 'val' or 'test'
    preprocess : bool
        Whether to apply preprocessing transforms (default: True)
    """

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        # ImageNet normalization as default (e.g., DINOv2)
        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])

        # Image input dimensions as expected by the backbone/encoder
        self.target_size = dataset_cfg.get("target_size", 224)
        self.image_size = 256  # Original RESISC45 image size
        self.num_channels = 3

        dataset_path = dataset_cfg.get("dataset_path", "./data")
        
        # Check if CSV files exist
        csv_path = os.path.join(dataset_path, f"{split}.csv")
        
        if os.path.exists(csv_path):
            # Load from CSV file
            print(f"Loading RESISC45 from CSV: {csv_path}")
            self.dataset = CSVImageDataset(
                csv_path=csv_path,
                dataset_path=dataset_path,
                transform=transforms.ToTensor(),
            )
            self.class_names = self.dataset.classes
        else:
            # Fall back to directory-based loading
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path) and os.path.isdir(split_path):
                # Use split-specific directory
                root_dir = split_path
            else:
                # Use root directory (assumes all classes are in root)
                root_dir = dataset_path

            print(f"Dataset path: {root_dir}")

            # Use ImageFolder to load images organized by class
            self.dataset = datasets.ImageFolder(
                root=root_dir,
                transform=transforms.ToTensor(),
            )
            self.class_names = self.dataset.classes

        self.num_classes = len(self.class_names)

        if preprocess:
            self.preprocess()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        image, label = self.dataset[idx]

        return {
            "x": image,
            "y": torch.tensor(label, dtype=torch.long),
        }

    @property
    def indices(self):
        return torch.arange(len(self))

    @property
    def labels(self):
        """Tensor of labels for each dataset index (for building per-class subsets)."""
        if isinstance(self.dataset, CSVImageDataset):
            return torch.tensor(
                self.dataset.df["label"].map(self.dataset.class_to_idx).values,
                dtype=torch.long,
            )
        return torch.tensor(self.dataset.targets, dtype=torch.long)

    def preprocess(self):
        # Handle both CSVImageDataset and ImageFolder
        if isinstance(self.dataset, CSVImageDataset):
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
        else:
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
