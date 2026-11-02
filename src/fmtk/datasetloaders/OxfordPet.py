import os

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

from fmtk.datasetloaders.base import VisionDataset

# Binary pet-segmentation classes: the Oxford-IIIT Pet trimap format encodes
# 1=Foreground(pet), 2=Background, 3=Not classified (boundary/ambiguous
# region) -- remapped here to {0: background, 1: pet}, with the boundary
# region mapped to ignore_index so it's excluded from loss/metrics, matching
# PascalVOC's ignore-region convention.
CLASSES = ["background", "pet"]
IGNORE_INDEX = 255

TRIMAP_FOREGROUND = 1
TRIMAP_BACKGROUND = 2
TRIMAP_UNCLASSIFIED = 3


class OxfordPetDataset(VisionDataset):
    """
    Oxford-IIIT Pet Dataset (Parkhi et al. 2012) binary segmentation: pet
    vs. background, read from the official release's `images/*.jpg` +
    `annotations/trimaps/*.png` (trimap-encoded: 1=foreground, 2=background,
    3=not classified -- remapped to {1: pet, 0: background, ignore_index:
    boundary}).

    The official release only ships two splits (trainval.txt/test.txt, no
    val) -- confirmed test.txt's trimaps are real, fully-annotated ground
    truth (unlike PascalVOC's official test split, which ships empty
    placeholder annotations -- do not assume any dataset's "test" split is
    usable without checking). "train"/"val" are carved from trainval.txt via
    a stratified (by breed class id) split; "test" reads test.txt directly.

    Parameters
    ----------
    dataset_cfg : dict
        dataset_path : str    root containing images/ and annotations/  (required)
        target_size  : int    output height/width after resize           (default: 224)
        val_fraction : float  fraction of trainval.txt held out as val    (default: 0.1)
        seed         : int    seed for the stratified train/val split     (default: 42)
    task_cfg : dict
        task_type : "segmentation"
    split : str
        "train", "val", or "test"
    """

    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        dataset_path = dataset_cfg.get("dataset_path", "./data")
        self.img_dir = os.path.join(dataset_path, "images")
        self.trimap_dir = os.path.join(dataset_path, "annotations", "trimaps")

        self.class_names = CLASSES
        self.num_classes = len(CLASSES)
        self.ignore_index = IGNORE_INDEX

        self.target_size = dataset_cfg.get("target_size", 224)
        self.mean = dataset_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = dataset_cfg.get("std", [0.229, 0.224, 0.225])
        val_fraction = dataset_cfg.get("val_fraction", 0.1)
        seed = dataset_cfg.get("seed", 42)

        self.filenames = self._load_split(dataset_path, split, val_fraction, seed)

        self.transform = None
        if preprocess:
            self.preprocess()

    def _load_split(self, dataset_path, split, val_fraction, seed):
        ann_dir = os.path.join(dataset_path, "annotations")
        if split == "test":
            with open(os.path.join(ann_dir, "test.txt")) as f:
                return [line.split()[0] for line in f if line.strip()]

        assert split in ("train", "val"), f"Unknown split {split!r}"
        with open(os.path.join(ann_dir, "trainval.txt")) as f:
            entries = [line.split() for line in f if line.strip()]
        names = [e[0] for e in entries]
        breed_ids = [int(e[1]) for e in entries]

        train_names, val_names = train_test_split(
            names, test_size=val_fraction, stratify=breed_ids, random_state=seed,
        )
        return train_names if split == "train" else val_names

    def __len__(self):
        return len(self.filenames)

    def preprocess(self):
        self.transform = transforms.Compose([
            transforms.Resize(
                (self.target_size, self.target_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def _load_mask(self, name):
        trimap = np.array(Image.open(os.path.join(self.trimap_dir, f"{name}.png")))
        mask = np.full(trimap.shape, self.ignore_index, dtype=np.uint8)
        mask[trimap == TRIMAP_FOREGROUND] = 1
        mask[trimap == TRIMAP_BACKGROUND] = 0
        return mask

    def __getitem__(self, index):
        name = self.filenames[index]
        image = Image.open(os.path.join(self.img_dir, f"{name}.jpg")).convert("RGB")

        mask = self._load_mask(name)
        mask_img = Image.fromarray(mask, mode="L").resize(
            (self.target_size, self.target_size), resample=Image.NEAREST
        )
        mask_tensor = torch.from_numpy(np.array(mask_img)).long()

        if self.transform is not None:
            image = self.transform(image)

        return {"x": image, "y": mask_tensor, "idx": index}

    def get_class_names(self):
        return self.class_names
