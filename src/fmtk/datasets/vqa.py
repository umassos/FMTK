import os
import json
from PIL import Image, ImageFile
from torchvision import transforms
from fmtk.datasets.base import TimeSeriesDataset

import os

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
dataset_path = os.path.join(root_dir, "dataset/val2014")

class VQADataset(TimeSeriesDataset):
    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)
        self.image_dir = os.path.join(dataset_path, "val2014")
        self.json_path = os.path.join(dataset_path, "val.json")
        self._read_data()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def _read_data(self):
        with open(self.json_path, "r") as f:
            raw_data = json.load(f)
            self.data = list(raw_data.values())
            self.data=self.data[:100]

    def __getitem__(self, index):
        item = self.data[index]
        image_id = item["image_id"]
        image_name = f"COCO_val2014_{image_id:012d}.jpg"
        image_path = os.path.join(self.image_dir, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Could not load image: {image_path} ({e})")
            return None

        image = self.transform(image)

        question = item["question"]
        gt_answer = item.get("answer", "").strip().lower()

        return {
            'x':image,
            'question':question,
            'y':gt_answer
        }

    def preprocess(self):
        pass
