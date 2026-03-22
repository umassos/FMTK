"""
Generic image dataset for all VLM tasks (activity, scene, ocr, vqa, traffic,
gesture, crowd, object_detection, image_classification).

dataset_cfg keys:
    dataset_path  : absolute path to the dataset root directory
    json_file     : filename of the annotations JSON (e.g. 'labels.json', 'val.json')
    image_subdir  : (optional) subdirectory under dataset_path containing images.
                    Required for VQA/COCO datasets where images live in 'val2014/'.

task_cfg keys:
    prompt        : prompt template string.
                    VQA is the only task with a {question} placeholder;
                    all other tasks use the prompt verbatim for every sample.

JSON normalisation (matches unified_inference.py exactly):
    - If the JSON root is a dict, take dict.values() as the sample list.
    - If a record has 'image_id' but no 'image_path', build the COCO filename:
        COCO_val2014_{image_id:012d}.jpg  under image_subdir.
    - If 'image_path' is relative, resolve it against dataset_path.
    - Normalise the label field: 'answer' -> 'label', 'categories' -> 'label'.

__getitem__ returns:
    {
        'x':        torch.Tensor  (C, H, W) — ToTensor only, no forced resize.
                    VLM backbones convert back to PIL before passing to their
                    processor, which applies model-specific resizing.
        'question': str           — formatted prompt (question inserted for VQA)
        'y':        str or list   — ground truth label(s)
    }

Split handling:
    VLM tasks set train=False; only the 'test' split is used at runtime.
    'train' and 'val' splits return empty datasets so InferencePipeline
    does not crash when it instantiates all three splits.
"""
import os
import json
from PIL import Image
from torchvision import transforms
from fmtk.datasetloaders.base import VisionDataset

MAX_SAMPLES = None  # overridden per split by task_cfg


def vlm_collate_fn(batch):
    """Custom collate that preserves per-sample ground-truth structure.

    PyTorch's default collate zips list-valued fields across the batch
    dimension, which destroys variable-length lists (e.g. object_detection
    categories).  This collate keeps 'y' as-is (list of str/list) and
    stacks image tensors normally.
    """
    import torch
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    return {
        'x':        torch.stack([b['x'] for b in batch]),
        'question': [b['question'] for b in batch],
        'y':        [b['y'] for b in batch],
    }


class VLMDataset(VisionDataset):

    def __init__(self, dataset_cfg, task_cfg, split):
        super().__init__(dataset_cfg, task_cfg, split)

        self.dataset_path = dataset_cfg['dataset_path']
        self.json_path    = os.path.join(self.dataset_path, dataset_cfg['json_file'])
        self.image_subdir = dataset_cfg.get('image_subdir', '')
        self.prompt       = task_cfg['prompt']

        # ToTensor only — no forced resize.
        # Each VLM processor applies its own model-specific resizing.
        self.to_tensor = transforms.ToTensor()
        self.split = split
        self.train_max = task_cfg.get('train_config', {}).get('max_samples', None)
        self.test_max = task_cfg.get('inference_config', {}).get('max_samples', None)
        self.train_ratio = task_cfg.get('train_ratio', 0.8)

        self._read_data()

    # ------------------------------------------------------------------
    def _read_data(self):
        with open(self.json_path, 'r') as f:
            raw = json.load(f)

        # Normalise root structure: dict of records -> list of records
        if isinstance(raw, dict):
            records = list(raw.values())
        else:
            records = raw

        normalised = []
        for rec in records:
            rec = dict(rec)  # shallow copy so we don't mutate the source

            # ---- image path resolution (mirrors unified_inference.py) ----
            if 'image_id' in rec and 'image_path' not in rec:
                # COCO/VQA convention: image lives in image_subdir
                img_id   = int(rec['image_id'])
                filename = f"COCO_val2014_{img_id:012d}.jpg"
                subdir   = os.path.join(self.dataset_path, self.image_subdir)
                rec['image_path'] = os.path.join(subdir, filename)
            elif 'image_path' in rec and not os.path.isabs(rec['image_path']):
                rec['image_path'] = os.path.join(self.dataset_path, rec['image_path'])

            # ---- label field normalisation (mirrors unified_inference.py) ----
            if 'answer' in rec and 'label' not in rec:
                rec['label'] = rec['answer']
            elif 'categories' in rec and 'label' not in rec:
                rec['label'] = rec['categories']

            # OCR labels are stored as int in MNIST JSON; convert to str
            if 'label' in rec and isinstance(rec['label'], int):
                rec['label'] = str(rec['label'])

            normalised.append(rec)

        split_idx = int(len(normalised) * self.train_ratio)
        if self.split == 'train':
            pool = normalised[:split_idx]
            self.data = pool[:self.train_max] if self.train_max is not None else pool
        else:
            pool = normalised[split_idx:]
            self.data = pool[:self.test_max] if self.test_max is not None else pool

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------------------
    def __getitem__(self, index):
        rec = self.data[index]

        # ---- image loading ----
        image_path = rec.get('image_path', '')
        if not os.path.exists(image_path):
            print(f"[VLMDataset] Warning: image not found, skipping: {image_path}")
            return None

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"[VLMDataset] Warning: could not load {image_path} ({e})")
            return None

        image_tensor = self.to_tensor(image)

        # ---- prompt formatting ----
        # VQA is the only task whose prompt contains {question}.
        # All other task prompts are fixed strings.
        if '{question}' in self.prompt and 'question' in rec:
            question = self.prompt.format(question=rec['question'])
        else:
            question = self.prompt

        # ---- ground truth ----
        gt = rec.get('label', '')

        return {
            'x':        image_tensor,
            'question': question,
            'y':        gt,
        }

    # ------------------------------------------------------------------
    def preprocess(self):
        pass
