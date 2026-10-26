# FMTK Dataset Loaders

Dataset loaders for the Foundation Model Toolkit (FMTK). Each loader wraps a specific dataset into a consistent interface that can be plugged directly into a PyTorch `DataLoader` and an FMTK `Pipeline`.

---

## Table of Contents

- [Base Classes](#base-classes)
- [Time Series Datasets](#time-series-datasets)
- [Vision Datasets](#vision-datasets)
- [VLM / Language Datasets](#vlm--language-datasets)
- [How to Add a New Dataset Loader](#how-to-add-a-new-dataset-loader)

---

## Base Classes

**File:** `base.py`

Three abstract base classes live here, one per modality. All share the same constructor signature and required abstract methods.

```python
class TimeSeriesDataset(ABC):
    def __init__(self, dataset_cfg: dict, task_cfg: dict, split: str): ...
    @abstractmethod def __len__(self): ...
    @abstractmethod def __getitem__(self, idx): ...
    @abstractmethod def preprocess(self): ...

class VisionDataset(ABC):   # same interface
class LLMDataset(ABC):      # same interface
```

### Constructor arguments (all loaders)

| Argument | Type | Description |
|---|---|---|
| `dataset_cfg` | `dict` | Dataset-level config: paths, normalization stats, image sizes, etc. |
| `task_cfg` | `dict` | Task-level config: `task_type`, batch sizes, label names, prompts, etc. |
| `split` | `str` | `"train"`, `"val"`, or `"test"` |

`preprocess()` is called automatically in `__init__` for vision loaders (controlled by a `preprocess=True` kwarg). Time series loaders call their own `_read_data()` / `load_data()` internally.

---

## Time Series Datasets

All time series loaders subclass `TimeSeriesDataset`. They read raw data in `_read_data()` or `load_data()`, apply `StandardScaler` normalization fitted on train only, and slice windows at `__getitem__` time.

### Common `dataset_cfg` keys

| Key | Default | Description |
|---|---|---|
| `dataset_path` | varies | Root directory of the dataset on disk |
| `seq_len` | `512` | Input context window length |

### Common `task_cfg` keys

| Key | Values | Description |
|---|---|---|
| `task_type` | `"forecasting"`, `"imputation"`, `"classification"`, `"regression"` | Controls what `__getitem__` returns |

---

### ETTh1 — `ETTh1Dataset`

**File:** `etth1.py` | **Task types:** `forecasting`, `imputation`

Electricity Transformer Temperature dataset (hourly). Single target column, sliding-window batches.

```python
from fmtk.datasetloaders.etth1 import ETTh1Dataset

dataset_cfg = {"dataset_path": "dataset/ETTh1", "seq_len": 512}
task_cfg    = {"task_type": "forecasting"}

ds = ETTh1Dataset(dataset_cfg, task_cfg, split="train", forecast_horizon=192)
```

**`__getitem__` output:**

| Task | Keys | Shape |
|---|---|---|
| `forecasting` | `x`, `y` | `(1, seq_len)`, `(1, horizon)` |
| `imputation` | `x`, `mask` | `(1, seq_len)`, `(seq_len,)` |

Dataset path is hard-coded relative to the package root: `dataset/ETTh1/ETTh1.csv`.

---

### ECL (Electricity) — `ECLDataset`

**File:** `ecl.py` | **Task types:** `forecasting`, `imputation`

UCI Electricity Load dataset. Train/val/test split is ratio-based (default 70/10/20).

```python
from fmtk.datasetloaders.ecl import ECLDataset

dataset_cfg = {"dataset_path": "dataset/ElectricityLoad-data", "seq_len": 512}
task_cfg    = {"task_type": "forecasting"}

ds = ECLDataset(dataset_cfg, task_cfg, split="train", forecast_horizon=192)
```

Same `__getitem__` contract as ETTh1.

---

### ECG5000 — `ECG5000Dataset`

**File:** `ecg5000.py` | **Task type:** `classification`

5-class ECG heartbeat classification. Data is in `.ts` format, loaded via `fmtk.utils.load_from_tsfile`. Labels are remapped to `{0, ..., 4}`.

```python
from fmtk.datasetloaders.ecg5000 import ECG5000Dataset

dataset_cfg = {"seq_len": 512}
task_cfg    = {"task_type": "classification"}

ds = ECG5000Dataset(dataset_cfg, task_cfg, split="train")
```

**`__getitem__` output:** `{"x": (1, seq_len), "mask": (seq_len,), "y": int}`

Dataset path is hard-coded relative to the package root: `dataset/ECG5000/`.

**Example:** `experiments/examples/moment_ecgclassify.py`, `experiments/examples/chronos_ecgclassify.py`

---

### PPG — `PPGDataset`

**File:** `ppg.py` | **Task type:** `regression`

Photoplethysmography dataset for blood-pressure regression. Multi-channel (PPG + VPG + APG). Applies pyPPG bandpass filtering and resamples from 1000 Hz to 125 Hz.

```python
from fmtk.datasetloaders.ppg import PPGDataset

dataset_cfg = {"seq_len": 1250, "num_channels": 3}
task_cfg    = {"task_type": "regression", "label": "sysbp"}
# label options: "sysbp", "diasbp", "hr", "age", "bmi"

ds = PPGDataset(dataset_cfg, task_cfg, split="train")
```

**`__getitem__` output:** `{"x": (num_channels, seq_len), "y": float}`

Dataset path is hard-coded relative to the package root: `dataset/PPG-data/`.

**Example:** `experiments/examples/moment_ppg.py`, `experiments/examples/chronos_ppg.py`

---

### FordA — `FordADataset`

**File:** `forda.py` | **Task type:** `classification`

Binary classification of engine fault from 500-step univariate time series. Labels `{-1, 1}` are remapped to `{0, 1}`.

```python
from fmtk.datasetloaders.forda import FordADataset

dataset_cfg = {"dataset_path": "dataset/FordA"}
task_cfg    = {"task_type": "classification"}

ds = FordADataset(dataset_cfg, task_cfg, split="train")
```

**`__getitem__` output:** `{"x": (1, 500), "y": int}`

Scaler statistics are saved/loaded from `dataset_path/forda_mean.npy` and `forda_scale.npy` so test-split can reuse training statistics without refitting.

---

### Other Time Series Loaders

The following datasets follow the same ETTh1-style sliding-window interface (`forecasting` or `imputation`):

| Class | File | Dataset |
|---|---|---|
| `WeatherDataset` | `weather.py` | Weather station forecasting |
| `TrafficDataset` | `traffic.py` | Road traffic occupancy |
| `IllnessDataset` | `illness.py` | ILI (influenza-like illness) |
| `ExchangeDataset` | `exchange.py` | Currency exchange rates |
| `REDDDataset` | `redd.py` | Household energy disaggregation |
| `UWaveGestureDataset` | `uwavegesture.py` | Accelerometer gesture recognition |

---

## Vision Datasets

All vision loaders subclass `VisionDataset`. They call `preprocess()` in `__init__` (if `preprocess=True`) to set up a `transforms.Compose` pipeline. The default normalization is ImageNet stats `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` and `target_size=224`.

### Common `dataset_cfg` keys

| Key | Default | Description |
|---|---|---|
| `dataset_path` | required | Root directory of the dataset |
| `target_size` | `224` | Resize H and W to this value |
| `mean` | ImageNet mean | Per-channel normalization mean |
| `std` | ImageNet std | Per-channel normalization std |

---

### CIFAR-10 — `CIFAR10Dataset`

**File:** `cifar10.py` | **Task type:** `classification`

10-class image classification. Wraps `torchvision.datasets.CIFAR10`; downloads automatically if `download=True`.

```python
from fmtk.datasetloaders.cifar10 import CIFAR10Dataset
from torch.utils.data import DataLoader

dataset_cfg = {"dataset_path": "dataset/cifar-10", "target_size": 224}
task_cfg    = {"task_type": "classification"}

train_ds = CIFAR10Dataset(dataset_cfg, task_cfg, split="train")
test_ds  = CIFAR10Dataset(dataset_cfg, task_cfg, split="test")
```

**`__getitem__` output:** `{"x": Tensor(3, 224, 224), "y": Tensor(long)}`

**Example (full pipeline):** `experiments/examples/dino_cifar10.py`

```python
from fmtk.pipeline import Pipeline
from fmtk.datasetloaders.cifar10 import CIFAR10Dataset
from fmtk.components.backbones.dinov2 import DinoV2Model
from fmtk.components.decoders.classification.logisticregression import LogisticRegressionDecoder
from fmtk.metrics import get_accuracy
from torch.utils.data import DataLoader

dataset_cfg  = {"dataset_path": "../dataset/cifar-10"}
task_cfg     = {"task_type": "classification"}
train_config = {"batch_size": 32, "shuffle": False, "epochs": 50, "lr": 1e-2}
infer_config = {"batch_size": 32, "shuffle": False}

train_loader = DataLoader(CIFAR10Dataset(dataset_cfg, task_cfg, "train"), batch_size=32)
test_loader  = DataLoader(CIFAR10Dataset(dataset_cfg, task_cfg, "test"),  batch_size=32)

P = Pipeline(DinoV2Model("cuda:0", "base", {}))
P.add_decoder(LogisticRegressionDecoder(max_iter=10000), load=True)
P.train(train_loader, parts_to_train=["decoder"], cfg=train_config)
y_test, y_pred = P.predict(test_loader, cfg=infer_config)
print(get_accuracy(y_test, y_pred))
```

---

### EuroSAT — `EuroSATDataset`

**File:** `EuroSAT.py` | **Task type:** `classification`

10-class satellite land-use classification. Expects a CSV per split (`train.csv`, `val.csv` or `validation.csv`, `test.csv`) with columns `Filename` and `Label`, and a `label_map.json` in `dataset_path`.

```python
from fmtk.datasetloaders.EuroSAT import EuroSATDataset

dataset_cfg = {"dataset_path": "/path/to/EuroSAT", "target_size": 224}
task_cfg    = {"task_type": "classification"}

train_ds = EuroSATDataset(dataset_cfg, task_cfg, split="train")
test_ds  = EuroSATDataset(dataset_cfg, task_cfg, split="test")
```

**`__getitem__` output:** `{"x": Tensor(3, 224, 224), "y": Tensor(long), "idx": int}`

Useful properties: `ds.labels` (numpy array), `ds.indices` (for building per-class `Subset`s), `ds.class_names`.

**Example:** `experiments/examples/dino_eurosat.py`, `experiments/examples/swin_eurosat.py`

---

### NYU Depth V2 — `NYUDepthV2Dataset`

**File:** `nyudepthv2.py` | **Task type:** `regression`

Monocular depth estimation. Expects paired `rgb/` and `depth/` subdirectories under `train/` and `test/`. Depth files can be `.npy` (float32, metres) or `.png` (uint16, millimetres).

```python
from fmtk.datasetloaders.nyudepthv2 import NYUDepthV2Dataset

dataset_cfg = {
    "dataset_path": "/path/to/NYUDepthV2",
    "target_size": 224,
    "max_depth": 10.0,
    "normalize_depth": True,   # scales depth -> [0, 1]
}
task_cfg = {"task_type": "regression"}

ds = NYUDepthV2Dataset(dataset_cfg, task_cfg, split="train")
```

**`__getitem__` output:** `{"x": Tensor(3, H, W), "y": Tensor(H, W) float, "idx": int}`

Depth uses nearest-neighbour resize to avoid boundary blending artifacts.

**Example:** `experiments/examples/dino_nyu.py`, `experiments/examples/swin_nyu.py`

---

### ShanghaiTech — `ShanghaiTechDataset`

**File:** `ShanghaiTech.py` | **Task type:** `regression` (crowd counting)

Crowd counting from images. Supports Part A and Part B. Ground-truth counts are loaded from `.mat` files.

```python
from fmtk.datasetloaders.ShanghaiTech import ShanghaiTechDataset

dataset_cfg = {
    "dataset_path": "/path/to/ShanghaiTech",
    "part": "A",   # or "B"
    "target_size": 224,
}
task_cfg = {"task_type": "regression"}

ds = ShanghaiTechDataset(dataset_cfg, task_cfg, split="train")
```

**`__getitem__` output:** `{"x": Tensor(3, H, W), "y": Tensor scalar (crowd count), "idx": int}`

**Example:** `experiments/examples/dino_shanghaitechcount.py`

---

### Pascal VOC 2012 — `VOC12Dataset`

**File:** `voc12.py` | **Task type:** `segmentation`

21-class semantic segmentation (background + 20 object classes). Expects **Supervisely format** on disk:

```
dataset_path/
    train/
        img/   *.jpg
        ann/   *.jpg.json   (Supervisely bitmap annotations)
    val/
        img/
        ann/
```

```python
from fmtk.datasetloaders.voc12 import VOC12Dataset

dataset_cfg = {"dataset_path": "/path/to/VOC12", "target_size": 512}
task_cfg    = {"task_type": "segmentation"}

ds = VOC12Dataset(dataset_cfg, task_cfg, split="train")
```

**`__getitem__` output:** `{"x": Tensor(3, 512, 512), "y": Tensor(512, 512) int64, "idx": int}`

`y` contains class indices 0–20; `255` is the ignore index. The `"test"` split maps to `"val"` on disk.

**Example:** `experiments/examples/swin_voc.py`, `experiments/examples/dinov3_voc.py`

---

### Food-101 — `Food101Dataset`

**File:** `Food101.py` | **Task type:** `classification`

101-class food image classification. Wraps `torchvision.datasets.Food101`.

```python
from fmtk.datasetloaders.Food101 import Food101Dataset

dataset_cfg = {"dataset_path": "dataset/food101", "target_size": 224}
task_cfg    = {"task_type": "classification"}

ds = Food101Dataset(dataset_cfg, task_cfg, split="train")
```

**`__getitem__` output:** `{"x": Tensor(3, 224, 224), "y": Tensor(long), "idx": int}`

---

## VLM / Language Datasets

### VLMDataset (generic, preferred)

**File:** `vlm_dataset.py` | **Supported tasks:** activity, scene, OCR, VQA, traffic, gesture, crowd counting, object detection, image classification

Generic multi-task VLM loader. Reads a JSON annotation file and resolves image paths. Handles both dict-root and list-root JSONs, COCO image naming, and variable-length ground-truth labels.

#### Required `dataset_cfg` keys

| Key | Description |
|---|---|
| `dataset_path` | Absolute path to the dataset root |
| `json_file` | Annotation JSON filename (e.g. `"labels.json"`, `"val.json"`) |
| `image_subdir` | (optional) Subdirectory under `dataset_path` where images live — required for COCO/VQA datasets |

#### Required `task_cfg` keys

| Key | Description |
|---|---|
| `prompt` | Prompt string. For VQA use `{question}` as a placeholder; all other tasks use it verbatim. |
| `inference_config.max_samples` | (optional) Cap the number of test samples |
| `train_config.max_samples` | (optional) Cap the number of train samples |
| `train_ratio` | (optional, default `0.8`) Fraction of data used for train split |

```python
from fmtk.datasetloaders.vlm_dataset import VLMDataset, vlm_collate_fn
from torch.utils.data import DataLoader

# VQA example
dataset_cfg = {
    "dataset_path": "/path/to/vqa",
    "json_file": "val.json",
    "image_subdir": "val2014",
}
task_cfg = {
    "prompt": "Answer the question in one word. Question: {question}",
    "inference_config": {"max_samples": 500},
    "train_ratio": 0.8,
}

ds     = VLMDataset(dataset_cfg, task_cfg, split="test")
loader = DataLoader(ds, batch_size=1, collate_fn=vlm_collate_fn)
```

**`__getitem__` output:** `{"x": Tensor(3, H, W), "question": str, "y": str | list}`

> Always use `vlm_collate_fn` with `DataLoader`. PyTorch's default collate breaks variable-length ground-truth lists (e.g. object detection returns multiple category labels per image).

#### JSON normalization rules (automatic)

- Dict root → `dict.values()` treated as sample list
- `image_id` present, `image_path` absent → COCO filename `COCO_val2014_{id:012d}.jpg` under `image_subdir`
- Relative `image_path` → resolved against `dataset_path`
- `"answer"` field → aliased to `"label"`
- `"categories"` field → aliased to `"label"`
- Integer labels → converted to `str`

**Example:** `experiments/examples/llava_vqa.py`, `experiments/examples/qwen_vqa.py`, `experiments/examples/moondream_vqa.py`

---

### VQADataset (legacy)

**File:** `vqa.py`

Older hardcoded VQA loader capped at 100 samples. Dataset path is hard-coded to `dataset/val2014/`. Prefer `VLMDataset` for all new work.

---

## How to Add a New Dataset Loader

1. **Choose the right base class** from `base.py` (`TimeSeriesDataset`, `VisionDataset`, or `LLMDataset`).

2. **Create a new file** in this directory, e.g. `mydata.py`.

3. **Implement the three required methods:**

```python
from fmtk.datasetloaders.base import VisionDataset  # or TimeSeriesDataset / LLMDataset

class MyDataset(VisionDataset):
    def __init__(self, dataset_cfg, task_cfg, split, preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)
        # read data using dataset_cfg['dataset_path']
        self.data = ...
        if preprocess:
            self.preprocess()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return a dict with at minimum:
        #   vision:     {"x": tensor,  "y": label}
        #   timeseries: {"x": array,   "y": target}  or  {"x": array, "mask": array}
        #   vlm:        {"x": tensor,  "question": str, "y": str}
        return {"x": ..., "y": ...}

    def preprocess(self):
        # build self.transform or normalize data in place
        pass
```

4. **Use the standard config keys** (`dataset_path`, `seq_len`, `target_size`, `mean`, `std`, `task_type`) so your loader is consistent with the rest of the codebase.

5. **Wire it into a Pipeline (Not generally required to implement):**

```python
from fmtk.pipeline import Pipeline
from fmtk.datasetloaders.mydata import MyDataset
from torch.utils.data import DataLoader

ds     = MyDataset(dataset_cfg, task_cfg, split="train")
loader = DataLoader(ds, batch_size=32, shuffle=True)

P = Pipeline(my_backbone)
P.add_decoder(my_decoder)
P.train(loader, parts_to_train=["decoder"], cfg=train_cfg)
y_test, y_pred = P.predict(test_loader, cfg=infer_cfg)
```
