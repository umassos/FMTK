# Quickstart

Every FMTK workflow has the same four steps:

1. Wrap a dataset in a PyTorch `DataLoader`.
2. Build a `Pipeline` around a backbone.
3. Attach the components you need (decoder, optionally an encoder or adapter).
4. `train` the parts you want, then `predict` and score.

The examples below come from [`examples/`](https://github.com/umassos/FMTK/tree/main/examples) and
[`experiments/examples/`](https://github.com/umassos/FMTK/tree/main/experiments/examples). They assume a GPU at
`cuda:0` and are run from inside `examples/`, so relative dataset paths point to `../dataset/`.

## ECG classification: Chronos + SVM

Download [ECG5000](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000) and place
`ECG5000_TRAIN.ts` and `ECG5000_TEST.ts` in `dataset/ECG5000/`.

```python
from torch.utils.data import DataLoader
from fmtk.pipeline import Pipeline
from fmtk.datasetloaders.ecg5000 import ECG5000Dataset
from fmtk.components.backbones.chronos import ChronosModel
from fmtk.components.decoders.classification.svm import SVMDecoder
from fmtk.metrics import get_accuracy
from fmtk.logger import Logger

device = "cuda:0"
task_cfg = {"task_type": "classification"}
dataset_cfg = {"dataset_path": "../dataset/ECG5000"}
train_config = {"batch_size": 32, "shuffle": False, "epochs": 50, "lr": 1e-2}
inference_config = {"batch_size": 32, "shuffle": False}

train_loader = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg, split="train"), batch_size=32)
test_loader = DataLoader(ECG5000Dataset(dataset_cfg, task_cfg, split="test"), batch_size=32)

logger = Logger(device, "chronos_ecg")            # optional: GPU memory / energy / timing
P = Pipeline(ChronosModel(device, "tiny"), logger)
P.add_decoder(SVMDecoder(), load=True)
P.train(train_loader, parts_to_train=["decoder"], cfg=train_config)
y_test, y_pred = P.predict(test_loader, cfg=inference_config)
print(get_accuracy(y_test, y_pred))
logger.save()                                      # writes ./logs/chronos_ecg.json
```

The backbone stays frozen: FMTK extracts embeddings once and fits the SVM on them.

## Forecasting: PaPaGei + MLP

Download [ETTh1](https://github.com/zhouhaoyi/ETDataset) into `dataset/ETTh1/ETTh1.csv`, and the PaPaGei weights as
described in the [PaPaGei repo](https://github.com/Nokia-Bell-Labs/papagei-foundation-model).

```python
from fmtk.datasetloaders.etth1 import ETTh1Dataset
from fmtk.components.backbones.papagei import PapageiModel
from fmtk.components.decoders.forecasting.mlp import MLPDecoder
from fmtk.metrics import get_mae

task_cfg = {"task_type": "forecasting"}
dataset_cfg = {"dataset_path": "../dataset/ETTh1"}
train_config = {"batch_size": 8, "shuffle": False, "epochs": 1, "lr": 1e-4}
inference_config = {"batch_size": 8, "shuffle": False}
# ... build train_loader / test_loader as above with ETTh1Dataset ...

model_cfg = {"in_channels": 1, "base_filters": 32, "kernel_size": 3, "stride": 2,
             "groups": 1, "n_block": 18, "n_classes": 512, "n_experts": 3}

P = Pipeline(PapageiModel(device, model_name="papagei_s", model_config=model_cfg))
P.add_decoder(MLPDecoder(device, cfg={"input_dim": 512, "output_dim": 192, "dropout": 0.1}), load=True)
P.train(train_loader, parts_to_train=["decoder"], cfg=train_config)
y_test, y_pred = P.predict(test_loader, cfg=inference_config)
print(get_mae(y_test, y_pred))
```

## PPG regression: MOMENT + encoder + LoRA + MLP

This example trains three parts together: a channel-combining **encoder** (3 channels → 1), a **LoRA adapter** inside
the backbone, and an **MLP decoder**. Download the
[PPG-BP dataset](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299) into `dataset/PPG-data/`.

```python
from peft import LoraConfig
from fmtk.datasetloaders.ppg import PPGDataset
from fmtk.components.backbones.moment import MomentModel
from fmtk.components.decoders.regression.mlp import MLPDecoder
from fmtk.components.encoders.diff import LinearChannelCombiner

task_cfg = {"task_type": "regression", "label": "hr"}   # or "sysbp", "diasbp"
dataset_cfg = {"dataset_path": "../dataset/PPG-data"}
train_config = {"batch_size": 32, "shuffle": False, "epochs": 50, "lr": 1e-2}
inference_config = {"batch_size": 32, "shuffle": False}
# ... build train_loader / test_loader with PPGDataset ...

lora_config = LoraConfig(r=64, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05)

P = Pipeline(MomentModel(device, "base"))
P.add_decoder(MLPDecoder(device, cfg={"input_dim": 768, "output_dim": 1, "hidden_dim": 128}), load=True)
P.add_encoder(LinearChannelCombiner(num_channels=3, new_num_channels=1), load=True)
P.add_adapter(lora_config)
P.train(train_loader, parts_to_train=["encoder", "decoder", "adapter"], cfg=train_config)
y_test, y_pred = P.predict(test_loader, cfg=inference_config)
print(get_mae(y_test, y_pred))
```

## Image classification: DINOv2 + logistic regression

```python
from fmtk.datasetloaders.cifar10 import CIFAR10Dataset
from fmtk.components.backbones.dinov2 import DinoV2Model
from fmtk.components.decoders.classification.logisticregression import LogisticRegressionDecoder

task_cfg = {"task_type": "classification"}
dataset_cfg = {"dataset_path": "../dataset/cifar-10"}
# ... build train_loader / test_loader with CIFAR10Dataset ...

P = Pipeline(DinoV2Model(device, "base", {}))
P.add_decoder(LogisticRegressionDecoder(max_iter=10000), load=True)
P.train(train_loader, parts_to_train=["decoder"], cfg=train_config)
y_test, y_pred = P.predict(test_loader, cfg=inference_config)
print(get_accuracy(y_test, y_pred))
```

## Visual question answering: a VLM with no decoder

Generative backbones (VLMs and LLMs) answer directly, so you don't attach a decoder. `predict` returns the ground-truth
labels and the generated strings.

```python
from fmtk.datasetloaders.vlm_dataset import VLMDataset, vlm_collate_fn
from fmtk.components.backbones.qwen import QwenModel

dataset_cfg = {"dataset_path": "/path/to/vqa", "json_file": "val.json", "image_subdir": "val2014"}
task_cfg = {"prompt": "Answer the question in one word. Question: {question}",
            "inference_config": {"max_samples": 500}}

test_loader = DataLoader(VLMDataset(dataset_cfg, task_cfg, split="test"),
                         batch_size=1, collate_fn=vlm_collate_fn)

P = Pipeline(QwenModel(device, "qwen-3B"))
y_test, y_pred = P.predict(test_loader, cfg={"batch_size": 1, "shuffle": False})
print(get_accuracy(y_test, y_pred))
```

## More examples

| Script | Shows |
|---|---|
| `experiments/examples/moment_etth1.py`, `chronos_etth1.py`, `papagei_etth1.py` | Forecasting with each TSFM |
| `experiments/examples/moment_adapter_mlpdecoder.py` | LoRA adapter + MLP decoder |
| `experiments/examples/dino_ade20k_linear.py`, `swin_voc.py`, `dinov3_voc.py` | Semantic segmentation |
| `experiments/examples/dino_nyu.py`, `swin_nyu.py` | Monocular depth estimation |
| `experiments/examples/dino_shanghaitechcount.py` | Crowd counting |
| `experiments/examples/llava_vqa.py`, `molmo_vqa.py`, `moondream_vqa.py`, … | VQA with other VLMs |
