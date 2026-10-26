# Core concepts

## The pipeline

A [`Pipeline`](api/pipeline.md) wraps one **backbone** and routes data through any components attached to it:

```
batch["x"] ──► encoder (optional) ──► backbone (+ adapter, optional) ──► decoder ──► prediction
```

| Component | Role | Attached with | Example |
|---|---|---|---|
| **Backbone** | Pretrained foundation model that produces embeddings (or text, for VLMs and LLMs) | `Pipeline(backbone)` | `MomentModel`, `DinoV2Model`, `QwenModel` |
| **Encoder** | Transforms the raw input before the backbone sees it | `P.add_encoder(...)` | `LinearChannelCombiner` (N channels → M) |
| **Adapter** | Parameter-efficient fine-tuning inside the backbone | `P.add_adapter(peft_cfg)` | `peft.LoraConfig` |
| **Decoder** | Task head that maps embeddings to outputs | `P.add_decoder(...)` | `SVMDecoder`, `MLPDecoder`, `LinearSemanticSegmenter` |

You can attach several decoders and encoders to one pipeline. Each `add_*` call returns an id (`"decoder_0"`,
`"decoder_1"`, …). The `load=True` argument makes that component the active one. You can then switch decoders without
reloading the backbone (see [Training and inference](guides/training.md#multiple-decoders-on-one-backbone)).

Generative backbones (VLMs and LLMs) usually run **without** a decoder: `predict` returns the generated text.

## Model categories

Every backbone sets `model_category`, which determines where checkpoints are read from and written to:

| `model_category` | Backbones | Checkpoints under |
|---|---|---|
| `tsfm` | Chronos, MOMENT, Mantis, PaPaGei | `models/tsfm/` |
| `vision` | DINOv2/v3, Swin, ConvNeXt, BEiT, MAE, ResNet, VGG | `models/vision/` |
| `vlm` | Qwen-VL, LLaVA, Molmo, MiniCPM, Moondream, Phi-Vision, Llama-Vision | `models/vlm/` |
| `llm` | Qwen (text), Llama, Mistral, Phi-3 and their vLLM variants | `models/llm/` |

## Configuration dicts

FMTK passes configuration around as plain dicts, not config classes.

| Dict | Used by | Common keys |
|---|---|---|
| `dataset_cfg` | dataset loader | `dataset_path`, `seq_len` (time series), `target_size`, `mean`, `std` (vision), `json_file`, `image_subdir` (VLM) |
| `task_cfg` | dataset loader | `task_type` (`classification`, `regression`, `forecasting`, `imputation`, `segmentation`), `label`, `prompt` |
| `model_config` | backbone constructor | backbone-specific, e.g. `{"return_all_tokens": True}` or `{"layers": [...]}` for DINOv2 |
| decoder `cfg` | decoder constructor | `input_dim`, `output_dim`, `hidden_dim`, `dropout`, `mode` (vision token selection) |
| `train_config` | `P.train(cfg=...)` | `batch_size`, `shuffle`, `epochs`, `lr`, `use_cache` |
| `inference_config` | `P.predict(cfg=...)` | `batch_size`, `shuffle` |

`train_config["lr"]` applies to every trainable parameter group (encoder, adapter, decoder).

## Batch format

Dataset loaders return dicts, and the pipeline reads these keys:

| Key | Meaning |
|---|---|
| `x` | Input: time series `(channels, seq_len)`, or image `(3, H, W)` |
| `y` | Target: class index, regression value, forecast window, segmentation mask, or answer string |
| `mask` | Optional input mask (time series padding / imputation) |
| `idx` | Optional sample index. Needed for the embedding cache (`use_cache`). |
| `question` | Prompt text for VLM / LLM loaders |

## Two kinds of decoders

The pipeline checks what a decoder provides and picks the training path to match:

- **Fit / predict decoders** (`SVMDecoder`, `KNNDecoder`, `LogisticRegressionDecoder`, `RandomForestDecoder`,
  `RidgeDecoder`) define `fit()` and `predict()`. The pipeline runs the frozen backbone once over the data, collects
  the embeddings, and hands them to scikit-learn. These decoders can't be trained jointly with an encoder or adapter,
  because no gradient flows through them.
- **Differentiable decoders** (`LinearDecoder`, `MLPDecoder`, segmentation and depth heads) are PyTorch modules. The
  pipeline trains them with Adam and the decoder's own `criterion`, optionally together with an encoder and/or adapter.

See [Adding a decoder](guides/decoders.md) for the exact interface of each.
