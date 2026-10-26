# Components

All paths are relative to `src/fmtk/components/`. Import a component with its module path, for example
`from fmtk.components.backbones.moment import MomentModel`.

## Backbones

### Time series (`tsfm`)

| Class | Module | `model_name` |
|---|---|---|
| `ChronosModel` | `backbones/chronos.py` | `tiny`, `mini`, `small`, `base`, `large` |
| `MomentModel` | `backbones/moment.py` | `small`, `base`, `large` |
| `MantisModel` | `backbones/mantis.py` | `8M` |
| `PapageiModel` | `backbones/papagei.py` | `papagei_p`, `papagei_s`, `papagei_s_svri` (needs `model_config`, see [Quickstart](../quickstart.md#forecasting-papagei-mlp)) |

### Vision (`vision`)

| Class | Module | `model_name` (default) |
|---|---|---|
| `DinoV2Model` | `backbones/dinov2.py` | `small`, `base`, `large`, `giant` (`base`) |
| `DinoV3Model` | `backbones/dinov3.py` | `vitb16` by default |
| `SwinModel` | `backbones/swin.py` | `tiny`, `small`, `base`, `large` (`base`) |
| `ConvNextModel` | `backbones/convnext.py` | `tiny` by default |
| `BEiTModel` | `backbones/beit.py` | `base` by default |
| `MAEModel` | `backbones/mae.py` | `base` by default |
| `ResNetVisionModel` | `backbones/resnet.py` | `base` by default |
| `VGGModel` | `backbones/vgg.py` | `vgg16` by default |

### Vision-language (`vlm`)

| Class | Module | `model_name` |
|---|---|---|
| `QwenModel` | `backbones/qwen.py` | `qwen-2B`, `qwen-3B`, `qwen-7B` |
| `LlavaModel` | `backbones/llava.py` | `llava-1.5-7b`, `llava-1.5-13b`, `llava-v1.6-13b` |
| `MolmoModel` | `backbones/molmo.py` | see module |
| `MinicpmModel` | `backbones/minicpm.py` | see module |
| `MoondreamModel` | `backbones/moondream.py` | see module |
| `PhiModel` | `backbones/phi.py` | see module |
| `LlamaVisionModel` | `backbones/llama_vision.py` | see module |

### Language (`llm`)

| Class | Module | `model_name` (default) |
|---|---|---|
| `QwenTextModel` | `backbones/qwen.py` | `qwen2.5-0.5b` … `qwen2.5-14b` (`qwen2.5-7b`) |
| `QwenVLLMModel` | `backbones/qwen_vllm.py` | vLLM runtime, multi-LoRA (`qwen2.5-7b`) |
| `LlamaModel` | `backbones/llama.py` | `llama-3.1-8b` by default |
| `MistralModel` | `backbones/mistral.py` | `mistral-7b` by default |
| `Phi3Model` | `backbones/phi3.py` | `phi3-mini` by default |
| `Phi3VLLMModel` | `backbones/phi3_vllm.py` | vLLM runtime (`phi3-mini`) |
| `PhiVLLMModel` | `backbones/phi.py` | vLLM runtime, `vlm` or `llm` depending on `model_name` (`phi3.5-vision`) |

## Decoders

**Kind**: *fit* = scikit-learn style `fit`/`predict` on frozen embeddings; *diff* = differentiable PyTorch head
(see [Two kinds of decoders](../concepts.md#two-kinds-of-decoders)).

| Task | Class | Module | Kind | Constructor |
|---|---|---|---|---|
| Classification | `SVMDecoder` | `decoders/classification/svm.py` | fit | `SVMDecoder(max_samples=10000)` |
| | `KNNDecoder` | `decoders/classification/knn.py` | fit | `KNNDecoder(n_neighbors=5)` |
| | `LogisticRegressionDecoder` | `decoders/classification/logisticregression.py` | fit | `LogisticRegressionDecoder(max_iter=10000)` |
| | `RandomForestDecoder` | `decoders/classification/randomforest.py` | fit | `RandomForestDecoder(n_estimators=100)` |
| | `LinearDecoder` | `decoders/classification/linear.py` | diff | `cfg`: `input_dim`, `output_dim`, `norm`, `mode` |
| | `MLPDecoder` | `decoders/classification/mlp.py` | diff | `cfg`: `input_dim`, `output_dim`, `hidden_dim` |
| Regression | `RidgeDecoder` | `decoders/regression/ridge.py` | fit | `RidgeDecoder()` |
| | `LinearDecoder` | `decoders/regression/linear.py` | diff | `cfg` |
| | `MLPDecoder` | `decoders/regression/mlp.py` | diff | `cfg`: `input_dim`, `output_dim`, `hidden_dim` |
| | `MonocularDepthDecoder` | `decoders/regression/monocular_depth.py` | diff | `cfg` |
| | `SpatialCountDecoder` | `decoders/regression/spatial_count.py` | diff | `cfg` |
| Forecasting | `MLPDecoder` | `decoders/forecasting/mlp.py` | diff | `cfg`: `input_dim`, `output_dim` (horizon), `dropout` |
| Segmentation | `LinearSemanticSegmenter` | `decoders/segmentation/LinearSemanticSegmenter.py` | diff | `cfg` |
| | `BNLinearSemanticSegmenter` | `decoders/segmentation/BNLinearSemanticSegmenter.py` | diff | `cfg` |

Every differentiable decoder takes `device` as its first argument, e.g. `MLPDecoder(device, cfg)`.

## Encoders

| Class | Module | Purpose |
|---|---|---|
| `LinearChannelCombiner` | `encoders/diff.py` | Learnable linear mix of `num_channels` → `new_num_channels` |

## Metrics

`fmtk.metrics` provides `get_accuracy`, `get_mae`, `get_mIoU`, `get_exact_match`, `get_token_f1`, `get_bleu`,
`get_rouge` and `get_gsm8k_accuracy`. See the [API page](../api/metrics.md) for details.
