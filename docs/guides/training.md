# Training and inference

## Choosing what to train

`Pipeline.train` takes a `parts_to_train` list. Anything left out of the list stays frozen.

```python
P.train(train_loader, parts_to_train=["decoder"], cfg=train_config)                          # linear probe / head only
P.train(train_loader, parts_to_train=["decoder", "adapter"], cfg=train_config)               # + LoRA in the backbone
P.train(train_loader, parts_to_train=["encoder", "decoder", "adapter"], cfg=train_config)    # everything trainable
```

| Situation | What happens |
|---|---|
| Decoder has `fit()` (e.g. SVM) | Embeddings are extracted once with the frozen backbone, then `decoder.fit(...)` runs. Only `["decoder"]` is allowed. |
| Differentiable decoder | A single Adam optimizer covers every listed part, at `cfg["lr"]`, for `cfg["epochs"]` epochs. |
| No decoder (VLM / LLM) | The backbone's own `train_step(batch)` loss is minimized with mixed precision, typically with an adapter. |

`predict` returns `(y_true, y_pred)` as NumPy arrays. For classification heads with `CrossEntropyLoss`, `y_pred` holds
class indices (argmax), not logits.

## Embedding cache

When you train **only** a differentiable decoder, the backbone output doesn't change between epochs. Set
`use_cache=True` to compute each sample's embedding once and reuse it in later epochs:

```python
train_config = {"batch_size": 32, "shuffle": True, "epochs": 50, "lr": 1e-3, "use_cache": True}
```

Requirements:

- the batches must include an `idx` key (most vision loaders provide one),
- `parts_to_train` must be exactly `["decoder"]`.

!!! warning "Don't cache with MOMENT time-series decoders"
    Decoders that set `requires_model = True` (e.g. the forecasting `MLPDecoder`) de-normalize their outputs with the
    backbone's RevIN statistics from the *most recent* forward pass. A cache hit skips that forward pass, so the
    statistics come from the wrong batch. Keep `use_cache=False` for MOMENT time-series tasks.

## Saving and loading

Pass `path` to `train` to save every trained part under `models/<model_category>/finetuned/<path>/`:

```python
logger = Logger(device, "ecg_run")
P = Pipeline(MomentModel(device, "small"), logger)
P.add_decoder(MLPDecoder(device, cfg), load=True)
P.train(train_loader, parts_to_train=["decoder"], cfg=train_config, path="ecgclass_momentsmall_mlp")
```

| File | Written when |
|---|---|
| `decoder.pth` | `"decoder"` was trained |
| `encoder.pth` | `"encoder"` was trained |
| `adapter.pth/` | `"adapter"` was trained (PEFT `save_pretrained` directory) |
| `pipeline.json` | always. Holds the logger's timing / memory / energy summary, **so a `Logger` is required when `path` is set**. |

Use the naming convention `{task}_{backbone}_{decoder}[_{encoder}][_{adapter}]`, e.g. `diasbp_chronosbase_mlp` or
`etth1_fore_momentsmall_mlp_lora`.

To load the saved parts later, pass `train=False` with the same `path`:

```python
P = Pipeline(MomentModel(device, "small"))
P.add_decoder(MLPDecoder(device, cfg), load=True, train=False, path="ecgclass_momentsmall_mlp")
P.add_adapter(lora_config, train=False, path="etth1_fore_momentsmall_mlp_lora")   # if one was trained
y_test, y_pred = P.predict(test_loader, cfg=inference_config)
```

If `decoder.pth` isn't on disk, FMTK tries to download it from the `umass-lass/fmtk-decoder-zoo` Hugging Face repo,
using the token in `hf-token.txt` at the repo root.

## Multiple decoders on one backbone

The backbone is the expensive part, so load it once and switch heads:

```python
P = Pipeline(MomentModel(device, "base"))
hr  = P.add_decoder(MLPDecoder(device, cfg_hr),  load=False, train=False, path="hr_momentbase_mlp")
bp  = P.add_decoder(MLPDecoder(device, cfg_bp),  load=False, train=False, path="sysbp_momentbase_mlp")

P.load_decoder(hr, swap=True)     # swap=True moves the old head to CPU and the new one to the GPU
hr_true, hr_pred = P.predict(test_loader, cfg=inference_config)

P.load_decoder(bp, swap=True)
bp_true, bp_pred = P.predict(test_loader, cfg=inference_config)
```

Adapters work the same way: `add_adapter` returns a name, `P.set_adapter(name)` activates it, and
`P.unload_adapter()` disables all adapter layers. For vLLM backbones (`QwenVLLMModel`, `Phi3VLLMModel`, …),
`add_adapter(..., path=...)` registers a saved LoRA directory with the runtime instead, so several adapters can serve
requests at the same time.

## Profiling with the logger

Pass a [`Logger`](../api/logger.md) to the pipeline to record wall time, GPU time, peak GPU memory, GPU energy and
CPU memory for each stage (`add_decoder_*`, `train`, `predict`, …):

```python
logger = Logger("cuda:0", run_name="chronos_ecg", save_dir="./logs")
P = Pipeline(ChronosModel("cuda:0", "tiny"), logger)
...
print(logger.summary())   # per-section averages
logger.save()             # appends to ./logs/chronos_ecg.json
```

You can also time your own code with `with logger.measure("my_section", device=logger.device): ...`.
