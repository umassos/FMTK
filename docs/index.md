# FMTK

**A modular toolkit for composable foundation model pipelines.**

Foundation models (FMs) can adapt to new tasks with little or no extra training. Practical pipelines usually wrap a
foundation model with task-specific **encoders**, **decoders** and **adapters**, but building these pipelines usually
means writing one-off, model-specific code. That makes the pipelines hard to reuse or reproduce.

FMTK is an open-source, lightweight and extensible toolkit for building and fine-tuning these pipelines through
standard backbone and component abstractions. It started with time-series foundation models (TSFMs) and now also
covers vision, vision-language (VLM) and language (LLM) backbones. A complete pipeline takes about seven lines of code:

```python
P = Pipeline(ChronosModel(device, "tiny"))                         # 1. pick a backbone
P.add_decoder(SVMDecoder(), load=True)                              # 2. attach a task head
P.train(train_loader, parts_to_train=["decoder"], cfg=train_cfg)    # 3. train only what you need
y_true, y_pred = P.predict(test_loader, cfg=infer_cfg)              # 4. predict
print(get_accuracy(y_true, y_pred))                                 # 5. evaluate
```

![FMTK architecture](images/architecture.jpg)

## What's in the box

| | |
|---|---|
| **Backbones** | Time series: Chronos, MOMENT, Mantis, PaPaGei · Vision: DINOv2/v3, Swin, ConvNeXt, BEiT, MAE, ResNet, VGG · VLM: Qwen-VL, LLaVA, Molmo, MiniCPM, Moondream, Phi-Vision, Llama-Vision · LLM: Qwen, Llama, Mistral, Phi-3 (incl. vLLM runtimes) |
| **Decoders** | Classification (linear, MLP, SVM, kNN, logistic regression, random forest), regression (linear, MLP, ridge, depth, counting), forecasting (MLP), segmentation (linear, BN-linear) |
| **Encoders** | Linear channel combiner for multi-channel time series |
| **Adapters** | LoRA and other [PEFT](https://huggingface.co/docs/peft) adapters, including multi-adapter serving |
| **Datasets** | 30+ loaders across time series, vision, VLM and NLP tasks |
| **Profiling** | Built-in logger for wall time, GPU time, GPU memory and energy |

See [Components](reference/components.md) for the full catalog.

## Where to go next

- [Installation](installation.md): set up the conda environment.
- [Quickstart](quickstart.md): runnable examples for classification, forecasting, regression, vision and VQA.
- [Core concepts](concepts.md): how a pipeline is put together and what each config dict does.
- [Adding a backbone](guides/backbones.md) / [decoder](guides/decoders.md) / [dataset](reference/datasets.md#how-to-add-a-new-dataset-loader): extend FMTK.
