# Adding a backbone

A backbone is a subclass of [`BaseModel`](../api/base.md) that wraps a pretrained model. `Pipeline` uses
duck typing, so it only calls the methods listed below. Put new backbones in `src/fmtk/components/backbones/`.

## Required interface

| Member | Purpose |
|---|---|
| `__init__(device, model_name=None, model_config=None)` | Load the weights. Map short names (`"small"`, `"base"`) to hub ids here. |
| `self.device` | Device the model runs on. |
| `self.model` | The underlying model. `Pipeline` calls `.eval()` on it before forward passes. |
| `self.model_category` | `"tsfm"`, `"vision"`, `"vlm"` or `"llm"`. It decides the `models/<category>/` checkpoint folder. `BaseModel` defaults to `"tsfm"`. |
| `forward(batch_x, mask=None)` | Return embeddings as a tensor. Differentiable decoders and encoder/adapter training use it. |
| `predict(dataloader)` | Return `(embeddings, labels)` as NumPy arrays over a whole loader. Fit/predict decoders (SVM, kNN, …) use it. |

`preprocess` and `postprocess` are optional hooks. By convention `forward` calls them to reshape inputs and outputs.

## Optional: adapters (PEFT)

To support `P.add_adapter(...)` and `parts_to_train=["adapter"]`, also provide:

| Member | Purpose |
|---|---|
| `self.peft_enable` | `False` until an adapter is attached. |
| `enable_peft(peft_cfg)` | Wrap `self.model` with `peft.get_peft_model` and set `peft_enable = True`. |
| `adapter_trainable_parameters()` | Yield the parameters with `requires_grad=True`. |

## Optional: generative backbones

VLM and LLM backbones run without a decoder:

- `forward((images, questions), None)` returns a list of generated strings, one per sample.
- `train_step(batch)` returns a scalar loss, so `Pipeline.train` can fine-tune the backbone (with an adapter) when no
  decoder is attached.
- vLLM runtimes may also implement `register_adapter(name, path)`. `add_adapter` then registers saved LoRA
  directories with the server instead of wrapping the weights with PEFT.

## Template

Based on `MomentModel` (`src/fmtk/components/backbones/moment.py`):

```python
import numpy as np
import torch
from peft import get_peft_model
from tqdm import tqdm
from fmtk.components.base import BaseModel

MODEL_IDS = {"small": "org/my-model-small", "base": "org/my-model-base"}


class MyModel(BaseModel):
    def __init__(self, device, model_name="base", model_config=None):
        super().__init__()
        self.device = device
        self.model_category = "tsfm"
        self.peft_enable = False
        self.model = load_my_pretrained(MODEL_IDS[model_name]).to(device)

    def preprocess(self, batch_x, mask=None):
        x = batch_x.float().to(self.device)          # (B, C, L) for time series
        return x, mask

    def forward(self, batch_x, mask=None):
        x, mask = self.preprocess(batch_x, mask)
        return self.model(x)                         # (B, ..., D) embeddings

    @torch.no_grad()
    def predict(self, dataloader):
        self.model.eval()
        embeddings, labels = [], []
        for batch in tqdm(dataloader):
            embeddings.append(self.forward(batch["x"]).cpu().float().numpy())
            labels.append(batch["y"])
        return np.vstack(embeddings), np.concatenate(labels)

    def enable_peft(self, peft_cfg):
        self.model = get_peft_model(self.model, peft_cfg)
        self.peft_enable = True

    def adapter_trainable_parameters(self):
        if not self.peft_enable:
            return []
        return (p for p in self.model.parameters() if p.requires_grad)
```

Then use it like any built-in backbone:

```python
P = Pipeline(MyModel("cuda:0", "base"))
```

## Checklist

- Embedding shape matches what your target decoders expect. Decoders average over extra token / channel dimensions,
  so `(B, D)`, `(B, T, D)` and `(B, C, T, D)` all work for the classification heads.
- Weights download into `models/<category>/pretrained/` (pass `cache_dir=`) rather than the default HF cache.
- Add a row to [Components](../reference/components.md) and an example under `experiments/examples/`.
- Add a short docstring to the class. It appears on the API pages automatically.
