# Adding a decoder or encoder

Decoders live in `src/fmtk/components/decoders/<task>/` (`classification`, `regression`, `forecasting`,
`segmentation`, `detection`), and encoders in `src/fmtk/components/encoders/`. As with backbones, `Pipeline` uses duck
typing, so what matters is which methods your class provides.

## Differentiable decoder

Use this for PyTorch heads that train with gradients, alone or jointly with an encoder or adapter.

| Member | Purpose |
|---|---|
| `self.device` | Device for the head. Targets are moved here before computing the loss. |
| `self.model` | The `nn.Module`. Its `state_dict` is what gets saved as `decoder.pth`. |
| `self.criterion` | Loss. With `CrossEntropyLoss`, `predict` returns argmax class ids. With `MSELoss` / `L1Loss` / `SmoothL1Loss`, outputs and targets are cast to float. |
| `forward(embeddings)` | Return logits / predictions. |
| `trainable_parameters()` | Parameters for the optimizer, usually `self.model.parameters()`. |
| `to_device()` / `to_cpu()` | Move the head. Used by `add_decoder(load=True)` and `load_decoder(swap=True)`. |
| `requires_model` *(optional)* | Set `True` to have outputs de-normalized with the backbone's `normalizer` (RevIN), as the forecasting `MLPDecoder` does. |

```python
import torch
import torch.nn as nn
from fmtk.components.base import BaseModel


class MyLinearDecoder(BaseModel):
    def __init__(self, device, cfg):
        self.device = device
        self.model = nn.Linear(cfg["input_dim"], cfg["output_dim"])
        self.criterion = nn.CrossEntropyLoss()

    def to_device(self):
        self.model.to(self.device)

    def to_cpu(self):
        self.model.to("cpu")

    def trainable_parameters(self):
        return self.model.parameters()

    def forward(self, batch_x):
        x = batch_x.to(torch.float32).to(self.device)
        while x.ndimension() > 2:        # average token / channel dims -> (B, D)
            x = x.mean(dim=1)
        return self.model(x)
```

### Vision token selection

Vision heads can subclass [`BaseVisionDecoder`](../api/base.md) and set `self.mode` from `cfg["mode"]`. Then
`self.select_embeddings(x)` picks the tokens the head uses:

| `mode` | Uses |
|---|---|
| `CLS` | the CLS token |
| `PATCH` | patch tokens only |
| `ALL` | every token |
| `CLS_PATCHMEAN` | CLS concatenated with the mean of the patch tokens |
| `MULTICLS_PATCHMEAN` | CLS tokens from several layers plus the last layer's patch mean (DINOv2 with `model_config={"layers": [...]}`) |

## Fit / predict decoder

Use this for scikit-learn style heads that train on precomputed embeddings in one shot.

| Member | Purpose |
|---|---|
| `fit(dataloader, cfg)` | Train. The loader yields `(embedding, label)` tensor tuples. |
| `predict(dataloader)` | Return `(labels, predictions)` as NumPy arrays. |

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from fmtk.components.base import BaseModel


class MyKNNDecoder(BaseModel):
    def __init__(self, k=5):
        self.model = KNeighborsClassifier(n_neighbors=k)

    def _collect(self, loader):
        xs, ys = [], []
        for x, y in loader:
            while x.ndimension() > 2:
                x = x.mean(dim=1)
            xs.append(x.numpy())
            ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)

    def fit(self, dataloader, cfg):
        self.model.fit(*self._collect(dataloader))

    def predict(self, dataloader):
        x, y = self._collect(dataloader)
        return y, self.model.predict(x)
```

!!! note
    A fit/predict decoder can't be trained together with an encoder or adapter. The pipeline raises a `ValueError`,
    because no gradient can flow back through it.

## Encoder

An encoder transforms `batch["x"]` before the backbone. To be trainable it needs:

| Member | Purpose |
|---|---|
| `self.model` | The `nn.Module`, saved as `encoder.pth`. |
| `forward(batch_x)` | Return the transformed input in the shape the backbone expects. |
| `trainable_parameters()` | Parameters for the optimizer. |

`LinearChannelCombiner` (`src/fmtk/components/encoders/diff.py`) is the reference implementation. It learns a linear
mix of `num_channels` input channels into `new_num_channels`, so you can feed multi-channel signals to a
single-channel backbone.

An encoder can instead define `fit(dataloader)` that returns a transformed loader. The pipeline then calls it once
before training.
