import torch
from abc import ABC, abstractmethod


class BaseVisionDecoder(ABC):
    def __init__(self):
        pass

    def select_embeddings(self, batch_x):
        if self.mode == "CLS":
            return batch_x[:, 0, :]
        elif self.mode == "PATCH":
            return batch_x[:, 1:, :]
        elif self.mode == "ALL":
            return batch_x
        elif self.mode == "CLS_PATCHMEAN":
            return torch.cat([batch_x[:, 0, :], batch_x[:, 1:, :].mean(dim=1)], dim=-1)
        elif self.mode == "MULTICLS_PATCHMEAN":
            # [B, n_layers, T, D] -> [cls_1 ... cls_n, mean(patches of last layer)]
            return torch.cat([batch_x[:, :, 0, :].flatten(1), batch_x[:, -1, 1:, :].mean(dim=1)], dim=-1)
        else:
            raise ValueError(f"[BaseVisionDecoder]Invalid mode: {self.mode}")
