import torch.nn as nn
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
        else:
            raise ValueError(f"[BaseVisionDecoder]Invalid mode: {self.mode}")
