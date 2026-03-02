import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from fmtk.components.base import BaseModel


class RepaLoss(object):
    def __init__(self, loss_fn, normalize=True):
        self.loss_fn = loss_fn
        self.normalize = normalize

    def to(self, device):
        self.loss_fn.to(device)

    def __call__(self, repr_x, repr_y):
        if self.normalize:
            repr_x = F.normalize(repr_x, dim=-1)
            repr_y = F.normalize(repr_y, dim=-1)
        return self.loss_fn(repr_x, repr_y)
        

class Repa(nn.Module):
    def __init__(self, device, stitch, criterion, preprocess_fn):
        super().__init__()
        self.device = device
        self.repa = stitch
        self.criterion = criterion
        self.preprocess_fn = preprocess_fn

    def forward(self, x):
        x = self.preprocess_fn(x).float()
        return self.repa(x)

    def save(self, path):
        torch.save(self.repa.state_dict(), path)
    
    def load(self, path):
        self.repa.load_state_dict(torch.load(path))


class RepaWrappedDecoder(nn.Module):
    def __init__(self, device, cfg, decoder):
        super().__init__()
        self.device = device
        self.repa = Repa(device, cfg)
        self.decoder = decoder
        self.model = decoder.model
        ignore_index = getattr(decoder, "ignore_index", 255)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def trainable_parameters(self):
        return self.parameters()

    def to_device(self):
        self.repa.to(self.device)
        self.decoder.to_device()
        self.criterion.to(self.device)

    def to_cpu(self):
        self.decoder.to_cpu()
        self.repa.to('cpu')
        self.criterion.to('cpu')

    def load_repa(self, path):
        self.repa.load(path)

    def load_decoder(self, path):
        self.decoder.model.load_state_dict(torch.load(path))

    def forward(self, x):
        x = x.to(device=self.device, dtype=torch.float32)
        x = self.repa(x)
        return self.decoder(x)