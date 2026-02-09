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
    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.repa = nn.Linear(in_features=cfg['input_dim'], out_features=cfg['repa_output_dim'], bias=False)
        self.criterion = RepaLoss(nn.MSELoss(), normalize=cfg.get('normalize', True))


    def forward(self, x):
        return self.repa(x)

    def save(self, path):
        torch.save(self.repa.state_dict(), path)
    
    def load(self, path):
        self.repa.load_state_dict(torch.load(path))


class RepaLinearDecoder(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.repa = Repa(device, cfg)
        self.decoder = nn.Linear(in_features=cfg['repa_output_dim'], out_features=cfg['output_dim'])
        self.criterion = nn.CrossEntropyLoss()

    def to_device(self):
        self.repa.to(self.device)
        self.decoder.to(self.device)
        self.criterion.to(self.device)

    def to_cpu(self):
        self.decoder.to('cpu')
        self.repa.to('cpu')
        self.criterion.to('cpu')
    
    def load_repa(self, path):
        self.repa.load(path)

    def load_decoder(self, path):
        self.decoder.load_state_dict(torch.load(path))

    def forward(self, x):
        x = self.repa(x)
        return self.decoder(x)