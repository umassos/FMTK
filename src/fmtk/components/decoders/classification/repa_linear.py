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
        self.norm_in = nn.LayerNorm(cfg['input_dim'])
        self.repa = nn.Linear(in_features=cfg['input_dim'], out_features=cfg['repa_output_dim'], bias=False)
        self.norm_out = nn.LayerNorm(cfg['repa_output_dim'])
        # self.repa = nn.Conv2d(in_channels=cfg['input_dim'], out_channels=cfg['repa_output_dim'], kernel_size=(1, 1), bias=False)
        self.criterion = RepaLoss(nn.MSELoss(), normalize=cfg.get('normalize', True))
        self.preprocess_fn = lambda x: x
        self.postprocess_fn = lambda x: x
        self.requires_model = True
        
    def forward(self, x):
        x = self.preprocess_fn(x).float()
        x = self.norm_in(x)
        x = self.repa(x)
        x = self.norm_out(x)
        x = self.postprocess_fn(x)
        return x

    def save(self, path):
        torch.save(self.repa.state_dict(), path)
    
    def load(self, path):
        self.repa.load_state_dict(torch.load(path))

    def preprocess(self, x):
        # for vision only 
        # if x.ndim == 3:
        #     x = x.reshape(-1, int(x.shape[1]**0.5), int(x.shape[1]**0.5), x.shape[2]).permute(0, 3, 1, 2)
        return x
        # return self.preprocess_fn(x)

    def set_preprocess_fn(self, preprocess_fn):
        self.preprocess_fn = preprocess_fn


class RepaWrappedDecoder(nn.Module):
    def __init__(self, device, cfg, decoder):
        super().__init__()
        self.device = device
        self.repa = Repa(device, cfg)
        self.decoder = decoder
        self.model = decoder.model
        ignore_index = getattr(decoder, "ignore_index", 255)
        # self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.criterion = nn.MSELoss()

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