import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from fmtk.components.base import BaseModel
from fmtk.components.decoders.base import BaseVisionDecoder


class LinearDecoder(nn.Module, BaseVisionDecoder):
    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.model = nn.Linear(in_features=cfg['input_dim'], out_features=cfg['output_dim'])
        self.criterion = nn.CrossEntropyLoss()
        self.mode = cfg.get('mode', None)

    def to_device(self):
        self.model.to(self.device)
        self.criterion.to(self.device)
        
    def to_cpu(self):
        self.model.to('cpu')

    def trainable_parameters(self):
        return self.model.parameters()
    
    def preprocess(self,batch_x):
        x=batch_x
        x=x.to(torch.float32).to(self.device)
        if self.mode is not None:
            x = self.select_embeddings(x)
        if x.ndimension() == 4:
            x=x.mean(dim=2)
        if x.ndimension() == 3:
            x=x.mean(dim=1)
        return x
    
    def postprocess(self,embedding):
        pass

    def forward(self, batch_x):
        x=self.preprocess(batch_x)
        return self.model(x)