import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from fmtk.components.base import BaseModel


class LinearDecoder(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.model = nn.Linear(in_features=cfg['input_dim'], out_features=cfg['output_dim'])
        self.criterion = nn.CrossEntropyLoss()
        self.reduction = 'concat'

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
        # shape will be [B, 3, 512] concat to [B, 3*512]
        if self.reduction == 'concat':
            x = x.view(x.size(0), -1)
        elif self.reduction == 'mean':
            if x.ndimension() == 4:
                x=x.mean(dim=2)
            if x.ndimension() == 3:
                x=x.mean(dim=1)
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")
        
        return x
    
    def postprocess(self,embedding):
        pass

    def forward(self, batch_x):
        x=self.preprocess(batch_x)
        return self.model(x)