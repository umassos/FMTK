import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy.typing as npt
from sklearn.utils.class_weight import compute_class_weight
from fmtk.components.base import BaseModel

class MLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class MLPDecoder(BaseModel):
    def __init__(self, device, cfg=None):
        self.device = device
        self.model = MLPHead(input_dim=cfg['input_dim'],output_dim=cfg['output_dim'],hidden_dim=cfg['hidden_dim'])
        self.criterion = nn.MSELoss().to(self.device)

    def to_device(self):
        self.model.to(self.device)

    def to_cpu(self):
        self.model.to('cpu')
        
    def trainable_parameters(self):
        return self.model.parameters()
    
    def preprocess(self,batch_x):
        x=batch_x
        x=x.to(self.device).to(torch.float32)
        if x.ndimension() == 4:
            x=x.mean(dim=2)
        if x.ndimension() == 3:
            x=x.mean(dim=1)
        return x
        
    def forward(self, batch_x):
        features=self.preprocess(batch_x)
        return self.model(features)
    