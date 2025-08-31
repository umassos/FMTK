import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm
import numpy.typing as npt
from fmtk.components.base import BaseModel

class MLPHead(nn.Module):
    def __init__(self, head_nf: int, forecast_horizon: int, head_dropout: float = 0.1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None):
        if x.ndimension() == 4:
            x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MLPDecoder(BaseModel):
    def __init__(self,device,cfg=None):
        self.device = device
        self.model = MLPHead(head_nf=cfg['input_dim'], forecast_horizon=cfg['output_dim'], head_dropout=cfg['dropout'])
        self.requires_model = True
        self.criterion = nn.MSELoss().to(self.device)

    def to_device(self):
        self.model.to(self.device)

    def preprocess(self,batch):
        x,y=batch
        x = x.to(self.device).to(torch.float32)
        return x,y

    def forward(self,batch):
        x,y=self.preprocess(batch)
        output = self.model(x)
        return output,y

    def postprocess(self,x):
        pass

    def trainable_parameters(self):
        return self.model.parameters()