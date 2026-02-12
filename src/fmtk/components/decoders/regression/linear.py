import torch
import torch.nn as nn
from fmtk.components.base import BaseModel


class LinearDecoder(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.model = nn.Linear(in_features=cfg['input_dim'], out_features=cfg['output_dim'])
        self.criterion = nn.MSELoss()

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
        if x.ndimension() == 4:
            x=x.mean(dim=2)
        if x.ndimension() == 3:
            x=x.mean(dim=1)
        return x
    
    def postprocess(self,embedding):
        pass

    def forward(self, batch_x):
        x=self.preprocess(batch_x)
        output = self.model(x)
        if self.cfg['output_dim'] == 1:
            return output.squeeze(-1)
        else:
            return output