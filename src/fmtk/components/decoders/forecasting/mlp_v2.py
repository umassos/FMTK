import torch
import torch.nn as nn
from fmtk.components.base import BaseModel

class MLPHead(nn.Module):
    def __init__(self, head_nf: int, forecast_horizon: int, hidden_dim: int = None):
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else head_nf // 2
        self.flatten = nn.Flatten(start_dim=-2)  # [B, C, P, D] -> [B, C, P*D]
        self.linear1 = nn.Linear(head_nf, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None):
        # x: [B, C, num_patches, d_model]
        x = self.flatten(x)      # [B, C, P*D]   — keeps channel dim
        x = self.linear1(x)      # [B, C, hidden] — same weights applied per channel
        x = self.relu(x)
        x = self.linear2(x)      # [B, C, forecast_horizon]
        return x.flatten(start_dim=1)  # [B, C * forecast_horizon]


class MLPDecoder(BaseModel):
    def __init__(self,device,cfg=None):
        self.device = device
        self.model = MLPHead(head_nf=cfg['input_dim'], forecast_horizon=cfg['output_dim'], hidden_dim=cfg.get('hidden_dim'))
        self.requires_model = False
        self.criterion = nn.MSELoss().to(self.device)

    def to_device(self):
        self.model.to(self.device)
        
    def to_cpu(self):
        self.model.to('cpu')

    def preprocess(self,batch_x):
        x=batch_x
        x = x.to(self.device).to(torch.float32)
        return x

    def forward(self,batch_x):
        x=self.preprocess(batch_x)
        output = self.model(x)
        return output

    def postprocess(self,x):
        pass

    def trainable_parameters(self):
        return self.model.parameters()