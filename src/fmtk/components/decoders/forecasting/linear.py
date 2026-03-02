import torch
import torch.nn as nn
from fmtk.components.base import BaseModel


class LinearDecoder(BaseModel):
    def __init__(self, device, cfg):
        self.device = device
        self.flatten = nn.Flatten(start_dim=-2)  # [B, C, P, D] -> [B, C, P*D]
        self.dropout = nn.Dropout(cfg.get("head_dropout", 0.1))
        self.model = nn.Linear(
            in_features=cfg["input_dim"], out_features=cfg["output_dim"]
        )
        self.criterion = nn.MSELoss()

    def to_device(self):
        self.flatten.to(self.device)
        self.dropout.to(self.device)
        self.model.to(self.device)
        self.criterion.to(self.device)

    def to_cpu(self):
        self.flatten.to("cpu")
        self.dropout.to("cpu")
        self.model.to("cpu")

    def trainable_parameters(self):
        return self.model.parameters()

    def forward(self, x):
        x = x.to(torch.float32).to(self.device)
        x = self.flatten(x)      # [B, C, P*D] — keeps channel dim
        x = self.dropout(x)
        x = self.model(x)        # [B, C, forecast_horizon] — same weights per channel
        return x.flatten(start_dim=1)  # [B, C * forecast_horizon]
