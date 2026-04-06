import torch.nn as nn
from fmtk.components.decoders.base import BaseVisionDecoder

class SpatialCountDecoder(nn.Module, BaseVisionDecoder):
    def __init__(self, device, cfg=None):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.mode = cfg.get('mode', None)
        # Standard regression head: features -> density map
        self.model = nn.Sequential(
            nn.Conv2d(
                self.cfg["input_dim"], self.cfg["hidden_dim"], kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(self.cfg["hidden_dim"], self.cfg["output_dim"], kernel_size=1),
            nn.ReLU(),  # Density cannot be negative
        )

        self.criterion = nn.MSELoss().to(self.device)

    def to_device(self):
        self.model.to(self.device)
        self.criterion.to(self.device)

    def to_cpu(self):
        self.model.to("cpu")
        self.criterion.to("cpu")

    def preprocess(self, embeddings):
        if self.mode is not None:
            embeddings = self.select_embeddings(embeddings)
        return embeddings

    def forward(self, embeddings):
        embeddings = self.preprocess(embeddings)
        # Assuming most common 224 x 224 input image size.

        B, N, C = embeddings.shape
        H, W = 224, 224
        h_feat, w_feat = H // 14, W // 14

        # Reshape [B, N, C] -> [B, C, h_feat, w_feat]
        x = embeddings.transpose(1, 2).reshape(B, C, h_feat, w_feat)

        # Generate the density map
        density_map = self.model(x)

        # Global Sum: The total count is the sum of the density map pixels
        count = density_map.sum(dim=(1, 2, 3))

        # TODO: Return density map which is a better metric for evaluation
        ## This would require differnt collate function for the dataloader
        return count

    def trainable_parameters(self):
        return self.model.parameters()
