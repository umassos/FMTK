import torch
import torch.nn as nn
from fmtk.components.decoders.base import BaseVisionDecoder


class MonocularDepthDecoder(nn.Module, BaseVisionDecoder):
    """
    Pixel-wise monocular depth decoder inspired by LinearSemanticSegmenter.

    Expects patch token embeddings from DINOv2 with return_all_tokens=True:
        input:  [B, N, C]  where N = height * width (patch grid, no CLS token)
        output: [B, pixel_height, pixel_width]  (single-channel depth map)

    cfg keys:
        input_dim    : backbone embed dim (e.g. 384 for DINOv2-small)
        height       : patch grid rows   (e.g. 16 for a 224px image with 14px patches)
        width        : patch grid cols
        pixel_height : output image height to upsample to (e.g. 224)
        pixel_width  : output image width  to upsample to (e.g. 224)
    """

    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.mode = cfg.get('mode', None)
        self.model = nn.Conv2d(
            in_channels=cfg["input_dim"],
            out_channels=1,
            kernel_size=(1, 1),
        )

        self.criterion = nn.L1Loss()

        self.height = cfg["height"]
        self.width = cfg["width"]
        self.pixel_height = cfg["pixel_height"]
        self.pixel_width = cfg["pixel_width"]

    def to_device(self):
        self.model.to(self.device)
        self.criterion.to(self.device)

    def to_cpu(self):
        self.model.to("cpu")

    def trainable_parameters(self):
        return self.model.parameters()

    def preprocess(self, batch_x):
        x = batch_x.to(self.device).float()
        if self.mode is not None:
            x = self.select_embeddings(x)
        if x.ndim == 3:
            B, N, C = x.shape
            # [B, N, C] -> [B, C, H_patch, W_patch]
            x = x.permute(0, 2, 1).reshape(B, C, self.height, self.width)
        return x

    def forward(self, batch_x):
        x = self.preprocess(batch_x)           # [B, C, H_patch, W_patch]
        x = self.model(x)                      # [B, 1, H_patch, W_patch]
        x = nn.functional.relu(x)             # depth values are non-negative
        x = nn.functional.interpolate(
            x,
            size=(self.pixel_height, self.pixel_width),
            mode="bilinear",
            align_corners=False,
        )                                      # [B, 1, pixel_H, pixel_W]
        return x.squeeze(1)                    # [B, pixel_H, pixel_W]
