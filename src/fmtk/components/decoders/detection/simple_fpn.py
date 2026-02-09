"""
Simple Feature Pyramid Network for plain ViT backbones.

Based on the ViTDet paper (Li et al., 2022): "Exploring Plain Vision Transformer
Backbones for Object Detection" (https://arxiv.org/abs/2203.16527).

Takes a single-scale feature map from a plain ViT backbone (stride 16) and builds
a multi-scale feature pyramid {1/4, 1/8, 1/16, 1/32} using parallel
deconvolutions, identity, and max pooling.
"""

import math
import torch
import torch.nn as nn
from collections import OrderedDict


class SimpleFeaturePyramid(nn.Module):
    """
    Simple Feature Pyramid that converts flat ViT patch tokens into
    multi-scale spatial feature maps suitable for detection heads.

    Input:  [B, N, D] patch tokens (no CLS token) from a ViT backbone
    Output: OrderedDict of {
        "p2": [B, out_channels, H/4,  W/4],   # 1/4  scale (stride  4)
        "p3": [B, out_channels, H/8,  W/8],   # 1/8  scale (stride  8)
        "p4": [B, out_channels, H/16, W/16],  # 1/16 scale (stride 16)
        "p5": [B, out_channels, H/32, W/32],  # 1/32 scale (stride 32)
    }

    Parameters
    ----------
    embed_dim : int
        Hidden dimension of the backbone (e.g. 768 for ViT-B).
    out_channels : int
        Number of output channels per pyramid level (default 256).
    """

    def __init__(self, embed_dim, out_channels=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        # --- Scale branches (applied to the 1/16 spatial feature map) ---

        # p2 (1/4 scale): two 2x2 deconvolutions with stride 2
        # First deconv followed by LayerNorm + GELU, then second deconv
        self.upsample_4x = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            _LayerNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        # p3 (1/8 scale): one 2x2 deconvolution with stride 2
        self.upsample_2x = nn.ConvTranspose2d(
            embed_dim, embed_dim, kernel_size=2, stride=2
        )

        # p4 (1/16 scale): identity -- use the ViT feature map as-is
        # (no module needed)

        # p5 (1/32 scale): stride-2 2x2 max pooling
        self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Per-level processing: 1x1 conv + LN, then 3x3 conv + LN ---
        self.level_proj = nn.ModuleDict()
        for name in ["p2", "p3", "p4", "p5"]:
            self.level_proj[name] = nn.Sequential(
                nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=False),
                _LayerNorm2d(out_channels),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                _LayerNorm2d(out_channels),
            )

    def forward(self, patch_tokens):
        """
        Parameters
        ----------
        patch_tokens : torch.Tensor
            Shape [B, N, D] where N = H_feat * W_feat (no CLS token).

        Returns
        -------
        OrderedDict[str, torch.Tensor]
            Multi-scale feature maps keyed by pyramid level name.
        """
        B, N, D = patch_tokens.shape

        # Reshape to spatial: [B, D, H_feat, W_feat]
        H_feat = W_feat = int(math.sqrt(N))
        assert (
            H_feat * W_feat == N
        ), f"Patch token count {N} is not a perfect square (got sqrt={math.sqrt(N)})"
        x = patch_tokens.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)

        # Build scale branches
        p2 = self.upsample_4x(x)   # [B, D, H_feat*4, W_feat*4]  = 1/4 of original
        p3 = self.upsample_2x(x)   # [B, D, H_feat*2, W_feat*2]  = 1/8 of original
        p4 = x                      # [B, D, H_feat,   W_feat]    = 1/16 of original
        p5 = self.downsample_2x(x)  # [B, D, H_feat/2, W_feat/2]  = 1/32 of original

        # Per-level projection to out_channels
        features = OrderedDict()
        features["p2"] = self.level_proj["p2"](p2)
        features["p3"] = self.level_proj["p3"](p3)
        features["p4"] = self.level_proj["p4"](p4)
        features["p5"] = self.level_proj["p5"](p5)

        return features


class _LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D feature maps (channels-first format).
    Normalizes over the channel dimension at each spatial position.
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
