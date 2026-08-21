import torch.nn as nn
import torch.nn.functional as F


class _FPNNet(nn.Module):
    """
    FPN (Lin et al. 2017) fusion + single-channel depth head. Same fusion
    structure as segmentation/FPNSemanticSegmenter's _FPNNet, but the final
    1x1 conv outputs a single depth channel instead of class logits.
    """

    def __init__(self, in_channels_list, fpn_channels=256):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, kernel_size=1) for c in in_channels_list
        ])
        self.smooths = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        self.depth_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)

    def forward(self, feats):
        # ReLU after the lateral projections and the smoothing conv: without
        # these, the lateral->fuse->smooth stack is a chain of purely linear
        # ops with no nonlinearity until the terminal ReLU applied outside
        # this module -- if that stack's pre-activation output drifts
        # negative everywhere early in training, the terminal ReLU (and its
        # gradient) goes to exactly zero everywhere and training permanently
        # dies. These intermediate activations keep the network from
        # collapsing into that single point of failure.
        laterals = [F.relu(lat(f)) for lat, f in zip(self.laterals, feats)]

        fused = [None] * len(laterals)
        fused[-1] = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(fused[i + 1], size=laterals[i].shape[-2:], mode="nearest")
            fused[i] = laterals[i] + upsampled

        finest = F.relu(self.smooths[0](fused[0]))
        return self.depth_head(finest)


class FPNMonocularDepthDecoder(nn.Module):
    """
    Pixel-wise monocular depth decoder over a hierarchical (multi-stage)
    backbone's feature pyramid, structured like MonocularDepthDecoder but
    fusing all stages via FPN instead of using only the deepest stage's
    single grid.

    Expects the backbone's forward() to return a list of per-stage feature
    maps [B, C_i, H_i, W_i], finest-to-coarsest resolution (see
    fmtk.components.backbones.swin.SwinModel's return_hierarchical option).
        output: [B, pixel_height, pixel_width]  (single-channel depth map)
    """

    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.model = _FPNNet(cfg["in_channels_list"], cfg.get("fpn_channels", 256))

        self.pixel_height = cfg["pixel_height"]
        self.pixel_width = cfg["pixel_width"]
        self.criterion = nn.L1Loss()

    def to_device(self):
        self.model.to(self.device)
        self.criterion.to(self.device)

    def to_cpu(self):
        self.model.to("cpu")

    def trainable_parameters(self):
        return self.model.parameters()

    def preprocess(self, batch_x):
        return [f.to(self.device) for f in batch_x]

    def forward(self, batch_x):
        feats = self.preprocess(batch_x)
        depth = self.model(feats)
        depth = F.relu(depth)
        depth = F.interpolate(depth, size=(self.pixel_height, self.pixel_width), mode="bilinear", align_corners=False)
        return depth.squeeze(1)
