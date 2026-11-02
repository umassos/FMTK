import torch.nn as nn
import torch.nn.functional as F


class _FPNNet(nn.Module):
    """
    FPN (Lin et al. 2017) fusion + segmentation head.

    Expects a list of feature maps [B, C_i, H_i, W_i] ordered
    finest-to-coarsest resolution (e.g. a hierarchical backbone's
    embedding/stage1/stage2/stage4 outputs). 1x1 lateral convs bring each
    level to a common width, a top-down pathway (2x upsample + add) fuses
    coarse semantic information into the finer levels, a 3x3 smoothing conv
    is applied per level, and the finest fused level feeds a 1x1 classifier.
    """

    def __init__(self, in_channels_list, output_dim, fpn_channels=256):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, kernel_size=1) for c in in_channels_list
        ])
        self.smooths = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        self.classifier = nn.Conv2d(fpn_channels, output_dim, kernel_size=1)

    def forward(self, feats):
        laterals = [lat(f) for lat, f in zip(self.laterals, feats)]

        fused = [None] * len(laterals)
        fused[-1] = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(fused[i + 1], size=laterals[i].shape[-2:], mode="nearest")
            fused[i] = laterals[i] + upsampled

        finest = self.smooths[0](fused[0])
        return self.classifier(finest)


class FPNSemanticSegmenter(nn.Module):
    """
    Semantic segmentation head over a hierarchical (multi-stage) backbone's
    feature pyramid, structured like LinearSemanticSegmenter but fusing all
    stages via FPN instead of using only the deepest stage's single grid.

    Expects the backbone's forward() to return a list of per-stage feature
    maps [B, C_i, H_i, W_i], finest-to-coarsest resolution (see
    fmtk.components.backbones.swin.SwinModel's return_hierarchical option).
    """

    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.model = _FPNNet(cfg["in_channels_list"], cfg["output_dim"], cfg.get("fpn_channels", 256))

        self.pixel_height = cfg["pixel_height"]
        self.pixel_width = cfg["pixel_width"]
        self.ignore_index = cfg.get("ignore_index", -100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

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
        logits = self.model(feats)
        logits = F.interpolate(logits, size=(self.pixel_height, self.pixel_width), mode="bilinear", align_corners=False)
        return logits
