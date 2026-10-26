from collections import OrderedDict
import torch.nn as nn
from fmtk.components.decoders.segmentation.LinearSemanticSegmenter import LinearSemanticSegmenter

class BNLinearSemanticSegmenter(LinearSemanticSegmenter):
    # BatchNorm + 1x1 conv, same layout as DINOv2's BNHead (bn.*, conv_seg.*)
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        self.model = nn.Sequential(OrderedDict(
            bn=nn.BatchNorm2d(cfg["input_dim"]),
            conv_seg=nn.Conv2d(cfg["input_dim"], cfg["output_dim"], kernel_size=1),
        ))
