import torch.nn as nn


class LinearSemanticSegmenter(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.model = nn.Conv2d(
            in_channels=cfg["input_dim"],
            out_channels=cfg["output_dim"],
            kernel_size=(1, 1),
        )
        self.ignore_index = cfg.get("ignore_index", 255)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
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
        x = batch_x.to(self.device)
        if x.ndim == 3:
            x = x.reshape(-1, self.height, self.width, self.cfg["input_dim"])
            x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, batch_x):
        embeddings = self.preprocess(batch_x)
        logits = self.model(embeddings)
        logits = nn.functional.interpolate(
            logits,
            size=(self.pixel_height, self.pixel_width),
            mode="bilinear",
            align_corners=False,
        )
        return logits
