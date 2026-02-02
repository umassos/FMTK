from fmtk.components.base import BaseModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from peft import get_peft_model, PeftModel

# import torch.nn as nn


MODEL_MAPPING = {
    "vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}

# EMBED_DIMS = {"giant": 1536, "large": 1024, "base": 768, "small": 384}


class DinoV3Model(BaseModel):
    """
    DINOv3 model for vision tasks.
    """

    def __init__(self, device, model_name="base", return_all_tokens=False):
        super().__init__()
        self.device = device
        self.return_all_tokens = return_all_tokens
        
        # Default to base model if not specified or not recognized
        if model_name not in MODEL_MAPPING:
            model_name = "base"
            print("Model name not recognized, using default facebook/dinov2-base")

        model_id = MODEL_MAPPING[model_name]
        # embed_dim = EMBED_DIMS[model_name]

        print(f"[DINO v3] Loading {model_id} on device {device}")

        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        # self.model.eval()
        self.model.to(device)
        # self.embed_dim = embed_dim

        self.peft_enable = False

    def preprocess(self, batch_x, mask=None):
        # Expect image tensors normalized and shaped [B, C, H, W].
        batch_x = batch_x.float()
        # print("batch_x in preprocess", batch_x.shape)
        self.B, self.C, self.H, self.W = batch_x.shape
        batch_x = batch_x.to(self.device)
        return batch_x, mask

    # TODO: Dataset/DataLoader should handle this
    # def preprocess_images(self, images):
    #     """
    #     Preprocess raw images using the Hugging Face image processor.
    #     Useful when you have raw PIL images or numpy arrays instead of tensors.
    #     """
    #     return self.processor(images, return_tensors="pt")["pixel_values"].to(
    #         self.device
    #     )

    def forward(self, batch_x, mask=None, adapters=[]):
        x, mask = self.preprocess(batch_x, mask)

        # The model returns a BaseModelOutputWithPooling object
        if isinstance(self.model, PeftModel) and len(adapters) > 0:
            outputs = self.model(x, adapters=adapters)
        else:
            outputs = self.model(x)

        if self.return_all_tokens:

            # Extract all tokens for a detr style decoder
            embeddings = outputs.last_hidden_state[:, 1:, :]
        else:
            # Extract the pooled output (CLS token representation)
            embeddings = outputs.pooler_output
            # If pooler_output is not available, use the last hidden state's CLS token
            if embeddings is None:
                last_hidden_state = outputs.last_hidden_state
                embeddings = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        return embeddings

    @torch.no_grad()
    def predict(self, batch_x, mask=None):
        self.model.eval()
        embeddings = self.forward(batch_x, mask)
        return embeddings

    def enable_peft(self, peft_cfg, load_path=None):
        if self.peft_enable:
            return

        self.peft_enable = True
        if load_path is None:
            self.model = get_peft_model(self.model, peft_cfg)
        else:
            self.model = PeftModel.from_pretrained(self.model, load_path)

        print(self.model.print_trainable_parameters())

    def adapter_trainable_parameters(self):
        if not self.peft_enable:
            return []
        return (p for p in self.model.parameters() if p.requires_grad)

    def save_adapter(self, path):
        if not self.peft_enable:
            return
        print(f"Saving adapter to {path}")
        self.model.save_pretrained(path)

    def set_adapter(self, adapter_name: str):
        assert self.peft_enable, "Backbone must be PEFT enabled for using adapters"
        self.model.set_adapter(adapter_name)
