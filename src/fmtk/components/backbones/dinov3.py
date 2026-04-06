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
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "convnext-small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "convnext-base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "convnext-tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "convnext-large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}
EMBED_DIMS = {
    "vit7b16": 4096,
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": 4096,
    "vits16": 384,
    "facebook/dinov3-vits16-pretrain-lvd1689m": 384,
    "convnext-small": 768,
    "facebook/dinov3-convnext-small-pretrain-lvd1689m": 768,
    "vitb16": 768,
    "facebook/dinov3-vitb16-pretrain-lvd1689m": 768,
    "convnext-base": 1024,
    "facebook/dinov3-convnext-base-pretrain-lvd1689m": 1024,
    "vits16plus": 768,
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": 768,
    "convnext-tiny": 768,
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m": 768,
    "vitl16": 1024,
    "facebook/dinov3-vitl16-pretrain-lvd1689m": 1024,
    "vith16plus": 1536,
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": 1536,
    "convnext-large": 1536,
    "facebook/dinov3-convnext-large-pretrain-lvd1689m": 1536,
}

def get_dinov3_model_id(model_name):
    if model_name in MODEL_MAPPING.values():
        return model_name
    elif model_name in MODEL_MAPPING.keys():
        return MODEL_MAPPING[model_name]
    else:
        raise ValueError(f"Model name {model_name} not recognized")
    
def get_dinov3_embed_dim(model_name):
    model_id = get_dinov3_model_id(model_name)
    return EMBED_DIMS[model_id]

class DinoV3Model(BaseModel):
    """
    DINOv3 model for vision tasks.
    """

    def __init__(self, device, model_name="vitb16", return_all_tokens=False):
        super().__init__()
        self.device = device
        self.model_category = 'vision'
        self.return_all_tokens = return_all_tokens
        
        # Default to base model if not specified or not recognized
        self.model_id = get_dinov3_model_id(model_name)
        self.embed_dim = get_dinov3_embed_dim(self.model_id)

        self.is_convnext = "convnext" in self.model_id.lower()

        print(f"[DINO v3] Loading {self.model_id} on device {device}")

        self.model = AutoModel.from_pretrained(self.model_id)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

        self.model.to(device)
        self.peft_enable = False

    def preprocess(self, batch_x, mask=None):
        # Expect image tensors normalized and shaped [B, C, H, W].
        batch_x = batch_x.float()
        # print("batch_x in preprocess", batch_x.shape)
        self.B, self.C, self.H, self.W = batch_x.shape
        batch_x = batch_x.to(self.device)
        return batch_x, mask

    def forward(self, batch_x, mask=None, adapters=[]):
        x, mask = self.preprocess(batch_x, mask)

        # The model returns a BaseModelOutputWithPooling object
        # if isinstance(self.model, PeftModel) and len(adapters) > 0:
        #     outputs = self.model(x, adapters=adapters)
        # else:
        #     outputs = self.model(x)
        #
        # if self.return_all_tokens:
        #     # Strip CLS token (index 0) and any register tokens
        #     num_register = getattr(self.model.config, "num_register_tokens", 0)
        #     embeddings = outputs.last_hidden_state[:, 1 + num_register:, :]
        # else:
        #     # Extract the pooled output (CLS token representation)
        #     embeddings = outputs.pooler_output
        #     # If pooler_output is not available, use the last hidden state's CLS token
        #     if embeddings is None:
        #         last_hidden_state = outputs.last_hidden_state
        #         embeddings = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        #
        # return embeddings

        outputs = self.model(x)
        embeddings = outputs.last_hidden_state
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
