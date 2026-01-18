from fmtk.components.base import BaseModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from peft import get_peft_model, PeftModel
from functools import singledispatchmethod

# import torch.nn as nn


MODEL_MAPPING = {
    "giant": "facebook/dinov2-giant",
    "large": "facebook/dinov2-large",
    "base": "facebook/dinov2-base",
    "small": "facebook/dinov2-small",
}

EMBED_DIMS = {"giant": 1536, "large": 1024, "base": 768, "small": 384}


class DinoV2Model(BaseModel):
    """
    DINOv2 model for vision tasks.
    """

    def __init__(self, device, model_name="base", model_config=None):
        super().__init__()
        self.device = device
        self.return_all_tokens = model_config.get("return_all_tokens", False)
        # Default to base model if not specified or not recognized
        if model_name not in MODEL_MAPPING:
            model_name = "base"
            print("Model name not recognized, using default facebook/dinov2-base")

        model_id = MODEL_MAPPING[model_name]
        embed_dim = EMBED_DIMS[model_name]

        print(f"[DINO] Loading {model_id} on device {device}")

        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        # self.model.eval()
        self.model.to(device)
        self.embed_dim = embed_dim

        self.peft_enable = False

    def preprocess(self, batch_x, mask=None):
        # Expect image tensors normalized and shaped [B, C, H, W].
        batch_x = batch_x.float()
        
        self.B, self.C, self.H, self.W = batch_x.shape
        batch_x = batch_x.to(self.device)
        return batch_x, mask

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

    @singledispatchmethod
    @torch.no_grad()
    def predict(self, data):
        # If data is of the form batch_x
        self.model.eval()
        embeddings = self.forward(data)
        return embeddings

    @predict.register
    @torch.no_grad()
    def _predict_from_dataloader(self, data: DataLoader):
        self.model.eval()
        all_embeddings, all_labels = [], []

        for batch in tqdm(data, total=len(data)):
            if isinstance(batch, dict):
                x = batch["x"]
                y = batch.get("y", None)
                mask = batch.get("mask", None)
            else:
                # Handle tuple format (x, y) or (x, mask, y)
                if len(batch) == 2:
                    x, y = batch
                    mask = None
                elif len(batch) == 3:
                    x, mask, y = batch
                else:
                    x = batch[0]
                    y = None
                    mask = None

            embeddings = self.forward(x, mask)
            all_embeddings.append(embeddings.cpu().detach().float().numpy())
            if y is not None:
                if isinstance(y, torch.Tensor):
                    all_labels.append(y.cpu().numpy())
                else:
                    all_labels.append(np.array(y))

        embeddings_np = np.vstack(all_embeddings)
        if all_labels:
            labels_np = np.concatenate(all_labels)
        else:
            labels_np = None

        return embeddings_np, labels_np

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
