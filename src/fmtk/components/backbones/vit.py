from functools import singledispatchmethod

import numpy as np
import torch
from torch.utils.data import DataLoader
from peft import PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

from fmtk.components.base import BaseModel

# Load model directly

# processor = AutoImageProcessor.from_pretrained("Andyrasika/vit-base-patch16-224-in21k-finetuned-lora-food101")
# model = AutoModelForImageClassification.from_pretrained("Andyrasika/vit-base-patch16-224-in21k-finetuned-lora-food101")


MODEL_MAPPING = {
    "base": "google/vit-base-patch16-224-in21k",
    "large": "google/vit-large-patch16-224-in21k",
    "huge": "google/vit-huge-patch14-224-in21k",
    "ft-lora-food101": "Andyrasika/vit-base-patch16-224-in21k-finetuned-lora-food101"
}

EMBED_DIMS = {
    "base": 768,
    "large": 1024,
    "huge": 1280,
    "ft-lora-food101": 768,
}

class ViTModel(BaseModel):
    """
    ViT model for vision tasks.
    Supports base/large/huge variants from the Hugging Face `google/vit-*` family.
    """

    def __init__(self, device, model_name="base", model_config=None):
        super().__init__()
        self.device = device
        self.return_all_tokens = model_config.get("return_all_tokens", False)

        if model_name not in MODEL_MAPPING:
            print(f"[ViT] Model name '{model_name}' not recognized, defaulting to 'base'")
            model_name = "base"
        model_id = MODEL_MAPPING[model_name]
        embed_dim = EMBED_DIMS[model_name]

        print(f"[ViT] Loading {model_id} on device {device}")

        if model_name == "ft-lora-food101":
            self.model = AutoModelForImageClassification.from_pretrained(model_id, num_labels=101)
        else: 
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


        if hasattr(self.model, 'vit'):
            # Use the base ViT model to get embeddings
            base_model = self.model.vit
            if isinstance(self.model, PeftModel) and len(adapters) > 0:
                outputs = base_model(x, adapters=adapters)
            else:
                outputs = base_model(x)
        else:
            # Regular AutoModel
            if isinstance(self.model, PeftModel) and len(adapters) > 0:
                outputs = self.model(x, adapters=adapters)
            else:
                outputs = self.model(x)


        if self.return_all_tokens:

            # Extract all tokens
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
            all_embeddings.append(embeddings.cpu().detach().float())
            if y is not None:
                if isinstance(y, torch.Tensor):
                    all_labels.append(y.cpu())
                else:
                    all_labels.append(y)

        # Use torch.cat instead of np.vstack (faster, no copy)
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        # Single numpy conversion at the end
        embeddings_np = embeddings_tensor.float().numpy()
        
        if all_labels:
            labels_tensor = torch.cat(all_labels, dim=0)
            labels_np = labels_tensor.numpy()
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
