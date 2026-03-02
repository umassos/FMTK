from fmtk.components.base import BaseModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from peft import get_peft_model, PeftModel
from functools import singledispatchmethod

def get_dinov2_model_id(model_name):
    if model_name in ['small', 'dinov2-small', 'facebook/dinov2-small']:
        return 'facebook/dinov2-small'
    elif model_name in ['base', 'dinov2-base', 'facebook/dinov2-base']:
        return 'facebook/dinov2-base'
    elif model_name in ['large', 'dinov2-large', 'facebook/dinov2-large']:
        return 'facebook/dinov2-large'
    elif model_name in ['giant', 'dinov2-giant', 'facebook/dinov2-giant']:
        return 'facebook/dinov2-giant'
    elif model_name in ['Karan007/facebook-dinov2-base-finetuned-orchid']:
        return 'Karan007/facebook-dinov2-base-finetuned-orchid'

def get_dinov2_embed_dim(model_id):
    if model_id in ['small', 'dinov2-small', 'facebook/dinov2-small']:
        return 384
    elif model_id in ['base', 'dinov2-base', 'facebook/dinov2-base']:
        return 768
    elif model_id in ['large', 'dinov2-large', 'facebook/dinov2-large']:
        return 1024
    elif model_id in ['giant', 'dinov2-giant', 'facebook/dinov2-giant']:
        return 1536
    elif model_id in ['Karan007/facebook-dinov2-base-finetuned-orchid']:
        return 768

class DinoV2Model(BaseModel):
    """
    DINOv2 model for vision tasks.
    """

    def __init__(self, device, model_name="base", model_config={}):
        super().__init__()
        self.device = device
        self.return_all_tokens = model_config.get("return_all_tokens", False)
        # For RF-DETR compatibility: output multi-scale features from these layer indices
        # e.g. [2, 4, 5, 9] for base (12 layers). Returns list of (B, D, H, W) per layer.
        self.out_feature_indexes = model_config.get("out_feature_indexes", None)
        # Default to base model if not specified or not recognized
        self.model_id = get_dinov2_model_id(model_name)

        print(f"[DINO] Loading {self.model_id} on device {device}")

        self.model = AutoModel.from_pretrained(self.model_id)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

        self.model.to(device)
        self.embed_dim = get_dinov2_embed_dim(self.model_id)

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
        output_hidden_states = self.out_feature_indexes is not None
        if isinstance(self.model, PeftModel) and len(adapters) > 0:
            outputs = self.model(x, adapters=adapters, output_hidden_states=output_hidden_states)
        else:
            outputs = self.model(x, output_hidden_states=output_hidden_states)

        if self.out_feature_indexes is not None:
            # Multi-scale output for RF-DETR: list of (B, D, H, W) per layer
            hidden_states = outputs.hidden_states  # tuple of (B, N+1, D)
            B, N_plus_1, D = hidden_states[0].shape
            h = w = int((N_plus_1 - 1) ** 0.5)  # exclude CLS
            feats = []
            for idx in self.out_feature_indexes:
                layer_out = hidden_states[idx][:, 1:, :]  # drop CLS: (B, N, D)
                feats.append(layer_out.permute(0, 2, 1).reshape(B, D, h, w))
            return feats

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
