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

def get_dinov2_embed_dim(model_id):
    if model_id in ['small', 'dinov2-small', 'facebook/dinov2-small']:
        return 384
    elif model_id in ['base', 'dinov2-base', 'facebook/dinov2-base']:
        return 768
    elif model_id in ['large', 'dinov2-large', 'facebook/dinov2-large']:
        return 1024
    elif model_id in ['giant', 'dinov2-giant', 'facebook/dinov2-giant']:
        return 1536

class DinoV2Model(BaseModel):
    """
    DINOv2 model for vision tasks.
    """

    def __init__(self, device, model_name="base", model_config={}):
        super().__init__()
        self.device = device
        self.model_category = 'vision'
        self.return_all_tokens = model_config.get("return_all_tokens", False)
        # hidden_states indices to return (0 = embeddings, -1 = last block); None = last_hidden_state
        self.layers = model_config.get("layers", None)
        # Default to base model if not specified or not recognized
        self.model_id = get_dinov2_model_id(model_name)

        print(f"[DINO] Loading {self.model_id} on device {device}")

        self.model = AutoModel.from_pretrained(self.model_id, cache_dir=model_config.get("cache_dir"))
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, cache_dir=model_config.get("cache_dir"))

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
        # if isinstance(self.model, PeftModel) and len(adapters) > 0:
        #     outputs = self.model(x, adapters=adapters)
        # else:
        #     outputs = self.model(x)

        # if self.return_all_tokens:

        #     # Extract all tokens for a detr style decoder
        #     embeddings = outputs.last_hidden_state[:, 1:, :]
        # else:
        #     # Extract the pooled output (CLS token representation)
        #     embeddings = outputs.pooler_output
        #     # If pooler_output is not available, use the last hidden state's CLS token
        #     if embeddings is None:
        #         last_hidden_state = outputs.last_hidden_state
        #         embeddings = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        if self.layers is None:
            outputs = self.model(x)
            embeddings = outputs.last_hidden_state
        else:
            outputs = self.model(x, output_hidden_states=True)
            embeddings = self.select_hidden_states(outputs, self.layers, norm=self.model.layernorm)
        return embeddings

    def select_hidden_states(self, outputs, layers, norm=None):
        hs = [outputs.hidden_states[i] for i in layers]
        if norm is not None:
            hs = [norm(h) for h in hs]
        return torch.stack(hs, dim=1)  # [B, len(layers), T, D]

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
