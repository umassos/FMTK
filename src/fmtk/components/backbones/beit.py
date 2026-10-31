from fmtk.components.base import BaseModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from peft import get_peft_model, PeftModel
from functools import singledispatchmethod


def get_beit_model_id(model_name):
    if model_name in ["base", "beitv2-base", "beitv2_base_patch16_224.in1k_ft_in22k_in1k"]:
        return "beitv2_base_patch16_224.in1k_ft_in22k_in1k"
    else:
        raise ValueError(f"Model name {model_name} not recognized")


def get_beit_embed_dim(model_name):
    if model_name in ["base", "beitv2-base", "beitv2_base_patch16_224.in1k_ft_in22k_in1k"]:
        return 768


class BEiTModel(BaseModel):
    """
    BEiT-v2 backbone for vision tasks.
    Uses timm; returns the token sequence [B, 1 + N, D] with the CLS token at index 0.
    """

    def __init__(self, device, model_name="base", model_config={}):
        super().__init__()

        self.device = device
        self.model_category = 'vision'

        self.model_id = get_beit_model_id(model_name)
        self.embed_dim = get_beit_embed_dim(self.model_id)

        print(f"[BEiT] Loading {self.model_id} on device {device}")

        self.model = timm.create_model(self.model_id, pretrained=True, num_classes=0, cache_dir=model_config.get("cache_dir"))
        self.model.to(device)

        self.peft_enable = False

    def preprocess(self, batch_x, mask=None):
        batch_x = batch_x.float().to(self.device)
        self.B, self.C, self.H, self.W = batch_x.shape
        return batch_x, mask

    def forward(self, batch_x, mask=None, adapters=[]):
        x, mask = self.preprocess(batch_x, mask)
        embeddings = self.model.forward_features(x)
        return embeddings

    @singledispatchmethod
    @torch.no_grad()
    def predict(self, data):
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
