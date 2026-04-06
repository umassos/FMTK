from fmtk.components.base import BaseModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from peft import get_peft_model, PeftModel
from functools import singledispatchmethod


def get_mae_model_id(model_name):
    if model_name in ["base", "mae-base", "facebook/vit-mae-base"]:
        return "facebook/vit-mae-base"
    elif model_name in ["large", "mae-large", "facebook/vit-mae-large"]:
        return "facebook/vit-mae-large"
    elif model_name in ["huge", "mae-huge", "facebook/vit-mae-huge"]:
        return "facebook/vit-mae-huge"


def get_mae_embed_dim(model_id):
    if model_id in ["base", "mae-base", "facebook/vit-mae-base"]:
        return 768
    elif model_id in ["large", "mae-large", "facebook/vit-mae-large"]:
        return 1024
    elif model_id in ["huge", "mae-huge", "facebook/vit-mae-huge"]:
        return 1280


class MAEModel(BaseModel):
    """
    ViT-MAE (Masked Autoencoder) backbone for vision tasks.
    Uses Hugging Face transformers ViTMAEModel; returns encoder [CLS] or all patch tokens.
    """

    def __init__(self, device, model_name="base", model_config={}):
        super().__init__()
        model_config = model_config or {}
        self.device = device
        self.model_category = 'vision'
        self.return_all_tokens = model_config.get("return_all_tokens", False)

        self.model_id = get_mae_model_id(model_name)
        self.embed_dim = get_mae_embed_dim(self.model_id)
        print(f"[MAE] Loading {self.model_id} on device {device}")

        self.model = AutoModel.from_pretrained(self.model_id)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

        self.model.to(device)
        self.peft_enable = False

    def preprocess(self, batch_x, mask=None):
        batch_x = batch_x.float().to(self.device)
        self.B, self.C, self.H, self.W = batch_x.shape
        return batch_x, mask

    def forward(self, batch_x, mask=None, adapters=[]):
        x, mask = self.preprocess(batch_x, mask)

        # if isinstance(self.model, PeftModel) and len(adapters) > 0:
        #     outputs = self.model(x, adapters=adapters)
        # else:
        #     outputs = self.model(x)
        #
        # last_hidden_state = outputs.last_hidden_state
        #
        # if self.return_all_tokens:
        #     embeddings = last_hidden_state[:, 1:, :]
        # else:
        #     embeddings = last_hidden_state[:, 0, :]
        #
        # return embeddings

        outputs = self.model(x)
        embeddings = outputs.last_hidden_state
        return embeddings

    # TODO: This should be moved to the abstract class
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
