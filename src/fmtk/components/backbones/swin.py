from fmtk.components.base import BaseModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from peft import get_peft_model, PeftModel
from functools import singledispatchmethod

MODEL_MAPPING = {
    "tiny": "microsoft/swin-tiny-patch4-window7-224",
    "small": "microsoft/swin-small-patch4-window7-224",
    "base": "microsoft/swin-base-patch4-window7-224",
    "large": "microsoft/swin-large-patch4-window7-224",
    "swin-tiny": "microsoft/swin-tiny-patch4-window7-224",
    "swin-small": "microsoft/swin-small-patch4-window7-224",
    "swin-base": "microsoft/swin-base-patch4-window7-224",
    "swin-large": "microsoft/swin-large-patch4-window7-224",
}

EMBED_DIMS = {
    "microsoft/swin-tiny-patch4-window7-224": 768,
    "microsoft/swin-small-patch4-window7-224": 768,
    "microsoft/swin-base-patch4-window7-224": 1024,
    "microsoft/swin-large-patch4-window7-224": 1536,
    "tiny": 768,
    "small": 768,
    "base": 1024,
    "large": 1536,
}


class SwinModel(BaseModel):
    """
    Swin Transformer backbone for vision tasks.
    Uses Hugging Face transformers (SwinModel with pooling).
    """

    def __init__(self, device, model_name="base", model_config=None):
        super().__init__()
        self.device = device
        self.return_all_tokens = model_config.get("return_all_tokens", False)

        if model_name in MODEL_MAPPING:
            model_id = MODEL_MAPPING[model_name]
        else:
            model_id = model_name    

        embed_dim = EMBED_DIMS[model_id]

        print(f"[Swin] Loading {model_id} on device {device}")

        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        self.model.to(device)
        self.embed_dim = embed_dim
        self.peft_enable = False

    def preprocess(self, batch_x, mask=None):
        batch_x = batch_x.float().to(self.device)
        self.B, self.C, self.H, self.W = batch_x.shape
        return batch_x, mask

    def forward(self, batch_x, mask=None, adapters=[]):
        x, mask = self.preprocess(batch_x, mask)

        if isinstance(self.model, PeftModel) and len(adapters) > 0:
            outputs = self.model(x, adapters=adapters)
        else:
            outputs = self.model(x)

        if self.return_all_tokens:
            embeddings = outputs.last_hidden_state
        else:
            embeddings = outputs.pooler_output
            if embeddings is None:
                last_hidden_state = outputs.last_hidden_state
                embeddings = last_hidden_state.mean(dim=1)

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
