from fmtk.components.base import BaseModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from peft import get_peft_model, PeftModel
from functools import singledispatchmethod

def get_swin_model_id(model_name):
    if model_name in ['tiny', 'swin-tiny', 'microsoft/swin-tiny-patch4-window7-224']:
        return 'microsoft/swin-tiny-patch4-window7-224'
    elif model_name in ['small', 'swin-small', 'microsoft/swin-small-patch4-window7-224']:
        return 'microsoft/swin-small-patch4-window7-224'
    elif model_name in ['base', 'swin-base', 'microsoft/swin-base-patch4-window7-224']:
        return 'microsoft/swin-base-patch4-window7-224'
    elif model_name in ['large', 'swin-large', 'microsoft/swin-large-patch4-window7-224']:
        return 'microsoft/swin-large-patch4-window7-224'

def get_swin_embed_dim(model_id):

    if model_id in ['tiny', 'swin-tiny', 'microsoft/swin-tiny-patch4-window7-224']:
        return 768
    elif model_id in ['small', 'swin-small', 'microsoft/swin-small-patch4-window7-224']:
        return 768
    elif model_id in ['base', 'swin-base', 'microsoft/swin-base-patch4-window7-224']:
        return 1024
    elif model_id in ['large', 'swin-large', 'microsoft/swin-large-patch4-window7-224']:
        return 1536

def get_swin_hierarchical_dims(model_id):
    """
    Per-stage channel counts for return_hierarchical=True's 4 feature maps
    (embedding + stages 1/2/4, at strides 4/8/16/32 -- see SwinModel.forward
    for why stage 3's entry is skipped). Derived from each variant's
    embed_dim C: stages are [C, 2C, 4C, 8C] (doubling at each patch-merge).
    Confirmed empirically for small: [96, 192, 384, 768].
    """
    if model_id in ['tiny', 'swin-tiny', 'microsoft/swin-tiny-patch4-window7-224',
                     'small', 'swin-small', 'microsoft/swin-small-patch4-window7-224']:
        return [96, 192, 384, 768]
    elif model_id in ['base', 'swin-base', 'microsoft/swin-base-patch4-window7-224']:
        return [128, 256, 512, 1024]
    elif model_id in ['large', 'swin-large', 'microsoft/swin-large-patch4-window7-224']:
        return [192, 384, 768, 1536]

class SwinModel(BaseModel):
    """
    Swin Transformer backbone for vision tasks.
    Uses Hugging Face transformers (SwinModel with pooling).
    """

    def __init__(self, device, model_name="base", model_config={}):
        super().__init__()
        self.device = device
        self.return_all_tokens = model_config.get("return_all_tokens", False)
        self.return_hierarchical = model_config.get("return_hierarchical", False)

        self.model_id = get_swin_model_id(model_name)
        self.embed_dim = get_swin_embed_dim(self.model_id)

        print(f"[Swin] Loading {self.model_id} on device {device}")

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

        if isinstance(self.model, PeftModel) and len(adapters) > 0:
            outputs = self.model(x, adapters=adapters, output_hidden_states=self.return_hierarchical)
        else:
            outputs = self.model(x, output_hidden_states=self.return_hierarchical)

        if self.return_hierarchical:
            # reshaped_hidden_states = (embedding, stage1, stage2, stage3, stage4),
            # at strides (4, 8, 16, 32, 32) -- stage4 has no downsample, so its
            # entry duplicates stage3's resolution. Use the 4 *distinct*
            # resolutions (embedding + stages 1/2/4, strides 4/8/16/32),
            # finest-to-coarsest, skipping the redundant stage3 entry.
            reshaped = outputs.reshaped_hidden_states
            return [reshaped[0], reshaped[1], reshaped[2], reshaped[4]]

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
