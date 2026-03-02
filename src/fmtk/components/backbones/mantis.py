from fmtk.components.base import BaseModel
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model


class MantisModel(BaseModel):
    def __init__(self, device, model_name="8M", model_config=None):
        super().__init__()
        self.device = device
        self.peft_enable = False

        # initialization step
        self.network = Mantis8M(device=self.device)
        self.network = Mantis8M.from_pretrained(f"paris-noah/Mantis-{model_name}")
        self.network.to(self.device)

        # next step is wrapping it with trainer
        self.model = MantisTrainer(device=self.device, network=self.network)
        self.peft_enable = False

    # preprocess step almost the same as MOMENTs since both expect (batch, channels, length),
    # and they both need float inputs
    def preprocess(self, batch_x, mask=None):

        # FMTK sends dict: {"x":..., "y":..., "mask":...}
        if mask is not None:
            mask = mask.to(self.device)

        x = batch_x.float().to(self.device)
        self.B, self.S, self.L = x.shape

        # resizing via interpolation, as suggested
        if self.L != 512:
            x = torch.nn.functional.interpolate(
                x, size=512, mode="linear", align_corners=False
            )

        # Mantis expects 1 input channel average multiple channels if present
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)

        return x, mask

    def forward(self, batch_x, mask=None):
        x, mask = self.preprocess(batch_x)

        vit = self.network.vit_unit

        # Get patch embeddings from token generator: [B, num_patches, hidden_dim]
        x_embeddings = self.network.tokgen_unit(x)

        # Manually run ViT to capture transformer-processed patch tokens
        b, n, d = x_embeddings.shape
        cls_tokens = vit.cls_token.unsqueeze(0).unsqueeze(0).expand(b, 1, -1)  # [B, 1, d]
        x_with_cls = torch.cat([cls_tokens, x_embeddings], dim=1)              # [B, n+1, d]
        x_with_cls = vit.pos_encoder(x_with_cls.transpose(0, 1)).transpose(0, 1)
        x_with_cls = vit.transformer(x_with_cls)

        # Drop CLS token, return patch tokens: [B, num_patches, hidden_dim]
        return x_with_cls[:, 1:, :]

    @torch.no_grad()
    def predict(self, dataloader: DataLoader):
        all_embeddings, all_labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader), desc="[Mantis] Embedding"):
            emb, y = self.forward(batch)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(y)
        embeddings_np = np.vstack(all_embeddings)
        labels_np = np.concatenate(all_labels)
        return embeddings_np, labels_np 

    def enable_peft(self, peft_cfg):
        self.model = get_peft_model(self.model, peft_cfg)
        self.peft_enable = True

    def adapter_trainable_parameters(self):
        if not self.peft_enable:
            return []
        return (p for p in self.model.parameters() if p.requires_grad)
