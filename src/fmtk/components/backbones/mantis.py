from fmtk.components.base import BaseModel
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model


class MantisModel(BaseModel):
    def __init__(self, device, model_name="8M"):
        super().__init__()
        self.device = device
        self.peft_enable = False

        # initialization step
        self.network = Mantis8M(device="cpu")
        self.network = Mantis8M.from_pretrained(f"paris-noah/Mantis-{model_name}")

        # next step is wrapping it with trainer
        self.model = MantisTrainer(device="cpu", network=self.network)
        self.peft_enable = False

    # preprocess step almost the same as MOMENTs since both expect (batch, channels, length),
    # and they both need float inputs
    def preprocess(self, batch):

        # FMTK sends dict: {"x":..., "y":..., "mask":...}
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device)

        else:
            # fallback for tuple mode
            if len(batch) == 3:
                x, mask, y = batch
                mask = mask.to(self.device)
            else:
                x, y = batch
                mask = None

        x = x.float().to(self.device)
        self.B, self.S, self.L = x.shape

        # resizing via interpolation, as suggested
        if self.L != 512:
            x = torch.nn.functional.interpolate(
                x, size=512, mode="linear", align_corners=False
            )

        # Mantis expects 1 input channel average multiple channels if present
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)

        return x, mask, y

    def forward(self, batch_x, mask=None):
        # changed this to match Chronos args (just pass batch)
        batch = batch_x
        
        x, mask, y = self.preprocess(batch)

        # x_np = x.cpu().numpy()
        # emb_np = self.model.transform(x_np)
        # embedding = torch.tensor(emb_np, dtype=torch.float32, device=self.device)

        self.network.to(self.device)
        self.network.train()  # enable training mode if needed
        outputs = self.network(x)

        # handle possible tuple/dict returns
        if isinstance(outputs, (tuple, list)):
            embedding = outputs[0]
        elif isinstance(outputs, dict) and "embedding" in outputs:
            embedding = outputs["embedding"]
        else:
            embedding = outputs

        return embedding, y

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
