from fmtk.components.base import BaseModel
from chronos import Chronos2Pipeline
import torch
import numpy as np
from tqdm import tqdm


def get_chronos2_embed_dim():
    """
    T5-style encoder hidden size (d_model) for the unified Chronos-2 model,
    confirmed empirically via Chronos2Pipeline.embed() output shape (last
    dim). Unlike the original Chronos, Chronos-2 ships as a single unified
    checkpoint (no tiny/mini/small/base/large variants -- only
    "amazon/chronos-2" exists on the Hub), so there's no size to select.
    """
    return 768


class Chronos2Model(BaseModel):
    """
    Chronos-2 (Amazon): unlike the original Chronos (T5-based, univariate
    only -- FMTK's ChronosModel flattens [B, C, L] to [B*C, L] and embeds
    each channel independently, with zero information sharing across
    channels), Chronos-2 natively supports multivariate time series --
    Chronos2Pipeline.embed() accepts [B, C, L] directly and shares
    information across the C channels internally (confirmed empirically: a
    10-channel batch embeds in one call, with cross-channel attention, not
    10 independent single-channel calls).

    Caveat confirmed empirically: Chronos2Pipeline.embed() always runs under
    an internal no-grad context, regardless of the model's train/eval mode
    or an outer torch.enable_grad() -- so this backbone can only be used
    frozen (baseline1-style decoder-only training), the same limitation as
    FMTK's ChronosModel. No LoRA/fine-tuning path is available.
    """

    def __init__(self, device, model_name=None, model_config=None):
        super().__init__()
        self.device = device
        model_path = "amazon/chronos-2"  # single unified checkpoint, no size variants
        print(f"[Chronos-2] Loading {model_path} on device {device}")
        self.model = Chronos2Pipeline.from_pretrained(model_path, device_map=self.device)

    def preprocess(self, batch_x, mask=None):
        if mask is not None:
            mask = mask.to(self.device)
        x = batch_x.float().to(self.device)
        return x, mask

    def forward(self, batch_x, mask=None):
        x, mask = self.preprocess(batch_x)
        embeddings, _ = self.model.embed(x)  # list of [C, num_patches+2, d_model], one per batch item
        output = torch.stack(embeddings, dim=0)  # [B, C, num_patches+2, d_model]
        return output

    def predict(self, dataloader):
        """
        Compute embeddings for a single split using a DataLoader.

        Args:
            dataloader: PyTorch DataLoader yielding dicts with "x"/"y" (and optionally "mask").

        Returns:
            embeddings: [N, C, num_patches+2, d_model] NumPy array
            labels: [N] NumPy array of ground truth labels (if available)
        """
        self.model.model.eval()
        all_embeddings, all_labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader)):
            x, y = batch["x"], batch["y"]
            with torch.no_grad():
                output = self.forward(x)
            all_embeddings.append(output.cpu().detach().float().numpy())
            all_labels.append(y)
        embeddings_np = np.concatenate(all_embeddings, axis=0)
        labels_np = np.concatenate(all_labels)
        return embeddings_np, labels_np
