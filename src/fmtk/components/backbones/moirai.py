import os

# uni2ts pulls in `jax` as a transitive dependency (via jaxtyping's runtime
# type-checking of Float[Tensor, ...]-style annotations in uni2ts's source),
# even though nothing in this module actually runs JAX computation. Left
# alone, JAX pre-allocates ~75%+ of GPU memory the moment it touches CUDA on
# import, competing directly with PyTorch for the same GPU and causing OOM
# regardless of batch size or model size -- confirmed by tracing which
# modules `from uni2ts.model.moirai.module import MoiraiModule` pulls in
# (jaxlib.gpu_solver, jaxlib.gpu_rnn, jaxlib.gpu_prng, etc. all get imported
# as a side effect). This must be set before uni2ts/jax is first imported
# anywhere in the process, since JAX reads it once at import time.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from fmtk.components.base import BaseModel
import numpy as np
import torch
from tqdm import tqdm
from uni2ts.model.moirai.module import MoiraiModule, mask_fill, packed_attention_mask

# Only moirai-1.0-R-small has been loaded and its encoder output verified
# empirically so far. Add more entries here (and verify their d_model the
# same way) before using other sizes.
MOIRAI_MODEL_IDS = {
    "small": "Salesforce/moirai-1.0-R-small",
}
MOIRAI_EMBED_DIMS = {
    "small": 384,  # confirmed via MoiraiModule.from_pretrained(...).d_model and a live encoder forward pass
}


def get_moirai_embed_dim(model_name):
    return MOIRAI_EMBED_DIMS[model_name]


class MoiraiModel(BaseModel):
    """
    Moirai (Salesforce, "Unified Training of Universal Time Series
    Forecasting Transformers"): a masked-encoder transformer over patch
    tokens that natively supports multivariate series via an "any-variate"
    packed-attention scheme (variate_id/time_id/sample_id token tags,
    rather than fixed per-channel embeddings) -- unlike FMTK's ChronosModel,
    which embeds each channel of a [B, C, L] input independently with zero
    cross-channel information sharing.

    MoiraiModule.forward() expects pre-"packed" patch-tokenized inputs
    (target/observed_mask/sample_id/time_id/variate_id/prediction_mask/
    patch_size), not a plain [B, C, L] tensor. This wrapper builds that
    packing internally from a standard FMTK [B, C, L] batch using a single
    fixed patch_size (default 8, matching this repo's MOMENT-based
    convention elsewhere), treating each batch item as one unpacked
    "sample" (constant sample_id per row) spanning all C channels x
    L//patch_size patches -- confirmed correct empirically via a live
    forward pass and shape check before this was written.

    Returns raw encoder representations (the transformer's output before
    the distribution-parameter projection), analogous to the other FMTK
    time-series backbones' embeddings -- not Moirai's native forecasting
    output. Only classification/embedding use is supported here.

    Unlike Chronos's `.embed()` (which internally forces no-grad regardless
    of train/eval mode), gradients DO flow through this encoder path --
    confirmed empirically with a real .backward() call -- so LoRA/
    fine-tuning is architecturally possible here, though this wrapper is
    currently wired for frozen-backbone (baseline1-style) use only; no
    enable_peft/adapter methods are implemented yet.
    """

    def __init__(self, device, model_name="small", model_config=None):
        super().__init__()
        self.device = device
        model_config = model_config or {}
        self.patch_size = model_config.get("patch_size", 8)

        assert model_name in MOIRAI_MODEL_IDS, (
            f"Unknown Moirai model_name {model_name!r}; only {list(MOIRAI_MODEL_IDS)} are verified so far"
        )
        model_path = MOIRAI_MODEL_IDS[model_name]
        print(f"[Moirai] Loading {model_path} on device {device}")
        self.model = MoiraiModule.from_pretrained(model_path)
        self.model.to(device)

        assert self.patch_size in self.model.patch_sizes, (
            f"patch_size {self.patch_size} not in model's supported patch_sizes {self.model.patch_sizes}"
        )
        self.max_patch = max(self.model.patch_sizes)

    def preprocess(self, batch_x, mask=None):
        if mask is not None:
            mask = mask.to(self.device)
        x = batch_x.float().to(self.device)
        return x, mask

    def forward(self, batch_x, mask=None):
        x, mask = self.preprocess(batch_x, mask)
        B, C, L = x.shape
        assert L % self.patch_size == 0, f"seq_len {L} must be a multiple of patch_size {self.patch_size}"
        N = L // self.patch_size

        # [B, C, L] -> [B, C*N, patch_size], right-padded to the model's max patch size,
        # with per-token time/variate ids marking each patch's channel and position.
        target = x.reshape(B, C, N, self.patch_size).reshape(B, C * N, self.patch_size)
        target = torch.nn.functional.pad(target, (0, self.max_patch - self.patch_size))

        observed_mask = torch.zeros(B, C * N, self.max_patch, dtype=torch.bool, device=self.device)
        observed_mask[:, :, :self.patch_size] = True

        sample_id = torch.ones(B, C * N, dtype=torch.long, device=self.device)
        time_id = torch.arange(N, device=self.device).repeat(C).unsqueeze(0).expand(B, -1)
        variate_id = torch.arange(C, device=self.device).repeat_interleave(N).unsqueeze(0).expand(B, -1)
        prediction_mask = torch.zeros(B, C * N, dtype=torch.bool, device=self.device)
        patch_size_t = torch.full((B, C * N), self.patch_size, dtype=torch.long, device=self.device)

        loc, scale = self.model.scaler(
            target, observed_mask * ~prediction_mask.unsqueeze(-1), sample_id, variate_id
        )
        scaled_target = (target - loc) / scale
        reprs = self.model.in_proj(scaled_target, patch_size_t)
        masked_reprs = mask_fill(reprs, prediction_mask, self.model.mask_encoding.weight)
        encoded = self.model.encoder(
            masked_reprs, packed_attention_mask(sample_id), time_id=time_id, var_id=variate_id
        )
        return encoded  # [B, C*N, d_model]

    def predict(self, dataloader):
        self.model.eval()
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
