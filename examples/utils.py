"""
Utility helpers for FMTK examples.

Provides `get_backbone` to automatically resolve a backbone class
from a human-readable model name (e.g. "dinov2", "swin") and a model id
(e.g. "facebook/dinov2-base", "swin-small").
"""

from fmtk.components.backbones.dinov2 import (
    DinoV2Model,
    get_dinov2_embed_dim,
    get_dinov2_model_id,
)
from fmtk.components.backbones.swin import SwinModel, EMBED_DIMS as SWIN_EMBED_DIMS
from fmtk.components.backbones.mae import MAEModel, EMBED_DIMS as MAE_EMBED_DIMS
from fmtk.components.backbones.vit import ViTModel, EMBED_DIMS as VIT_EMBED_DIMS
from fmtk.components.backbones.resnet import (
    ResNetVisionModel,
    get_resnet_vision_embed_dim,
    get_resnet_vision_model_id,
)


# ── registry ────────────────────────────────────────────────────────────
BACKBONE_REGISTRY = {
    "dinov2": DinoV2Model,
    "dino": DinoV2Model,
    "swin": SwinModel,
    "mae": MAEModel,
    "vit": ViTModel,
    "resnet": ResNetVisionModel,
}

MODEL_ID_REGISTRY = {
    "dinov2": get_dinov2_model_id,
    "dino": get_dinov2_model_id,
    "resnet": get_resnet_vision_model_id,
}

EMBED_DIMS_REGISTRY = {
    "dinov2": get_dinov2_embed_dim,
    "swin": SWIN_EMBED_DIMS,
    "mae": MAE_EMBED_DIMS,
    "vit": VIT_EMBED_DIMS,
    "resnet": get_resnet_vision_embed_dim,
}


def get_backbone(model_name, model_id, device, model_cfg={}):

    key = model_name.lower().strip()
    if key not in BACKBONE_REGISTRY:
        available = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model name '{model_name}'. " f"Available names: {available}"
        )

    BackboneClass = BACKBONE_REGISTRY[key]
    backbone = BackboneClass(device, model_id, model_cfg)
    return backbone


def get_embed_dim(model_name, model_id):
    key = model_name.lower().strip()
    if key not in EMBED_DIMS_REGISTRY:
        available = ", ".join(sorted(EMBED_DIMS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model name '{model_name}' for embed dim lookup. "
            f"Available names: {available}"
        )

    embed_dim_fn = EMBED_DIMS_REGISTRY[key]
    model_id_fn = MODEL_ID_REGISTRY.get(
        key, lambda x: x
    )  # Identity if no specific model_id_fn
    model_id = model_id_fn(model_id)
    embed_dim = embed_dim_fn(model_id)
    return embed_dim
