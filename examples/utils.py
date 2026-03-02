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
from fmtk.components.backbones.swin import (
    SwinModel,
    get_swin_embed_dim,
    get_swin_model_id,
)
from fmtk.components.backbones.mae import (
    MAEModel,
    get_mae_embed_dim,
    get_mae_model_id,
)
from fmtk.components.backbones.resnet import (
    ResNetVisionModel,
    get_resnet_vision_embed_dim,
    get_resnet_vision_model_id,
)
# from fmtk.components.backbones.vgg import (
#     VGGModel,
#     get_vgg_embed_dim,
#     get_vgg_model_id,
# )
from fmtk.components.backbones.dinov3 import DinoV3Model, get_dinov3_embed_dim, get_dinov3_model_id
from fmtk.components.backbones.chronos import ChronosModel
from fmtk.components.backbones.moment import MomentModel
from fmtk.components.backbones.mantis import MantisModel

def get_chronos_embed_dim(model_id):
    if model_id=='large':
        return 1024
    elif model_id=='base':
        return 768
    elif model_id=='small':
        return 512
    else:
        return 1024

def get_moment_embed_dim(model_id):
    if model_id=='large':
        return 1024
    elif model_id=='base':
        return 768
    elif model_id=='small':
        return 512
    else:
        return 1024

def get_mantis_embed_dim(model_id):
    if model_id=='8M':
        return 256
    else:
        return 1024

# ── registry ────────────────────────────────────────────────────────────
BACKBONE_REGISTRY = {
    "dinov2": DinoV2Model,
    "dino": DinoV2Model,
    "dinov3": DinoV3Model,
    "swin": SwinModel,
    "mae": MAEModel,
    "resnet": ResNetVisionModel,
    # "vgg": VGGModel,
    "moment": MomentModel,
    "chronos": ChronosModel,
    "mantis": MantisModel,
}

MODEL_ID_REGISTRY = {
    "dinov2": get_dinov2_model_id,
    "dino": get_dinov2_model_id,
    "resnet": get_resnet_vision_model_id,
    "swin": get_swin_model_id,
    "mae": get_mae_model_id,
    # "vgg": get_vgg_model_id,
    "dinov3": get_dinov3_model_id,
    "moment": lambda x: x,
    "chronos": lambda x: x,
    "mantis": lambda x: x,
}

EMBED_DIMS_REGISTRY = {
    "dinov2": get_dinov2_embed_dim,
    "swin": get_swin_embed_dim,
    "mae": get_mae_embed_dim,
    "resnet": get_resnet_vision_embed_dim,
    # "vgg": get_vgg_embed_dim,
    "dinov3": get_dinov3_embed_dim,
    "moment": get_moment_embed_dim,
    "chronos": get_chronos_embed_dim,
    "mantis": get_mantis_embed_dim,
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


