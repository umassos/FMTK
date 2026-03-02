"""
RF-DETR decoder for object detection in FMTK.

Receives list of tensors (feats) from backbone, runs projector + LWDETR
transformer + detection heads, outputs {"pred_logits", "pred_boxes"}.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.backbone import Joiner
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.transformer import build_transformer
from rfdetr.models.position_encoding import PositionEmbeddingSine
from rfdetr.util.misc import NestedTensor
from rfdetr.models import build_criterion_and_postprocessors


class FeatsNestedTensor(NestedTensor):
    """NestedTensor that carries pre-extracted feats for FeatsBackbone."""

    def __init__(
        self,
        feats: list[torch.Tensor],
        mask: torch.Tensor | None,
    ):
        self.feat_list = feats
        tensors = feats[0] if feats else torch.empty(0)
        if mask is None and feats:
            b, _, h, w = feats[0].shape
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=feats[0].device)
        super().__init__(tensors, mask)


class FeatsBackbone(nn.Module):
    """
    Backbone that takes FeatsNestedTensor (pre-extracted encoder feats)
    and runs projector + NestedTensor wrapping for Joiner.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        projector_scale: list[str],
        layer_norm: bool = True,
        rms_norm: bool = False,
    ):
        super().__init__()
        level2scalefactor = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}
        scale_factors = [level2scalefactor[lvl] for lvl in projector_scale]
        self.projector = MultiScaleProjector(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

    def forward(self, tensor_list: FeatsNestedTensor) -> list[NestedTensor]:
        feats = tensor_list.feat_list
        mask = tensor_list.mask
        projected = self.projector(feats)
        out: list[NestedTensor] = []
        for feat in projected:
            b, _, h, w = feat.shape
            if mask is not None:
                m = F.interpolate(
                    mask[None].float(), size=(h, w)
                ).to(torch.bool)[0]
            else:
                m = torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            out.append(NestedTensor(feat, m))
        return out


class RFDetrHead(nn.Module):
    """
    RF-DETR detection head: projector + LWDETR (transformer + class/bbox heads).

    Receives list of tensors (B, C, H, W) from backbone, outputs
    {"pred_logits", "pred_boxes"}.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        hidden_dim = cfg["hidden_dim"]
        num_classes = cfg["num_classes"] + 1  # +1 for no-object
        in_channels = cfg["in_channels"]
        projector_scale = cfg["projector_scale"]
        if isinstance(projector_scale, str):
            projector_scale = [projector_scale]

        feats_backbone = FeatsBackbone(
            in_channels=in_channels,
            out_channels=hidden_dim,
            projector_scale=projector_scale,
            layer_norm=cfg.get("layer_norm", True),
            rms_norm=cfg.get("rms_norm", False),
        )
        position_embedding = PositionEmbeddingSine(
            hidden_dim // 2,
            normalize=True,
        )
        backbone = Joiner(feats_backbone, position_embedding)

        class Args:
            pass

        args = Args()
        args.hidden_dim = hidden_dim
        args.sa_nheads = cfg.get("sa_nheads", 8)
        args.ca_nheads = cfg.get("ca_nheads", 16)
        args.num_queries = cfg.get("num_queries", 300)
        args.dropout = cfg.get("dropout", 0.0)
        args.dim_feedforward = cfg.get("dim_feedforward", 2048)
        args.dec_layers = cfg.get("dec_layers", 3)
        args.group_detr = cfg.get("group_detr", 1)
        args.two_stage = cfg.get("two_stage", False)
        args.dec_n_points = cfg.get("dec_n_points", 4)
        args.lite_refpoint_refine = cfg.get("lite_refpoint_refine", True)
        args.decoder_norm = cfg.get("decoder_norm", "LN")
        args.bbox_reparam = cfg.get("bbox_reparam", True)
        args.num_feature_levels = len(projector_scale)

        transformer = build_transformer(args)

        self.model = LWDETR(
            backbone=backbone,
            transformer=transformer,
            segmentation_head=None,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=cfg.get("aux_loss", True),
            group_detr=args.group_detr,
            two_stage=args.two_stage,
            lite_refpoint_refine=args.lite_refpoint_refine,
            bbox_reparam=args.bbox_reparam,
        )

    def forward(
        self,
        feats: list[torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        samples = FeatsNestedTensor(feats, mask)
        out = self.model(samples, targets=None)
        return {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}


class RFDetrDecoder(nn.Module):
    """
    RF-DETR decoder for FMTK pipeline.

    Receives (feats) - list of tensors from backbone - in forward(x).
    Outputs {"pred_logits", "pred_boxes"}. No predict method.
    """

    def __init__(self, device: str | torch.device, cfg: dict[str, Any]):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cfg = cfg
        embed_dim = cfg.get("embed_dim", 384)
        out_feature_indexes = cfg.get("out_feature_indexes", [3, 6, 9, 12])
        head_cfg = {
            "hidden_dim": cfg.get("hidden_dim", 256),
            "num_classes": cfg.get("num_classes", 80),
            "in_channels": [embed_dim] * len(out_feature_indexes),
            "projector_scale": cfg.get("projector_scale", cfg.get("projecter_scale", ["P4"])),
            "dec_layers": cfg.get("dec_layers", 3),
            "dec_n_points": cfg.get("dec_n_points", 4),
            "num_queries": cfg.get("num_queries", 300),
        }
        self.model = RFDetrHead(head_cfg)
        self.model.to(self.device)

        class Args:
            pass

        args = Args()
        args.num_classes = cfg.get("num_classes", 80)
        args.set_cost_class = cfg.get("set_cost_class", 2)
        args.set_cost_bbox = cfg.get("set_cost_bbox", 5)
        args.set_cost_giou = cfg.get("set_cost_giou", 2)
        args.focal_alpha = cfg.get("focal_alpha", 0.25)
        args.cls_loss_coef = cfg.get("cls_loss_coef", 2.0)
        args.bbox_loss_coef = cfg.get("bbox_loss_coef", 5.0)
        args.giou_loss_coef = cfg.get("giou_loss_coef", 2.0)
        args.aux_loss = cfg.get("aux_loss", True)
        args.group_detr = cfg.get("group_detr", 1)
        args.sum_group_losses = cfg.get("sum_group_losses", False)
        args.use_varifocal_loss = cfg.get("use_varifocal_loss", False)
        args.use_position_supervised_loss = cfg.get("use_position_supervised_loss", False)
        args.ia_bce_loss = cfg.get("ia_bce_loss", False)
        args.segmentation_head = False
        args.device = str(self.device)
        args.dec_layers = cfg.get("dec_layers", 3)
        args.num_select = cfg.get("num_select", 100)
        args.two_stage = cfg.get("two_stage", False)

        self.criterion, _ = build_criterion_and_postprocessors(args)
        self.criterion.to(self.device)

    def forward(self, x: list[torch.Tensor] | tuple) -> dict[str, torch.Tensor]:
        # Handle (feats, mask) 2-tuple vs plain list of feature tensors from backbone.
        # A mask tensor is at most 3-D (B, H, W); feature tensors are 4-D (B, C, H, W).
        mask = None
        if (
            isinstance(x, tuple)
            and len(x) == 2
            and isinstance(x[1], torch.Tensor)
            and x[1].ndim <= 3
        ):
            raw_feats = x[0]
            mask = x[1]
        else:
            raw_feats = x

        if isinstance(raw_feats, torch.Tensor):
            feats = [raw_feats]
        else:
            feats = list(raw_feats)

        feats = [f.to(self.device) for f in feats]
        if mask is not None:
            mask = mask.to(self.device)
        return self.model(feats, mask)

    def trainable_parameters(self):
        return self.model.parameters()

    def load_pretrained_rfdetr_weights(self, checkpoint_path: str) -> dict[str, int]:
        """Load RF-DETR checkpoint into decoder (projector + transformer + heads)."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        decoder_prefixes = (
            "backbone.0.projector.",
            "backbone.1.",
            "transformer.",
            "class_embed.",
            "bbox_embed.",
            "refpoint_embed.",
            "query_feat.",
        )
        decoder_state = {
            k: v
            for k, v in state.items()
            if any(k.startswith(p) for p in decoder_prefixes)
        }
        loaded = 0
        if decoder_state:
            load_result = self.model.load_state_dict(decoder_state, strict=False)
            # load_state_dict returns _IncompatibleKeys namedtuple; use attribute access
            loaded = len(decoder_state) - len(load_result.missing_keys)
        return {"loaded": loaded}
