"""
Faster R-CNN decoder for plain ViT backbones using Simple Feature Pyramid.

Wraps the SimpleFeaturePyramid (from simple_fpn.py) with torchvision's
RPN and RoI box heads to perform object detection (bounding boxes only).

Based on the ViTDet paper (Li et al., 2022): "Exploring Plain Vision Transformer
Backbones for Object Detection" (https://arxiv.org/abs/2203.16527).
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm

from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RPNHead,
    RegionProposalNetwork,
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from fmtk.components.decoders.detection.simple_fpn import SimpleFeaturePyramid


class FasterRCNNDecoder(nn.Module):
    """
    Faster R-CNN detection decoder for plain ViT backbones.

    Takes patch tokens [B, N, D] from a ViT backbone, builds a simple feature
    pyramid, then runs torchvision's RPN + RoI box heads.

    Parameters
    ----------
    device : str
        Target device (e.g. "cuda:0").
    cfg : dict
        Configuration with keys:
        - embed_dim (int): backbone hidden dim (e.g. 768 for ViT-B)
        - num_classes (int): number of object classes (NOT including background;
          background is added internally as torchvision convention)
        - out_channels (int, optional): FPN output channels (default 256)
        - image_size (int, optional): expected input image size (default 1024)

        RPN config (optional):
        - rpn_pre_nms_top_n_train (int, default 2000)
        - rpn_pre_nms_top_n_test (int, default 1000)
        - rpn_post_nms_top_n_train (int, default 2000)
        - rpn_post_nms_top_n_test (int, default 1000)
        - rpn_nms_thresh (float, default 0.7)
        - rpn_fg_iou_thresh (float, default 0.7)
        - rpn_bg_iou_thresh (float, default 0.3)
        - rpn_batch_size_per_image (int, default 256)
        - rpn_positive_fraction (float, default 0.5)

        RoI config (optional):
        - box_roi_pool_output_size (int, default 7)
        - box_roi_pool_sampling_ratio (int, default 2)
        - box_head_fc_dim (int, default 1024)
        - box_score_thresh (float, default 0.05)
        - box_nms_thresh (float, default 0.5)
        - box_detections_per_img (int, default 100)
        - box_fg_iou_thresh (float, default 0.5)
        - box_bg_iou_thresh (float, default 0.5)
        - box_batch_size_per_image (int, default 512)
        - box_positive_fraction (float, default 0.25)
    """

    def __init__(self, device, cfg):
        super().__init__()
        self.device = device

        embed_dim = cfg["embed_dim"]
        num_classes = cfg["num_classes"] + 1  # +1 for background (torchvision convention)
        out_channels = cfg.get("out_channels", 256)
        self.image_size = cfg.get("image_size", 1024)

        # --- Simple Feature Pyramid ---
        self.fpn = SimpleFeaturePyramid(embed_dim=embed_dim, out_channels=out_channels)

        # --- Region Proposal Network ---
        anchor_sizes = ((32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios
        )

        rpn_head = RPNHead(
            in_channels=out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )

        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=cfg.get("rpn_fg_iou_thresh", 0.7),
            bg_iou_thresh=cfg.get("rpn_bg_iou_thresh", 0.3),
            batch_size_per_image=cfg.get("rpn_batch_size_per_image", 256),
            positive_fraction=cfg.get("rpn_positive_fraction", 0.5),
            pre_nms_top_n=dict(
                training=cfg.get("rpn_pre_nms_top_n_train", 2000),
                testing=cfg.get("rpn_pre_nms_top_n_test", 1000),
            ),
            post_nms_top_n=dict(
                training=cfg.get("rpn_post_nms_top_n_train", 2000),
                testing=cfg.get("rpn_post_nms_top_n_test", 1000),
            ),
            nms_thresh=cfg.get("rpn_nms_thresh", 0.7),
        )

        # --- RoI Box Head ---
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5"],
            output_size=cfg.get("box_roi_pool_output_size", 7),
            sampling_ratio=cfg.get("box_roi_pool_sampling_ratio", 2),
        )

        box_head_fc_dim = cfg.get("box_head_fc_dim", 1024)
        resolution = cfg.get("box_roi_pool_output_size", 7)
        box_head = TwoMLPHead(
            in_channels=out_channels * resolution * resolution,
            representation_size=box_head_fc_dim,
        )

        box_predictor = FastRCNNPredictor(
            in_channels=box_head_fc_dim,
            num_classes=num_classes,
        )

        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.get("box_fg_iou_thresh", 0.5),
            bg_iou_thresh=cfg.get("box_bg_iou_thresh", 0.5),
            batch_size_per_image=cfg.get("box_batch_size_per_image", 512),
            positive_fraction=cfg.get("box_positive_fraction", 0.25),
            bbox_reg_weights=cfg.get("bbox_reg_weights", (10.0, 10.0, 5.0, 5.0)),
            score_thresh=cfg.get("box_score_thresh", 0.05),
            nms_thresh=cfg.get("box_nms_thresh", 0.5),
            detections_per_img=cfg.get("box_detections_per_img", 100),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, patch_tokens, image_sizes, targets=None):
        """
        Parameters
        ----------
        patch_tokens : torch.Tensor
            [B, N, D] patch tokens from the backbone (no CLS token).
        image_sizes : list[tuple[int, int]]
            Original (H, W) for each image in the batch. Needed by RPN/RoI
            to clip boxes and map proposals to the correct image dimensions.
        targets : list[dict], optional
            Each dict has:
            - "boxes": FloatTensor [num_objects, 4] in (x1, y1, x2, y2)
            - "labels": Int64Tensor [num_objects]
            Required during training; omit for inference.

        Returns
        -------
        In training mode:
            dict[str, Tensor] -- losses:
                loss_objectness, loss_rpn_box_reg, loss_classifier, loss_box_reg
        In eval mode:
            list[dict] -- per-image detections, each with:
                boxes: [num_det, 4], scores: [num_det], labels: [num_det]
        """
        # Build multi-scale features from patch tokens
        features = self.fpn(patch_tokens)

        # torchvision RPN/RoI heads expect ImageList-like metadata.
        # We build a minimal representation: list of image sizes.
        image_list = _ImageListStub(image_sizes, device=patch_tokens.device)

        # RPN: generate proposals (and RPN losses in training)
        proposals, rpn_losses = self.rpn(image_list, features, targets)

        # RoI heads: classify and regress proposals (and box losses in training)
        detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)

        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses
        else:
            return detections

    # ------------------------------------------------------------------
    # FMTK-compatible interface
    # ------------------------------------------------------------------

    def to_device(self):
        self.to(self.device)

    def to_cpu(self):
        self.to("cpu")

    def trainable_parameters(self):
        return self.parameters()

    # ------------------------------------------------------------------
    # Standalone training / evaluation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, backbone, data_loader, device):
        """
        Run inference on a data_loader and return all predictions.

        Parameters
        ----------
        backbone : BaseModel
            The ViT backbone (must have return_all_tokens=True).
        data_loader : DataLoader
            Yields dicts with "x" (images), "targets" (list of target dicts).
        device : str

        Returns
        -------
        list[dict] -- all detections across all images.
        """
        backbone_was_training = backbone.model.training
        backbone.model.eval()
        self.eval()

        all_detections = []
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch["x"].to(device)
            B = images.shape[0]
            image_sizes = [(images.shape[2], images.shape[3])] * B

            patch_tokens = backbone.forward(images)
            detections = self.forward(patch_tokens, image_sizes)
            all_detections.extend(detections)

        if backbone_was_training:
            backbone.model.train()

        return all_detections

    def train_one_epoch(self, backbone, data_loader, optimizer, device, freeze_backbone=True):
        """
        Train for one epoch.

        Parameters
        ----------
        backbone : BaseModel
            The ViT backbone (must have return_all_tokens=True).
        data_loader : DataLoader
            Yields dicts with "x" (images), "targets" (list of target dicts).
            Each target dict has "boxes" and "labels".
        optimizer : torch.optim.Optimizer
        device : str
        freeze_backbone : bool
            If True (default), backbone is in eval mode and not updated.

        Returns
        -------
        float -- average total loss for the epoch.
        """
        self.train()
        if freeze_backbone:
            backbone.model.eval()
        else:
            backbone.model.train()

        total_loss = 0.0
        for batch in tqdm(data_loader, desc="Training"):
            images = batch["x"].to(device)
            targets = batch["targets"]
            # Move target tensors to device
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            B = images.shape[0]
            image_sizes = [(images.shape[2], images.shape[3])] * B

            # Extract patch tokens from backbone
            if freeze_backbone:
                with torch.no_grad():
                    patch_tokens = backbone.forward(images)
            else:
                patch_tokens = backbone.forward(images)

            # Forward through detection heads
            optimizer.zero_grad()
            loss_dict = self.forward(patch_tokens, image_sizes, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        return avg_loss


class _ImageListStub:
    """
    Minimal stand-in for torchvision's ImageList.
    RPN and RoIHeads only need .image_sizes and .tensors (for device).
    """

    def __init__(self, image_sizes, device):
        self.image_sizes = image_sizes
        # Tensors attribute is only needed for .device in some codepaths
        self.tensors = torch.empty(0, device=device)
