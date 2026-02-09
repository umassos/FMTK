"""
DINOv2 + Faster R-CNN on COCO detection.

Uses the DINOv2 backbone (return_all_tokens=True) with the Simple Feature Pyramid
and torchvision-based Faster R-CNN detection heads.
"""

import timeit
import torch
import gc

from fmtk.components.backbones.dinov2 import DinoV2Model, EMBED_DIMS
from fmtk.components.decoders.detection.faster_rcnn import FasterRCNNDecoder
from fmtk.datasets.COCO import COCODetectionDataset, coco_collate_fn
from torch.utils.data import DataLoader

device = "cuda:0"
seed = 42
torch.manual_seed(seed)
generator = torch.Generator()
generator.manual_seed(seed)


def train(
    backbone,
    decoder,
    dataloader_train,
    dataloader_val,
    train_config,
    device,
):
    """Train the detection decoder for multiple epochs."""
    optimizer = torch.optim.AdamW(
        decoder.trainable_parameters(),
        lr=train_config["lr"],
        weight_decay=train_config.get("weight_decay", 0.1),
    )

    freeze_backbone = train_config.get("freeze_backbone", True)
    num_epochs = train_config["epochs"]

    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")

        start = timeit.default_timer()
        avg_loss = decoder.train_one_epoch(
            backbone, dataloader_train, optimizer, device,
            freeze_backbone=freeze_backbone,
        )
        elapsed = timeit.default_timer() - start
        print(f"Epoch {epoch}  loss: {avg_loss:.4f}  time: {elapsed:.1f}s")

        # Run evaluation every eval_interval epochs
        eval_interval = train_config.get("eval_interval", 5)
        if epoch % eval_interval == 0 or epoch == num_epochs:
            print("Running evaluation...")
            detections = decoder.evaluate(backbone, dataloader_val, device)
            print(f"  Total detections: {sum(len(d['boxes']) for d in detections)}")
            # For full COCO mAP evaluation, use pycocotools COCOeval
            # with dataset.get_coco_api() -- see below.

    return decoder


def evaluate_coco_map(decoder, backbone, dataloader_val, dataset_val, device):
    """
    Compute COCO mAP using pycocotools.

    Requires: pip install pycocotools
    """
    from pycocotools.cocoeval import COCOeval
    import json
    import numpy as np

    detections = decoder.evaluate(backbone, dataloader_val, device)

    # Convert detections to COCO results format
    coco_results = []
    for idx, det in enumerate(detections):
        image_id = dataset_val.get_image_id(idx)
        boxes = det["boxes"].cpu().numpy()
        scores = det["scores"].cpu().numpy()
        labels = det["labels"].cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            # Convert back to COCO format [x, y, w, h]
            coco_results.append({
                "image_id": int(image_id),
                "category_id": int(
                    # Map contiguous label back to COCO category ID
                    list(dataset_val._cat_id_to_label.keys())[
                        list(dataset_val._cat_id_to_label.values()).index(int(labels[i]))
                    ]
                ),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[i]),
            })

    if len(coco_results) == 0:
        print("No detections to evaluate.")
        return

    coco_gt = dataset_val.get_coco_api()
    coco_dt = coco_gt.loadRes(coco_results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    model_id = "base"  # DINOv2 variant: "small", "base", "large", "giant"
    model_cfg = {"return_all_tokens": True}  # Required for detection

    dataset_cfg = {
        "dataset_path": "/datasets/ai/coco",
        "target_size": 1024,  # Use 224 to match ViT pretrain size; increase for better detection
    }
    task_cfg = {"task_type": "detection"}

    train_config = {
        "batch_size": 4,
        "epochs": 10,
        "lr": 1e-4,
        "weight_decay": 0.1,
        "freeze_backbone": True,
        "eval_interval": 5,
    }
    inference_config = {"batch_size": 4}

    decoder_cfg = {
        "embed_dim": EMBED_DIMS[f"facebook/dinov2-{model_id}"],
        "num_classes": 80,  # COCO has 80 object classes
        "out_channels": 256,
        "image_size": dataset_cfg["target_size"],
    }

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("Loading COCO train dataset...")
    train_data = COCODetectionDataset(dataset_cfg, task_cfg, split="train")

    print("Loading COCO val dataset...")
    val_data = COCODetectionDataset(dataset_cfg, task_cfg, split="val")

    dataloader_train = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=coco_collate_fn,
        num_workers=4,
        generator=generator,
    )

    dataloader_val = DataLoader(
        val_data,
        batch_size=inference_config["batch_size"],
        shuffle=False,
        collate_fn=coco_collate_fn,
        num_workers=4,
    )

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    print("Loading DINOv2 backbone...")
    backbone = DinoV2Model(device, model_id, model_cfg)

    print("Building Faster R-CNN decoder...")
    decoder = FasterRCNNDecoder(device, decoder_cfg)
    decoder.to_device()

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    print(f"\nTraining Faster R-CNN with DINOv2-{model_id} backbone on COCO")
    print(f"  Image size:      {dataset_cfg['target_size']}")
    print(f"  Batch size:      {train_config['batch_size']}")
    print(f"  Epochs:          {train_config['epochs']}")
    print(f"  Learning rate:   {train_config['lr']}")
    print(f"  Freeze backbone: {train_config['freeze_backbone']}")

    decoder = train(
        backbone, decoder,
        dataloader_train, dataloader_val,
        train_config, device,
    )

    # -------------------------------------------------------------------------
    # Final evaluation (COCO mAP)
    # -------------------------------------------------------------------------
    print("\n--- Final COCO mAP evaluation ---")
    stats = evaluate_coco_map(decoder, backbone, dataloader_val, val_data, device)
    if stats is not None:
        print(f"AP @[IoU=0.50:0.95]: {stats[0]:.4f}")
        print(f"AP @[IoU=0.50]:      {stats[1]:.4f}")
        print(f"AP @[IoU=0.75]:      {stats[5]:.4f}")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
