# TODO: Maybe broken code, not completely tested
import timeit
import numpy as np
import torch
import gc
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

start_time = timeit.default_timer()
from fmtk.pipeline import Pipeline

end_time = timeit.default_timer()
print(f"Time taken to import fmtk pipeline: {end_time - start_time} seconds")

from fmtk.components.backbones.dinov2 import DinoV2Model, get_dinov2_embed_dim
from fmtk.components.decoders.detection.rfdetr_decoder import RFDetrDecoder
from fmtk.datasets.COCO import COCODetectionDataset, coco_collate_fn

device = "cuda:0"
seed = 42
generator = torch.Generator()
generator.manual_seed(seed)

# Resolution and per-model configs that match RF-DETR checkpoints
RFDETR_BACKBONE_CONFIG = {
    "nano": {"model_id": "small", "resolution": 384, "out_feature_indexes": [3, 6, 9, 12]},
    "base": {"model_id": "small", "resolution": 560, "out_feature_indexes": [2, 5, 8, 11]},
}

DECODER_CHECKPOINT = {"nano": "rf-detr-nano.pth", "base": "rf-detr-base.pth"}

# RF-DETR decoder configs: must match checkpoint architecture
RFDETR_CONFIG = {
    "nano": {
        "out_feature_indexes": [3, 6, 9, 12],
        "projector_scale": ["P4"],
        "dec_layers": 2,
        "dec_n_points": 2,
    },
    "base": {
        "out_feature_indexes": [2, 5, 8, 11],
        "projector_scale": ["P4"],
        "dec_layers": 3,
        "dec_n_points": 2,
    },
}


def evaluate_detection(pipeline, test_loader, device):
    """Run detection on test set and compute COCO mAP."""
    try:
        from rfdetr.models import PostProcess
        from rfdetr.datasets.coco_eval import CocoEvaluator
    except ImportError as e:
        print(f"Could not evaluate: rfdetr required ({e})")
        return None

    coco_gt = test_loader.dataset.get_coco_api()
    pipeline.set_eval_mode()
    cat_ids = sorted(coco_gt.getCatIds())
    coco_gt.label2cat = {i: cid for i, cid in enumerate(cat_ids)}

    postprocess = PostProcess(num_select=100)
    coco_evaluator = CocoEvaluator(coco_gt, ["bbox"], max_dets=100)

    total_preds = 0
    total_fg = 0
    sample_scores = []
    for batch in tqdm(test_loader, desc="Evaluating"):
        x = batch["x"].to(device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch["y"]]

        with torch.no_grad():
            outputs = pipeline.forward(x, mask=mask, idx=None, use_cache=False)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_all = postprocess(outputs, orig_target_sizes)
        res = {t["image_id"].item(): out for t, out in zip(targets, results_all)}
        for output in results_all:
            labels = output["labels"].cpu()
            scores = output["scores"].cpu()
            total_preds += len(labels)
            fg_mask = labels < 80
            total_fg += fg_mask.sum().item()
            if len(sample_scores) < 100:
                sample_scores.extend(scores[fg_mask].tolist())
        coco_evaluator.update(res)

    print(f"[Debug] Total preds: {total_preds}, foreground (label<80): {total_fg}")
    if sample_scores:
        print(f"[Debug] Sample FG scores (first 10): {sample_scores[:10]}")

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval["bbox"].stats
    mAP = float(stats[0]) if len(stats) > 0 else 0.0
    mAP_50 = float(stats[1]) if len(stats) > 1 else 0.0
    mAP_75 = float(stats[2]) if len(stats) > 2 else 0.0
    print(f"COCO mAP: {mAP:.4f}  mAP@50: {mAP_50:.4f}  mAP@75: {mAP_75:.4f}")
    return {"mAP": mAP, "mAP_50": mAP_50, "mAP_75": mAP_75}


def coco_detection_metric_fn(coco_dataset, device):
    """Build metric_fn for COCO detection. Use with pipeline train_eval."""

    def metric_fn(y_true, y_pred):
        if not y_true or not y_pred:
            return 0.0
        try:
            from rfdetr.models import PostProcess
            from rfdetr.datasets.coco_eval import CocoEvaluator
        except ImportError:
            return 0.0
        coco_gt = coco_dataset.get_coco_api()
        cat_ids = sorted(coco_gt.getCatIds())
        coco_gt.label2cat = {i: cid for i, cid in enumerate(cat_ids)}
        postprocess = PostProcess(num_select=100)
        coco_evaluator = CocoEvaluator(coco_gt, ["bbox"], max_dets=100)
        for targets_batch, outputs_batch in zip(y_true, y_pred):
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets_batch]
            outputs_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in outputs_batch.items()}
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results_all = postprocess(outputs_device, orig_target_sizes)
            res = {t["image_id"].item(): out for t, out in zip(targets, results_all)}
            coco_evaluator.update(res)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        stats = coco_evaluator.coco_eval["bbox"].stats
        return float(stats[1]) if len(stats) > 1 else 0.0

    return metric_fn


def train_model(
    dataloader_train,
    dataloader_test,
    model_id,
    decoder_id,
    model_cfg,
    train_config,
    inference_config,
    device,
):
    rf_cfg = RFDETR_CONFIG.get(decoder_id, RFDETR_CONFIG["base"])
    out_feature_indexes = rf_cfg["out_feature_indexes"]
    backbone_cfg = RFDETR_BACKBONE_CONFIG.get(decoder_id, RFDETR_BACKBONE_CONFIG["nano"])
    model_cfg_aligned = {
        **model_cfg,
        "out_feature_indexes": out_feature_indexes,
    }
    backbone = DinoV2Model(device, backbone_cfg["model_id"], model_cfg_aligned)
    P = Pipeline(backbone)
    embed_dim = get_dinov2_embed_dim(backbone_cfg["model_id"])
    decoder_cfg = {
        "embed_dim": embed_dim,
        "num_classes": 80,
        "out_feature_indexes": out_feature_indexes,
        "projector_scale": rf_cfg["projector_scale"],
        "dec_layers": rf_cfg["dec_layers"],
        "dec_n_points": rf_cfg.get("dec_n_points", 2),
    }
    rfdetr_decoder = RFDetrDecoder(device, cfg=decoder_cfg)
    P.add_decoder(rfdetr_decoder, load=True)
    end_time = timeit.default_timer()
    print(f"Time taken to load model: {end_time - start_time} seconds")

    checkpoint = DECODER_CHECKPOINT.get(decoder_id, "rf-detr-base.pth")
    # try:
    #     report = rfdetr_decoder.load_pretrained_rfdetr_weights(checkpoint)
    #     print(f"Loaded decoder pretrained weights: {report['loaded']} keys")
    # except Exception as e:
    #     print(f"Could not load decoder pretrained weights: {e}")

    metric_fn = coco_detection_metric_fn(dataloader_test.dataset, device)

    print("Training...")
    P.train_eval(
        dataloader_train,
        dataloader_test,
        parts_to_train=["decoder"],
        train_cfg=train_config,
        inference_cfg=inference_config,
        path="dino_coco_rfdetrnano",
        metric_fn=metric_fn,
        mlflow_cfg={
            "experiment_name": "dino-coco-rfdetrnano",
            "run_name": "dino-coco-rfdetrnano",
            "extra_params": {
                "decoder_id": decoder_id,
                "model_cfg": str(model_cfg),
                "train_config": str(train_config),
            },
        },
    )

    print("Evaluating after training...")
    evaluate_detection(P, dataloader_test, device)

    gc.collect()
    del P, rfdetr_decoder, backbone
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    decoder_id = "nano"
    train_config = {"batch_size": 32, "shuffle": True, "epochs": 10, "lr": 1e-4}
    inference_config = {"batch_size": 128, "shuffle": False}
    model_cfg = {}

    backbone_cfg = RFDETR_BACKBONE_CONFIG.get(decoder_id, RFDETR_BACKBONE_CONFIG["nano"])
    dataset_cfg = {
        "dataset_path": "/datasets/ai/coco",
        "target_size": backbone_cfg["resolution"],
    }
    task_cfg = {}

    train_data = COCODetectionDataset(dataset_cfg, task_cfg, split="train")
    test_data = COCODetectionDataset(dataset_cfg, task_cfg, split="val")

    print("Loading dataloaders...")
    dataloader_test = DataLoader(
        test_data,
        batch_size=inference_config["batch_size"],
        shuffle=inference_config["shuffle"],
        collate_fn=coco_collate_fn,
        generator=generator,
    )
    dataloader_train = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=train_config["shuffle"],
        collate_fn=coco_collate_fn,
        generator=generator,
    )

    train_model(
        dataloader_train,
        dataloader_test,
        decoder_id,
        decoder_id,
        model_cfg,
        train_config,
        inference_config,
        device,
    )
    print("Done.")
