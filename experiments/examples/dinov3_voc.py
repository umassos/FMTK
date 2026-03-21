import timeit
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt

start_time = timeit.default_timer()
from fmtk.pipeline import Pipeline

end_time = timeit.default_timer()
print(f"Time taken to import fmtk pipeline: {end_time - start_time} seconds")

from fmtk.components.backbones.dinov2 import DinoV2Model, get_dinov2_embed_dim
from fmtk.components.decoders.segmentation.LinearSemanticSegmenter import (
    LinearSemanticSegmenter,
)
from fmtk.metrics import get_mIoU
from torch.utils.data import DataLoader
from fmtk.datasets.VOC12 import VOC12Dataset, VOC_CLASSES

VOC_CMAP = np.array([
    [0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],
    [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
    [64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],
    [192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],
    [0,64,128],
], dtype=np.uint8)

device = "cuda:0"
seed = 42
NUM_CLASSES = 21
TARGET_SIZE = 448
PATCH_SIZE = 14
# ViT: 512/16 = 32x32 = 1024 tokens. ConvNeXt: 16x16 = 256 tokens
GRID_SIZE_VIT = TARGET_SIZE // PATCH_SIZE  # 32
GRID_SIZE_CONVNEXT = 16

generator = torch.Generator()
generator.manual_seed(seed)


def save_segmentation_example(P, dataset, save_path="segmentation_example.png"):
    """Run inference on one sample and save image / GT / prediction side-by-side."""
    sample = dataset[0]
    img_tensor = sample["x"].unsqueeze(0)    # [1, 3, H, W]
    gt = sample["y"].numpy()                 # [H, W]

    with torch.no_grad():
        logits = P.forward(img_tensor, use_cache=False)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # [H, W]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean).clip(0, 1)

    def colorize(mask):
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id in range(len(VOC_CMAP)):
            rgb[mask == cls_id] = VOC_CMAP[cls_id]
        rgb[mask == 255] = [224, 224, 224]
        return rgb

    gt_rgb = colorize(gt)
    pred_rgb = colorize(pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved segmentation example to {save_path}")


def train_model(
    dataloader_train,
    dataloader_test,
    model_id,
    model_cfg,
    train_config,
    inference_config,
    device,
):
    # backbone = DinoV3Model(device, model_id, model_cfg)
    backbone = DinoV2Model(device, model_id, model_cfg)
    # embed_dim = get_dinov3_embed_dim(model_id)
    embed_dim = get_dinov2_embed_dim(model_id)

    is_convnext = "convnext" in model_id.lower()
    grid_size = GRID_SIZE_CONVNEXT if is_convnext else GRID_SIZE_VIT

    decoder_cfg = {
        "input_dim": embed_dim,
        "output_dim": NUM_CLASSES,
        "height": grid_size,
        "width": grid_size,
        "pixel_height": TARGET_SIZE,
        "pixel_width": TARGET_SIZE,
        "ignore_index": 255,
    }

    P = Pipeline(backbone)
    linear_decoder = P.add_decoder(
        LinearSemanticSegmenter(device, cfg=decoder_cfg),
        load=True,
        path="vocseg_dinobase_linsemseg",
    )
    end_time = timeit.default_timer()
    print(f"Time taken to load model: {end_time - start_time} seconds")

    print("Training...")
    P.train(
        dataloader_train,
        parts_to_train=["decoder"],
        cfg=train_config,
        path="vocseg_dinobase_linsemseg",
    )

    y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
    result = get_mIoU(y_test, y_pred, num_classes=NUM_CLASSES, ignore_index=255)
    print("mIoU:", result["mIoU"])
    print("Per-class IoU:", result["per_class_iou"])

    save_segmentation_example(P, dataloader_test.dataset, save_path="results/dinov2_voc_base_segmentation_example.png")

    gc.collect()
    del P, linear_decoder, backbone
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result["mIoU"], result["per_class_iou"]


if __name__ == "__main__":
    task_cfg = {"task_type": "segmentation"}
    train_config = {
        "batch_size": 8,
        "shuffle": True,
        "epochs": 10,
        "lr": 1e-3,
        "scheduler": {"type": "cosine", "T_max": 10, "eta_min": 0},
    }
    inference_config = {"batch_size": 8, "shuffle": False}
    dataset_cfg = {
        "dataset_path": "/work/pi_shenoy_umass_edu/kgudipaty/datasets/PASCAL-VOC",
        "target_size": TARGET_SIZE,
    }
    model_cfg = {"return_all_tokens": True}

    # model_id = "vitb16"
    model_id = "base"

    train_data = VOC12Dataset(dataset_cfg, task_cfg, split="trainval")
    test_data = VOC12Dataset(dataset_cfg, task_cfg, split="test")

    print("Loading test dataloader...")
    dataloader_test = DataLoader(
        test_data,
        batch_size=inference_config["batch_size"],
        shuffle=inference_config["shuffle"],
        generator=generator,
    )
    print("Loading train dataloader...")
    dataloader_train = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=train_config["shuffle"],
        generator=generator,
    )

    miou, per_class = train_model(
        dataloader_train,
        dataloader_test,
        model_id,
        model_cfg,
        train_config,
        inference_config,
        device,
    )
    print("Final mIoU:", miou)
