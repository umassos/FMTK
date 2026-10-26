import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fmtk.pipeline import Pipeline
from fmtk.components.backbones.dinov2 import DinoV2Model, get_dinov2_embed_dim
from fmtk.components.decoders.segmentation.BNLinearSemanticSegmenter import BNLinearSemanticSegmenter
from fmtk.datasetloaders.ade20k import ADE20KDataset, NUM_CLASSES
from fmtk.metrics import seg_intersect_union, get_mIoU_from_areas
from fmtk.tasks.segmentation_utils import slide_inference
from fmtk.logger import Logger

device = "cuda:0"

if __name__ == "__main__":
    model_id = "small"
    # DINOv2 ADE20K linear eval: short side 512 (keep ratio), slide 512/341, mIoU at original resolution
    crop_size, stride, patch_size = 512, 341, 14
    grid = math.ceil(crop_size / patch_size)  # 37 after center padding to 518

    task_cfg = {"task_type": "segmentation"}
    dataset_cfg = {
        "dataset_path": "/work/pi_shenoy_umass_edu/kgudipaty/datasets/ADEChallengeData2016",
        "target_size": 512,
        "resize_mode": "short_side",
        "interpolation": "bilinear",
        "antialias": False,
    }
    dataloader_test = DataLoader(ADE20KDataset(dataset_cfg, task_cfg, split="val"), batch_size=1, shuffle=False, num_workers=4)

    backbone = DinoV2Model(device, model_id, {})
    P = Pipeline(backbone, Logger(device, "ade20k_logger"))
    embed_dim = get_dinov2_embed_dim(model_id)
    P.add_decoder(
        BNLinearSemanticSegmenter(device, cfg={
            "input_dim": embed_dim, "output_dim": NUM_CLASSES, "mode": "PATCH",
            "height": grid, "width": grid,
            "pixel_height": crop_size, "pixel_width": crop_size, "ignore_index": 255,
        }),
        load=True,
        train=False,
        path=f"adeseg_dino{model_id}_linear",
    )
    P.active_decoder.eval()  # BatchNorm must use running stats

    area_inter = np.zeros(NUM_CLASSES)
    area_union = np.zeros(NUM_CLASSES)
    with torch.no_grad():
        for batch in tqdm(dataloader_test):
            y = batch["y"][0]
            logits = slide_inference(P, batch["x"], y.shape, NUM_CLASSES, crop_size, stride, patch_size, device)
            inter, union = seg_intersect_union(logits.argmax(dim=1)[0], y.to(device), NUM_CLASSES)
            area_inter += inter
            area_union += union

    print("mIoU: ", get_mIoU_from_areas(area_inter, area_union)["mIoU"])
