import torch
from torch.utils.data import DataLoader

from fmtk.pipeline import Pipeline
from fmtk.components.backbones.dinov2 import DinoV2Model, get_dinov2_embed_dim
from fmtk.components.decoders.classification.linear import LinearDecoder
from fmtk.datasetloaders.imagenet1k import ImageNet1kDataset
from fmtk.metrics import get_accuracy
from fmtk.logger import Logger

device = "cuda:0"

if __name__ == "__main__":
    task_cfg = {"task_type": "classification"}
    inference_config = {"batch_size": 64, "shuffle": False}
    dataset_cfg = {"dataset_path": "/datasets/ai/imagenet/classification/split", "resize_size": 256}
    model_id = "small"
    layers = [-4, -3, -2, -1]

    dataloader_test = DataLoader(
        ImageNet1kDataset(dataset_cfg, task_cfg, split="val"),
        batch_size=inference_config["batch_size"],
        shuffle=inference_config["shuffle"],
        num_workers=4,
    )

    backbone = DinoV2Model(device, model_id, {"layers": layers})
    P = Pipeline(backbone, Logger(device, "imagenet_logger"))
    embed_dim = get_dinov2_embed_dim(model_id)
    P.add_decoder(
        LinearDecoder(device, cfg={"input_dim": (len(layers) + 1) * embed_dim, "output_dim": 1000, "mode": "MULTICLS_PATCHMEAN"}),
        load=True,
        train=False,
        path=f"imagenetclass_dino{model_id}_linear4",
    )

    with torch.no_grad():
        y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
    print("Accuracy: ", get_accuracy(y_test, y_pred))
