import argparse
import torch
from torch.utils.data import DataLoader

from fmtk.pipeline import Pipeline
from fmtk.components.backbones.beit import BEiTModel
from fmtk.components.backbones.convnext import ConvNextModel
from fmtk.components.backbones.swin import SwinModel
from fmtk.components.decoders.classification.linear import LinearDecoder
from fmtk.datasetloaders.imagenet1k import ImageNet1kDataset
from fmtk.metrics import get_accuracy
from fmtk.logger import Logger

device = "cuda:0"

# name -> backbone, decoder token mode / norm eps, eval resize (crop is 224)
MODELS = {
    "beitv2base": dict(backbone=(BEiTModel, "base"), embed_dim=768, mode="PATCH", norm_eps=1e-6, resize_size=248),
    "convnextv2tiny": dict(backbone=(ConvNextModel, "v2-tiny"), embed_dim=768, mode="ALL", norm_eps=1e-12, resize_size=256),
    "convnexttiny": dict(backbone=(ConvNextModel, "tiny"), embed_dim=768, mode="ALL", norm_eps=1e-12, resize_size=256),
    "swintiny": dict(backbone=(SwinModel, "tiny"), embed_dim=768, mode="ALL", norm_eps=None, resize_size=256),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="beitv2base", choices=list(MODELS))
    args = parser.parse_args()
    spec = MODELS[args.model]

    task_cfg = {"task_type": "classification"}
    inference_config = {"batch_size": 64, "shuffle": False}
    dataset_cfg = {
        "dataset_path": "/datasets/ai/imagenet/classification/split",
        "target_size": 224,
        "resize_size": spec["resize_size"],
    }

    dataloader_test = DataLoader(
        ImageNet1kDataset(dataset_cfg, task_cfg, split="val"),
        batch_size=inference_config["batch_size"],
        shuffle=inference_config["shuffle"],
        num_workers=4,
    )

    backbone_cls, backbone_name = spec["backbone"]
    backbone = backbone_cls(device, backbone_name)
    P = Pipeline(backbone, Logger(device, "imagenet_logger"))
    P.add_decoder(
        LinearDecoder(device, cfg={
            "input_dim": spec["embed_dim"], "output_dim": 1000, "mode": spec["mode"],
            "norm": spec["norm_eps"] is not None, "norm_eps": spec["norm_eps"],
        }),
        load=True,
        train=False,
        path=f"imagenetclass_{args.model}_linear",
    )

    with torch.no_grad():
        y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
    print("Accuracy: ", get_accuracy(y_test, y_pred))
