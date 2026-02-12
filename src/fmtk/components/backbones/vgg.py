from fmtk.components.base import BaseModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from functools import singledispatchmethod


def get_vgg_model_id(model_name):
    if model_name in ["vgg11", "timm/vgg11.tv_in1k"]:
        return "timm/vgg11.tv_in1k"
    elif model_name in ["vgg13", "timm/vgg13.tv_in1k"]:
        return "timm/vgg13.tv_in1k"
    elif model_name in ["vgg16", "timm/vgg16.tv_in1k"]:
        return "timm/vgg16.tv_in1k"
    elif model_name in ["vgg19", "timm/vgg19.tv_in1k"]:
        return "timm/vgg19.tv_in1k"
    elif model_name in ["vgg11_bn", "timm/vgg11_bn.tv_in1k"]:
        return "timm/vgg11_bn.tv_in1k"
    elif model_name in ["vgg13_bn", "timm/vgg13_bn.tv_in1k"]:
        return "timm/vgg13_bn.tv_in1k"
    elif model_name in ["vgg16_bn", "timm/vgg16_bn.tv_in1k"]:
        return "timm/vgg16_bn.tv_in1k"
    elif model_name in ["vgg19_bn", "timm/vgg19_bn.tv_in1k"]:
        return "timm/vgg19_bn.tv_in1k"


def get_vgg_embed_dim(model_name):
    # All VGG variants return (512, 7, 7) .flatten() = 25088 -> linear -> 4096
    return 4096


class VGGModel(BaseModel):

    def __init__(self, device, model_name="vgg16", model_config={}):
        super().__init__()

        self.device = device

        self.model_id = get_vgg_model_id(model_name)
        self.embed_dim = get_vgg_embed_dim(
            model_name
        )  # All VGG variants have 4096 hidden dim features

        print(f"[VGG] Loading {self.model_id} on device {device}")

        self.model = timm.create_model(self.model_id, pretrained=True, num_classes=0)
        self.model.to(device)

        self.peft_enable = False

    def preprocess(self, batch_x, mask=None):
        batch_x = batch_x.float().to(self.device)
        self.B, self.C, self.H, self.W = batch_x.shape
        return batch_x, mask

    def forward(self, batch_x, mask=None):
        x, mask = self.preprocess(batch_x, mask)
        embeddings = self.model(x)
        return embeddings

    @singledispatchmethod
    @torch.no_grad()
    def predict(self, data):
        self.model.eval()
        embeddings = self.forward(data)
        return embeddings

    @predict.register
    @torch.no_grad()
    def _predict_from_dataloader(self, data: DataLoader):
        self.model.eval()
        all_embeddings, all_labels = [], []

        for batch in tqdm(data, total=len(data)):
            if isinstance(batch, dict):
                x = batch["x"]
                y = batch.get("y", None)
                mask = batch.get("mask", None)
            else:
                if len(batch) == 2:
                    x, y = batch
                    mask = None
                elif len(batch) == 3:
                    x, mask, y = batch
                else:
                    x = batch[0]
                    y = None
                    mask = None

            embeddings = self.forward(x, mask)
            all_embeddings.append(embeddings.cpu().detach().float().numpy())
            if y is not None:
                if isinstance(y, torch.Tensor):
                    all_labels.append(y.cpu().numpy())
                else:
                    all_labels.append(np.array(y))

        embeddings_np = np.vstack(all_embeddings)
        if all_labels:
            labels_np = np.concatenate(all_labels)
        else:
            labels_np = None

        return embeddings_np, labels_np
