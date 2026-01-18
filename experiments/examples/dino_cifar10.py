from fmtk.pipeline import Pipeline
from fmtk.datasets.cifar10 import CIFAR10Dataset
from fmtk.components.backbones.dinov2 import DinoV2Model, EMBED_DIMS
from fmtk.components.decoders.classification.logisticregression import (
    LogisticRegressionDecoder,
)
from fmtk.components.encoders.diff import LinearChannelCombiner
from fmtk.metrics import get_accuracy
from torch.utils.data import DataLoader
from peft import LoraConfig

device = "cuda:0"

task_cfg = {"task_type": "classification"}
# 'hr' for heart rate, 'sysbp' for systolic blood pressure, 'diasbp' for diastolic blood pressure
train_config = {"batch_size": 32, "shuffle": False, "epochs": 50, "lr": 1e-2}
inference_config = {"batch_size": 32, "shuffle": False}
dataset_cfg = {"dataset_path": "../dataset/cifar-10"}
model_cfg = {"return_all_tokens": True}

dataloader_train = DataLoader(
    CIFAR10Dataset(dataset_cfg, task_cfg, split="train"),
    batch_size=train_config["batch_size"],
    shuffle=train_config["shuffle"],
)
dataloader_test = DataLoader(
    CIFAR10Dataset(dataset_cfg, task_cfg, split="test"),
    batch_size=inference_config["batch_size"],
    shuffle=inference_config["shuffle"],
)
model_id = "base"

backbone = DinoV2Model(device, model_id, model_cfg)

P = Pipeline(backbone)
logistic_regression_decoder = P.add_decoder(
    LogisticRegressionDecoder(max_iter=10000), load=True
)
P.train(dataloader_train, parts_to_train=["decoder"], cfg=train_config)
y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
result = get_accuracy(y_test, y_pred)
print(result)
