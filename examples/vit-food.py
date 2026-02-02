from fmtk.pipeline import Pipeline
from fmtk.datasets.food101 import Food101Dataset
from fmtk.components.backbones.vit import ViTModel, EMBED_DIMS
from fmtk.components.decoders.classification.logisticregression import (
    LogisticRegressionDecoder,
)
from fmtk.components.encoders.diff import LinearChannelCombiner
from fmtk.metrics import get_accuracy
from torch.utils.data import DataLoader, Subset
from peft import LoraConfig
from itertools import islice

device = "cuda:0"

task_cfg = {"task_type": "classification"}
# 'hr' for heart rate, 'sysbp' for systolic blood pressure, 'diasbp' for diastolic blood pressure
train_config = {"batch_size": 32, "shuffle": False, "epochs": 50, "lr": 1e-2}
inference_config = {"batch_size": 32, "shuffle": False}
dataset_cfg = {"dataset_path": "../dataset/food-101"}
# model_cfg = {"return_all_tokens": True}
model_cfg = {}

print("Loading dataset...")

dataset = Food101Dataset(dataset_cfg, task_cfg, split="train")

# Uncomment this only for testing
# dataset = Subset(dataset, range(5))
# -------------------------------------------------------------

dataloader_train = DataLoader(
    dataset, batch_size=train_config["batch_size"], shuffle=train_config["shuffle"]
)

dataloader_test = DataLoader(
    Food101Dataset(dataset_cfg, task_cfg, split="test"),
    batch_size=inference_config["batch_size"],
    shuffle=inference_config["shuffle"],
)
# model_name = "base"
model_name = 'ft-lora-food101'


print("Loading model...")
backbone = ViTModel(device, model_name, model_cfg)

print("Loading pipeline...")
P = Pipeline(backbone)

print("Loading decoder...")
logistic_regression_decoder = P.add_decoder(
    LogisticRegressionDecoder(max_iter=10000), load=True
)

print("Training...")
P.train(dataloader_train, parts_to_train=["decoder"], cfg=train_config)

print("Testing...")
y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
result = get_accuracy(y_test, y_pred)
print(result)
