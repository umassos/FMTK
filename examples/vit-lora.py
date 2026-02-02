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
from transformers import AutoModelForImageClassification
from tqdm import tqdm
import numpy as np

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

model_id = "Andyrasika/vit-base-patch16-224-in21k-finetuned-lora-food101"
model = AutoModelForImageClassification.from_pretrained(model_id, num_labels=101)
model.to(device)

logits = []
labels = []
for batch in tqdm(dataloader_test, total=len(dataloader_test)):
    x = batch['x']
    y = batch['y']
    x = x.to(device)

    outputs = model(x).logits
    logits.append(outputs.cpu().numpy())
    labels.append(y.cpu().numpy())
    

logits = np.concatenate(logits, axis=0)
labels = np.concatenate(labels, axis=0)
print(logits.shape, labels.shape)
predictions = np.argmax(logits, axis=1)
result = get_accuracy(labels, predictions)
print(result)
