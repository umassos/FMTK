from fmtk.pipeline import Pipeline
from fmtk.datasets.resisc45 import RESIC45Dataset
from fmtk.components.backbones.dinov2 import DinoV2Model, EMBED_DIMS
from fmtk.components.decoders.classification.linear import LinearDecoder
from fmtk.components.encoders.diff import LinearChannelCombiner
from fmtk.metrics import get_accuracy
from torch.utils.data import DataLoader, Subset
from peft import LoraConfig
import torch.nn as nn

device = "cuda:0"

task_cfg = {"task_type": "classification"}
# 'hr' for heart rate, 'sysbp' for systolic blood pressure, 'diasbp' for diastolic blood pressure
train_config = {
    "batch_size": 128,
    "shuffle": False,
    "epochs": 20,
    "lr": 1e-3,
    "scheduler": {"type": "cosine", "T_max": 10, "eta_min": 0},
}
inference_config = {"batch_size": 64, "shuffle": False}
dataset_cfg = {"dataset_path": "../dataset/RESISC45"}
model_cfg = {"return_all_tokens": True}

lora_config = LoraConfig(
    r=16,  # Rank: 16 is a good balance (can use 8, 16, 32, or 64)
    lora_alpha=16,  # Scaling: typically equal to r or 2*r (16 or 32)
    target_modules=["query", "value"],  # Standard for vision transformers
    lora_dropout=0.05,
)

data_train = RESIC45Dataset(dataset_cfg, task_cfg, split="train")
data_test = RESIC45Dataset(dataset_cfg, task_cfg, split="test")


dataloader_train = DataLoader(
    data_train,
    batch_size=train_config["batch_size"],
    shuffle=train_config["shuffle"],
)
dataloader_test = DataLoader(
    data_test,
    batch_size=inference_config["batch_size"],
    shuffle=inference_config["shuffle"],
)
model_id = "base"

backbone = DinoV2Model(device, model_id, model_cfg)

P = Pipeline(backbone)
linear_decoder = P.add_decoder(
    LinearDecoder(device, cfg={"input_dim": EMBED_DIMS[model_id], "output_dim": 45}),
    load=True,
    train=False,
    path="imgclass_dinobase_resic45",
)
# peft_adapter = P.add_adapter(lora_config)
peft_adapter = P.add_adapter(lora_config, path="imgclass_dinobase_resic45_lora")
# P.train(dataloader_train, parts_to_train=["decoder", "adapter"], cfg=train_config, path="imgclass_dinobase_resic45_lora")
y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
result = get_accuracy(y_test, y_pred)
print(result)
