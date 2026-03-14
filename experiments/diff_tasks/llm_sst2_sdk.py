"""
Task 2: Sentiment Analysis (SST-2)
Models: LLaMA-3.1-8B, Mistral-7B, Phi-3-mini, Qwen2.5-7B
Metric: Accuracy
"""
from torch.utils.data import DataLoader
from fmtk.pipeline import Pipeline
from fmtk.datasetloaders.sst2 import SST2Dataset
from fmtk.components.backbones.llama import LlamaModel
from fmtk.components.backbones.mistral import MistralModel
from fmtk.components.backbones.phi3 import Phi3Model
from fmtk.components.backbones.qwen_text import QwenTextModel
from fmtk.metrics import get_accuracy

device = 'cuda:0'

task_cfg = {
    'task_type': 'sentiment',
    'inference_config': {'batch_size': 8, 'shuffle': False},
}
dataset_cfg = {
    'max_samples': 10,
}

dataloader_test = DataLoader(
    SST2Dataset(dataset_cfg, task_cfg, split='test'),
    batch_size=task_cfg['inference_config']['batch_size'],
    shuffle=task_cfg['inference_config']['shuffle'],
    collate_fn=lambda x: {'x': [i['x'] for i in x], 'y': [i['y'] for i in x]},
)

# ── Phi-3 ───────────────────────────────────────────────────────────────────────
P = Pipeline(Phi3Model(device, 'phi3-mini', model_config={'max_new_tokens': 5}))
labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
print(f"[Phi-3-mini]    SST-2 Accuracy: {get_accuracy(labels, preds):.4f}")

# # ── LLaMA ──────────────────────────────────────────────────────────────────────
# P = Pipeline(LlamaModel(device, 'llama-3.1-8b', model_config={'max_new_tokens': 5}))
# labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
# print(f"[LLaMA-3.1-8B]  SST-2 Accuracy: {get_accuracy(labels, preds):.4f}")

# # ── Mistral ─────────────────────────────────────────────────────────────────────
# P = Pipeline(MistralModel(device, 'mistral-7b', model_config={'max_new_tokens': 5}))
# labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
# print(f"[Mistral-7B]    SST-2 Accuracy: {get_accuracy(labels, preds):.4f}")


# # ── Qwen ────────────────────────────────────────────────────────────────────────
# P = Pipeline(QwenTextModel(device, 'qwen2.5-7b', model_config={'max_new_tokens': 5}))
# labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
# print(f"[Qwen2.5-7B]    SST-2 Accuracy: {get_accuracy(labels, preds):.4f}")

# # ── Qwen ────────────────────────────────────────────────────────────────────────
P = Pipeline(QwenTextModel(device, 'qwen2.5-0.5b', model_config={'max_new_tokens': 5}))
labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
print(f"[Qwen2.5-0.5B]    SST-2 Accuracy: {get_accuracy(labels, preds):.4f}")
