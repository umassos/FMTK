"""
Task: Sentiment Analysis (SST-2) — vLLM runtime
Models: Phi-3-mini, Qwen2.5 backed by vLLM for faster batched inference
Metric: Accuracy

Usage:
    python experiments/diff_tasks/llm_sst2_vllm.py
"""
from torch.utils.data import DataLoader
from fmtk.pipeline import Pipeline
from fmtk.datasetloaders.sst2 import SST2Dataset
from fmtk.components.backbones.phi3_vllm import Phi3VLLMModel
from fmtk.components.backbones.qwen_vllm import QwenVLLMModel
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

# # ── Phi-3-mini (vLLM) ───────────────────────────────────────────────────────
# P = Pipeline(Phi3VLLMModel(device, 'phi3-mini', model_config={'max_new_tokens': 5}))
# labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
# print(f"[Phi-3-mini vLLM]  SST-2 Accuracy: {get_accuracy(labels, preds):.4f}")

# ── Qwen2.5-0.5B (vLLM) ─────────────────────────────────────────────────────
P = Pipeline(QwenVLLMModel(device, 'qwen2.5-0.5b', model_config={'max_new_tokens': 5}))
labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
print(f"[Qwen2.5-0.5B vLLM]  SST-2 Accuracy: {get_accuracy(labels, preds):.4f}")

# ── Qwen2.5-7B (vLLM) ───────────────────────────────────────────────────────
# P = Pipeline(QwenVLLMModel(device, 'qwen2.5-7b', model_config={'max_new_tokens': 5}))
# labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
# print(f"[Qwen2.5-7B vLLM]   SST-2 Accuracy: {get_accuracy(labels, preds):.4f}")
