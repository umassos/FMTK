"""
Task 5: Summarization (CNN / DailyMail)
Models: LLaMA-3.1-8B, Mistral-7B, Phi-3-mini, Qwen2.5-7B
Metrics: ROUGE-1, ROUGE-2, ROUGE-L
"""
from torch.utils.data import DataLoader
from fmtk.pipeline import Pipeline
from fmtk.datasetloaders.cnn_dailymail import CNNDailyMailDataset
from fmtk.components.backbones.llama import LlamaModel
from fmtk.components.backbones.mistral import MistralModel
from fmtk.components.backbones.phi3 import Phi3Model
from fmtk.components.backbones.qwen_text import QwenTextModel
from fmtk.metrics import get_rouge

device = 'cuda:0'

task_cfg = {
    'task_type': 'summarization',
    'inference_config': {'batch_size': 2, 'shuffle': False},
}
dataset_cfg = {
    'max_samples': 10,
}

dataloader_test = DataLoader(
    CNNDailyMailDataset(dataset_cfg, task_cfg, split='test'),
    batch_size=task_cfg['inference_config']['batch_size'],
    shuffle=task_cfg['inference_config']['shuffle'],
    collate_fn=lambda x: {'x': [i['x'] for i in x], 'y': [i['y'] for i in x]},
)

# # ── LLaMA ──────────────────────────────────────────────────────────────────────
# P = Pipeline(LlamaModel(device, 'llama-3.1-8b', model_config={'max_new_tokens': 128}))
# labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
# rouge = get_rouge(labels, preds)
# print(f"[LLaMA-3.1-8B]  CNN/DM  R1={rouge['rouge1']:.4f}  R2={rouge['rouge2']:.4f}  RL={rouge['rougeL']:.4f}")

# # ── Mistral ─────────────────────────────────────────────────────────────────────
# P = Pipeline(MistralModel(device, 'mistral-7b', model_config={'max_new_tokens': 128}))
# labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
# rouge = get_rouge(labels, preds)
# print(f"[Mistral-7B]    CNN/DM  R1={rouge['rouge1']:.4f}  R2={rouge['rouge2']:.4f}  RL={rouge['rougeL']:.4f}")

# ── Phi-3 ───────────────────────────────────────────────────────────────────────
P = Pipeline(Phi3Model(device, 'phi3-mini', model_config={'max_new_tokens': 128}))
labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
rouge = get_rouge(labels, preds)
print(f"[Phi-3-mini]    CNN/DM  R1={rouge['rouge1']:.4f}  R2={rouge['rouge2']:.4f}  RL={rouge['rougeL']:.4f}")

# # ── Qwen ────────────────────────────────────────────────────────────────────────
# P = Pipeline(QwenTextModel(device, 'qwen2.5-7b', model_config={'max_new_tokens': 128}))
# labels, preds = P.predict(dataloader_test, cfg=task_cfg['inference_config'])
# rouge = get_rouge(labels, preds)
# print(f"[Qwen2.5-7B]    CNN/DM  R1={rouge['rouge1']:.4f}  R2={rouge['rouge2']:.4f}  RL={rouge['rougeL']:.4f}")
