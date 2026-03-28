from transformers import AutoModel, AutoTokenizer
import os
from pathlib import Path

import torch
import re
from fmtk.components.base import BaseModel
from peft import get_peft_model

from torchvision import transforms

_MODEL_CACHE = str(Path(__file__).resolve().parents[4] / "models" / "vlm" / "pretrained")

class MinicpmModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlm'
        models_directory = _MODEL_CACHE
        if model_name=="minicpm":
            model_id='openbmb/MiniCPM-V-2_6'
        elif model_name=="minicpm-2b":
            model_id='openbmb/MiniCPM-V-2'
        self.peft_enable = False
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=models_directory)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=models_directory, attn_implementation="flash_attention_2", device_map={"": self.device}).eval()
        self.processor = tokenizer

    def preprocess(self,batch_x,mask=None):
        pass

    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            # Resize very small images to avoid NaN in MiniCPM's image processor
            w, h = image.size
            if w < 32 or h < 32:
                image = image.resize((max(w, 32), max(h, 32)))
            msgs = [{"role": "user", "content": [image, question]}]
            try:
                response = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.processor,
                )
            except (RuntimeError, ValueError) as e:
                print(f"  MiniCPM sample error ({w}x{h}): {e}")
                response = ""
            responses.append(response)
        return responses

    def train_step(self, batch):
        """Single training step — returns scalar loss via causal LM objective."""
        image, question, label = batch['x'], batch['question'], batch['y']
        losses = []
        for img, q, lbl in zip(image, question, label):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            w, h = img.size
            if w < 32 or h < 32:
                img = img.resize((max(w, 32), max(h, 32)))
            msgs = [{"role": "user", "content": [img, q]}]
            full_text = q + " " + lbl
            msgs_full = [{"role": "user", "content": [img, full_text]}]
            # encode prompt only to get its length for masking
            prompt_enc = self.processor(q, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[1]
            inputs = self.model.get_vllm_embedding(msgs_full, self.processor)
            labels = inputs["input_ids"].clone().to(self.device)
            labels[:, :prompt_len] = -100
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()}, labels=labels)
            losses.append(outputs.loss)
        return torch.stack(losses).mean()

    def enable_peft(self, peft_cfg):
        self.model = self.model.to(self.device)
        self.model = get_peft_model(self.model, peft_cfg)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
        self.peft_enable = True

    def adapter_trainable_parameters(self):
        if not self.peft_enable:
            return []
        return (p for p in self.model.parameters() if p.requires_grad)

    def load_adapter(self, adapter_dir, peft_cfg=None):
        """Load a saved PEFT adapter for inference."""
        if not self.peft_enable:
            self.enable_peft(peft_cfg)
        self.model.load_adapter(adapter_dir, adapter_name='loaded')

    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.strip().split()[0]
            answers.append(answer)
        return answers
