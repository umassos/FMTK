from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path

import torch
import re
from fmtk.components.base import BaseModel
from peft import get_peft_model

from torchvision import transforms

_MODEL_CACHE = str(Path(__file__).resolve().parents[4] / "models" / "vlm" / "pretrained")

class MoondreamModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlm'
        models_directory = _MODEL_CACHE
        if model_name=="moondream":
            model_id='vikhyatk/moondream2'
        self.peft_enable = False
        self.processor = AutoTokenizer.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True).to(self.device)

    def preprocess(self,batch_x,mask=None):
        pass
    
    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            response = self.model.query(image, question)["answer"]
            responses.append(response)
        return responses
    
    def train_step(self, batch):
        """Single training step — returns scalar loss via causal LM objective."""
        image, question, label = batch['x'], batch['question'], batch['y']
        losses = []
        for img, q, lbl in zip(image, question, label):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            # encode image via moondream's internal vision encoder
            image_embeds = self.model.encode_image(img)
            # build prompt and full text token ids
            prompt_tokens = self.processor(q, return_tensors="pt")["input_ids"].to(self.device)
            full_tokens = self.processor(q + lbl + self.processor.eos_token, return_tensors="pt")["input_ids"].to(self.device)
            prompt_len = prompt_tokens.shape[1]
            labels = full_tokens.clone()
            labels[:, :prompt_len] = -100
            outputs = self.model(input_ids=full_tokens, image_embeds=image_embeds, labels=labels)
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

    def postprocess(self, embeddings):
        answers = []
        for embedding in embeddings:
            answer = embedding.split()[0] if embedding else ""
            answers.append(answer)
        return answers

