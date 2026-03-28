from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
from pathlib import Path

import torch
import re
from fmtk.components.base import BaseModel
from peft import get_peft_model

from torchvision import transforms

_MODEL_CACHE = str(Path(__file__).resolve().parents[4] / "models" / "vlm" / "pretrained")

class LlamaVisionModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlm'
        models_directory = _MODEL_CACHE
        if model_name=="llama-vision":
            model_id='meta-llama/Llama-3.2-11B-Vision-Instruct'
        self.peft_enable = False
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
        self.model = MllamaForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager", device_map={"": self.device})

    def preprocess(self,batch_x,mask=None):
        pass

    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            messages = [
                {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}]}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(image, prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
            generate_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
            outputs = self.model.generate(**generate_inputs, max_new_tokens=20)
            response = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
            # Extract assistant response
            match = re.search(r"assistant\s*:?[\s\n]*(.*?)(?:\n+user|$)", response, re.IGNORECASE | re.DOTALL)
            if match:
                response = match.group(1).strip()
            responses.append(response)
        return responses

    def train_step(self, batch):
        """Single training step — returns scalar loss via causal LM objective."""
        image, question, label = batch['x'], batch['question'], batch['y']
        losses = []
        for img, q, lbl in zip(image, question, label):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            full_text = text_prompt + lbl + self.processor.tokenizer.eos_token
            inputs = self.processor(img, full_text, add_special_tokens=False, return_tensors="pt").to(self.device)
            # mask prompt tokens so loss is only on the answer
            prompt_inputs = self.processor(img, text_prompt, add_special_tokens=False, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100
            outputs = self.model(**inputs, labels=labels)
            losses.append(outputs.loss)
        return torch.stack(losses).mean()

    def enable_peft(self, peft_cfg):
        # dispatch hooks from device_map interfere with get_peft_model — remove them first
        self.model = self.model.to(self.device)
        self.model = get_peft_model(self.model, peft_cfg)
        # LoRA params must be fp32 for GradScaler to unscale correctly
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
            match = re.search(r"assistant\s*:?[\s\n]*(.*?)(?:\n+user|$)", embedding, re.IGNORECASE | re.DOTALL)
            if match:
                answer_chunk = match.group(1).strip()
                answer = answer_chunk.split()[0] if answer_chunk else ""
            else:
                answer = embedding.strip().split()[0]
            answers.append(answer)
        return answers
