from transformers import AutoModelForVision2Seq, AutoProcessor
import os
from pathlib import Path

import torch
import re
from fmtk.components.base import BaseModel
from peft import get_peft_model

from torchvision import transforms
from qwen_vl_utils import process_vision_info

_MODEL_CACHE = str(Path(__file__).resolve().parents[4] / "models" / "vlm" / "pretrained")

class QwenModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlm'
        models_directory = _MODEL_CACHE
        if model_name=="qwen-2B":
            model_id='Qwen/Qwen2-VL-2B-Instruct'
        elif model_name=="qwen-3B":
            model_id='Qwen/Qwen2.5-VL-3B-Instruct'
        elif model_name=="qwen-7B":
            model_id='Qwen/Qwen2.5-VL-7B-Instruct'
        self.peft_enable = False
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True, device_map={"": self.device})

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
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",).to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=20, min_new_tokens=1)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, outputs)]
            response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
            responses.append(response)
        return responses

    def train_step(self, batch):
        """Single training step — returns scalar loss via causal LM objective."""
        image, question, label = batch['x'], batch['question'], batch['y']
        losses = []
        for img, q, lbl in zip(image, question, label):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": q},
            ]}]
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = text_prompt + lbl + self.processor.tokenizer.eos_token
            image_inputs, video_inputs = process_vision_info(messages)
            # encode prompt only to get its length for masking
            prompt_inputs = self.processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]
            inputs = self.processor(text=[full_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
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
            answer = embedding.split()[0] if embedding else ""
            answers.append(answer)
        return answers
