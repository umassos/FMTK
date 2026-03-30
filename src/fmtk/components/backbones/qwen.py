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

QWEN_TEXT_MODELS = {
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-3b':   'Qwen/Qwen2.5-3B-Instruct',
    'qwen2.5-7b':   'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-14b':  'Qwen/Qwen2.5-14B-Instruct',
}
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
class QwenTextModel(BaseModel):
    def __init__(self, device, model_name='qwen2.5-7b', model_config=None):
        super().__init__()
        self.device = device
        self.model_category = 'llm'
        model_config = model_config or {}
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.abspath(os.path.join(base_dir, '../../../../models/llm/pretrained'))

        model_id = QWEN_TEXT_MODELS.get(model_name, model_name)
        print(f"[Qwen] Loading {model_id} on device {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=models_directory
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=models_directory,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
        self.model.eval()

        self.max_new_tokens = model_config.get('max_new_tokens', 64)

    def preprocess(self, batch_x, mask=None):
        return batch_x, mask

    def forward(self, batch_x, mask=None):
        """
        batch_x: list of prompt strings.
        Returns list of raw generated strings.
        """
        (_,batch_x_prompt)=batch_x
        responses = []
        for prompt in batch_x_prompt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = outputs[0][input_len:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            responses.append(response)
        return responses

    def postprocess(self, responses):
        return responses

    def predict(self, dataloader):
        all_preds, all_labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader)):
            prompts = batch['x']
            labels = batch['y']
            responses = self.forward(prompts)
            responses = self.postprocess(responses)
            all_preds.extend(responses)
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(list(labels))
        return np.array(all_labels, dtype=object), np.array(all_preds, dtype=object)