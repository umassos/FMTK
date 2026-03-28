from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
import os
from pathlib import Path

import torch
import re
from fmtk.components.base import BaseModel
from peft import get_peft_model

from torchvision import transforms

# ── PyTorch 2.0 compat: torch.all() doesn't support dim=tuple ──
_orig_torch_all = torch.all
def _patched_torch_all(input, *args, **kwargs):
    dim = kwargs.get('dim', args[0] if args else None)
    if isinstance(dim, tuple):
        keepdim = kwargs.get('keepdim', False)
        result = input
        for d in sorted(dim, reverse=True):
            result = _orig_torch_all(result, dim=d, keepdim=keepdim)
        return result
    return _orig_torch_all(input, *args, **kwargs)
torch.all = _patched_torch_all

_MODEL_CACHE = str(Path(__file__).resolve().parents[4] / "models" / "vlm" / "pretrained")

class MolmoModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlm'
        models_directory = _MODEL_CACHE
        if model_name=="molmo":
            model_id='allenai/Molmo-7B-D-0924'

        self.peft_enable = False
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager", device_map={"": self.device})

    def preprocess(self,batch_x,mask=None):
        pass

    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            processed = self.processor.process(images=[image], text=question)
            inputs = {k: v.to(self.device).unsqueeze(0) for k, v in processed.items()}
            input_len = inputs["input_ids"].shape[1]
            gen_config = GenerationConfig(max_new_tokens=20, stop_strings="<|endoftext|>")
            output_ids = self.model.generate_from_batch(inputs, gen_config, tokenizer=self.processor.tokenizer)
            generated_ids = output_ids[0, input_len:]
            response = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            responses.append(response)
        return responses

    def train_step(self, batch):
        """Single training step — returns scalar loss via causal LM objective."""
        image, question, label = batch['x'], batch['question'], batch['y']
        losses = []
        for img, q, lbl in zip(image, question, label):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            full_text = q + " " + lbl
            processed = self.processor.process(images=[img], text=full_text)
            inputs = {k: v.to(self.device).unsqueeze(0) for k, v in processed.items()}
            # build labels masking the prompt tokens
            prompt_processed = self.processor.process(images=[img], text=q)
            prompt_len = prompt_processed["input_ids"].shape[0]
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

    def postprocess(self, embeddings):
        answers = []
        for embedding in embeddings:
            answer = embedding.split()[0] if embedding else ""
            answers.append(answer)
        return answers
