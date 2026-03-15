from transformers import AutoProcessor, AutoModelForCausalLM
import os
from pathlib import Path

import torch
import re
from fmtk.components.base import BaseModel

from torchvision import transforms

_MODEL_CACHE = str(Path(__file__).resolve().parents[4] / "models" / "vlm")

class PhiModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlms'
        models_directory = _MODEL_CACHE
        if model_name=="phi":
            model_id='microsoft/Phi-3.5-vision-instruct'
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map={"": self.device})

    def preprocess(self,batch_x,mask=None):
        pass

    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            prompt = f"<|image_1|>\n{question}"
            messages = [{"role": "user", "content": prompt}]

            text_prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text_prompt, [image], return_tensors="pt").to(self.device)

            outputs = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=20,
                do_sample=False,
                use_cache=False,
            )

            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            responses.append(response)
        return responses

    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.split()[0] if embedding else ""
            answers.append(answer)
        return answers
