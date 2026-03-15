from transformers import LlavaForConditionalGeneration, AutoProcessor, LlavaNextForConditionalGeneration
import os
from pathlib import Path

import torch
import re
from fmtk.components.base import BaseModel

from torchvision import transforms

_MODEL_CACHE = str(Path(__file__).resolve().parents[4] / "models" / "vlm")

class LlavaModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlms'
        models_directory = _MODEL_CACHE
        if model_name=="llava-1.5-7b":
            model_id='llava-hf/llava-1.5-7b-hf'
        elif model_name=="llava-1.5-13b":
            model_id='llava-hf/llava-1.5-13b-hf'
        elif model_name=="llava-v1.6-13b":
            model_id='llava-hf/llava-v1.6-vicuna-13b-hf'

        if model_name in ("llava-1.5-7b", "llava-1.5-13b"):
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory)
            self.model = LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map={"": self.device})
        elif model_name=="llava-v1.6-13b":
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True, attn_implementation="flash_attention_2", device_map={"": self.device})

    def preprocess(self,batch_x,mask=None):
        pass

    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=20)
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            responses.append(response)
        return responses

    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.split("ASSISTANT:")[-1].strip().split()[0]
            answers.append(answer)
        return answers
