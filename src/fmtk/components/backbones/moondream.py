from transformers import AutoModelForCausalLM
import os
from pathlib import Path

import torch
import re
from fmtk.components.base import BaseModel

from torchvision import transforms

_MODEL_CACHE = str(Path(__file__).resolve().parents[4] / "models" / "vlm")

class MoondreamModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlms'
        models_directory = _MODEL_CACHE
        if model_name=="moondream":
            model_id='vikhyatk/moondream2'
        self.processor = None #AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True)
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
    
    def postprocess(self,embeddings):
        pass

