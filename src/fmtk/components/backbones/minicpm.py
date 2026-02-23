from transformers import AutoModel, AutoTokenizer
import os

import torch
import re
from fmtk.components.base import BaseModel

from torchvision import transforms

class MinicpmModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../../..', 'FMaaS-motivation/vqa/updated/models')
        if model_name=="minicpm":
            model_id='openbmb/MiniCPM-V-2_6'
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=models_directory)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=models_directory, attn_implementation="flash_attention_2").eval().to(self.device)
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

    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.strip().split()[0]
            answers.append(answer)
        return answers
