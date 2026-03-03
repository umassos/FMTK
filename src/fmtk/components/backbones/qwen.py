from transformers import AutoModelForVision2Seq, AutoProcessor
import os

import torch
import re
from fmtk.components.base import BaseModel

from torchvision import transforms
from qwen_vl_utils import process_vision_info

class QwenModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlms'
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../../..', 'FMaaS-motivation/vqa/updated/models')
        if model_name=="qwen-3B":
            model_id='Qwen/Qwen2.5-VL-3B-Instruct'
        elif model_name=="qwen-7B":
            model_id='Qwen/Qwen2.5-VL-7B-Instruct'
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

    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.split()[0] if embedding else ""
            answers.append(answer)
        return answers
