from transformers import MllamaForConditionalGeneration, AutoProcessor
import os

import torch
import re
from fmtk.components.base import BaseModel

from torchvision import transforms

class LlamaVisionModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        self.model_category = 'vlms'
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../../..', 'FMaaS-motivation/vqa/updated/models')
        if model_name=="llama-vision":
            model_id='meta-llama/Llama-3.2-11B-Vision-Instruct'
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
