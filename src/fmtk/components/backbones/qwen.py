from transformers import AutoModelForVision2Seq, AutoProcessor
import os
import torch
import re
from fmtk.components.base import BaseModel
from tqdm import tqdm
from torchvision import transforms
from qwen_vl_utils import process_vision_info

class QwenModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../weights')
        if model_name=="qwen-3B":
            model_id='Qwen/Qwen2.5-VL-3B-Instruct'
        elif model_name=="qwen-7B":
            model_id='Qwen/Qwen2.5-VL-7B-Instruct'
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="flash_attention_2", device_map={"": "cuda:0"})

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
                        {"type": "text", "text": question+" Please answer in one word."},
                    ],}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",).to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=20)
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

    
    def predict(self,dataloader):
        """
        Compute embeddings for a single split using a DataLoader.
        
        Args:
            dataloader: PyTorch DataLoader yielding (x, y) or just x.
            pipeline: model or wrapper with a `.embed()` method.
            device: torch device.
        
        Returns:
            embeddings: [N, E] NumPy array (where E = embedding dimension)
            labels: [N] NumPy array of ground truth labels (if available)
        """
        embeddings_np=[]
        labels_np=[]
        for batch in tqdm(dataloader,total=len(dataloader)):
            image,question,gt = batch 
            with torch.no_grad():
                embeddings=self.forward((image,question))
                answer=self.postprocess(embeddings)
                embeddings_np.append(answer)
                labels_np.append(gt)
        return embeddings_np,labels_np               

    