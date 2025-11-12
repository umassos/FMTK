from transformers import LlavaForConditionalGeneration, LlavaProcessor, AutoProcessor,LlavaNextForConditionalGeneration
import os
import torch
import re
from fmtk.components.base import BaseModel
from tqdm import tqdm
from torchvision import transforms

class LlavaModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../weights')
        if model_name=="llava-1.5-7b-hf":
            model_id='llava-hf/llava-1.5-7b-hf'
        elif model_name=="llava-1.5-13b-hf":
            model_id='llava-hf/llava-1.5-13b-hf' 
        elif model_name=="llava-v1.6-vicuna-13b-hf":
            model_id='llava-hf/llava-v1.6-vicuna-13b-hf' 
        
        if model_name=="llava-1.5-7b-hf" or  model_name=="llava-1.5-13b-hf":
            self.processor = LlavaProcessor.from_pretrained(model_id, cache_dir=models_directory, use_fast=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map={"": self.device})
        elif model_name=="llava-v1.6-vicuna-13b-hf":
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True, attn_implementation="flash_attention_2", device_map={"": "cuda:0"}).to("cuda")            

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
                    {"type": "text", "text": question + " Please answer in one word."}]}]
            prompt = f"USER: <image>\n{question}\nPlease answer in one word.\nASSISTANT:"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
            input_tokens = inputs["input_ids"].shape[1]
            outputs = self.model.generate(**inputs, max_new_tokens=10)
            response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            responses.append(response)
        return responses
    
    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.split("ASSISTANT:")[-1].strip().split()[0]
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

    