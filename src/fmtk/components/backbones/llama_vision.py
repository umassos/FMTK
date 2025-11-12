from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
import torch
import re
from fmtk.components.base import BaseModel
from tqdm import tqdm
from torchvision import transforms

class LlamaVisionModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../weights')
        if model_name=="llama-vision":
            model_id='meta-llama/Llama-3.2-11B-Vision-Instruct'
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
        self.model = MllamaForConditionalGeneration.from_pretrained(model_id, cache_dir=models_directory, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager", device_map={"": "cuda:0"})
        
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
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(image,prompt,add_special_tokens=False,return_tensors="pt").to(self.device)
            generate_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
            outputs = self.model.generate(**generate_inputs, max_new_tokens=20)
            response = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
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

    