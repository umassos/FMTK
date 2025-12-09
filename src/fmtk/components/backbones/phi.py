from transformers import AutoProcessor, AutoModelForCausalLM
import os
import torch
import re
from fmtk.components.base import BaseModel
from tqdm import tqdm
from torchvision import transforms

class PhiModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../weights')
        if model_name=="phi":
            model_id='microsoft/Phi-3.5-vision-instruct'
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map={"": "cuda:0"})

    def preprocess(self,batch_x,mask=None):
        pass
    
    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            prompt = f"<|image_1|>\n{question} Please answer in one word."
            messages = [{"role": "user", "content": prompt}]
            
            # Create chat-style prompt with special tokens
            text_prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize + process image(s) together
            inputs = self.processor(text_prompt, [image], return_tensors="pt").to(self.device)

            # Run inference
            outputs = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=20,
                temperature=0.0,
                do_sample=False,
            )

            # Strip off prompt tokens
            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            responses.append(response)
        return responses
    
    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
            answer = embedding.split()[0] if embedding else ""
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
            image,question,gt = batch['x'],batch['question'],batch['y'] 
            with torch.no_grad():
                embeddings=self.forward((image,question))
                answer=self.postprocess(embeddings)
                embeddings_np.append(answer)
                labels_np.append(gt)
        return embeddings_np,labels_np               

    