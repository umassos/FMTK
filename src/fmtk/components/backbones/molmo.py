from transformers import AutoProcessor, AutoModelForCausalLM
import os
import torch
import re
from fmtk.components.base import BaseModel
from tqdm import tqdm
from torchvision import transforms

class MolmoModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../weights')
        if model_name=="molmo":
            model_id='allenai/Molmo-7B-D-0924'

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=models_directory, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager", device_map={"": self.device})

    def preprocess(self,batch_x,mask=None):
        pass
    
    def forward(self, batch_x, mask=None):
        batch_x_image,batch_x_question=batch_x
        responses=[]
        for image, question in zip(batch_x_image, batch_x_question):
            if isinstance(image, torch.Tensor):
                to_pil = transforms.ToPILImage()
                image = to_pil(image)
            processed = self.processor.process(images=[image], text=question+ " Please answer in one word.")
            inputs = {k: v.to(self.device).unsqueeze(0) for k, v in processed.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            generated_token_ids = torch.cat([inputs["input_ids"], next_token_id[:, None]], dim=1)
            response = self.processor.tokenizer.decode(generated_token_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            responses.append(response)
        return responses
    
    def postprocess(self,embeddings):
        pass

    
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
                answer=self.forward((image,question))
                embeddings_np.append(answer)
                labels_np.append(gt)
        return embeddings_np,labels_np               

    