from transformers import AutoModelForCausalLM
import os
import torch
import re
from fmtk.components.base import BaseModel
from tqdm import tqdm
from torchvision import transforms

class MoondreamModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../weights')
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
            response = self.model.query(image, question + " Please answer in one word.")["answer"]
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
            image,question,gt = batch['x'],batch['question'],batch['y'] 
            with torch.no_grad():
                answer=self.forward((image,question))
                embeddings_np.append(answer)
                labels_np.append(gt)
        return embeddings_np,labels_np               

    