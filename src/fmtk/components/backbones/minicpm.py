from transformers import AutoModel, AutoTokenizer
import os
import torch
import re
from fmtk.components.base import BaseModel
from tqdm import tqdm
from torchvision import transforms

class MinicpmModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        models_directory = os.path.join(base_dir, '../../../weights')
        if model_name=="minicpm":
            model_id='openbmb/MiniCPM-V-2_6'
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=models_directory)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=models_directory, attn_implementation="flash_attention_2").eval().cuda()
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
            question_prompt = question + " Please answer in one word."
            msgs = [{"role": "user", "content": [image, question_prompt]}]
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.processor,  # tokenizer = processor in this case
            )   

            responses.append(response)
        return responses
    
    def postprocess(self,embeddings):
        answers=[]
        for embedding in embeddings:
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
            image,question,gt = batch['x'],batch['question'],batch['y']
            with torch.no_grad():
                embeddings=self.forward((image,question))
                answer=self.postprocess(embeddings)
                embeddings_np.append(answer)
                labels_np.append(gt)
        return embeddings_np,labels_np               

    