from timeseries.components.base import BaseModel
from chronos import ChronosPipeline
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

class ChronosModel(BaseModel):
    def __init__(self,device,model_name=None):
        super().__init__()
        self.device=device
        if model_name=='large':
            model_path='amazon/chronos-t5-large'
        elif model_name=='base':
            model_path='amazon/chronos-t5-base'
        elif model_name=='small':     
            model_path='amazon/chronos-t5-small'
        elif model_name=='mini':
            model_path='amazon/chronos-t5-mini'
        elif model_name=='tiny':     
            model_path='amazon/chronos-t5-tiny' 
        else:
            model_path='amazon/chronos-t5-large'
            print("Model name not recognized, using default amazon/chronos-t5-large")

        print(f"[Chronos] Loading {model_path} on device {device}")
        self.model = ChronosPipeline.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16
        )
    
    def preprocess(self,batch):
        if len(batch)==3:
            x, mask, y = batch
            self.B, self.S, self.L = x.shape
            x = x.view(-1, self.L)
            x = x.float()
            return x, mask, y
        elif len(batch)==2:
            x, y = batch
            self.B, self.S, self.L = x.shape
            x = x.view(-1, self.L)
            x = x.float()
            return x,None, y      
    
    def forward(self, batch):
        x, mask, y=self.preprocess(batch)
        embedding, _ = self.model.embed(x)
        output=self.postprocess(embedding)
        return output,y
    
    def postprocess(self,embedding):
        _,E,_=embedding.shape #[batch size*segment size,token size, length]
        output =embedding.view(self.B,self.S,E,-1)
        return output
    
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
        # self.model.eval()
        self.model.model.eval()
        all_embeddings,all_labels = [],[]
        for batch in tqdm(dataloader,total=len(dataloader)):
            with torch.no_grad():
                output,y=self.forward(batch)   
            all_embeddings.append(output.cpu().detach().float().numpy())
            all_labels.append(y)
        embeddings_np = np.vstack(all_embeddings)
        labels_np = np.concatenate(all_labels)
        return embeddings_np, labels_np               

    