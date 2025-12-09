from fmtk.components.base import BaseModel
from momentfm import MOMENTPipeline
import torch
import numpy as np
import pandas as pd
import inspect
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

class MomentModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        if model_name=='large':
            model_path='AutonLab/MOMENT-1-large'
        elif model_name=='base':
            model_path='AutonLab/MOMENT-1-base'
        elif model_name=='small':     
            model_path='AutonLab/MOMENT-1-small'
        else:
            model_path='AutonLab/MOMENT-1-large'
            print("Model name not recognized, using default AutonLab/MOMENT-1-large")

        self.peft_enable=False
        print(f"[Moment] Loading {model_path} on device {device}")
        self.model=MOMENTPipeline.from_pretrained(
            f'{model_path}', 
            model_kwargs={'task_name': 'embedding','enable_gradient_checkpointing': False}, # We are loading the model in `embedding` mode to learn representations
            )
        
        self.model.init()     
        self.model.to(device)     

    def preprocess(self,batch_x,mask=None):
        """
        Match the shape and preprocess before sending it to model.
        Args:
            batch: batch from dataloader
        Returns:s 
            x: input
            y: output
        """
        if mask is not None:
            mask=mask.to(self.device)
        
        x=batch_x.float()
        self.B, self.S, self.L = x.shape
        x=x.to(self.device)
        return x, mask   


    def forward(self, batch_x, mask=None):
        x, mask=self.preprocess(batch_x,mask)
        if mask is None:
            embedding=self.model(x_enc=x, reduction="none").embeddings
        else:
            embedding=self.model(x_enc=x, input_mask=mask, reduction="none").embeddings
        return embedding
    
    def postprocess(self, embedding):
        pass

    @torch.no_grad()
    def predict(self, dataloader: DataLoader):
        """
        Compute embeddings (no grad) for decoder-only training/inference.
        Returns: embeddings_np [N,E], labels_np [N]
        """
        self.model.eval()
        all_embeddings, all_labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader)):
            if len(batch)==3:
                x, mask, y = batch["x"], batch["mask"], batch["y"]
            else:
                x, y = batch["x"], batch["y"]
                mask=None
            with torch.no_grad():
                output=self.forward(x)   
            all_embeddings.append(output.cpu().float().numpy())
            all_labels.append(y)
        embeddings_np = np.vstack(all_embeddings)
        labels_np = np.concatenate(all_labels)
        return embeddings_np, labels_np
    
    def enable_peft(self,peft_cfg):
        self.model = get_peft_model(self.model, peft_cfg)
        self.peft_enable=True
    
    def adapter_trainable_parameters(self):
        if not self.peft_enable:
            return []
        return (p for p in self.model.parameters() if p.requires_grad)

    