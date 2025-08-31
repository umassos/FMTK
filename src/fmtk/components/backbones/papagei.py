from fmtk.components.base  import BaseModel
import numpy as np
import torch
from fmtk.components.backbones.resnet import ResNet1D, ResNet1DMoE
from torch_ecg._preprocessors import Normalize
from fractions import Fraction
from scipy.signal import filtfilt, resample_poly
from math import gcd
from dotmap import DotMap
import os

class PapageiModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
        super().__init__()
        self.device=device
        base_dir = os.path.dirname(__file__)
        if model_name==None:
            model_name='papagei_p'
            
        if model_config==None:
            model_config={'in_channels':1, 
                'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512}
        if model_name=='papagei_p':
            model_path = os.path.join(base_dir, '../../../../weights', 'papagei_p.pt')
            model = ResNet1D(**model_config)
        elif model_name=='papagei_s':
            model_path = os.path.join(base_dir, '../../../../weights', 'papagei_s.pt')
            model = ResNet1DMoE(**model_config)
        elif model_name=='papagei_s_svri':
            model_path = os.path.join(base_dir, '../../../../weights', 'papagei_s_svri.pt')
            model = ResNet1D(**model_config)                       
        self.model = self.load_model_without_module_prefix(model,model_path )
        self.model.to(self.device)

    def preprocess(self,batch):
        """
        Match the shape and preprocess before sending it to model.
        Args:
            batch: batch from dataloader
        Returns:s 
            x: input
            y: output
        """
        if len(batch)==3:
            x, mask, y = batch
            x=x.float()
            self.B, self.S, self.L = x.shape
            x = x.view(-1, 1, self.L).to(self.device)
            return x, mask, y
        elif len(batch)==2:
            x, y = batch
            x=x.float()
            self.B, self.S, self.L = x.shape
            x = x.view(-1, 1, self.L).to(self.device)
            return x, None, y           

    def forward(self, batch):
        x, mask, y = self.preprocess(batch)
        embedding = self.model(x) 
        output=self.postprocess(embedding)
        return output,y

    def postprocess(self,embedding):
        embeddings = embedding[0]
        output = embeddings.view(self.B, self.S, -1)
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
        self.model.eval()
        all_embeddings,all_labels = [],[]

        # with torch.inference_mode():
        for batch in dataloader:
            with torch.no_grad():
                output,y = self.forward(batch) 
            all_embeddings.append(output.cpu().numpy())
            all_labels.append(y)

        embeddings_np = np.vstack(all_embeddings)
        labels_np = np.concatenate(all_labels)
        return embeddings_np, labels_np        

    
    def load_model_without_module_prefix(self, model, checkpoint_path):
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Create a new state_dict with the `module.` prefix removed
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_key = k[7:]  # Remove `module.` prefix
            else:
                new_key = k
            new_state_dict[new_key] = v
        
        # Load the new state_dict into the model
        model.load_state_dict(new_state_dict)

        return model
    