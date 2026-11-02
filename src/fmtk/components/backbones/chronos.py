from fmtk.components.base import BaseModel
from chronos import ChronosPipeline
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_chronos_embed_dim(model_name):
    """
    T5 encoder hidden size (d_model) for each Chronos-T5 variant, confirmed
    empirically via ChronosModel.forward() output shape (last dim) for
    tiny/mini/small/base; large follows the same published Chronos-T5 sizing.
    """
    return {
        "tiny": 256,
        "mini": 384,
        "small": 512,
        "base": 768,
        "large": 1024,
    }[model_name]


class ChronosModel(BaseModel):
    def __init__(self,device,model_name=None,model_config=None):
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
    
    def preprocess(self,batch_x,mask=None):

        x=batch_x.float()
        self.B, self.S, self.L = x.shape
        x = x.view(-1, self.L)

        if mask is not None:
            # FMTK's mask is [B, L] (one mask shared across all S channels,
            # e.g. PEMSDataset's history_len padding), 0 = padded/not-real.
            # Chronos has no separate mask argument -- its tokenizer instead
            # treats NaN values in the context as padding (excluded from its
            # mean-scaling stats and marked pad in the T5 attention_mask it
            # builds internally, confirmed via MeanScaleUniformBins._input_transform:
            # `attention_mask = ~torch.isnan(context)`), so the FMTK mask is
            # converted to that convention here rather than being dropped.
            mask = mask.to(x.device)
            mask_flat = mask.repeat_interleave(self.S, dim=0)  # [B*S, L], matches x's flattening order
            x = x.masked_fill(mask_flat == 0, float("nan"))

        return x, None

    def forward(self, batch_x, mask=None):
        x, mask=self.preprocess(batch_x, mask)
        embedding, _ = self.model.embed(x)
        output=self.postprocess(embedding)
        return output
    
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
            if len(batch)==3:
                x, mask, y = batch["x"], batch["mask"], batch["y"]
            else:
                x, y = batch["x"], batch["y"]
                mask=None
            with torch.no_grad():
                output=self.forward(x)   
            all_embeddings.append(output.cpu().detach().float().numpy())
            all_labels.append(y)
        embeddings_np = np.vstack(all_embeddings)
        labels_np = np.concatenate(all_labels)
        return embeddings_np, labels_np               

    