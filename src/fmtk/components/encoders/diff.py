#code from Mantis
import torch
from timeseries.components.base import BaseModel
from torch import nn


class LinearChannelCombiner(BaseModel,nn.Module):
    """
    A differentiable adapter that implements a linear projector along the channel axis.
    Given time series dataset with `num_channels`, it transforms it into a dataset with `new_num_channels`,
    where each new channel is a linear combination of the original ones.
    This adapter is a pytorch module, which can be trained together with the prediction head 
    or during the full fine-tuning of Mantis.

    Parameters
    ----------
    num_channels: int
        The original number of channels in a time series dataset.
    new_num_channels: int
        The number of channels after transformation.
    """
    def __init__(self, num_channels, new_num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.new_num_channels = new_num_channels
        self.model = nn.Linear(num_channels, new_num_channels)

    def preprocess(self,batch):
        x,y=batch
        x=x.to(torch.float32)
        # transpose time and channel dimensions to apply linear layer
        x_transposed = x.transpose(1, 2)
        return x_transposed,y

    def forward(self, batch):
        x_transposed,y=self.preprocess(batch)
        x_transformed = self.model(x_transposed)
        return self.postprocess(x_transformed), y
    
    def postprocess(self,embedding):
        # return the output transposing back the dimensions
        return embedding.transpose(1, 2)

    def trainable_parameters(self):
        return self.model.parameters()