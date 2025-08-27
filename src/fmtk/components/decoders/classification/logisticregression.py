from sklearn.linear_model import LogisticRegression
from timeseries.components.base import BaseModel
from torch.utils.data import DataLoader
import numpy.typing as npt
import numpy as np
import torch
import torch.nn as nn

class LogisticRegressionDecoder(BaseModel):
    def __init__(self, max_iter: int = 10000):
        self.model = LogisticRegression(solver='lbfgs',max_iter=max_iter,multi_class='ovr')

    def preprocess(self,batch):
        pass
    
    def postprocess(self,embedding):
        pass
    
    def forward(self, batch):
        pass

    def _collect_from_loader(self, dataloader: DataLoader):
        features, labels = [], []
        for x, y in dataloader:
            if x.ndimension() == 4:
                x=x.mean(dim=2)
            if x.ndimension() == 3:
                x=x.mean(dim=1)
            features.append(x.cpu().numpy())
            labels.append(y.cpu().numpy())
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    def fit(self, dataloader: DataLoader,cfg):
        X, y = self._collect_from_loader(dataloader)
        self.model.fit(X, y)

    def predict(self, dataloader: DataLoader) -> tuple[npt.NDArray, npt.NDArray]:
        X, y_true = self._collect_from_loader(dataloader)
        y_pred = self.model.predict(X)
        return y_true, y_pred
