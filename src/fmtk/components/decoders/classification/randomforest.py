from sklearn.ensemble import RandomForestClassifier
from timeseries.components.base import BaseModel
from torch.utils.data import DataLoader
import numpy.typing as npt
import numpy as np
import torch
import torch.nn as nn

class RandomForestDecoder(BaseModel):
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

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
