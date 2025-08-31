import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm
from fmtk.components.base import BaseModel

class SVMDecoder(BaseModel):
    def __init__(self, max_samples: int = 10000):
        self.model = None
        self.max_samples = max_samples

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
        features, y = self._collect_from_loader(dataloader)
        nb_classes = np.unique(y, return_counts=True)[1].shape[0]
        train_size = features.shape[0]

        svm = SVC(C=100000, gamma="scale")
        if train_size // nb_classes < 5 or train_size < 50:
            self.model = svm.fit(features, y)
        else:
            grid_search = GridSearchCV(
                svm,
                {
                    "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                    "kernel": ["rbf"],
                    "degree": [3],
                    "gamma": ["scale"],
                    "coef0": [0],
                    "shrinking": [True],
                    "probability": [False],
                    "tol": [0.001],
                    "cache_size": [200],
                    "class_weight": [None],
                    "verbose": [False],
                    "max_iter": [10000000],
                    "decision_function_shape": ["ovr"],
                },
                cv=5,
                n_jobs=10,
            )
            if train_size > self.max_samples:
                features, _, y, _ = train_test_split(
                    features, y, train_size=self.max_samples, random_state=0, stratify=y
                )
            grid_search.fit(features, y)
            self.model = grid_search.best_estimator_

    def predict(self, dataloader: DataLoader):
        features, labels = self._collect_from_loader(dataloader)
        if self.model is None:
            raise ValueError("SVMDecoder: model not trained. Call fit() first.")
        return labels, self.model.predict(features)