import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy.typing as npt
from fmtk.components.base import BaseModel

class RidgeDecoder(BaseModel):
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

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

    def fit(self, dataloader: DataLoader, cfg):
        X_train, Y_train = self._collect_from_loader(dataloader)
        X_train = self.scaler.fit_transform(X_train)
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],  # Regularization strength
            'solver': ['auto', 'cholesky', 'sparse_cg']  # Solver to use in the computational routines
        }
        estimator = Ridge()
        grid_search = GridSearchCV(estimator=estimator,
                                   param_grid=param_grid,
                                   cv=4,
                                   scoring='neg_mean_squared_error',
                                   verbose=2,
                                   n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        self.model = grid_search.best_estimator_

    def predict(self, dataloader: DataLoader) -> tuple[npt.NDArray, npt.NDArray]:
        X_test, Y_test = self._collect_from_loader(dataloader)
        X_test = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test)
        return Y_test, y_pred

    def preprocess(self,batch):
        pass
    
    def postprocess(self,embedding):
        pass
    
    def forward(self, batch):
        pass