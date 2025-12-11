from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from fmtk.datasets.base import TimeSeriesDataset
import os

class FordADataset(TimeSeriesDataset):
    def __init__(
        self,
        dataset_cfg,
        task_cfg,
        split,
        data_stride_len: int = 1,
        random_seed: int = 13,
    ):
        """
        FordA Dataset for binary classification.
        Each row = one 500-length univariate time series sample.
        Labels are {-1, 1} → {0, 1}.
        """
        super().__init__(dataset_cfg, task_cfg, split)
        self.seq_len = 500
        self.data_stride_len = data_stride_len
        self.random_seed = random_seed
        self.task_name = self.task_cfg["task_type"]
        self.full_file_path_and_name = (
            f"{self.dataset_cfg['dataset_path']}/FordA_{split.upper()}.txt"
        )
        self._read_data()

    def _read_data(self):
        data = np.loadtxt(self.full_file_path_and_name)
        np.random.seed(self.random_seed)

        #shuffle only training set for better generalization
        if self.split == "train":
            np.random.shuffle(data)

        y = data[:, 0].astype(int)
        X = data[:, 1:].astype(np.float32)

        # mapping because -1 caused issues
        self.labels = np.where(y == 1, 1, 0)

        #paths to save or load scaling statistics
        mean_path = os.path.join(self.dataset_cfg["dataset_path"], "forda_mean.npy")
        scale_path = os.path.join(self.dataset_cfg["dataset_path"], "forda_scale.npy")

        #fit scaler only on training data
        if self.split == "train":
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            np.save(mean_path, self.scaler.mean_)
            np.save(scale_path, self.scaler.scale_)
        else:
            mean = np.load(mean_path)
            scale = np.load(scale_path)
            self.scaler = StandardScaler()
            self.scaler.mean_ = mean
            self.scaler.scale_ = scale

        X = self.scaler.transform(X)
        self.data = X.astype(np.float32)

    def __getitem__(self, index):
        """
        Each sample = one 500-length univariate time series.
        Returns (1, 500) tensor-like array and corresponding label.
        """
        #each index returns one complete ts as in one row, with each row corresponding to one label, these were missaligned previously
        timeseries = self.data[index].reshape(1, self.seq_len) 
        #pper-series normalization
        timeseries = (timeseries - timeseries.mean()) / (timeseries.std() + 1e-8)
        label = self.labels[index]
        return {
            "x": timeseries,
            "y": label,
        }

    def __len__(self):
        return len(self.labels)

    def preprocess(self):
        pass
