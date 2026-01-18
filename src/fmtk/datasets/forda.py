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
    ):
        """
        FordA Dataset for binary classification.
        Each row = one 500-length univariate time series sample.
        Labels are {-1, 1} → {0, 1}.
        """
        super().__init__(dataset_cfg, task_cfg, split)
        self.seq_len = 500
        # moved into dataset_cfg for convention consistency
        self.data_stride_len = dataset_cfg.get("data_stride_len", 1)
        self.random_seed = dataset_cfg.get("random_seed", 13)
        self.task_name = self.task_cfg["task_type"]
        self.full_file_path_and_name = (
            f"{self.dataset_cfg['dataset_path']}/FordA_{split.upper()}.txt"
        )
        self._read_data()

    def _read_data(self):
        data = np.loadtxt(self.full_file_path_and_name)
        np.random.seed(self.random_seed)

        # shuffle only training set for better generalization
        if self.split == "train":
            np.random.shuffle(data)

        y = data[:, 0].astype(int)
        X = data[:, 1:].astype(np.float32)

        # mapping because -1 caused issues
        self.labels = np.where(y == 1, 1, 0)

        # paths to save or load scaling statistics
        mean_path = os.path.join(self.dataset_cfg["dataset_path"], "forda_mean.npy")
        scale_path = os.path.join(self.dataset_cfg["dataset_path"], "forda_scale.npy")

        # fit scaler only on training data, but compute if missing
        if self.split == "train" or not (os.path.exists(mean_path) and os.path.exists(scale_path)):
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
        timeseries = self.data[index].reshape(1, self.seq_len)
        # per-series normalization
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
