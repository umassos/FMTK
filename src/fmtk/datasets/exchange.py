from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fmtk.datasets.base import TimeSeriesDataset

import os

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
dataset_path = os.path.join(root_dir, "dataset/Exchange")

class ExchangeDataset(TimeSeriesDataset):

    def __init__(
        self,
        dataset_cfg,
        task_cfg,
        split,
        forecast_horizon: Optional[int] = 192,
        data_stride_len: int = 1,
        random_seed: int = 13,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        target_col: Optional[str] = "OT",
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        split : str
            'train', 'val', or 'test'.
        data_stride_len : int
            Stride for sliding windows.
        random_seed : int
            For reproducibility (if needed).
        two_year_start : str or None
            Start of the 2-year window (YYYY-MM-DD). If None, inferred from data start.
        hourly_agg : {'sum','mean'}
            Aggregation when going from 15-min to hourly. For energy (kWh) use 'sum' and
            multiply 15-min kW by 0.25 before summing.
        """
        super().__init__(dataset_cfg, task_cfg, split)
        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = f"{dataset_path}/exchange_rate.csv"
        self.data_stride_len = data_stride_len
        self.task_name = self.task_cfg['task_type']
        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio     
        self.target_col = target_col   
        self._read_data()

    def _get_borders(self):
        n_train = int(self.train_ratio * self.length_timeseries_original)
        n_test = int(self.test_ratio * self.length_timeseries_original)
        n_val = self.length_timeseries_original - n_train - n_test

        train_end = n_train
        val_start = train_end - self.seq_len
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        val = slice(val_start, val_end)
        test = slice(test_start, test_end)
        return train, val, test

    def _read_data(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects().interpolate(method="cubic")

        df = df[[self.target_col]]
        self.n_channels = 1

        data_splits = self._get_borders()

        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.split == "train":
            self.data = df[data_splits[0], :]
        elif self.split == "val":
            self.data = df[data_splits[1], :]
        elif self.split == "test":
            self.data = df[data_splits[2], :]
        
        self.length_timeseries = self.data.shape[0]
    
    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T

            return {
                'x':timeseries,
                'y':forecast,
            }

        elif self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T

            return {
                'x':timeseries,
                'mask':input_mask,
            }
    def __len__(self):
        if self.task_name == "imputation":
            return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        elif self.task_name == "forecasting":
            return (
                self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1

    def preprocess(self):
        pass