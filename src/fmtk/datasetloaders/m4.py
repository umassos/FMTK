import os

import numpy as np
import pandas as pd

from fmtk.datasetloaders.base import TimeSeriesDataset

# Standard M4 competition forecast horizon per frequency subset (fixed by
# the competition, matches the column count of each Test/{freq}-test.csv).
FREQUENCY_HORIZONS = {
    "Yearly": 6,
    "Quarterly": 8,
    "Monthly": 18,
    "Weekly": 13,
    "Daily": 14,
    "Hourly": 48,
}


class M4Dataset(TimeSeriesDataset):
    """
    M4 competition dataset (Makridakis et al. 2018): ~100,000 univariate
    time series across 6 frequency subsets (Yearly/Quarterly/Monthly/
    Weekly/Daily/Hourly), each with its own fixed forecast horizon.

    Read from the official M4-methods CSV release:
        {dataset_path}/Train/{frequency}-train.csv
        {dataset_path}/Test/{frequency}-test.csv
    Each row is one series (ragged, NaN-padded to a common column count).
    The train file holds each series' history; the test file holds
    exactly `horizon` held-out future values per series -- the real M4
    competition evaluation target.

    Per-series z-score normalization (mean/std computed from that
    series' own context window) is applied to both x and y, since M4
    mixes series of wildly different scales/units -- without it,
    cross-series metrics would be dominated by whichever series happen
    to have the largest magnitude. Reported metrics (sMAPE/MAE) are
    therefore in per-series-normalized units: useful for comparing
    baselines against each other within this project, but NOT directly
    comparable to official M4 leaderboard sMAPE (computed in each
    series' original units).

    split == "train": context = series[:-horizon], target = series[-horizon:]
        (last-block-out on the train file only -- never touches the real
        test file, so there is no leakage).
    split == "val"/"test": context = full train-file series (all
        available history), target = the real M4 test-file horizon for
        that series (the actual competition ground truth).

    Parameters
    ----------
    dataset_cfg : dict
        dataset_path : str  root containing Train/ and Test/           (required)
        frequency    : str  "Yearly" | "Quarterly" | "Monthly" |
                             "Weekly" | "Daily" | "Hourly"              (default: "Monthly")
        seq_len      : int  context length fed to the model; shorter
                             series are left-padded with zeros (masked) (default: 512)
    task_cfg : dict
        task_type : "forecasting"
    split : str
        "train", "val", or "test"
    """

    FREQUENCIES = tuple(FREQUENCY_HORIZONS.keys())

    def __init__(self, dataset_cfg, task_cfg, split="train", preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.dataset_path = dataset_cfg.get("dataset_path", "")
        assert self.dataset_path, "dataset_cfg['dataset_path'] must be set."

        self.frequency = dataset_cfg.get("frequency", "Monthly")
        assert self.frequency in self.FREQUENCIES, f"Unknown frequency {self.frequency!r}"
        self.forecast_horizon = FREQUENCY_HORIZONS[self.frequency]

        self.seq_len = dataset_cfg.get("seq_len", 512)
        self.n_channels = 1  # M4 series are univariate

        self._samples = []  # list of (x [1, seq_len], mask [seq_len], y [1, horizon]), all normalized
        self._read_data()

    def _read_data(self):
        train_path = os.path.join(self.dataset_path, "Train", f"{self.frequency}-train.csv")
        assert os.path.isfile(train_path), f"Train file not found: {train_path}"
        train_df = pd.read_csv(train_path, index_col=0)

        test_df = None
        if self.split in ("val", "test"):
            test_path = os.path.join(self.dataset_path, "Test", f"{self.frequency}-test.csv")
            assert os.path.isfile(test_path), f"Test file not found: {test_path}"
            test_df = pd.read_csv(test_path, index_col=0)

        for series_id, row in train_df.iterrows():
            values = row.dropna().to_numpy(dtype=np.float32)
            if len(values) <= self.forecast_horizon:
                continue  # too short to hold out a full horizon as context

            if self.split == "train":
                context = values[:-self.forecast_horizon]
                target = values[-self.forecast_horizon:]
            else:
                context = values
                target = test_df.loc[series_id].dropna().to_numpy(dtype=np.float32)

            mean = context.mean()
            std = context.std()
            if std < 1e-6:
                std = 1.0
            context_norm = (context - mean) / std
            target_norm = (target - mean) / std

            L = len(context_norm)
            if L >= self.seq_len:
                window = context_norm[-self.seq_len:]
                mask = np.ones(self.seq_len, dtype=np.float32)
            else:
                pad = self.seq_len - L
                window = np.concatenate([np.zeros(pad, dtype=np.float32), context_norm]).astype(np.float32)
                mask = np.concatenate([np.zeros(pad, dtype=np.float32), np.ones(L, dtype=np.float32)])

            self._samples.append((
                window[np.newaxis, :],          # [1, seq_len]
                mask,                            # [seq_len]
                target_norm[np.newaxis, :].astype(np.float32),  # [1, horizon]
            ))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        x, mask, y = self._samples[index]
        return {"x": x, "mask": mask, "y": y, "idx": index}

    def preprocess(self):
        pass
