import os

import numpy as np

from fmtk.datasetloaders.base import TimeSeriesDataset


class PEMSDataset(TimeSeriesDataset):
    """
    PEMS08 traffic flow dataset (Song et al. 2020, STSGCN): 170 Caltrans
    PeMS loop-detector sensors in the San Bernardino area, 5-minute
    interval, ~2 months (17,856 timesteps) of traffic flow/occupancy/
    speed. Only the flow channel is used (the standard single-feature
    setup for this benchmark across STSGCN/ASTGCN/Graph WaveNet and the
    LTSF-Transformer papers that reuse it), with each of the 170 sensors
    treated as an independent channel of one shared multivariate series --
    consistent with how MOMENT-based scripts elsewhere in this repo
    process channels independently.

    Read from the raw benchmark release (see download.py):
        {dataset_path}/PEMS08.npz  (numpy array 'data', shape [17856, 170, 3])

    A chronological 6:2:2 train/val/test split (Song et al.'s own
    convention, reused by every PEMS03/04/07/08 forecasting baseline) is
    applied first; sliding windows of length seq_len + pred_len are then
    cut independently within each split (stride=1 by default) so no
    window spans two splits.

    Per-channel z-score normalization uses train-split statistics only
    (fit once, applied to val/test) -- the standard LTSF convention.
    Unlike M4Dataset's per-window normalization (appropriate there
    because M4 mixes series of wildly different scales), PEMS08's 170
    sensors form one shared series where global per-channel stats are
    the right normalization unit.

    The full per-split series is kept as a single [N, split_T] array and
    windows are sliced lazily in __getitem__, rather than materializing
    every window eagerly -- at stride=1 the train split alone has
    ~10k overlapping 512+pred_len windows, which would otherwise mean
    duplicating most of the array dozens of times over.

    Parameters
    ----------
    dataset_cfg : dict
        dataset_path : str   directory containing PEMS08.npz            (required)
        seq_len      : int   context length fed to the model            (default: 512)
        pred_len     : int   forecast horizon                           (default: 12)
        stride       : int   step between consecutive windows           (default: 1)
    task_cfg : dict
        task_type : "forecasting"
    split : str
        "train", "val", or "test"
    """

    TRAIN_FRAC = 0.6
    VAL_FRAC = 0.2

    def __init__(self, dataset_cfg, task_cfg, split="train", preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.dataset_path = dataset_cfg.get("dataset_path", "")
        assert self.dataset_path, "dataset_cfg['dataset_path'] must be set."

        self.seq_len = dataset_cfg.get("seq_len", 512)
        self.pred_len = dataset_cfg.get("pred_len", 12)
        self.stride = dataset_cfg.get("stride", 1)
        self.forecast_horizon = self.pred_len  # matches M4Dataset's attribute name

        npz_path = os.path.join(self.dataset_path, "PEMS08.npz")
        assert os.path.isfile(npz_path), f"PEMS08.npz not found at {npz_path} (run download.py first)"
        raw = np.load(npz_path)["data"][:, :, 0].astype(np.float32)  # [T, N] -- flow channel only
        self.n_channels = raw.shape[1]

        T = raw.shape[0]
        n_train = int(T * self.TRAIN_FRAC)
        n_val = int(T * self.VAL_FRAC)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, T),
        }
        assert split in bounds, f"Unknown split {split!r}, expected one of {list(bounds)}"

        train_start, train_end = bounds["train"]
        train_slice = raw[train_start:train_end]
        self.mean = train_slice.mean(axis=0, keepdims=True)  # [1, N]
        self.std = train_slice.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0

        start, end = bounds[split]
        normed = (raw[start:end] - self.mean) / self.std  # [split_T, N]
        self.series = normed.T.copy()  # [N, split_T], channel-major for fast time-slicing

        window = self.seq_len + self.pred_len
        n_windows = max(0, (self.series.shape[1] - window) // self.stride + 1)
        self._starts = [i * self.stride for i in range(n_windows)]

    def __len__(self):
        return len(self._starts)

    def __getitem__(self, index):
        t = self._starts[index]
        x = self.series[:, t: t + self.seq_len]  # [N, seq_len]
        y = self.series[:, t + self.seq_len: t + self.seq_len + self.pred_len]  # [N, pred_len]
        mask = np.ones(self.seq_len, dtype=np.float32)
        return {"x": x, "mask": mask, "y": y, "idx": index}

    def preprocess(self):
        pass
