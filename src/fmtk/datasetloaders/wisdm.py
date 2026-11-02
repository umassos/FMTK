import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from fmtk.datasetloaders.base import TimeSeriesDataset

RAW_FILENAME = "WISDM_ar_v1.1_raw.txt"


class WISDMDataset(TimeSeriesDataset):
    """
    WISDM v1.1 (Kwapisz et al. 2010) triaxial phone-accelerometer dataset,
    read from the raw `[user],[activity],[timestamp],[x],[y],[z];` file and
    windowed into fixed-length sequences for time-series tasks.

    The raw file has no session boundaries beyond (user, activity) changing,
    so contiguous runs of rows sharing the same (user, activity) are treated
    as one continuous recording and windowed independently -- windows are
    never slid across a user/activity change. Splits are by user (not by
    window) so no recording leaks between train and test.

    Currently only "imputation" is implemented as a task_type: __getitem__
    returns the raw window itself (`x`); masking for the reconstruction
    objective is applied by the training script, not the dataset.

    Parameters
    ----------
    dataset_cfg : dict
        dataset_path    : str   directory containing WISDM_ar_v1.1_raw.txt (required)
        seq_len         : int   window length in timesteps      (default: 512)
        data_stride_len : int   stride between windows           (default: seq_len, non-overlapping)
        train_frac      : float fraction of users used for train (default: 0.8)
        random_seed     : int   seed for the user train/test split (default: 42)
    task_cfg : dict
        task_type : "imputation"
    split : str
        "train" or "test"
    """

    ACTIVITIES = ["Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs"]

    def __init__(self, dataset_cfg, task_cfg, split="train", preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.dataset_path = dataset_cfg.get("dataset_path", "")
        assert self.dataset_path, "dataset_cfg['dataset_path'] must be set."

        self.seq_len = dataset_cfg.get("seq_len", 512)
        self.data_stride_len = dataset_cfg.get("data_stride_len", self.seq_len)
        self.train_frac = dataset_cfg.get("train_frac", 0.8)
        self.random_seed = dataset_cfg.get("random_seed", 42)
        self.n_channels = 3  # x, y, z accelerometer axes

        self.task_name = task_cfg["task_type"]

        self.scaler = None
        self._windows = []  # list of (channel_first_array [3, session_len], start)

        self._read_data()

        if preprocess:
            self.preprocess()

    def _parse_raw_file(self):
        """Yields (user, activity, x, y, z) tuples, skipping malformed rows."""
        path = os.path.join(self.dataset_path, RAW_FILENAME)
        assert os.path.isfile(path), f"WISDM raw file not found: {path}"

        with open(path) as f:
            for line in f:
                line = line.strip().rstrip(";").rstrip(",")
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 6:
                    continue
                try:
                    user = int(parts[0])
                    activity = parts[1]
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                except ValueError:
                    continue
                if activity not in self.ACTIVITIES:
                    continue
                yield user, activity, x, y, z

    def _read_data(self):
        # Group contiguous same-(user, activity) rows into sessions, per user.
        sessions_by_user = {}
        current_key = None
        current_rows = []

        def flush():
            if current_key is not None and len(current_rows) >= self.seq_len:
                user = current_key[0]
                arr = np.asarray(current_rows, dtype=np.float32).T  # [3, session_len]
                sessions_by_user.setdefault(user, []).append(arr)

        for user, activity, x, y, z in self._parse_raw_file():
            key = (user, activity)
            if key != current_key:
                flush()
                current_key = key
                current_rows = []
            current_rows.append((x, y, z))
        flush()

        users = sorted(sessions_by_user.keys())
        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(users)
        n_train = max(1, int(len(users) * self.train_frac))
        train_users = set(users[:n_train])
        test_users = set(users[n_train:])

        split_users = train_users if self.split == "train" else test_users

        # Always fit the scaler on train-user sessions, even when building
        # the test split, so test windows are normalized with train stats.
        train_concat = np.concatenate(
            [sess for u in train_users for sess in sessions_by_user[u]], axis=1
        )
        self.scaler = StandardScaler()
        self.scaler.fit(train_concat.T)

        for u in split_users:
            for sess in sessions_by_user[u]:
                sess_norm = self.scaler.transform(sess.T).T.astype(np.float32)  # [3, session_len]
                session_len = sess_norm.shape[1]
                n_windows = (session_len - self.seq_len) // self.data_stride_len + 1
                for w in range(n_windows):
                    start = w * self.data_stride_len
                    self._windows.append((sess_norm, start))

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, index):
        arr, start = self._windows[index]
        window = arr[:, start:start + self.seq_len]  # [3, seq_len]
        return {"x": window, "idx": index}

    def preprocess(self):
        pass
