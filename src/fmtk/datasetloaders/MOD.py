import os

import numpy as np
import torch
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

from fmtk.datasetloaders.base import TimeSeriesDataset

# Per mod_task label cardinality, confirmed by scanning every split's .pt
# labels: vehicle_classification labels are {0..6} (7 vehicle/pedestrian
# types); distance_classification/speed_classification labels are {0..2}
# (3 bucketed ranges each), pulled from the "distance"/"speed" key of the
# label dict rather than the scalar label vehicle_classification uses.
NUM_CLASSES = {
    "vehicle_classification": 7,
    "distance_classification": 3,
    "speed_classification": 3,
}
LABEL_KEYS = {
    "distance_classification": "distance",
    "speed_classification": "speed",
}


class MODDataset(TimeSeriesDataset):
    """
    MOD dataset: multimodal (acoustic + seismic) roadside "shake" sensor
    recordings of passing vehicles/pedestrians, read from per-event .pt
    files indexed by `partitions/{mod_task}/{split}_index.txt` (paths
    relative to `dataset_path`).

    Each .pt file holds `{"label": ..., "data": {"shake": {"audio": [10, 1600],
    "seismic": [10, 20]}}}`. The 10 rows of each modality are 10 independent
    sensor channels (not sequential frames), matching this repo's WISDM
    x/y/z-channel convention -- so a sample becomes [n_channels=10, seq_len].

    Audio's native length (1600) exceeds MOMENT's pretrained context
    (seq_len=512), so it's linearly resampled down to `seq_len` by default;
    seismic (native 20) is left as-is unless `seq_len` is set explicitly.

    modality="both" concatenates audio and seismic along the channel axis
    (10 + 10 = 20 channels) instead of using just one -- since the two
    modalities have different native lengths, both are resampled to the
    same `seq_len` (default 512, matching audio's own default) before
    concatenation, so every channel in the combined tensor shares one
    timeline.

    Which of the three mod_task partitions is used determines both the
    index files read and how the label is extracted:
      - "vehicle_classification": label is a scalar int (7 classes)
      - "distance_classification": label is a dict; classify its "distance" key (3 classes)
      - "speed_classification": label is a dict; classify its "speed" key (3 classes)

    Parameters
    ----------
    dataset_cfg : dict
        dataset_path : str  root containing data/ and partitions/ (required)
        mod_task     : str  "vehicle_classification" | "distance_classification"
                             | "speed_classification"        (default: "vehicle_classification")
        modality     : str  "audio" | "seismic" | "both"      (default: "audio")
        seq_len      : int | None  output length per channel; None keeps
                             the modality's native length (ignored -- forced
                             to 512 -- when modality="both", since seismic
                             must be resampled to match audio's timeline)
                             (default: 512 for audio/both, None for seismic)
    task_cfg : dict
        task_type : "classification"
    split : str
        "train", "val", or "test"
    """

    MOD_TASKS = tuple(NUM_CLASSES.keys())
    MODALITIES = ("audio", "seismic", "both")

    def __init__(self, dataset_cfg, task_cfg, split="train", preprocess=True):
        super().__init__(dataset_cfg, task_cfg, split)

        self.dataset_path = dataset_cfg.get("dataset_path", "")
        assert self.dataset_path, "dataset_cfg['dataset_path'] must be set."

        self.mod_task = dataset_cfg.get("mod_task", "vehicle_classification")
        assert self.mod_task in self.MOD_TASKS, f"Unknown mod_task {self.mod_task!r}"
        self.num_classes = NUM_CLASSES[self.mod_task]

        self.modality = dataset_cfg.get("modality", "audio")
        assert self.modality in self.MODALITIES, f"Unknown modality {self.modality!r}"

        default_seq_len = 512 if self.modality in ("audio", "both") else None
        self.seq_len = dataset_cfg.get("seq_len", default_seq_len)

        index_path = os.path.join(
            self.dataset_path, "partitions", self.mod_task, f"{split}_index.txt"
        )
        assert os.path.isfile(index_path), f"Index file not found: {index_path}"
        with open(index_path) as f:
            self._rel_paths = [line.strip() for line in f if line.strip()]

        self.scaler = None
        self._data = None  # [N, n_channels, seq_len]
        self._labels = None  # [N]

        self._read_data()
        if preprocess:
            self.preprocess()

    def _extract_label(self, raw_label):
        if self.mod_task == "vehicle_classification":
            return int(raw_label)
        return int(raw_label[LABEL_KEYS[self.mod_task]])

    def _read_one_modality(self, obj, modality):
        x = obj["data"]["shake"][modality].numpy().astype(np.float32)
        x = x.reshape(-1, x.shape[-1])  # [10, native_len]
        if self.seq_len is not None and x.shape[-1] != self.seq_len:
            x = resample(x, self.seq_len, axis=-1).astype(np.float32)
        return x

    def _read_data(self):
        arrays, labels = [], []
        for rel in self._rel_paths:
            obj = torch.load(os.path.join(self.dataset_path, rel), map_location="cpu")
            if self.modality == "both":
                x = np.concatenate(
                    [self._read_one_modality(obj, "audio"), self._read_one_modality(obj, "seismic")],
                    axis=0,
                )  # [20, seq_len]
            else:
                x = self._read_one_modality(obj, self.modality)
            arrays.append(x)
            labels.append(self._extract_label(obj["label"]))

        self.n_channels = arrays[0].shape[0]
        self._data = np.stack(arrays, axis=0)  # [N, n_channels, seq_len]
        self._labels = np.asarray(labels, dtype=np.int64)

    def preprocess(self):
        n, c, l = self._data.shape
        flat = self._data.transpose(0, 2, 1).reshape(-1, c)  # [N*L, C]
        self.scaler = StandardScaler()
        flat = self.scaler.fit_transform(flat).astype(np.float32)
        self._data = flat.reshape(n, l, c).transpose(0, 2, 1)  # [N, C, L]

    def __len__(self):
        return len(self._rel_paths)

    def __getitem__(self, index):
        assert index < len(self)
        x = self._data[index]  # [n_channels, seq_len]
        y = self._labels[index]
        mask = np.ones(x.shape[-1], dtype=np.float32)
        return {"x": x, "mask": mask, "y": y, "idx": index}
