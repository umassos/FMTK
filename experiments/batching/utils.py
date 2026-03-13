"""
Utility helpers for the batching experiment.

  ensure_decoder_weights()  — creates random-init decoder.pth if missing
  make_synthetic_pool()     — random float32 arrays of shape (N, 1, seq_len)
  make_synthetic_pool_short()— same but content only in first short_len cols
  load_uwavegesture_pool()  — load UWaveGesture test set, crop/pad to seq_len
  load_ecg5000_pool()       — load ECG5000 test set, crop/pad to seq_len
  make_short_pool()         — zero-out the tail of an existing pool (for diff-len runs)
  construct_batch()         — sample from data pools and assemble one batch

NOTE: sys.path must already include FMaaS-motivation/serving before importing
      this module (run_batching.py handles that at the top).
"""

import os
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Decoder weight helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_decoder_weights(decoder_specs: list, backbone: str, fmtk_root: str) -> list:
    """
    For every decoder spec, check whether decoder.pth exists under
    models/tsfm/finetuned/<path>/decoder.pth.  If it does not exist,
    initialise the MLP with random weights and save it so that
    ModelLoader.load_models(train=False) can proceed.

    Returns the list of paths where dummy weights were created (empty if all
    pre-trained weights were present).

    Latency measurements remain valid even with random weights; accuracy
    measurements with random weights are meaningless and will be flagged.
    """
    # Lazy imports — device.config requires FMaaS-motivation/serving on sys.path
    from device.config import DECODERS
    from fmtk.components.decoders.regression.mlp import MLPDecoder as RegressionMLP
    from fmtk.components.decoders.classification.mlp import MLPDecoder as ClassificationMLP

    models_dir = os.path.join(fmtk_root, 'models', 'tsfm', 'finetuned')
    cpu = torch.device('cpu')
    created = []

    for spec in decoder_specs:
        task  = spec['task']
        dtype = spec['type']
        path  = spec['path']

        decoder_pth = os.path.join(models_dir, path, 'decoder.pth')
        if os.path.exists(decoder_pth):
            continue

        print(f"[ensure_weights] No trained weights at {decoder_pth}")
        print(f"[ensure_weights]  → Creating random-init weights for '{path}'")
        os.makedirs(os.path.join(models_dir, path), exist_ok=True)

        if dtype == 'regression':
            key = f'mlp_{backbone}_regression'
            cfg = DECODERS[key]['decoder_config']['cfg']
            dec = RegressionMLP(device=cpu, cfg=cfg)
        elif dtype == 'classification':
            key = f'mlp_{backbone}_{task}'
            cfg = DECODERS[key]['decoder_config']['cfg']
            dec = ClassificationMLP(device=cpu, cfg=cfg)
        else:
            raise ValueError(f"Unknown decoder type: {dtype}")

        torch.save(dec.model.state_dict(), decoder_pth)
        # Write a marker file so subsequent runs can distinguish dummy weights
        # from genuinely pre-trained weights.
        with open(decoder_pth + '.dummy', 'w') as _f:
            _f.write('random_init\n')
        created.append(path)

    return created


def is_dummy_decoder(path: str, fmtk_root: str) -> bool:
    """Return True if the decoder.pth at this path was created by ensure_decoder_weights
    (i.e., it is randomly initialized, not pre-trained)."""
    pth = os.path.join(fmtk_root, 'models', 'tsfm', 'finetuned', path, 'decoder.pth')
    return os.path.exists(pth + '.dummy')


# ─────────────────────────────────────────────────────────────────────────────
# Data pool builders
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_pool(
    n_samples: int, seq_len: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (x, mask, y) where:
      x    : (N, 1, seq_len)  float32  standard-normal
      mask : (N, seq_len)     float32  all ones
      y    : (N,)             float32  zeros (no real labels)
    """
    rng  = np.random.default_rng(seed)
    x    = rng.standard_normal((n_samples, 1, seq_len)).astype(np.float32)
    mask = np.ones((n_samples, seq_len), dtype=np.float32)
    y    = np.zeros(n_samples, dtype=np.float32)
    return x, mask, y


def make_synthetic_pool_short(
    n_samples: int, seq_len: int, short_len: int, seed: int = 43
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Like make_synthetic_pool but real content only in [:, :, :short_len].
    Positions short_len: are zero-padded with mask = 0.

    Returns (x, mask, y) same shapes as make_synthetic_pool.
    """
    rng  = np.random.default_rng(seed)
    x    = np.zeros((n_samples, 1, seq_len), dtype=np.float32)
    x[:, :, :short_len] = rng.standard_normal((n_samples, 1, short_len)).astype(np.float32)
    mask = np.zeros((n_samples, seq_len), dtype=np.float32)
    mask[:, :short_len] = 1.0
    y    = np.zeros(n_samples, dtype=np.float32)
    return x, mask, y


def _normalise_to_seqlen(
    x: np.ndarray, m: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure a single sample (1, L) / (L,) pair is exactly (1, seq_len) / (seq_len,).
    Crops from the end if longer; zero-pads the front if shorter (left-pad,
    consistent with the dataset loaders).
    """
    L = x.shape[-1]
    if L == seq_len:
        return x, m
    if L > seq_len:
        return x[:, -seq_len:], m[-seq_len:]
    # L < seq_len: left-pad
    pad    = seq_len - L
    x_new  = np.zeros((1, seq_len), dtype=np.float32)
    x_new[:, pad:] = x
    m_new  = np.zeros(seq_len, dtype=np.float32)
    m_new[pad:] = m
    return x_new, m_new


def load_uwavegesture_pool(
    fmtk_root: str, seq_len: int = 512, split: str = 'test'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load UWaveGestureLibraryAll from the FMTK dataset directory.
    All sequences are normalised to (1, seq_len) with matching mask.

    Returns (x, mask, y):
      x    : (N, 1, seq_len)  float32
      mask : (N, seq_len)     float32
      y    : (N,)             int64    class indices 0-7
    """
    from fmtk.datasetloaders.uwavegesture import UWaveGestureLibraryALLDataset

    ds = UWaveGestureLibraryALLDataset(
        dataset_cfg={'dataset_type': 'UWaveGestureLibraryAll'},
        task_cfg={'task_type': 'classification'},
        split=split,
    )

    xs, masks, ys = [], [], []
    for i in range(len(ds)):
        item = ds[i]
        x, m, y = item['x'], item['mask'], item['y']   # (1,L), (L,), int
        x, m = _normalise_to_seqlen(x.astype(np.float32), m.astype(np.float32), seq_len)
        xs.append(x)
        masks.append(m)
        ys.append(y)

    return np.stack(xs), np.stack(masks), np.array(ys, dtype=np.int64)


def load_ecg5000_pool(
    fmtk_root: str, seq_len: int = 512, split: str = 'test'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ECG5000 from the FMTK dataset directory.
    ECG5000 native length (~140) is already padded to seq_len=512 in the loader.

    Returns (x, mask, y):
      x    : (N, 1, seq_len)  float32
      mask : (N, seq_len)     float32
      y    : (N,)             int64    class indices 0-4
    """
    from fmtk.datasetloaders.ecg5000 import ECG5000Dataset

    ds = ECG5000Dataset(
        dataset_cfg={'dataset_type': 'ECG5000'},
        task_cfg={'task_type': 'classification'},
        split=split,
    )

    xs, masks, ys = [], [], []
    for i in range(len(ds)):
        item = ds[i]
        x, m, y = item['x'], item['mask'], item['y']
        x, m = _normalise_to_seqlen(x.astype(np.float32), m.astype(np.float32), seq_len)
        xs.append(x)
        masks.append(m)
        ys.append(int(y))

    return np.stack(xs), np.stack(masks), np.array(ys, dtype=np.int64)


def make_short_pool(
    x_pool: np.ndarray,
    mask_pool: np.ndarray,
    y_pool: np.ndarray,
    short_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return a copy of the pool where content beyond short_len is zeroed
    and the mask marks those positions as invalid (0).

    Useful to create "diff-seq-len" variants from an existing full-length pool.
    """
    x    = x_pool.copy()
    mask = mask_pool.copy()
    x[:, :, short_len:]  = 0.0
    mask[:, short_len:]  = 0.0
    return x, mask, y_pool.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Batch construction
# ─────────────────────────────────────────────────────────────────────────────

def construct_batch(
    data_pools: dict,
    task_sequence: list,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Sample batch_size items from data_pools by cycling through task_sequence.

    task_sequence entries ending in "_short" draw from the corresponding
    short pool (e.g. "heartrate_short" → data_pools["heartrate_short"]) but
    route to the base task name for decoder lookup
    (e.g. task_names will contain "heartrate", not "heartrate_short").

    Args:
        data_pools    : dict  task_key → (x_arr, mask_arr, y_arr)
        task_sequence : list  e.g. ["heartrate", "gestureclass"] — cycled mod batch_size
        batch_size    : int
        rng           : numpy Generator for reproducible sampling

    Returns:
        x_batch    : (batch_size, 1, seq_len)  float32
        mask_batch : (batch_size, seq_len)     float32
        task_names : list[str]  length batch_size — routing task names (no "_short")
        y_batch    : list  length batch_size — ground-truth labels
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x_list, mask_list, task_names, y_list = [], [], [], []

    for i in range(batch_size):
        pool_key     = task_sequence[i % len(task_sequence)]
        routing_task = pool_key.replace('_short', '')

        x_pool, m_pool, y_pool = data_pools[pool_key]
        idx = int(rng.integers(0, len(x_pool)))

        x_list.append(x_pool[idx])
        mask_list.append(m_pool[idx])
        task_names.append(routing_task)
        y_list.append(y_pool[idx])

    x_batch    = np.stack(x_list)    # (batch_size, 1, seq_len)
    mask_batch = np.stack(mask_list) # (batch_size, seq_len)
    return x_batch, mask_batch, task_names, y_list
