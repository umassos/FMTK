#!/usr/bin/env python3
"""
TSFM Batching Experiment
========================
Measures how batching impacts inference latency, memory, and accuracy for
TSFM workloads sharing a MOMENT-base backbone.

Five experiment runs:
  run1_baseline          — single task / single decoder, varying batch size
  run2_same_dec_same_len — two regression tasks (heartrate + sysbp), same arch
  run3_same_dec_diff_len — same decoder, different effective seq_len (512 vs 256)
  run4_diff_dec_same_len — regression + classification decoder, same seq_len
  run5_diff_dec_diff_len — regression + classification decoder, mixed seq_len

Metrics captured per batch (from BatchRunResult):
  backbone_latency_ms     — backbone forward time
  total_latency_ms        — end-to-end time (backbone + all decoders)
  per_request_latency_ms  — total_latency / batch_size
  mean_decoder_latency_ms — average per-sample decoder forward time
  sum_decoder_latency_ms  — total decoder time across all samples in batch
  mean_swap_latency_ms    — average per-sample decoder lookup/swap overhead
  gpu_peak_mb             — peak GPU memory during batch

Usage examples:
  # Full experiment (all 5 runs, default 20 reps):
  python run_batching.py

  # Quick smoke test (run1 only, batch sizes 1 and 4, 3 reps):
  python run_batching.py --runs run1_baseline --batch-sizes 1 4 --reps 3

  # Skip accuracy pass:
  python run_batching.py --no-accuracy
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
# Must happen before any serving-code or fmtk imports.

_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FMTK_ROOT      = os.path.abspath(os.path.join(_EXPERIMENT_DIR, '..', '..'))

# FMaaS-motivation/serving must be on sys.path so that
# `from device.runtime import ...` and `from device.config import ...` resolve.
_FMAAS_SERVING = os.path.abspath(
    os.path.join(_FMTK_ROOT, '..', 'FMaaS-motivation', 'serving')
)
for _p in [_FMAAS_SERVING]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Config / utils imports ────────────────────────────────────────────────────
from config import (
    BACKBONE, SEQ_LEN, SHORT_SEQ_LEN,
    BATCH_SIZES, NUM_REPS, WARMUP_REPS, N_POOL_SAMPLES,
    DECODER_SPECS, RUN_CONFIGS,
    FMTK_ROOT, OUTPUT_DIR, TIMING_CSV, ACCURACY_CSV,
)
from utils import (
    ensure_decoder_weights, is_dummy_decoder,
    make_synthetic_pool, make_synthetic_pool_short,
    load_uwavegesture_pool, load_ecg5000_pool,
    make_short_pool, construct_batch,
)

# ── Serving-code imports (no modification to these files) ────────────────────
from device.runtime     import PyTorchRuntime
from device.model_loader import ModelLoader


# ─────────────────────────────────────────────────────────────────────────────
# Data pool builder
# ─────────────────────────────────────────────────────────────────────────────

def build_data_pools(fmtk_root: str, seq_len: int, short_len: int,
                     n_synthetic: int) -> dict:
    """
    Build all data pools needed by the five experiment runs.

    Returns a dict: pool_key → (x_arr, mask_arr, y_arr)
      x_arr    : (N, 1, seq_len)  float32
      mask_arr : (N, seq_len)     float32
      y_arr    : (N,)             numeric labels
    """
    print("\n[data] Building data pools...")
    pools = {}

    # ── heartrate / sysbp (regression on PPG-data) ────────────────────────────
    # PPG preprocessing requires pyPPG/torch_ecg; no decoder.pth exists yet.
    # We use synthetic random data — timing measurements are unaffected.
    x_hr, m_hr, y_hr = make_synthetic_pool(n_synthetic, seq_len, seed=1)
    x_sy, m_sy, y_sy = make_synthetic_pool(n_synthetic, seq_len, seed=2)
    pools['heartrate'] = (x_hr, m_hr, y_hr)
    pools['sysbp']     = (x_sy, m_sy, y_sy)
    print(f"  heartrate : {x_hr.shape}  (synthetic — no PPG weights)")
    print(f"  sysbp     : {x_sy.shape}  (synthetic — no PPG weights)")

    # ── heartrate_short (same regression decoder, effective len=short_len) ────
    x_hs, m_hs, y_hs = make_synthetic_pool_short(n_synthetic, seq_len, short_len, seed=3)
    pools['heartrate_short'] = (x_hs, m_hs, y_hs)
    print(f"  heartrate_short : {x_hs.shape}  "
          f"(synthetic, real content in [:, :, :{short_len}])")

    # ── gestureclass (classification on UWaveGestureLibraryAll) ──────────────
    try:
        x_gc, m_gc, y_gc = load_uwavegesture_pool(fmtk_root, seq_len=seq_len, split='test')
        pools['gestureclass'] = (x_gc, m_gc, y_gc)
        print(f"  gestureclass : {x_gc.shape}  (UWaveGesture test set, real data)")

        x_gcs, m_gcs, y_gcs = make_short_pool(x_gc, m_gc, y_gc, short_len)
        pools['gestureclass_short'] = (x_gcs, m_gcs, y_gcs)
        print(f"  gestureclass_short : {x_gcs.shape}  (UWaveGesture, tail zeroed)")
    except Exception as exc:
        print(f"  WARNING: UWaveGesture load failed ({exc}). Using synthetic data.")
        x_gc, m_gc, y_gc = make_synthetic_pool(n_synthetic, seq_len, seed=4)
        pools['gestureclass'] = (x_gc, m_gc, y_gc)
        x_gcs, m_gcs, y_gcs = make_synthetic_pool_short(n_synthetic, seq_len, short_len, seed=5)
        pools['gestureclass_short'] = (x_gcs, m_gcs, y_gcs)

    # ── ecgclass (classification on ECG5000) — not used in default runs but
    #    available for accuracy checks if the caller wants it.
    try:
        x_ec, m_ec, y_ec = load_ecg5000_pool(fmtk_root, seq_len=seq_len, split='test')
        pools['ecgclass'] = (x_ec, m_ec, y_ec)
        print(f"  ecgclass : {x_ec.shape}  (ECG5000 test set, real data)")
    except Exception as exc:
        print(f"  NOTE: ECG5000 load failed ({exc}). ecgclass pool skipped.")

    print()
    return pools


# ─────────────────────────────────────────────────────────────────────────────
# Timing experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_timing_experiment(
    runtime: PyTorchRuntime,
    run_config: dict,
    data_pools: dict,
    warmup: int = WARMUP_REPS,
    num_reps: int = NUM_REPS,
) -> list[dict]:
    """
    Sweep over batch sizes for one run configuration.

    Returns a list of dicts, one per (batch_size, rep) measurement.
    """
    name     = run_config['name']
    desc     = run_config['description']
    task_seq = run_config['task_sequence']
    rng      = np.random.default_rng(42)

    print(f"\n{'─'*62}")
    print(f"  {name}")
    print(f"  {desc}")
    print(f"{'─'*62}")

    records = []

    for bs in run_config['batch_sizes']:
        for rep in range(warmup + num_reps):
            x_batch, mask_batch, task_names, y_batch = construct_batch(
                data_pools, task_seq, bs, rng=rng
            )

            result = runtime.run_batch(x_batch, task_names, mask_batch)

            if rep < warmup:
                continue   # discard warmup

            # Optional GPU barrier for wall-clock accuracy.
            # (runtime.py does not synchronize internally — consistent treatment.)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_ns = result.end_time_ns - result.start_time_ns

            records.append({
                'run':                     name,
                'description':             desc,
                'batch_size':              bs,
                'rep':                     rep - warmup,
                'task_sequence':           '→'.join(task_seq),
                'tasks_in_batch':          ','.join(task_names),
                'backbone_latency_ms':     result.proc_time_ns / 1e6,
                'total_latency_ms':        total_ns / 1e6,
                'per_request_latency_ms':  total_ns / bs / 1e6,
                'mean_decoder_latency_ms': float(np.mean(result.decoder_time_ns)) / 1e6,
                'sum_decoder_latency_ms':  float(np.sum(result.decoder_time_ns)) / 1e6,
                'mean_swap_latency_ms':    float(np.mean(result.swap_time_ns)) / 1e6,
                'gpu_peak_mb':             result.gpu_alloc_peak_mb,
            })

        # Per-batch-size summary line
        recs_bs = [r for r in records if r['batch_size'] == bs]
        if recs_bs:
            m_total  = np.mean([r['total_latency_ms']       for r in recs_bs])
            m_bb     = np.mean([r['backbone_latency_ms']    for r in recs_bs])
            m_per_rq = np.mean([r['per_request_latency_ms'] for r in recs_bs])
            m_gpu    = np.mean([r['gpu_peak_mb']            for r in recs_bs])
            print(f"  bs={bs:2d} | total={m_total:7.2f}ms  "
                  f"backbone={m_bb:7.2f}ms  "
                  f"per_req={m_per_rq:6.2f}ms  "
                  f"gpu={m_gpu:6.1f}MB")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy experiment
# ─────────────────────────────────────────────────────────────────────────────

# Tasks that produce real (non-synthetic) ground-truth labels
_REAL_DATA_TASKS = {'gestureclass', 'ecgclass'}

# Map: routing-task-name → metric type
_TASK_METRIC = {
    'heartrate':    'mae',
    'sysbp':        'mae',
    'gestureclass': 'accuracy',
    'ecgclass':     'accuracy',
}


def run_accuracy_experiment(
    runtime: PyTorchRuntime,
    run_config: dict,
    data_pools: dict,
    batch_size: int = 32,
) -> list[dict]:
    """
    Run the full data pool through run_batch() and compute accuracy / MAE.

    NOTE: If decoder weights are random (created by ensure_decoder_weights),
    the numbers will be meaningless — flagged in the 'weight_status' column.
    """
    name = run_config['name']
    # Collect unique routing tasks in this run
    routing_tasks = list({t.replace('_short', '') for t in run_config['task_sequence']})
    records = []

    for task in routing_tasks:
        if task not in data_pools:
            continue

        x_pool, m_pool, y_pool = data_pools[task]
        n = len(x_pool)

        all_preds  = []
        all_labels = []

        for start in range(0, n, batch_size):
            end        = min(start + batch_size, n)
            bs         = end - start
            x_b        = x_pool[start:end]
            m_b        = m_pool[start:end]
            task_names = [task] * bs
            y_b        = y_pool[start:end]

            result = runtime.run_batch(x_b, task_names, m_b)
            # Each output in result.outputs is already argmax'd for classifiers
            preds = np.array([out.flatten()[0] for out in result.outputs])
            all_preds.append(preds)
            all_labels.append(y_b)

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        metric     = _TASK_METRIC.get(task, 'mae')
        has_labels = task in _REAL_DATA_TASKS and task in data_pools

        if metric == 'accuracy':
            value = float(np.mean(all_preds == all_labels))
        else:
            value = float(np.mean(np.abs(all_preds - all_labels)))

        weight_status = 'pretrained' if _has_pretrained_weights(task) else 'random_init'
        data_status   = 'real' if has_labels else 'synthetic'

        records.append({
            'run':           name,
            'task':          task,
            'n_samples':     n,
            'metric':        metric,
            'value':         value,
            'weight_status': weight_status,
            'data_status':   data_status,
        })
        note = (f"  [{name}] {task:15s}  {metric}={value:.4f}"
                f"  weights={weight_status}  data={data_status}")
        print(note)

    return records


def _has_pretrained_weights(task: str) -> bool:
    """Return True if the decoder for this task has genuine pre-trained weights
    (i.e. was NOT created by ensure_decoder_weights with random init)."""
    from config import DECODER_SPECS, FMTK_ROOT
    for spec in DECODER_SPECS:
        if spec['task'] == task:
            return not is_dummy_decoder(spec['path'], FMTK_ROOT)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# First-milestone helper
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(runtime: PyTorchRuntime, data_pools: dict):
    """
    Run 1 forward pass (batch_size=1) through the heartrate decoder and
    print the BatchRunResult.  Useful for verifying the pipeline is wired up.
    """
    print("\n[milestone] Smoke test: single heartrate request through runtime...")
    rng = np.random.default_rng(0)
    x_b, m_b, t_names, y_b = construct_batch(data_pools, ['heartrate'], 1, rng=rng)

    result = runtime.run_batch(x_b, t_names, m_b)

    print(f"  backbone_latency  : {result.proc_time_ns / 1e6:.2f} ms")
    print(f"  decoder_latency   : {result.decoder_time_ns[0] / 1e6:.2f} ms")
    print(f"  swap_latency      : {result.swap_time_ns[0] / 1e6:.2f} ms")
    total = (result.end_time_ns - result.start_time_ns) / 1e6
    print(f"  total e2e latency : {total:.2f} ms")
    print(f"  gpu_peak_mb       : {result.gpu_alloc_peak_mb:.2f} MB")
    print(f"  output shape      : {result.outputs[0].shape}")
    print("[milestone] Smoke test passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="TSFM batching experiment — latency, memory, accuracy"
    )
    p.add_argument('--batch-sizes', nargs='+', type=int, default=None,
                   help="Override batch sizes, e.g. --batch-sizes 1 4 8")
    p.add_argument('--reps', type=int, default=None,
                   help="Override number of measured repetitions")
    p.add_argument('--warmup', type=int, default=None,
                   help="Override warmup iterations")
    p.add_argument('--runs', nargs='+', default=None,
                   help="Run only named configs, e.g. --runs run1_baseline run4_diff_dec_same_len")
    p.add_argument('--no-accuracy', action='store_true',
                   help="Skip accuracy measurement pass")
    p.add_argument('--smoke-test', action='store_true',
                   help="Run first-milestone smoke test only (1 sample, batch=1)")
    return p.parse_args()


def main():
    args = parse_args()

    batch_sizes = args.batch_sizes or BATCH_SIZES
    num_reps    = args.reps    if args.reps    is not None else NUM_REPS
    warmup_reps = args.warmup  if args.warmup  is not None else WARMUP_REPS

    print("=" * 62)
    print("  TSFM Batching Experiment")
    print(f"  Backbone    : {BACKBONE}")
    print(f"  Batch sizes : {batch_sizes}")
    print(f"  Reps        : {num_reps} + {warmup_reps} warmup")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        device_str = f"cuda ({torch.cuda.get_device_name(0)})"
    print(f"  Device      : {device_str}")
    print("=" * 62)

    # ── Step 0: Ensure decoder weights ───────────────────────────────────────
    print("\n[setup] Checking decoder weights...")
    dummy_paths = ensure_decoder_weights(DECODER_SPECS, BACKBONE, FMTK_ROOT)
    if dummy_paths:
        print(f"  WARNING: Created random-init weights for: {dummy_paths}")
        print("  Latency / memory numbers are valid. Accuracy numbers are NOT.")
    else:
        print("  All decoder weights are pre-trained.")

    # ── Step 1: Build data pools ──────────────────────────────────────────────
    data_pools = build_data_pools(FMTK_ROOT, SEQ_LEN, SHORT_SEQ_LEN, N_POOL_SAMPLES)

    # ── Step 2: Initialise runtime ────────────────────────────────────────────
    print("[setup] Loading backbone + decoders...")
    t0 = time.time()
    runtime = PyTorchRuntime()
    runtime.load(BACKBONE, DECODER_SPECS)
    load_sec = time.time() - t0
    print(f"[setup] Model loaded in {load_sec:.1f}s.\n")

    # ── Step 3 (optional): First-milestone smoke test ─────────────────────────
    smoke_test(runtime, data_pools)
    if args.smoke_test:
        print("[smoke-test] --smoke-test flag set, exiting after first milestone.")
        return

    # ── Step 4: Timing experiments ────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_timing   = []
    all_accuracy = []

    active_runs = RUN_CONFIGS
    if args.runs:
        names_wanted = set(args.runs)
        active_runs  = [r for r in RUN_CONFIGS if r['name'] in names_wanted]
        if not active_runs:
            print(f"ERROR: No run configs matched {args.runs}. "
                  f"Available: {[r['name'] for r in RUN_CONFIGS]}")
            return

    for run_cfg in active_runs:
        cfg = dict(run_cfg)
        cfg['batch_sizes'] = batch_sizes   # honour CLI override

        timing_records = run_timing_experiment(
            runtime, cfg, data_pools,
            warmup=warmup_reps, num_reps=num_reps,
        )
        all_timing.extend(timing_records)

        if not args.no_accuracy:
            acc_records = run_accuracy_experiment(runtime, cfg, data_pools, batch_size=32)
            all_accuracy.extend(acc_records)

    # ── Step 5: Save results ──────────────────────────────────────────────────
    timing_df = pd.DataFrame(all_timing)
    timing_df.to_csv(TIMING_CSV, index=False)
    print(f"\n[results] Timing   → {TIMING_CSV}  ({len(timing_df)} rows)")

    if all_accuracy:
        acc_df = pd.DataFrame(all_accuracy)
        acc_df.to_csv(ACCURACY_CSV, index=False)
        print(f"[results] Accuracy → {ACCURACY_CSV}  ({len(acc_df)} rows)")

    # ── Step 6: Summary table ─────────────────────────────────────────────────
    if len(timing_df) > 0:
        print("\n" + "=" * 62)
        print("  Summary: median total latency (ms) by run × batch_size")
        print("=" * 62)
        pivot = (timing_df
                 .groupby(['run', 'batch_size'])['total_latency_ms']
                 .median()
                 .unstack('batch_size'))
        print(pivot.to_string())

    return timing_df, all_accuracy


if __name__ == '__main__':
    main()
