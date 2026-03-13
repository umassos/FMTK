#!/usr/bin/env python3
"""
Validates batching experiment results for sanity and stability.
Prints a report and returns exit code 0 if all checks pass, 1 if any fail.
"""
import sys
import numpy as np
import pandas as pd

TIMING_CSV   = "experiments/batching/results/batching_results.csv"
ACCURACY_CSV = "experiments/batching/results/accuracy_results.csv"
REPORT_FILE  = "experiments/batching/results/validation_report.txt"

issues = []
passes = []

def check(condition, pass_msg, fail_msg):
    if condition:
        passes.append(f"PASS: {pass_msg}")
    else:
        issues.append(f"FAIL: {fail_msg}")

# Load data
t = pd.read_csv(TIMING_CSV)
a = pd.read_csv(ACCURACY_CSV)

# ── 1. Backbone latency is roughly constant across batch sizes ────────────
for run in t['run'].unique():
    sub = t[t['run'] == run].groupby('batch_size')['backbone_latency_ms'].median()
    ratio = sub.max() / sub.min()
    check(ratio < 3.0,
          f"{run}: backbone latency ratio max/min={ratio:.2f} < 3.0 (roughly constant)",
          f"{run}: backbone latency varies too much across batch sizes (max/min={ratio:.2f})")

# ── 2. Total latency scales sub-linearly (per-request latency decreases) ──
for run in t['run'].unique():
    sub = t[t['run'] == run].groupby('batch_size')['per_request_latency_ms'].median()
    bs_vals = sorted(sub.index)
    if len(bs_vals) >= 2:
        ratio = sub[bs_vals[0]] / sub[bs_vals[-1]]
        check(ratio > 1.5,
              f"{run}: per-request latency at bs=1 / bs={bs_vals[-1]} = {ratio:.2f} > 1.5 (good batching benefit)",
              f"{run}: no batching benefit — per-request latency barely decreases (ratio={ratio:.2f})")

# ── 3. Coefficient of variation < 15% per (run, batch_size) ──────────────
for (run, bs), grp in t.groupby(['run', 'batch_size']):
    cv = grp['total_latency_ms'].std() / grp['total_latency_ms'].mean()
    check(cv < 0.15,
          f"{run} bs={bs}: CV={cv:.3f} < 0.15 (stable)",
          f"{run} bs={bs}: high variance CV={cv:.3f} >= 0.15 (unstable)")

# ── 4. Accuracy checks ────────────────────────────────────────────────────
for _, row in a.iterrows():
    task   = row['task']
    metric = row['metric']
    value  = row['value']
    wt     = row['weight_status']

    if task == 'gestureclass' and metric == 'accuracy':
        if wt == 'pretrained':
            check(value > 0.15,
                  f"gestureclass accuracy={value:.4f} > 0.15 (beats random 0.125)",
                  f"gestureclass accuracy={value:.4f} <= 0.15 (at or below random chance for 8 classes)")
    elif task == 'ecgclass' and metric == 'accuracy':
        if wt == 'pretrained':
            check(value > 0.25,
                  f"ecgclass accuracy={value:.4f} > 0.25 (beats random 0.20)",
                  f"ecgclass accuracy={value:.4f} <= 0.25 (at or below random chance for 5 classes)")
    elif metric == 'mae':
        check(np.isfinite(value) and value > 0,
              f"{task} MAE={value:.4f} is finite and positive",
              f"{task} MAE={value} is invalid (NaN/inf/zero)")

# ── 5. Cross-run: backbone latency consistent across runs ─────────────────
bb_medians = t.groupby('run')['backbone_latency_ms'].median()
bb_ratio = bb_medians.max() / bb_medians.min()
check(bb_ratio < 2.0,
      f"Backbone latency consistent across runs (max/min={bb_ratio:.2f})",
      f"Backbone latency inconsistent across runs (max/min={bb_ratio:.2f})")

# ── Report ─────────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 60)
lines.append("BATCHING EXPERIMENT VALIDATION REPORT")
lines.append("=" * 60)
lines.append(f"\nPASSED ({len(passes)}):")
for p in passes:
    lines.append(f"  {p}")
lines.append(f"\nFAILED ({len(issues)}):")
for i in issues:
    lines.append(f"  {i}")
lines.append("\nSUMMARY: median total latency (ms) by run x batch_size")
pivot = t.groupby(['run','batch_size'])['total_latency_ms'].median().unstack('batch_size')
lines.append(pivot.to_string())
lines.append("\nACCURACY:")
lines.append(a[['run','task','metric','value','weight_status']].to_string(index=False))
lines.append("=" * 60)

report = "\n".join(lines)
print(report)
with open(REPORT_FILE, 'w') as f:
    f.write(report)

sys.exit(0 if not issues else 1)
