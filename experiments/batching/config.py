"""
Batching experiment configuration.

Paths, backbone, decoder specs, run definitions, and tuning knobs all live here.
Edit NUM_REPS / WARMUP_REPS for quick tests; use 50/10 for publication-quality results.
"""
import os

# ── Directory layout ─────────────────────────────────────────────────────────
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
FMTK_ROOT      = os.path.abspath(os.path.join(EXPERIMENT_DIR, '..', '..'))
FMAAS_SERVING  = os.path.abspath(
    os.path.join(FMTK_ROOT, '..', 'FMaaS-motivation', 'serving')
)
OUTPUT_DIR     = os.path.join(EXPERIMENT_DIR, 'results')
TIMING_CSV     = os.path.join(OUTPUT_DIR, 'batching_results.csv')
ACCURACY_CSV   = os.path.join(OUTPUT_DIR, 'accuracy_results.csv')

# ── Backbone ─────────────────────────────────────────────────────────────────
BACKBONE = "momentbase"   # embedding dim: 768

# ── Sequence lengths ─────────────────────────────────────────────────────────
SEQ_LEN       = 512   # MOMENT-base native context length
SHORT_SEQ_LEN = 256   # "short" variant: real content, rest zero-padded + masked

# ── Timing knobs ─────────────────────────────────────────────────────────────
BATCH_SIZES  = [1, 2, 4, 8, 16, 32]
NUM_REPS     = 50    # measured repetitions per (run, batch_size)  → use 50 for final
WARMUP_REPS  = 10    # warmup iterations discarded                 → use 10 for final

# ── Synthetic data pool ───────────────────────────────────────────────────────
N_POOL_SAMPLES = 500   # synthetic samples drawn from for non-PPG tasks

# ── Decoder specs ─────────────────────────────────────────────────────────────
# 'task'  : key used in runtime.decoders and in task_names list passed to run_batch()
# 'type'  : decoder architecture selector  ("regression" | "classification")
# 'path'  : sub-directory under models/tsfm/finetuned/ where decoder.pth lives
DECODER_SPECS = [
    {"task": "heartrate",    "type": "regression",     "path": "heartrate_momentbase_mlp"},
    {"task": "sysbp",        "type": "regression",     "path": "sysbp_momentbase_mlp"},
    {"task": "gestureclass", "type": "classification", "path": "gestureclass_momentbase_mlp"},
    {"task": "ecgclass",     "type": "classification", "path": "ecgclass_momentbase_mlp"},
]

# ── Run definitions ───────────────────────────────────────────────────────────
# task_sequence is cycled modulo batch_size to fill the batch.
# Names that end in "_short" route to the base task name for decoder lookup
# (e.g. "heartrate_short" → decoder "heartrate") but draw from the short pool.
RUN_CONFIGS = [
    {
        "name":         "run1_baseline",
        "description":  "Single task, single decoder — heartrate regression, seq_len=512",
        "batch_sizes":  BATCH_SIZES,
        "task_sequence": ["heartrate"],
    },
    {
        "name":         "run2_same_dec_same_len",
        "description":  "Two regression tasks, same decoder arch, same seq_len "
                        "(heartrate + sysbp interleaved, both MLP 768→1)",
        "batch_sizes":  BATCH_SIZES,
        "task_sequence": ["heartrate", "sysbp"],
    },
    {
        "name":         "run3_same_dec_diff_len",
        "description":  "Same decoder (heartrate MLP), mixed seq_len "
                        "(full 512 + zero-padded 256, interleaved)",
        "batch_sizes":  BATCH_SIZES,
        "task_sequence": ["heartrate", "heartrate_short"],
    },
    {
        "name":         "run4_diff_dec_same_len",
        "description":  "Different decoders, same seq_len "
                        "(heartrate regression + gestureclass classification, interleaved)",
        "batch_sizes":  BATCH_SIZES,
        "task_sequence": ["heartrate", "gestureclass"],
    },
    {
        "name":         "run5_diff_dec_diff_len",
        "description":  "Different decoders, mixed seq_len "
                        "(heartrate@512 + gestureclass@256, interleaved)",
        "batch_sizes":  BATCH_SIZES,
        "task_sequence": ["heartrate", "gestureclass_short"],
    },
]
