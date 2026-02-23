"""
Quick smoke test: run one TSFM task and one VLM task through the unified
InferencePipeline to verify both flows work and TSFMs are not disrupted.

Usage:
    conda run -n fmtk python test_both_flows.py
    conda run -n fmtk python test_both_flows.py --vlm-only     # skip TSFM (if dataset unavailable)
    conda run -n fmtk python test_both_flows.py --tsfm-only    # skip VLM
"""

import subprocess, json, os, csv, sys, argparse

# ── TSFM task: diasbp + momentsmall (2 epochs for speed) ──────────────
tsfm_task_info = {
    "task_type": "regression",
    "datasets": ["PPG-data"],
    "label": "diasbp",
    "train": True,
    "inference_config": {"batch_size": 1, "shuffle": False},
    "train_config": {
        "batch_size": 32,
        "shuffle": False,
        "epochs": 2,          # reduced from 50
        "lr": 1e-2,
    },
}

tsfm_pipeline = {
    "backbone": "momentsmall",
    "paths": [
        {
            "decoder": "mlp_momentsmall_regression",
            "encoder": "linear",
            "parts_to_train": ["decoder", "encoder"],
            "path": "test_diasbp_momentsmall_mlp",
        }
    ],
}

# ── VLM task: vlm_ocr + phi ──────────────────────────────────────────
vlm_task_info = {
    "task_type": "vlm",
    "vlm_task_key": "ocr",
    "datasets": ["vlm_ocr"],
    "train": False,
    "parser": "parse_ocr_digit",
    "evaluator": "evaluate_ocr",
    "inference_config": {"batch_size": 1, "shuffle": False},
}

vlm_pipeline = {
    "backbone": "phi",
    "paths": [{}],
}

# ── Run a task through worker.py ──────────────────────────────────────
def run_task(label, task_name, task_info, pipeline):
    print(f"\n{'='*60}")
    print(f"  {label}: task={task_name}, backbone={pipeline['backbone']}")
    print(f"{'='*60}")
    result = subprocess.run(
        ["python3", "worker.py", json.dumps({
            "task_name": task_name,
            "task_info": task_info,
            "pipeline": pipeline,
        })],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        print(f"  *** {label} FAILED (exit code {result.returncode}) ***")
        return False
    print(f"  {label} completed successfully.")
    return True

def check_dataset(dataset_path):
    """Check if dataset exists relative to experiments/run_all/."""
    base = os.path.dirname(os.path.abspath(__file__))
    full = os.path.normpath(os.path.join(base, dataset_path))
    return os.path.isdir(full)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-only", action="store_true", help="Skip TSFM test")
    parser.add_argument("--tsfm-only", action="store_true", help="Skip VLM test")
    args = parser.parse_args()

    ok = True
    tsfm_ran = False
    vlm_ran = False

    # 1) TSFM
    if not args.vlm_only:
        ds_path = "../../../dataset/PPG-data"  # from config.py
        if check_dataset(ds_path):
            tsfm_ran = True
            ok &= run_task("TSFM", "diasbp", tsfm_task_info, tsfm_pipeline)
        else:
            print(f"\n  TSFM SKIPPED: PPG-data dataset not found at {ds_path}")
            print(f"  (The _run_tsfm() code path is unchanged from the original run() method)")

    # 2) VLM
    if not args.tsfm_only:
        vlm_ran = True
        ok &= run_task("VLM", "vlm_ocr", vlm_task_info, vlm_pipeline)

    # 3) Verify CSV outputs
    print(f"\n{'='*60}")
    print("  Verifying CSV outputs")
    print(f"{'='*60}")

    base = os.path.dirname(os.path.abspath(__file__))
    tsfm_csv = os.path.join(base, "combined_metrics.csv")
    vlm_csv = os.path.join(base, "vlm_metrics.csv")

    for label, path, ran in [("TSFM CSV", tsfm_csv, tsfm_ran),
                              ("VLM CSV", vlm_csv, vlm_ran)]:
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            print(f"  {label}: {path}")
            print(f"    Columns ({len(reader.fieldnames)}): {reader.fieldnames}")
            print(f"    Total rows: {len(rows)}")
            if rows:
                print(f"    Last row: {dict(rows[-1])}")
        elif ran:
            print(f"  {label}: NOT FOUND at {path} (unexpected!)")
            ok = False
        else:
            print(f"  {label}: not checked (task was skipped)")

    print(f"\n{'='*60}")
    status = "PASS" if ok else "FAIL"
    if not tsfm_ran and not args.vlm_only:
        status += " (TSFM skipped — no dataset)"
    print(f"  RESULT: {status}")
    print(f"{'='*60}")
    sys.exit(0 if ok else 1)
