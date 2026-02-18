"""
VLM profiling experiment for FMTK.

Mirrors the profiling methodology of FMaaS-motivation/unified_inference.py
but uses FMTK's Pipeline, Logger, VLMDataset, and backbone abstractions.

Runs VLM model(s) across ALL 9 VLM tasks, appending one CSV row per
(model, task) combination — identical schema to unified_metrics.csv.

Usage:
    python experiments/vlm_profiling/run_vlm_profile.py                   # all models
    python experiments/vlm_profiling/run_vlm_profile.py --models moondream phi  # subset
"""

import sys, os, gc, csv, time, argparse, importlib

# ── make FMTK importable when running from the repo root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# ── point HF cache at FMaaS-motivation's cached models ────
_FMTK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
_HF_MODELS = os.path.join(_FMTK_ROOT, '..', 'FMaaS-motivation', 'vqa', 'updated', 'models')
os.environ["HF_HOME"] = _HF_MODELS
os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_MODELS
os.environ["TRANSFORMERS_CACHE"] = _HF_MODELS

import torch
from torch.utils.data import DataLoader

# ── PyTorch compat shims for moondream2 on PyTorch < 2.5 ──
if not hasattr(torch, '_dynamo'):
    import types
    torch._dynamo = types.ModuleType('torch._dynamo')
    torch._dynamo.mark_dynamic = lambda *a, **kw: None

import torch.nn.functional as F
_orig_sdpa = F.scaled_dot_product_attention
def _patched_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _orig_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = _patched_sdpa

from fmtk.datasets.vlm_dataset import VLMDataset, vlm_collate_fn
from fmtk.tasks.vlm_utils import (
    get_vlm_dataset_config,
    get_parser,
    get_evaluator,
    TASK_REGISTRY,
)
from fmtk.pipeline import Pipeline
from fmtk.logger import Logger

# ── Model registry: short_name -> (module, class, HF model ID) ──
VLM_MODELS = {
    "moondream":      ("moondream",    "MoondreamModel",    "vikhyatk/moondream2"),
    "llava-1.5-7b":   ("llava",        "LlavaModel",        "llava-hf/llava-1.5-7b-hf"),
    "llava-1.5-13b":  ("llava",        "LlavaModel",        "llava-hf/llava-1.5-13b-hf"),
    "llava-v1.6-13b": ("llava",        "LlavaModel",        "llava-hf/llava-v1.6-vicuna-13b-hf"),
    "qwen-3B":        ("qwen",         "QwenModel",         "Qwen/Qwen2.5-VL-3B-Instruct"),
    "qwen-7B":        ("qwen",         "QwenModel",         "Qwen/Qwen2.5-VL-7B-Instruct"),
    "phi":            ("phi",          "PhiModel",          "microsoft/Phi-3.5-vision-instruct"),
    "molmo":          ("molmo",        "MolmoModel",        "allenai/Molmo-7B-D-0924"),
    "llama-vision":   ("llama_vision", "LlamaVisionModel",  "meta-llama/Llama-3.2-11B-Vision-Instruct"),
    "minicpm":        ("minicpm",      "MinicpmModel",      "openbmb/MiniCPM-V-2_6"),
}

# ── CSV schema — identical to FMaaS-motivation/unified_metrics.csv ──
VLM_METRICS_CSV = os.path.join(os.path.dirname(__file__), "vlm_metrics.csv")
CSV_COLUMNS = [
    "model_name", "dataset_name", "device", "model_load_duration_sec",
    "gpu_load_memory_mb", "avg_cpu_memory_usage_mb", "avg_cpu_usage_percent",
    "avg_gpu_usage_percent", "avg_gpu_memory_usage_mb", "total_prompt_tokens",
    "total_generated_tokens", "ttft_ms", "avg_latency_ms", "throughput_tps",
    "accuracy", "total_time", "num_samples", "gpu_name",
]

# ── configuration ──────────────────────────────────────────
DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
BATCH_SIZE = 1

# All 9 VLM tasks (same order as FMaaS-motivation/run_all_models_tasks.py)
ALL_TASKS = [
    "crowd", "scene", "ocr", "vqa", "traffic",
    "gesture", "activity", "object_detection", "image_classification",
]

_dev_idx = int(DEVICE.split(":")[-1])
GPU_NAME = torch.cuda.get_device_name(_dev_idx) if torch.cuda.is_available() else ""


def load_vlm_model(model_name, device):
    """Dynamically import and instantiate a VLM backbone by short name."""
    module_name, class_name, hf_id = VLM_MODELS[model_name]
    mod = importlib.import_module(f"fmtk.components.backbones.{module_name}")
    cls = getattr(mod, class_name)
    return cls(device, model_name), hf_id


def run_task(task_name, model, model_hf_id, model_name,
             model_load_duration_sec, gpu_load_memory_mb):
    """Run a single (model, task) profiling experiment and append to CSV."""

    print(f"\n{'='*60}")
    print(f"  Task: {task_name}")
    print(f"{'='*60}")

    # ── dataset ────────────────────────────────────────────
    dataset_cfg, task_cfg = get_vlm_dataset_config(task_name)
    dataset = VLMDataset(dataset_cfg, task_cfg, split="test")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=vlm_collate_fn)
    print(f"  Dataset: {len(dataset)} samples from .../{os.path.basename(dataset_cfg['dataset_path'])}")

    # ── logger (fresh per task) ────────────────────────────
    logger = Logger(DEVICE, run_name=f"vlm_{model_name}_{task_name}", save_dir="./logs")

    # ── inference ──────────────────────────────────────────
    pipeline = Pipeline(model, logger=logger)

    with logger.measure("inference_total", device=DEVICE):
        labels_raw, preds_raw = pipeline.predict(
            dataloader, cfg={"batch_size": BATCH_SIZE, "shuffle": False}
        )

    # Flatten: predict() returns list-of-lists (one list per batch)
    all_preds = [p for batch in preds_raw for p in batch]
    all_labels = [l for batch in labels_raw for l in batch]

    # ── evaluation ─────────────────────────────────────────
    task_info = TASK_REGISTRY[task_name]
    parser_fn = get_parser(task_info["parser"])
    evaluator_fn = get_evaluator(task_info["evaluator"])

    correct = 0
    total = 0
    for pred, gt in zip(all_preds, all_labels):
        parsed = parser_fn(pred)
        is_correct = evaluator_fn(parsed, gt)
        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0

    # ── collect metrics ────────────────────────────────────
    summary = logger.summary()
    vlm = summary.get("vlm", {})

    inference_rec = next(
        (r for r in logger.records if r["section"] == "inference_total"), {}
    )
    total_time = inference_rec.get("wall_time_sec", 0)
    avg_cpu_memory_usage_mb = inference_rec.get("cpu_rss_delta", 0) / 1e6

    csv_row = {
        "model_name":              model_hf_id,
        "dataset_name":            dataset_cfg["dataset_path"],
        "device":                  DEVICE,
        "model_load_duration_sec": model_load_duration_sec,
        "gpu_load_memory_mb":      gpu_load_memory_mb,
        "avg_cpu_memory_usage_mb": avg_cpu_memory_usage_mb,
        "avg_cpu_usage_percent":   0,  # not tracked per-sample
        "avg_gpu_usage_percent":   vlm.get("avg_gpu_util_pct", 0),
        "avg_gpu_memory_usage_mb": vlm.get("avg_gpu_mem_delta_mb", 0),
        "total_prompt_tokens":     vlm.get("total_prompt_tokens", 0),
        "total_generated_tokens":  vlm.get("total_gen_tokens", 0),
        "ttft_ms":                 0,  # not tracked
        "avg_latency_ms":          vlm.get("avg_latency_ms", 0),
        "throughput_tps":          vlm.get("throughput_tps", 0),
        "accuracy":                accuracy,
        "total_time":              total_time,
        "num_samples":             total,
        "gpu_name":                GPU_NAME,
    }

    # ── append to CSV ──────────────────────────────────────
    file_exists = os.path.isfile(VLM_METRICS_CSV)
    with open(VLM_METRICS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_row)

    # Save raw Logger JSON
    logger.save()

    # ── print summary ──────────────────────────────────────
    print(f"  Accuracy:    {correct}/{total} = {accuracy:.2%}")
    print(f"  Latency:     {csv_row['avg_latency_ms']:.1f} ms/sample")
    print(f"  Throughput:  {csv_row['throughput_tps']:.1f} tok/s")
    print(f"  Total time:  {total_time:.1f}s")
    print(f"  GPU util:    {csv_row['avg_gpu_usage_percent']:.1f}%")

    return csv_row


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM profiling across models and tasks")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Model(s) to profile. Choices: {list(VLM_MODELS.keys())}. Default: all."
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Clear existing CSV before starting (default: append)."
    )
    args = parser.parse_args()

    models_to_run = args.models if args.models else list(VLM_MODELS.keys())

    # Validate model names
    for m in models_to_run:
        if m not in VLM_MODELS:
            print(f"ERROR: Unknown model '{m}'. Available: {list(VLM_MODELS.keys())}")
            sys.exit(1)

    print(f"VLM Profiling: {len(models_to_run)} model(s) x {len(ALL_TASKS)} tasks")
    print(f"Models: {models_to_run}")
    print(f"Device: {DEVICE} ({GPU_NAME})")
    print(f"Output: {VLM_METRICS_CSV}")

    # ── CSV is always append-only; use --fresh to start over ──
    if args.fresh and os.path.exists(VLM_METRICS_CSV):
        os.remove(VLM_METRICS_CSV)
        print("Cleared previous CSV.")

    all_results = []

    for model_name in models_to_run:
        _, _, model_hf_id = VLM_MODELS[model_name]

        print(f"\n{'#'*60}")
        print(f"  MODEL: {model_name} ({model_hf_id})")
        print(f"{'#'*60}")

        # ── load model ────────────────────────────────────
        print(f"\nLoading model: {model_name}...")
        try:
            load_logger = Logger(DEVICE, run_name=f"model_load_{model_name}", save_dir="./logs")

            with load_logger.measure("model_load", device=DEVICE):
                model, model_hf_id = load_vlm_model(model_name, DEVICE)

            load_rec = load_logger.records[0]
            model_load_duration_sec = load_rec["wall_time_sec"]
            gpu_load_memory_mb = load_rec["gpu_alloc_peak"] / 1e6
            print(f"Model loaded: {model_load_duration_sec:.2f}s, {gpu_load_memory_mb:.0f} MB GPU")
        except Exception as e:
            print(f"\n  ERROR loading model '{model_name}': {e}")
            import traceback
            traceback.print_exc()
            continue

        # ── run all tasks ─────────────────────────────────
        model_results = []
        for task_name in ALL_TASKS:
            try:
                row = run_task(task_name, model, model_hf_id, model_name,
                               model_load_duration_sec, gpu_load_memory_mb)
                model_results.append(row)
            except Exception as e:
                print(f"\n  ERROR on task '{task_name}': {e}")
                import traceback
                traceback.print_exc()
                continue

        # ── per-model summary ─────────────────────────────
        print(f"\n{'='*60}")
        print(f"  {model_name}: {len(model_results)}/{len(ALL_TASKS)} tasks")
        print(f"{'='*60}")
        print(f"{'Task':<25} {'Accuracy':>10} {'Latency':>12} {'Samples':>10}")
        print(f"{'-'*25} {'-'*10} {'-'*12} {'-'*10}")
        for r in model_results:
            task = os.path.basename(r['dataset_name'])
            print(f"{task:<25} {r['accuracy']:>9.2%} {r['avg_latency_ms']:>9.1f} ms {r['num_samples']:>10}")

        all_results.extend(model_results)

        # ── free GPU memory before next model ─────────────
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── final summary ─────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  ALL DONE: {len(all_results)} rows ({len(models_to_run)} models x {len(ALL_TASKS)} tasks)")
    print(f"{'#'*60}")
    print(f"CSV: {VLM_METRICS_CSV}")
