import time, json, os
from collections import defaultdict
from contextlib import contextmanager

try:
    import torch
    HAS_TORCH = True
except:
    HAS_TORCH = False

try:
    import psutil, os as _os
    PROC = psutil.Process(os.getpid())
    HAS_PSUTIL = True
except:
    HAS_PSUTIL = False

try:
    import pynvml as nvml
    nvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

class Logger:
    def __init__(self, device, run_name="run", save_dir="./logs"):
        self.device=device
        self.run_name = run_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.records = []  
        self.scalars = defaultdict(list)

    def log_scalar(self, key, value, step=None, section=None):
        self.scalars[key].append({"step": step, "value": float(value), "section": section})

    def log_dict(self, dct: dict, step=None, section=None):
        for k, v in dct.items():
            if isinstance(v, (int, float)):
                self.log_scalar(k, v, step=step, section=section)

    @contextmanager
    def measure(self, section, device=None, cuda_sync=True):
        rec = {"section": section}

        # CPU memory
        rss_before = PROC.memory_info().rss if HAS_PSUTIL else None
        
        # GPU energy
        nvml_handle = None
        energy_before_mJ = None
        power_start_mW = None
        gpu_index = device.index if isinstance(device, torch.device) else int(str(device).split(":")[1])
        if HAS_NVML and gpu_index is not None:
            try:
                nvml_handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
                energy_before_mJ = nvml.nvmlDeviceGetTotalEnergyConsumption(nvml_handle)
            except nvml.NVMLError:
                nvml_handle = None

        # GPU memory and time
        if HAS_TORCH and torch.cuda.is_available() and device is not None and "cuda" in str(device):
            if cuda_sync: torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            alloc_before = torch.cuda.memory_allocated(device)
            reserv_before = torch.cuda.memory_reserved(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            alloc_before = reserv_before = None
            start_event = end_event = None

        # time
        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            t1 = time.perf_counter()
            rec["wall_time_sec"] = t1 - t0

            # GPU measures
            if HAS_TORCH and torch.cuda.is_available() and device is not None and "cuda" in str(device):
                end_event.record()
                if cuda_sync: 
                    torch.cuda.synchronize()
                rec["gpu_time_ms"] = start_event.elapsed_time(end_event)
                rec["gpu_alloc_before"] = alloc_before
                rec["gpu_alloc_peak"] = torch.cuda.max_memory_allocated(device)
                rec["gpu_reserved_before"] = reserv_before
                rec["gpu_reserved_peak"] = torch.cuda.max_memory_reserved(device)
            
            if nvml_handle is not None:
                try:
                    if cuda_sync: torch.cuda.synchronize()
                    energy_after_mJ = nvml.nvmlDeviceGetTotalEnergyConsumption(nvml_handle)
                    if energy_after_mJ >= energy_before_mJ:
                        rec["gpu_energy_mJ"] = int(energy_after_mJ - energy_before_mJ)
                    else:
                        rec["gpu_energy_mJ"] = None
                except nvml.NVMLError:
                    rec["gpu_energy_mJ"] = None

            # CPU RSS delta
            if HAS_PSUTIL:
                rss_after = PROC.memory_info().rss
                rec["cpu_rss_before"] = rss_before
                rec["cpu_rss_after"] = rss_after
                rec["cpu_rss_delta"] = (rss_after - rss_before)

            self.records.append(rec)

    def save(self):
        out = {
            "run_name": self.run_name,
            "records": self.records,
            "scalars": dict(self.scalars),
        }
        path = os.path.join(self.save_dir, f"{self.run_name}.json")

        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    old = json.load(f)
            except Exception:
                old = {}

            # merge records (append new ones)
            if "records" in old:
                out["records"] = old["records"] + out["records"]

            # merge scalars (append to each list)
            if "scalars" in old:
                merged = old["scalars"]
                for k, v in out["scalars"].items():
                    merged[k] = merged.get(k, []) + v
                out["scalars"] = merged

        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        return path


    def summary(self):
        s = {}
        grouped = defaultdict(list)

        # group records by section
        for r in self.records:
            grouped[r['section']].append(r)

        # compute averages for each section
        for section, records in grouped.items():
            s[section] = {}
            n = len(records)
            if any("wall_time_sec" in r for r in records):
                wall = sum(r.get('wall_time_sec', 0) for r in records) / n
                line = f"[{section}] wall={wall:.3f}s"
                s[section].update({"wall time":wall*1000})

            if any("gpu_time_ms" in r for r in records):
                gpu = sum(r.get("gpu_time_ms", 0) for r in records) / n
                line += f", gpu={gpu:.2f}ms"
                s[section].update({"gpu time":gpu})

            if any("gpu_alloc_peak" in r for r in records):
                gpu_peak = sum(r.get("gpu_alloc_peak", 0) for r in records) / n
                line += f", gpu_peak={gpu_peak/1e6:.1f}MB"
                s[section].update({"gpu peak":gpu_peak/1e6})

            if any("gpu_energy_mJ" in r and r["gpu_energy_mJ"] is not None for r in records):
                gpu_energy = sum((r.get("gpu_energy_mJ") or 0) for r in records) / n
                line += f", gpu_energy={gpu_energy/1000:.3f}J"
                s[section].update({"gpu energy":gpu_energy/1000})

            if any("cpu_rss_delta" in r for r in records):
                cpu = sum(r.get("cpu_rss_delta", 0) for r in records) / n
                line += f", cpu_dRSS={cpu/1e6:.1f}MB"
                s[section].update({"cpu dRSS":cpu/1e6})

        return s
