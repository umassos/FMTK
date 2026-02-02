import json
import matplotlib.pyplot as plt
# ==== CONFIGURE YOUR INPUT FILES HERE ====
files = {
    "Our": "../result/our.jsonl",
    "sLora": "../result/slora.jsonl"
}

# ==== LOAD DATA ====
data = {method: [] for method in files}

for method, filename in files.items():
    with open(filename, "r") as f:
        for line in f:
            entry = json.loads(line)
            cfg = entry["config"]
            res = entry["result"]
            data[method].append({
                "num_tasks": cfg["num_tasks"],
                "req_rate": cfg["req_rate"],
                "throughput": res["throughput"]
            })
print(data)
# ==== GET UNIQUE VALUES ====
all_req_rates = sorted({e["req_rate"] for entries in data.values() for e in entries})
all_num_tasks = sorted({e["num_tasks"] for entries in data.values() for e in entries})

# =====================================================
# 1) Throughput vs Num Tasks — separate subplot per req_rate
# =====================================================
fig, axes = plt.subplots(1, len(all_req_rates), figsize=(6 * len(all_req_rates), 5), squeeze=False)

for idx, rr in enumerate(all_req_rates):
    ax = axes[0][idx]
    for method, entries in data.items():
        xs = sorted(e["num_tasks"] for e in entries if e["req_rate"] == rr)
        ys = [e["throughput"] for e in sorted(entries, key=lambda x: x["num_tasks"]) if e["req_rate"] == rr]
        ax.plot(xs, ys, marker='o', label=method)

    ax.set_title(f"req_rate = {rr}")
    ax.set_xlabel("Num Tasks")
    ax.set_ylabel("Throughput (req/sec)")
    
    ax.grid(True)
    ax.legend()



plt.tight_layout()
plt.savefig("ThroughputvsNumTask.pdf")
plt.show()

# =====================================================
# 2) Throughput vs Request Rate — separate subplot per num_tasks
# =====================================================
fig, axes = plt.subplots(1, len(all_num_tasks), figsize=(6 * len(all_num_tasks), 5), squeeze=False)

for idx, nt in enumerate(all_num_tasks):
    ax = axes[0][idx]
    for method, entries in data.items():
        xs = sorted(e["req_rate"] for e in entries if e["num_tasks"] == nt)
        ys = [e["throughput"] for e in sorted(entries, key=lambda x: x["req_rate"]) if e["num_tasks"] == nt]
        ax.plot(xs, ys, marker='o', label=method)

    ax.set_title(f"num_tasks = {nt}")
    ax.set_xlabel("Request Rate")
    ax.set_ylabel("Throughput (req/sec)")
    ax.grid(True)
    ax.legend()
    

plt.tight_layout()
plt.savefig("ThroughputvsRequestRate.pdf")
plt.show()
