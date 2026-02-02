import matplotlib.pyplot as plt
import numpy as np

# Data
tasks = ["Task1", "Task2", "Task3"]

throughput = {
    "Normal": [7.980066445182724, 7.976744186046512, 7.970099667774086],
    "Sharing without batching": [5.840531561461794, 5.833887043189368, 5.8305647840531565],
    "Sharing with batching": [7.501661129568106, 7.501661129568106, 7.501661129568106]
}

latency = {
    "Normal": [123.38504515321527, 123.47735071718469, 123.58002078289685],
    "Sharing without batching": [166.98082491535104, 166.75334381351166, 166.08914622554073],
    "Sharing with batching": [132.30479175375035, 132.28198635145242, 132.31447156917955]
}

memory={
    "Normal": 4554.9569748294,
    "Sharing without batching": 1518.3189916098,
    "Sharing with batching": 1518.3189916098
}

# Plot setup
x = np.arange(len(tasks))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Throughput plot
axes[0].bar(x - width, throughput["Normal"], width, label="Normal")
axes[0].bar(x, throughput["Sharing without batching"], width, label="Sharing w/o batching")
axes[0].bar(x + width, throughput["Sharing with batching"], width, label="Sharing with batching")

axes[0].set_xticks(x)
axes[0].set_xticklabels(tasks)
axes[0].set_ylabel("Throughput (req/s)")
axes[0].set_title("Throughput Comparison")
axes[0].legend()

# Latency plot
axes[1].bar(x - width, latency["Normal"], width, label="Normal")
axes[1].bar(x, latency["Sharing without batching"], width, label="Sharing w/o batching")
axes[1].bar(x + width, latency["Sharing with batching"], width, label="Sharing with batching")

axes[1].set_xticks(x)
axes[1].set_xticklabels(tasks)
axes[1].set_ylabel("Latency (ms)")
axes[1].set_title("Latency Comparison")
axes[1].legend()

plt.tight_layout()
plt.show()
plt.savefig("throughput_memory.png")

fig, ax = plt.subplots(figsize=(6, 5))

methods = list(memory.keys())
values = list(memory.values())

ax.bar(methods, values, color=['tab:blue', 'tab:orange', 'tab:green'])
ax.set_ylabel("Memory (MB)")
ax.set_title("Memory Usage Comparison")

plt.tight_layout()
plt.show()
plt.savefig("memory_comparison.png")