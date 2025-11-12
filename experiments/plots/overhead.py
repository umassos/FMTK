import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data
heartbeat_classification = {'base': 31.24, 'FMTK': 34.59}
energy_forecasting = {'base': 45.41, 'FMTK': 46.42}

tasks = ['heartbeat\nclassification', 'energy\nforecasting']
methods = list(heartbeat_classification.keys())

values = [
    [heartbeat_classification[m] for m in methods],
    [energy_forecasting[m] for m in methods]
]

x = np.arange(len(tasks))  # task positions
width = 0.35               # bar width

fig, ax = plt.subplots(figsize=(8,4))
rects1 = ax.bar(x - width/2, [v[0] for v in values], width, label='base',color='#faa275')
rects2 = ax.bar(x + width/2, [v[1] for v in values], width, label='FMTK',color='#ce6a85')

# Labels
ax.set_ylabel('Inf. time (ms)',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylim(0,48)
ax.set_yticks(np.arange(0, 49, 12)) 
ax.tick_params(axis='both', labelsize=20) 
ax.legend(prop={'size': 20},ncol=2,bbox_to_anchor=(0.5,1.3),loc='upper center')
sns.despine()
plt.tight_layout()
plt.savefig("./plots/inference_time.pdf")
plt.show()

# Data
heartbeat_classification = {'base': 562.62, 'FMTK':562.62 }
energy_forecasting = {'base': 797.07, 'FMTK': 740.80}

tasks = ['heartbeat\nclassification', 'energy\nforecasting']
methods = list(heartbeat_classification.keys())

values = [
    [heartbeat_classification[m] for m in methods],
    [energy_forecasting[m] for m in methods]
]

x = np.arange(len(tasks))  # task positions
width = 0.35               # bar width

fig, ax = plt.subplots(figsize=(8,4))
rects1 = ax.bar(x - width/2, [v[0] for v in values], width, label='base',color='#faa275')
rects2 = ax.bar(x + width/2, [v[1] for v in values], width, label='FMTK',color='#ce6a85')

# Labels
ax.set_ylabel('Inf. GPU \n memory (MB)',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylim(0,800)
ax.set_yticks(np.arange(0, 801, 400)) 
ax.tick_params(axis='both', labelsize=20) 
ax.legend(prop={'size': 20},ncol=2,bbox_to_anchor=(0.5,1.3),loc='upper center')
sns.despine()
plt.tight_layout()
plt.savefig("./plots/inference_gpu.pdf")
plt.show()