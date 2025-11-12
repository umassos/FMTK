import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your dataframe
df = pd.DataFrame({
    'model': ['Papagei','MOMENT','Chronos'],
    'MLP': [91.26, 93.66, 90.06],
    'SVM': [90.86, 94.15, 93.86],
    'KNN': [87.82, 93, 91.62],
    'Random\nForest': [88.28, 92.64, 91.62],
    'Logistic\nRegression': [90.84, 91.26, 58.37]
})

# --- Radar chart setup ---
categories = list(df.columns[1:])  
N = len(categories)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # complete the loop

# Plot each row
fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))

for i, row in df.iterrows():
    values = row.drop('model').tolist()
    values += values[:1]  # repeat first value
    ax.plot(angles, values, label=row['model'])
    ax.fill(angles, values, alpha=0.1)

# Category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Y-labels
ax.set_rlabel_position(30)
ax.set_yticks([60, 80, 100])
ax.set_yticklabels(["60", "80", "100"])
ax.set_ylim(50, 100)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19),ncol=3)
plt.savefig('diversedecoders.pdf')
plt.show()
