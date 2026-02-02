import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example raw data: rows=models, columns=tasks (mixed metrics)
# accuracy-like: higher is better; mae-like: lower is better
df = pd.DataFrame({
    'model': ['Papagei','MOMENT','Chronos'],
    'SysBP': [16.24, 24.28, 18.62],     # higher-better
    'DiaSysBP': [9.68, 13.67, 11.05],        # lower-better
    'HR': [6.02, 5.63, 9.06],     # higher-better
    'ECG': [91.26, 93.66, 90.06], 
    'ETTH1':[0.598,0.596,0.6]
})

# Declare metric direction per task (True = higher is better)
metric_meta = {
    'SysBP': False,
    'DiaSysBP': False,
    'HR': False,
    'ECG': True,
    'ETTH1': False,

}

def minmax_unitize(df_in: pd.DataFrame, meta: dict) -> pd.DataFrame:
    dfn = df_in.copy()
    for col, hib in meta.items():
        vals = dfn[col].to_numpy(dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if np.isclose(vmax, vmin):
            # Avoid divide-by-zero: set all to 1.0 (or 0.5) if identical
            scaled = np.ones_like(vals)
        else:
            scaled = (vals - vmin) / (vmax - vmin)
        if not hib:
            scaled = 1.0 - scaled  # flip lower-better to higher-better
        dfn[col] = scaled
    return dfn

# Normalize to [0,1], consistent direction
categories = [c for c in df.columns if c != 'model']
df_norm = df[['model'] + categories].copy()
df_norm[categories] = minmax_unitize(df_norm[categories], metric_meta)

# ---- Radar chart ----
cats = categories
N = len(cats)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))

for _, row in df_norm.iterrows():
    vals = row[cats].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, label=row['model'])
    ax.fill(angles, vals, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats, fontsize=10)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'])
ax.set_ylim(0,1)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19),ncol=3)
plt.savefig('diversetask.pdf')
plt.show()

