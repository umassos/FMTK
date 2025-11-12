# # d_e_a
# 8.085966901081365
# 98.06262564659119
# 0.05816078186035156
# 0.30655384063720703

# #d
# 8.125248792694837
# 124.60682344436646
# 0.10486888885498047
# 0.8356707096099854

# #d_e
# 7.851133020912728
# 92.82743573188782
# 0.046140074729919434
# 0.2920846939086914

# #d_a
# 7.711800179830411
# 278.99744963645935
# 0.34123289585113525
# 1.2092335224151611


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

labels = ['decoder (d)', 'd \n + adapter (a)', 'd \n + encoder (e)', 'd+e+a']  
inference_time=[104.86,341.23,46.14,58.16]
mae=[8.12,7.71,7.85,8.08]

x = np.arange(len(labels))
width = 0.4

fig, ax1 = plt.subplots(figsize=(8, 4))

# Left axis: inference_time bars (shifted left)
b1 = ax1.bar(x - width/2, inference_time, width, label='Inference time',color='#faa275')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel('Inference time (ms)')
ax1.set_ylim(0,350)

# Right axis: MAE bars (shifted right)
ax2 = ax1.twinx()
b2 = ax2.bar(x + width/2, mae, width, label='MAE',color='#ce6a85')
ax2.set_ylabel('MAE')
ax2.set_ylim(0,9)
# One legend for both axes
handles = [b1, b2]
labels_legend = [h.get_label() for h in handles]
ax1.legend(handles, labels_legend,bbox_to_anchor=(0.5, 1.19),loc='upper center',ncol=2)

plt.tight_layout()
sns.despine(fig, right=False, top=True)
plt.savefig("diversecomponent.pdf")
plt.show()


