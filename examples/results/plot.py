import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.rcParams['text.usetex'] = False
plt.style.use(["science", "no-latex"])


font = {"weight": "bold", "size": 12}
# plt.rc("font", **font)

# plt.figure(figsize=(4,3))
dfs = []
for i in [1,5,10,20]:
    df = pd.read_csv(f"dino_resic45_accuracy_epochs_{i}.csv")
    dfs.append(df)


plt.plot(dfs[0]["samples_per_class"] * 10, dfs[0]["accuracy"], label="1 epoch")
plt.plot(dfs[1]["samples_per_class"] * 10, dfs[1]["accuracy"], label="5 epoch")
plt.plot(dfs[2]["samples_per_class"] * 10, dfs[2]["accuracy"], label="10 epoch")
plt.plot(dfs[3]["samples_per_class"] * 10, dfs[3]["accuracy"], label="20 epoch")

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])

plt.axhline(y=0.87, color='red', linestyle='--')



plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")


plt.legend()
plt.savefig("dino_resisc45_accuracy.pdf")
plt.show()