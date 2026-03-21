import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from collections import defaultdict
import numpy as np

plt.rcParams["text.usetex"] = False
plt.style.use(["science", "no-latex"])


font = {"weight": "bold", "size": 12}
# plt.rc("font", **font)


# plt.figure(figsize=(4,3))
def eurosat_decoder_accuracy():
    dfs = []
    for i in [1, 5, 10, 20]:
        df = pd.read_csv(f"dino_resic45_accuracy_epochs_{i}.csv")
        dfs.append(df)

    plt.plot(dfs[0]["samples_per_class"] * 10, dfs[0]["accuracy"], label="1 epoch")
    plt.plot(dfs[1]["samples_per_class"] * 10, dfs[1]["accuracy"], label="5 epoch")
    plt.plot(dfs[2]["samples_per_class"] * 10, dfs[2]["accuracy"], label="10 epoch")
    plt.plot(dfs[3]["samples_per_class"] * 10, dfs[3]["accuracy"], label="20 epoch")

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    plt.axhline(y=0.87, color="red", linestyle="--")

    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.savefig("dino_resisc45_accuracy.pdf")
    plt.show()


def plot_repa_decoder_accuracy(epochs, samples, trials=10):

    mins = defaultdict(list)
    maxs = defaultdict(list)
    means = defaultdict(list)
    medians = defaultdict(list)
    for num in samples:
        file = f"repa/dino_base_to_small_accuracy_num_samples_{num}.csv"
        df = pd.read_csv(file, header=None)
        df.drop(columns=[2], inplace=True)
        df = df.groupby(0)

        for index, group in df:
            mins[index].append(group[1].min())
            maxs[index].append(group[1].max())
            means[index].append(group[1].mean())
            medians[index].append(group[1].median())

    for epoch in epochs:
        # plt.plot(samples, means[epoch], label=f"Epoch {epoch}")
        plt.errorbar(
            samples,
            means[epoch],
            yerr=[
                np.array(means[epoch]) - np.array(mins[epoch]),
                np.array(maxs[epoch]) - np.array(means[epoch]),
            ],
            label=f"Epoch {epoch}",
            capsize=1,
            capthick=.5,
            elinewidth=.5,
            # ecolor="black",
        )
        # plt.plot(samples, medians[epoch], label=f"Epoch {epoch}")

    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")

    plt.axhline(y=0.9444, color="red", linestyle="--")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.yticks([0.2, 0.4, 0.6, 0.8])

    ticks = [*plt.gca().get_yticks(), 0.9444]
    labels = [*plt.gca().get_yticks(), "orig."] 
    plt.gca().set_yticks(ticks)
    plt.gca().set_yticklabels(labels)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles, labels)
    plt.savefig(f"repa/repa_experiment.pdf")
    plt.show()

def plot_repa_backbone_comparison(samples, epoch=10):
    """
    One line per backbone (at a fixed epoch), x-axis = num training samples.

    Backbones:
        1. DINOv2 base → small   (dino_base_to_small)
        2. MAE base → DINOv2 base (mae-base_to_dinov2-base)
        3. Swin small → DINOv2 base (swin-small_to_dinov2-base)
    """
    # backbones = {
    #     "moment-base": "repa/uwave_moment-base_to_moment-small_accuracy_num_samples_",
    #     "moment-large": "repa/uwave_moment-large_to_moment-small_accuracy_num_samples_",
    #     "mantis-8M":   "repa/uwave_mantis-8M_to_moment-small_accuracy_num_samples_",
        
    # }

    backbones = {
        "moment-small": "repa/etth1/moment-small_to_moment-base_accuracy_num_samples_",
        
    }

    def _parse_accuracy(val):
        """Extract scalar from either a plain number or a stringified tuple like '(0.04, [...])'."""
        if isinstance(val, str) and val.startswith("("):
            return float(val.split(",", 1)[0].lstrip("("))
        return float(val)

    for label, prefix in backbones.items():
        means, mins, maxs = [], [], []
        for num in samples:
            file = f"{prefix}{num}.csv"
            df = pd.read_csv(file, header=None,nrows=25)
            df.columns = ["epoch", "accuracy", "loss"]
            df["accuracy"] = df["accuracy"].apply(_parse_accuracy)
            group = df[df["epoch"] == epoch]["accuracy"]
            means.append(group.mean())
            mins.append(group.min())
            maxs.append(group.max())

        means = np.array(means)
        mins = np.array(mins)
        maxs = np.array(maxs)

        plt.errorbar(
            samples,
            means,
            yerr=[means - mins, maxs - means],
            label=label,
            capsize=1,
            capthick=0.5,
            elinewidth=0.5,
        )

    # last_mean = means
    # last_min = mins
    # last_max = maxs
    
    # df = pd.read_csv(f"{backbones['DINOv3-small']}{2000}.csv", header=None,skiprows=25)
    # df.columns = ["epoch", "accuracy", "loss"]
    # df["accuracy"] = df["accuracy"].apply(_parse_accuracy)
    # group = df[df["epoch"] == epoch]["accuracy"]
    # last_mean[-1] = group.mean()
    # last_min[-1] = group.min()
    # last_max[-1] = group.max()

    # df = pd.read_csv(f"{backbones['DINOv3-small']}{3000}.csv", header=None,skiprows=25)
    # df.columns = ["epoch", "accuracy", "loss"]
    # df["accuracy"] = df["accuracy"].apply(_parse_accuracy)
    # group = df[df["epoch"] == epoch]["accuracy"]
    # last_mean[-2] = group.mean()
    # last_min[-2] = group.min()
    # last_max[-2] = group.max()

    # plt.errorbar(
    #         samples,
    #         last_mean,
    #         yerr=[last_mean - last_min, last_max - last_mean],
    #         label="DINOv3-small-1x1",
    #         capsize=1,
    #         capthick=0.5,
    #         elinewidth=0.5,
    #     )

    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")

    plt.axhline(y=0.4257, color="red", linestyle="--")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

    ticks = [*plt.gca().get_yticks(), 0.95]
    tick_labels = [*plt.gca().get_yticks(), "orig."]
    plt.gca().set_yticks(ticks)
    plt.gca().set_yticklabels(tick_labels)

    handles, legend_labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles, legend_labels, loc="lower right")
    plt.savefig("repa/repa_etth1fore.pdf")
    plt.show()


if __name__ == "__main__":
    samples = [1, 5, 10, 50, 100, 500, 1000, 2000, 3000]

    # per-epoch plot (existing)
    # epochs = [1, 5, 10, 20]
    # plot_repa_decoder_accuracy(epochs, samples)

    # backbone comparison at epoch 10
    plot_repa_backbone_comparison(samples, epoch=10)
