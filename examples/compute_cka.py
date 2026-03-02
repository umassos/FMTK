import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# CKA utilities
# -------------------------
@torch.no_grad()
def _center_gram(K: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """
    Center an (n x n) Gram matrix K.
    If unbiased=True, uses the unbiased estimator (more expensive / can be noisier).
    """
    n = K.size(0)
    if K.dim() != 2 or K.size(0) != K.size(1):
        raise ValueError(f"Expected square Gram matrix, got {tuple(K.shape)}")

    if not unbiased:
        # Biased (standard) centering: Kc = H K H
        mean_row = K.mean(dim=0, keepdim=True)
        mean_col = K.mean(dim=1, keepdim=True)
        mean_all = K.mean()
        return K - mean_row - mean_col + mean_all

    # Unbiased centering (per Kornblith et al. / Szekely-Rizzo style)
    # Zero diagonal, then apply correction terms
    K = K.clone()
    K.fill_diagonal_(0)
    mean_row = K.sum(dim=0, keepdim=True) / (n - 2)
    mean_col = K.sum(dim=1, keepdim=True) / (n - 2)
    mean_all = K.sum() / ((n - 1) * (n - 2))
    Kc = K - mean_row - mean_col + mean_all
    Kc.fill_diagonal_(0)
    return Kc


@torch.no_grad()
def linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    unbiased: bool = False,
    eps: float = 1e-12,
) -> float:
    """
    Compute linear CKA between representations X and Y.

    Args:
        X: (n, dx) tensor
        Y: (n, dy) tensor
        unbiased: whether to use unbiased centering of Gram matrices
        eps: numerical stability

    Returns:
        cka in [0, 1] (typically)
    """
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError("X and Y must be 2D tensors of shape (n, d).")
    if X.size(0) != Y.size(0):
        raise ValueError(f"Number of samples must match: {X.size(0)} vs {Y.size(0)}")

    # Ensure float
    X = X.float()
    Y = Y.float()

    # Compute Gram matrices (n x n)
    K = X @ X.t()
    L = Y @ Y.t()

    # Center them
    Kc = _center_gram(K, unbiased=unbiased)
    Lc = _center_gram(L, unbiased=unbiased)

    # HSIC terms
    hsic = (Kc * Lc).sum()
    norm_x = torch.sqrt((Kc * Kc).sum()).clamp_min(eps)
    norm_y = torch.sqrt((Lc * Lc).sum()).clamp_min(eps)

    cka_val = (hsic / (norm_x * norm_y)).item()
    return cka_val


# -------------------------
# Procrustes utilities
# -------------------------
@torch.no_grad()
def orthogonal_procrustes(
    X: torch.Tensor,
    Y: torch.Tensor,
    allow_scale: bool = True,
    center: bool = True,
    eps: float = 1e-12,
) -> dict:
    """
    Orthogonal Procrustes: min_{R^T R = I} || X - s Y R ||_F^2

    Requires X and Y to have the same shape (n, d).

    Returns:
        {
            "R": (d, d) orthogonal matrix,
            "s": float scale (1.0 if allow_scale=False),
            "error": normalized Frobenius error ||X - sYR||_F / ||X||_F,
            "mse": mean squared error per entry
        }
    """
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError("X and Y must be 2D (n, d).")
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape for orthogonal Procrustes. Got {X.shape} vs {Y.shape}")

    X = X.float()
    Y = Y.float()

    if center:
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

    # Cross-covariance
    M = Y.t() @ X  # (d, d)

    # SVD: M = U S V^T
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh  # (d, d)

    if allow_scale:
        numerator = torch.trace(X.t() @ (Y @ R))
        denom = (Y * Y).sum().clamp_min(eps)
        s = (numerator / denom).item()
    else:
        s = 1.0

    Y_aligned = (Y @ R) * s
    diff = X - Y_aligned

    frob_X = torch.linalg.norm(X, ord="fro").clamp_min(eps)
    error = (torch.linalg.norm(diff, ord="fro") / frob_X).item()
    mse = (diff * diff).mean().item()

    return {"R": R, "s": s, "error": error, "mse": mse}


@torch.no_grad()
def _pca_project(Z: torch.Tensor, k: int) -> torch.Tensor:
    """
    PCA projection of Z (n, d) to (n, k) using SVD.
    Returns PCA coordinates U[:, :k] * S[:k].
    """
    if Z.dim() != 2:
        raise ValueError("Z must be 2D (n, d).")
    Z = Z.float()
    Z = Z - Z.mean(dim=0, keepdim=True)
    U, S, _ = torch.linalg.svd(Z, full_matrices=False)
    return U[:, :k] * S[:k]


@torch.no_grad()
def procrustes_mismatch_dims(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int | None = None,
    allow_scale: bool = True,
) -> dict:
    """
    Procrustes for mismatched dims by projecting both X and Y to shared k dims (PCA),
    then running orthogonal Procrustes in that shared space.

    Args:
        X: (n, dx)
        Y: (n, dy)
        k: shared projection dim. Default = min(dx, dy)
        allow_scale: fit scale term s as well

    Returns:
        Same dict as orthogonal_procrustes plus "k".
    """
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError("X and Y must be 2D (n, d).")
    if X.size(0) != Y.size(0):
        raise ValueError("X and Y must have the same number of samples (n).")

    dx = X.size(1)
    dy = Y.size(1)
    if k is None:
        k = min(dx, dy)
    if not (1 <= k <= min(dx, dy)):
        raise ValueError(f"k must be in [1, {min(dx, dy)}], got {k}")

    Xk = _pca_project(X, k)
    Yk = _pca_project(Y, k)

    out = orthogonal_procrustes(Xk, Yk, allow_scale=allow_scale, center=False)
    out["k"] = k
    return out


# -------------------------
# Plotting helpers
# -------------------------
def plot_cka(names):
    dfs = []
    labels = []
    for name1, name2 in names:
        cka_path = f"/home/kgudipaty_umass_edu/FMTK/examples/results/uwave_{name1}_to_{name2}_cka_mean.csv"
        cka_df = pd.read_csv(cka_path)
        dfs.append(cka_df[1:])
        labels.append(f"{name1} to {name2}")

    for i, df in enumerate(dfs):
        plt.plot(df["num_samples"], df["cka"], marker="o", label=labels[i])

    plt.xlabel("Number of training samples")
    plt.ylabel("CKA")
    plt.legend()
    plt.savefig(f"/home/kgudipaty_umass_edu/FMTK/examples/results/UWAVE_CKA.png")
    plt.show()


def plot_procrustes(names):
    dfs = []
    labels = []
    for name1, name2 in names:
        proc_path = f"/home/kgudipaty_umass_edu/FMTK/examples/results/uwave_{name1}_to_{name2}_procrustes_mean.csv"
        proc_df = pd.read_csv(proc_path)
        dfs.append(proc_df[1:])
        labels.append(f"{name1} to {name2}")

    for i, df in enumerate(dfs):
        plt.plot(df["num_samples"], df["procrustes_error"], marker="o", label=labels[i])

    plt.xlabel("Number of training samples")
    plt.ylabel("Procrustes error (||X - sYR||_F / ||X||_F)")
    plt.legend()
    plt.savefig(f"/home/kgudipaty_umass_edu/FMTK/examples/results/UWAVE_PROCRUSTES.png")
    plt.show()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    task_cfg = {"task_type": "classification"}
    train_config = {
        "batch_size": 32,
        "shuffle": False,
        "epochs": 50,
        "lr": 1e-3,
        "scheduler": {"type": "cosine", "T_max": 10, "eta_min": 0},
        "use_cache": True,
    }
    inference_config = {"batch_size": 32, "shuffle": False}
    dataset_cfg = {
        "dataset_path": "../datasets/UWaveGestureLibrary",
        # "model_id": "facebook/dinov2-base",
        "model_id": "AutonLab/MOMENT-1-small",
    }
    model_cfg = {"return_all_tokens": False}
    print("Loading features from disk...")

    names = [
        ("moment-base", "moment-small"),
        ("moment-large", "moment-small"),
        ("mantis-8M", "moment-small"),
        ("chronos-small", "moment-small"),
    ]

    # ----------------------------------------------------
    # Compute CKA + Procrustes (uncomment to recompute CSVs)
    # ----------------------------------------------------
    for name1, name2 in names[1:]:
        model_features_path = f"/home/kgudipaty_umass_edu/FMTK/features/uwave/{name1}_to_{name2}_features.pt"
        data = torch.load(model_features_path)
    
        num_samples_list = [1, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
        num_runs = 5
    
        # --- CKA rows ---
        cka_rows = []
        # --- Procrustes rows ---
        proc_rows = []
    
        for num_samples in num_samples_list:
            print(f"Processing {name1} to {name2} with {num_samples} samples")
            for run in range(num_runs):
                print(f"Processing run {run} of {num_runs}")
                x, y = data[name1], data[name2]
                keys = list(x.keys())
                n = min(num_samples, len(keys))
                chosen_positions = np.random.choice(len(keys), n, replace=False)
                chosen_keys = [keys[i] for i in chosen_positions]
    
                x_sub = torch.stack([x[k] for k in chosen_keys], dim=0)
                y_sub = torch.stack([y[k] for k in chosen_keys], dim=0)
    
                # Flatten if 3D/4D (tokens) -> sample vectors
                if x_sub.ndim > 2:
                    x_sub = x_sub.flatten(1)
                if y_sub.ndim > 2:
                    y_sub = y_sub.flatten(1)
    
                # ---- CKA ----
                # cka = linear_cka(x_sub, y_sub, unbiased=False)
                # cka_rows.append({"num_samples": num_samples, "run": run, "cka": cka})
    
                # ---- Procrustes ----
                # Use direct orthogonal Procrustes if dims match; else PCA->k->Procrustes
                if x_sub.size(1) == y_sub.size(1):
                    proc = orthogonal_procrustes(x_sub, y_sub, allow_scale=True, center=True)
                else:
                    # default k=min(dx,dy) is fine; you can also set k explicitly
                    proc = procrustes_mismatch_dims(x_sub, y_sub, k=min(x_sub.size(1), y_sub.size(1)), allow_scale=True)
    
                proc_rows.append(
                    {
                        "num_samples": num_samples,
                        "run": run,
                        "procrustes_error": proc["error"],
                        "procrustes_mse": proc["mse"],
                        "procrustes_scale": proc["s"],
                        "k": proc.get("k", x_sub.size(1)),
                    }
                )
    
        # Save aggregated CKA
        # cka_df = pd.DataFrame(cka_rows).groupby("num_samples").agg({"cka": "mean"}).reset_index()
        # cka_df.to_csv(
        #     f"/home/kgudipaty_umass_edu/FMTK/examples/results/uwave_{name1}_to_{name2}_cka_mean.csv",
        #     index=False,
        # )
    
        # Save aggregated Procrustes
        proc_df = (
            pd.DataFrame(proc_rows)
            .groupby("num_samples")
            .agg(
                {
                    "procrustes_error": "mean",
                    "procrustes_mse": "mean",
                    "procrustes_scale": "mean",
                    "k": "mean",
                }
            )
            .reset_index()
        )
        proc_df.to_csv(
            f"/home/kgudipaty_umass_edu/FMTK/examples/results/uwave_{name1}_to_{name2}_procrustes_mean.csv",
            index=False,
        )

    # Plot existing CSVs
    # plot_cka(names)
    plot_procrustes(names)