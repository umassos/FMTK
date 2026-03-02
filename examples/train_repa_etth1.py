"""
RepA training for ETTh1 forecasting: transfer from Mantis to Chronos.

Transforms Mantis embeddings to Chronos embeddings so a pre-trained Chronos
decoder can be used with the Mantis backbone for inference.

Prerequisites:
  1. Train a Chronos decoder on ETTh1 forecasting first, e.g.:
     - Create chronos_etth1.py (or use moment_etth1.py as template with Chronos backbone)
     - Train and save decoder to src/fmtk/saved/etth1_fore_chronossmall_linear/decoder.pth
"""
import timeit
import argparse
import os
import logging
import csv
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

start_time = timeit.default_timer()
from fmtk.pipeline import Pipeline

end_time = timeit.default_timer()
print(f"Time taken to import fmtk pipeline: {end_time - start_time} seconds")

from fmtk.datasets.etth1 import ETTh1Dataset
from fmtk.components.decoders.forecasting.linear import LinearDecoder
from fmtk.components.decoders.classification.repa_linear import Repa, RepaWrappedDecoder
import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s",
)
logger = logging.getLogger("repa_etth1")

device = "cuda:0"
seed = 42
generator = torch.Generator()
generator.manual_seed(seed)

# ETTh1: 7 channels, forecast_horizon 96
N_CHANNELS = 7
FORECAST_HORIZON = 96


def get_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FeatureDataset(Dataset):
    def __init__(self, feats_x, feats_y):
        super().__init__()
        self.keys = list(feats_x.keys())
        self.dataset = {}
        for k in self.keys:
            self.dataset[k] = (feats_x[k], feats_y[k])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        repr_x, repr_y = self.dataset[self.keys[idx]]
        return {"x": repr_x, "y": repr_y, "idx": idx}


class RepaWrappedForecastingDecoder(RepaWrappedDecoder):
    """RepA-wrapped decoder for forecasting. Reshapes RepA output to [B, C, D] for Chronos."""

    def __init__(self, device, cfg, decoder, n_channels=N_CHANNELS):
        super().__init__(device, cfg, decoder)
        self.n_channels = n_channels
        self.repa_output_dim = cfg["repa_output_dim"]
        # Use decoder's criterion (MSELoss) for forecasting
        self.criterion = decoder.criterion

    def forward(self, x):
        x = x.to(device=self.device, dtype=torch.float32)
        # Mantis outputs [B, 32, 256]; flatten to [B, 8192] for RepA
        if x.ndim == 3:
            x = x.flatten(1)
        x = self.repa(x)
        # Reshape [B, n_channels * embed_dim] -> [B, n_channels, embed_dim] for decoder
        embed_dim = self.repa_output_dim // self.n_channels
        x = x.reshape(x.shape[0], self.n_channels, embed_dim)
        return self.decoder(x)


def _extract_mantis(dataloader, model_cfg, device):
    """Extract Mantis features. Output: [B, 32, 256] -> flatten to [B, 8192] per sample."""
    backbone = utils.get_backbone(
        model_cfg["model_from_name"], model_cfg["model_from_id"], device, model_cfg
    )
    feats = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting [Mantis]"):
            x = batch["x"]
            idx = batch["idx"]
            mask = batch.get("mask", None)
            out = backbone.forward(x, mask)
            # Mantis: [B, 32, 256] -> [B, 8192]
            out = out.flatten(1)
            for i in range(len(idx)):
                feats[idx[i].item()] = out[i].cpu().float()
    del backbone
    torch.cuda.empty_cache()
    return feats


def _extract_chronos(dataloader, model_cfg, device):
    """Extract Chronos features. Output: [B*7, E] -> reshape to [B, 7*E] per sample."""
    backbone = utils.get_backbone(
        model_cfg["model_to_name"], model_cfg["model_to_id"], device, model_cfg
    )
    feats = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting [Chronos]"):
            x = batch["x"]
            idx = batch["idx"]
            mask = batch.get("mask", None)
            out = backbone.forward(x, mask)
            # Chronos: [B*7, E] -> [B, 7*E]
            B = len(idx)
            E = out.shape[-1]
            out = out.reshape(B, N_CHANNELS * E)
            for i in range(B):
                feats[idx[i].item()] = out[i].cpu().float()
    del backbone
    torch.cuda.empty_cache()
    return feats


def extract_features(
    dataloader_from, dataloader_to, model_cfg, device, model_features_path
):
    start_time = timeit.default_timer()
    feats_from = _extract_mantis(dataloader_from, model_cfg, device)
    feats_to = _extract_chronos(dataloader_to, model_cfg, device)
    end_time = timeit.default_timer()
    print(f"Time taken to extract features: {end_time - start_time} seconds")

    logger.info(f"Saving features to {model_features_path}")
    data = {
        model_cfg["name1"]: feats_from,
        model_cfg["name2"]: feats_to,
    }
    os.makedirs(os.path.dirname(model_features_path), exist_ok=True)
    torch.save(data, model_features_path)
    return data


def train_one_epoch(dataloader_train, model, optimizer, device):
    loss_meter = AverageMeter()
    for batch in tqdm(dataloader_train):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if x.ndim == 3:
            x = x.flatten(1)
        if y.ndim == 3:
            y = y.flatten(1)
        feats = model(x)
        if feats.ndim > 2:
            feats = feats.flatten(1)
        if y.ndim > 2:
            y = y.flatten(1)
        loss = model.criterion(feats, y)
        loss_meter.update(loss.item(), x.size(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss_meter.avg


def run_repa_validation(
    repa_path,
    P,
    dataloader_test,
    inference_config,
):
    P.active_decoder.load_repa(repa_path)
    y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
    return get_mse(y_test, y_pred)


def run_repa_training(
    dataloader_train,
    model_cfg,
    train_config,
    repa_cfg,
    device,
    dataloader_test,
    inference_config,
    decoder_path,
    log_path="",
):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(
            f"Chronos decoder not found at {decoder_path}. "
            "Train a Chronos decoder on ETTh1 first (e.g. chronos_etth1.py)."
        )
    repa_dir = f"{base_dir}/src/fmtk/saved/repa/etth1_{model_cfg['name1']}_to_{model_cfg['name2']}"
    os.makedirs(repa_dir, exist_ok=True)
    last_save_path = f"{repa_dir}/last.pth"
    best_save_path = f"{repa_dir}/best.pth"

    chronos_embed_dim = utils.get_embed_dim(
        model_cfg["model_to_name"], model_cfg["model_to_id"]
    )
    decoder = LinearDecoder(
        device,
        cfg={
            "input_dim": chronos_embed_dim,
            "output_dim": FORECAST_HORIZON,
            "head_dropout": 0.0,
        },
    )
    decoder = RepaWrappedForecastingDecoder(device, repa_cfg, decoder)
    decoder.load_decoder(decoder_path)

    backbone = utils.get_backbone(
        model_cfg["model_from_name"], model_cfg["model_from_id"], device, model_cfg
    )

    P_validation = Pipeline(backbone)
    P_validation.add_decoder(decoder, load=True)

    model = Repa(device, repa_cfg)
    model.repa.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    logger.info("Training RepA model (Mantis -> Chronos)...")
    best_mse = float("inf")
    for epoch in range(train_config["epochs"]):
        start_time = timeit.default_timer()
        loss = train_one_epoch(dataloader_train, model, optimizer, device)
        end_time = timeit.default_timer()
        print(f"Epoch {epoch + 1} time: {end_time - start_time:.2f}s")
        logger.info(f"Epoch {epoch + 1}/{train_config['epochs']} RepA Loss: {loss:.4f}")
        model.save(last_save_path)
        if loss < best_mse:
            best_mse = loss
            model.save(best_save_path)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            mse = run_repa_validation(
                best_save_path,
                P_validation,
                dataloader_test,
                inference_config,
            )
            logger.info(f"Epoch {epoch + 1} Test MSE: {mse:.4f}")
            if log_path:
                with open(log_path, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, mse, loss])

    final_mse = run_repa_validation(
        best_save_path, P_validation, dataloader_test, inference_config
    )
    logger.info(f"Best Test MSE: {final_mse:.4f}")
    return final_mse


# Argument parsing
config_parser = argparse.ArgumentParser(
    description="RepA ETTh1 Config", add_help=False
)
config_parser.add_argument(
    "-c", "--config", default="", type=str, metavar="FILE", help="YAML config file"
)

parser = argparse.ArgumentParser(
    description="RepA Training: ETTh1 forecasting Mantis -> Chronos"
)
model_group = parser.add_argument_group("Model parameters")
model_group.add_argument(
    "--model-from-name",
    type=str,
    default="mantis",
    help="Source model (mantis)",
)
model_group.add_argument(
    "--model-to-name",
    type=str,
    default="chronos",
    help="Target model (chronos)",
)
model_group.add_argument(
    "--model-from-id",
    type=str,
    default="8M",
    help="Mantis model id",
)
model_group.add_argument(
    "--model-to-id",
    type=str,
    default="small",
    help="Chronos model id (small/base/large)",
)
model_group.add_argument(
    "--name1", type=str, default=None, help="Short name for source (default: model-from-id)"
)
model_group.add_argument(
    "--name2", type=str, default=None, help="Short name for target (default: model-to-id)"
)
model_group.add_argument(
    "--decoder-path",
    type=str,
    default=None,
    help="Path to pre-trained Chronos decoder (default: src/fmtk/saved/etth1_fore_chronossmall_linear/decoder.pth)",
)

dataset_group = parser.add_argument_group("Dataset parameters")
dataset_group.add_argument(
    "--dataset-path",
    type=str,
    default="../datasets/ETTh1",
    help="Path to ETTh1 dataset",
)
dataset_group.add_argument(
    "--forecast-horizon",
    type=int,
    default=96,
    help="Forecast horizon",
)

train_group = parser.add_argument_group("Training parameters")
train_group.add_argument("--num-experiments", type=int, default=1)
train_group.add_argument("--train-batch-size", type=int, default=32)
train_group.add_argument("--train-shuffle", type=bool, default=True)
train_group.add_argument("--epochs", type=int, default=50)
train_group.add_argument("--lr", type=float, default=1e-3)

inference_group = parser.add_argument_group("Inference parameters")
inference_group.add_argument("--inference-batch-size", type=int, default=32)
inference_group.add_argument("--inference-shuffle", type=bool, default=False)

repa_group = parser.add_argument_group("Repa parameters")
repa_group.add_argument("--normalize", type=bool, default=True)

parser.add_argument(
    "--num-samples-list",
    type=lambda s: [int(x) for x in s.split(",")],
    default=[1000, 5000, 10000],
    help="List of sample counts for training",
)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args, yaml.safe_dump(args.__dict__, default_flow_style=False)


if __name__ == "__main__":
    args, args_text = _parse_args()
    print(args_text)

    task_cfg = {"task_type": "forecasting"}
    train_config = {
        "batch_size": args.train_batch_size,
        "shuffle": args.train_shuffle,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    inference_config = {
        "batch_size": args.inference_batch_size,
        "shuffle": args.inference_shuffle,
    }
    model_cfg = {
        "model_from_id": args.model_from_id,
        "model_to_id": args.model_to_id,
        "model_from_name": args.model_from_name,
        "model_to_name": args.model_to_name,
        "name1": args.name1 or args.model_from_id,
        "name2": args.name2 or args.model_to_id,
    }

    # Mantis: [B, 32, 256] -> 8192; Chronos: [B, 7, 512] -> 3584 for small
    mantis_dim = utils.get_embed_dim(args.model_from_name, args.model_from_id)
    chronos_dim = utils.get_embed_dim(args.model_to_name, args.model_to_id)
    # Mantis flattens to 32*256=8192; Chronos to 7*512=3584
    repa_input_dim = 32 * mantis_dim  # Mantis num_patches * embed_dim
    repa_output_dim = N_CHANNELS * chronos_dim

    repa_cfg = {
        "input_dim": repa_input_dim,
        "repa_output_dim": repa_output_dim,
        "normalize": args.normalize,
    }

    base_dir = os.path.dirname(os.path.dirname(__file__))
    decoder_path = args.decoder_path or (
        f"{base_dir}/src/fmtk/saved/etth1_fore_chronos{args.model_to_id}_linear/decoder.pth"
    )
    model_features_path = (
        f"../features/etth1/{model_cfg['name1']}_to_{model_cfg['name2']}_features.pt"
    )

    dataset_cfg = {
        "dataset_path": args.dataset_path,
        "model_id": "paris-noah/Mantis-8M",
    }

    train_data = ETTh1Dataset(
        dataset_cfg, task_cfg, split="train", forecast_horizon=args.forecast_horizon
    )
    test_data = ETTh1Dataset(
        dataset_cfg, task_cfg, split="test", forecast_horizon=args.forecast_horizon
    )

    dataloader_train_from = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=False,
        generator=generator,
    )
    dataloader_train_to = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=False,
        generator=generator,
    )
    dataloader_test = DataLoader(
        test_data,
        batch_size=inference_config["batch_size"],
        shuffle=inference_config["shuffle"],
        generator=generator,
    )

    if os.path.exists(model_features_path):
        logger.info("Loading features from disk...")
        data = torch.load(model_features_path)
    else:
        logger.info("Extracting features...")
        data = extract_features(
            dataloader_train_from,
            dataloader_train_to,
            model_cfg,
            device,
            model_features_path,
        )

    for num_samples in args.num_samples_list:
        log_path = f"results/repa/etth1_{model_cfg['name1']}_to_{model_cfg['name2']}_mse_num_samples_{num_samples}.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        for exp in range(args.num_experiments):
            x, y = data[model_cfg["name1"]], data[model_cfg["name2"]]
            keys = list(x.keys())
            n = min(num_samples, len(keys))
            chosen = np.random.choice(len(keys), n, replace=False)
            chosen_keys = [keys[i] for i in chosen]
            x_sub = {k: x[k] for k in chosen_keys}
            y_sub = {k: y[k] for k in chosen_keys}
            data_train = FeatureDataset(x_sub, y_sub)
            dataloader_train = DataLoader(
                data_train,
                batch_size=train_config["batch_size"],
                shuffle=train_config["shuffle"],
                generator=generator,
            )
            run_repa_training(
                dataloader_train,
                model_cfg,
                train_config,
                repa_cfg,
                device,
                dataloader_test,
                inference_config,
                decoder_path,
                log_path=log_path,
            )
