import timeit
import pandas as pd
import torch
import gc
from torch.utils.data import ConcatDataset
import numpy as np

start_time = timeit.default_timer()
from fmtk.pipeline import Pipeline
import argparse

end_time = timeit.default_timer()
print(f"Time taken to import fmtk pipeline: {end_time - start_time} seconds")

from fmtk.datasets.etth1 import ETTh1Dataset
from fmtk.metrics import get_mse
from fmtk.components.decoders.forecasting.linear import LinearDecoder
from torch.utils.data import DataLoader, Subset, Dataset
from peft import LoraConfig
from fmtk.components.decoders.classification.repa_linear import (
    Repa,
    RepaWrappedDecoder,
)
import os
from tqdm import tqdm
import logging
import csv
import yaml
import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s",
)
logger = logging.getLogger("repa_train")

device = "cuda:0"
seed = 42
generator = torch.Generator()
generator.manual_seed(seed)


class AverageMeter:
    """Computes and stores the average and current value"""

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


def _extract_single_model(backbone_name, model_id, model_cfg, device, dataloader):
    """Extract features from one backbone, then free its GPU memory."""
    backbone = utils.get_backbone(backbone_name, model_id, device, model_cfg)
    P = Pipeline(backbone)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting [{model_id}]"):
            x, idx = batch["x"], batch["idx"]
            mask = batch.get("mask", None)
            P.forward(x, mask, idx=idx, use_cache=True)

    feats = P.embedding_cache._cache
    del P, backbone
    torch.cuda.empty_cache()
    return feats


def extract_features(
    dataloader_from, dataloader_to, model_cfg, device, model_features_path
):
    start_time = timeit.default_timer()

    feats_from = _extract_single_model(
        model_cfg["model_from_name"],
        model_cfg["model_from_id"],
        model_cfg,
        device,
        dataloader_from,
    )
    feats_to = _extract_single_model(
        model_cfg["model_to_name"],
        model_cfg["model_to_id"],
        model_cfg,
        device,
        dataloader_to,
    )

    end_time = timeit.default_timer()
    print(f"Time taken to extract features: {end_time - start_time} seconds")

    logger.info(f"Saving features to {model_features_path}")
    data = {
        model_cfg["name1"]: feats_from,
        model_cfg["name2"]: feats_to,
    }
    os.makedirs(os.path.dirname(model_features_path), exist_ok=True)
    # torch.save(data, model_features_path)
    return data


def train_one_epoch(dataloader_train, model, optimizer, device):
    loss_meter = AverageMeter()
    for batch in tqdm(dataloader_train):
        x, y = batch["x"].to(device), batch["y"].to(device)
        feats = model(x)
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
    print(y_test.shape, y_pred.shape)
    result = get_mse(y_test, y_pred)
    return result


def run_repa_training(
    dataloader_train,
    model_cfg,
    train_config,
    repa_cfg,
    device,
    dataloader_test,
    inference_config,
    log_path="",
):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    decoder_path = f"{base_dir}/src/fmtk/saved/etth1_fore_momentbase_linear/decoder.pth"
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(
            f"Moment decoder not found at {decoder_path}. "
            "Train a Moment decoder on ETTh1 first (e.e. via moment_etth1.py)."
        )
    repa_dir = f"{base_dir}/src/fmtk/saved/repa/etth1_{model_cfg['name1']}_to_{model_cfg['name2']}"
    os.makedirs(repa_dir, exist_ok=True)
    last_save_path = f"{repa_dir}/last.pth"
    best_save_path = f"{repa_dir}/best.pth"

    decoder = LinearDecoder(
        device, cfg={"input_dim": 49152, "output_dim": 96}
    )
    decoder = RepaWrappedDecoder(device, repa_cfg, decoder)
    decoder.load_decoder(decoder_path)

    backbone = utils.get_backbone(
        model_cfg["model_from_name"], model_cfg["model_from_id"], device, model_cfg
    )

    P_validation = Pipeline(backbone)
    P_validation.add_decoder(decoder, load=True)

    model = Repa(device, repa_cfg)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    logger.info("Training RepA model...")
    logger.info(f"RepA device: {next(model.parameters()).device}")
    best_loss = float("inf")
    for epoch in range(train_config["epochs"]):
        start_time = timeit.default_timer()
        loss = train_one_epoch(dataloader_train, model, optimizer, device)
        end_time = timeit.default_timer()
        print(f"Time taken for epoch {epoch + 1}: {end_time - start_time} seconds")
        logger.info(f"Epoch {epoch + 1}/{train_config['epochs']} Loss: {loss:.4f}")
        model.save(last_save_path)
        if loss < best_loss:
            best_loss = loss
            model.save(best_save_path)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            best_accuracy = run_repa_validation(
                best_save_path,
                P_validation,
                dataloader_test,
                inference_config,
            )
            logger.info(
                f"Epoch {epoch + 1}/{train_config['epochs']} Best Test accuracy: {best_accuracy}"
            )
            if log_path:
                with open(log_path, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, best_accuracy, best_loss])

    best_accuracy = run_repa_validation(
        best_save_path,
        P_validation,
        dataloader_test,
        inference_config,
    )
    logger.info(f"Best accuracy: {best_accuracy}")
    logger.info(f"Best loss: {best_loss}")


config_parser = parser = argparse.ArgumentParser(
    description="RepA Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)

parser = argparse.ArgumentParser(description="RepA Training (ETTh1 Chronos->Moment)")
model_group = parser.add_argument_group("Model parameters")

model_group.add_argument(
    "--model-from-name",
    type=str,
    default="moment",
    help="Model to convert from",
)
model_group.add_argument(
    "--model-to-name",
    type=str,
    default="moment",
    help="Model to convert to",
)
model_group.add_argument(
    "--model-from-id",
    type=str,
    default="small",
    help="Model to convert from (chronos: small/base/large)",
)
model_group.add_argument(
    "--model-to-id",
    type=str,
    default="small",
    help="Model to convert to (moment: small/base/large)",
)
model_group.add_argument(
    "--name1",
    type=str,
    default=None,
    help="Short name for the from-model (defaults to model-from-id)",
)
model_group.add_argument(
    "--name2",
    type=str,
    default=None,
    help="Short name for the to-model (defaults to model-to-id)",
)
model_group.add_argument(
    "--return-all-tokens", action="store_true", help="Return all tokens"
)
model_group.add_argument(
    "--model-features-path",
    type=str,
    default="../dataset/feature_dataset.pt",
    help="Path to the features from the model",
)

dataset_group = parser.add_argument_group("Dataset parameters")
dataset_group.add_argument(
    "--dataset-path",
    type=str,
    default="../datasets/ETTh1",
    help="Path to the dataset (folder containing ETTh1.csv)",
)
dataset_group.add_argument(
    "--dataset-name",
    type=str,
    default="etth1",
    help="Name of the dataset",
)

train_group = parser.add_argument_group("Training parameters")
train_group.add_argument(
    "--num-experiments", type=int, default=10, help="Number of experiments"
)
train_group.add_argument("--train-batch-size", type=int, default=64, help="Batch size")
train_group.add_argument(
    "--train-shuffle", type=bool, default=False, help="Shuffle the dataset"
)
train_group.add_argument("--epochs", type=int, default=50, help="Number of epochs")
train_group.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
train_group.add_argument("--scheduler", type=str, default="cosine", help="Scheduler")
train_group.add_argument(
    "--scheduler-T-max", type=int, default=10, help="T_max for the scheduler"
)
train_group.add_argument(
    "--scheduler-eta-min", type=float, default=0, help="eta_min for the scheduler"
)

inference_group = parser.add_argument_group("Inference parameters")
inference_group.add_argument(
    "--inference-batch-size", type=int, default=32, help="Batch size"
)
inference_group.add_argument(
    "--inference-shuffle", type=bool, default=False, help="Shuffle the dataset"
)

repa_group = parser.add_argument_group("Repa parameters")
repa_group.add_argument("--output-dim", type=int, default=10, help="Output dimension")
repa_group.add_argument(
    "--normalize", type=bool, default=True, help="Normalize the features"
)

parser.add_argument(
    "--num-samples-list",
    type=lambda s: [int(x) for x in s.split(",")],
    default=[1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000],
    help="List of number of samples to train on",
)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == "__main__":
    args, args_text = _parse_args()
    print(args_text)

    task_cfg = {"task_type": "forecasting"}
    train_config = {
        "batch_size": args.train_batch_size,
        "shuffle": args.train_shuffle,
        "epochs": args.epochs,
        "lr": args.lr,
        "scheduler": {
            "type": args.scheduler,
            "T_max": args.scheduler_T_max,
            "eta_min": args.scheduler_eta_min,
        },
        "use_cache": False,
    }
    inference_config = {
        "batch_size": args.inference_batch_size,
        "shuffle": args.inference_shuffle,
    }
    model_cfg = {
        "return_all_tokens": args.return_all_tokens,
        "model_from_id": args.model_from_id,
        "model_to_id": args.model_to_id,
        "model_from_name": args.model_from_name,
        "model_to_name": args.model_to_name,
        "name1": args.name1,
        "name2": args.name2,
    }

    if args.name1 is None:
        args.name1 = args.model_from_id
    if args.name2 is None:
        args.name2 = args.model_to_id

    args.input_dim = utils.get_embed_dim(args.model_from_name, args.model_from_id)
    args.repa_output_dim = utils.get_embed_dim(args.model_to_name, args.model_to_id)
    repa_cfg = {
        "input_dim": args.input_dim,
        "repa_output_dim": args.repa_output_dim,
        "output_dim": args.output_dim,
        # "normalize": args.normalize,
        "normalize": False,
    }

    model_features_path = (
        f"/project/pi_shenoy_umass_edu/kgudipaty/features/{args.dataset_name}/{args.name1}_to_{args.name2}_features.pt"
    )
    print(f"Model features path: {model_features_path}")

    dataset_cfg = {"dataset_path": args.dataset_path}

    train_data_from = ETTh1Dataset(
        dataset_cfg, task_cfg, split="train", forecast_horizon=96
    )
    train_data_to = ETTh1Dataset(
        dataset_cfg, task_cfg, split="train", forecast_horizon=96
    )
    test_data = ETTh1Dataset(dataset_cfg, task_cfg, split="test", forecast_horizon=96)

    print("Loading dataloaders...")
    dataloader_train_from = DataLoader(
        train_data_from,
        batch_size=train_config["batch_size"],
        shuffle=False,
        generator=generator,
    )
    dataloader_train_to = DataLoader(
        train_data_to,
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

    # if os.path.exists(model_features_path):
    #     logger.info("Loading features from disk...")
    #     data = torch.load(model_features_path)
    # else:
    #     logger.info("Creating feature dataset...")
    data = extract_features(
        dataloader_train_from,
        dataloader_train_to,
        model_cfg,
        device,
        model_features_path,
    )


    # ---------------------
    log_base_dir = f"results/repa/etth1"
    os.makedirs(log_base_dir, exist_ok=True)

    num_samples_list = args.num_samples_list
    for num_samples in num_samples_list:
        log_path = f"{log_base_dir}/{args.name1}_to_{args.name2}_accuracy_num_samples_{num_samples}.csv"
        for _ in range(args.num_experiments):
            x, y = data[model_cfg["name1"]], data[model_cfg["name2"]]
            keys = list(x.keys())
            n = min(num_samples, len(keys))
            chosen_positions = np.random.choice(len(keys), n, replace=False)
            chosen_keys = [keys[i] for i in chosen_positions]
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
                log_path=log_path,
            )
