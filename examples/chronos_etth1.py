"""
Train Chronos decoder on ETTh1 forecasting.

Chronos flattens channels (B,7,512) -> (B*7,512), so its output is [B*7, E].
This script uses a decoder that reshapes to [B, 7, E] before the linear layer.

Prerequisite for train_repa_etth1.py (Mantis -> Chronos transfer).
"""
import timeit
import torch
import gc
import numpy as np

start_time = timeit.default_timer()
from fmtk.pipeline import Pipeline

end_time = timeit.default_timer()
print(f"Time taken to import fmtk pipeline: {end_time - start_time} seconds")

from torch.utils.data import DataLoader
from fmtk.datasets.etth1 import ETTh1Dataset
from fmtk.components.backbones.chronos import ChronosModel
from fmtk.components.decoders.forecasting.linear import LinearDecoder

device = "cuda:0"
seed = 42
generator = torch.Generator()
generator.manual_seed(seed)

N_CHANNELS = 7
FORECAST_HORIZON = 96


def get_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def get_chronos_embed_dim(model_id):
    if model_id == "small":
        return 512
    elif model_id == "base":
        return 768
    elif model_id == "large":
        return 1024
    elif model_id in ("mini", "tiny"):
        return 256 if model_id == "tiny" else 384
    return 512


class ChronosETTh1LinearDecoder(LinearDecoder):
    """Linear decoder for Chronos on ETTh1. Reshapes [B*7, E] -> [B, 7, E] before linear."""

    def forward(self, x):
        x = x.to(torch.float32).to(self.device)
        B7, E = x.shape
        B = B7 // N_CHANNELS
        x = x.reshape(B, N_CHANNELS, E)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.model(x)
        return x.flatten(start_dim=1)


def train_model(
    dataloader_train,
    dataloader_test,
    model_id,
    model_cfg,
    train_config,
    inference_config,
    device,
    forecast_horizon=96,
):
    backbone = ChronosModel(device, model_id, model_cfg)
    P = Pipeline(backbone)
    embed_dim = get_chronos_embed_dim(model_id)
    decoder = P.add_decoder(
        ChronosETTh1LinearDecoder(
            device,
            cfg={
                "input_dim": embed_dim,
                "output_dim": forecast_horizon,
                "head_dropout": 0.0,
            },
        ),
        load=True,
    )
    end_time = timeit.default_timer()
    print(f"Time taken to load model: {end_time - start_time} seconds")

    path = f"etth1_fore_chronos{model_id}_linear"
    print("Training...")
    P.train_eval(
        dataloader_train,
        dataloader_test,
        parts_to_train=["decoder"],
        train_cfg=train_config,
        inference_cfg=inference_config,
        path=path,
        metric_fn=get_mse,
        mlflow_cfg={
            "experiment_name": path,
            "run_name": path,
            "extra_params": {
                "model_id": model_id,
                "model_cfg": model_cfg,
                "train_config": train_config,
            },
        },
    )

    y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
    mse = get_mse(y_test, y_pred)
    print("MSE:", mse)
    gc.collect()
    del P, decoder, backbone
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return mse


if __name__ == "__main__":
    task_cfg = {"task_type": "forecasting"}
    model_id = "small"
    forecast_horizon = FORECAST_HORIZON
    train_config = {
        "batch_size": 32,
        "shuffle": True,
        "epochs": 50,
        "lr": 1e-3,
        "scheduler": {"type": "cosine", "T_max": 10, "eta_min": 0},
        "use_cache": True,
    }
    inference_config = {"batch_size": 32, "shuffle": False}
    dataset_cfg = {
        "dataset_path": "../datasets/ETTh1",
        "model_id": "amazon/chronos-t5-small",
    }
    model_cfg = {}

    train_data = ETTh1Dataset(
        dataset_cfg, task_cfg, split="train", forecast_horizon=forecast_horizon
    )
    test_data = ETTh1Dataset(
        dataset_cfg, task_cfg, split="test", forecast_horizon=forecast_horizon
    )

    dataloader_test = DataLoader(
        test_data,
        batch_size=inference_config["batch_size"],
        shuffle=inference_config["shuffle"],
        generator=generator,
    )
    dataloader_train = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=train_config["shuffle"],
        generator=generator,
    )

    mse = train_model(
        dataloader_train,
        dataloader_test,
        model_id,
        model_cfg,
        train_config,
        inference_config,
        device,
        forecast_horizon=forecast_horizon,
    )
    print("Final MSE:", mse)
