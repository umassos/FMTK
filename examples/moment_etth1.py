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
from fmtk.components.backbones.moment import MomentModel
from fmtk.components.decoders.forecasting.mlp import MLPDecoder
from fmtk.components.decoders.forecasting.linear import LinearDecoder

from fmtk.metrics import get_mae

device = "cuda:0"
seed = 42
generator = torch.Generator()
generator.manual_seed(seed)


def get_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def get_moment_embed_dim(model_id):
    if model_id == "small":
        return 512
    if model_id == "base":
        return 768
    elif model_id == "large":
        return 1024
    else:
        raise ValueError(f"Invalid model_id: {model_id}")


def train_model(
    dataloader_train,
    dataloader_test,
    model_id,
    model_cfg,
    train_config,
    inference_config,
    device,
    n_channels=7,
    forecast_horizon=96,
):
    backbone = MomentModel(device, model_id, model_cfg)
    P = Pipeline(backbone)
    num_patches = 512 // 8
    input_dim = get_moment_embed_dim(model_id) * num_patches  # per-channel
    decoder = P.add_decoder(
        LinearDecoder(
            device,
            cfg={
                "input_dim": input_dim,
                "output_dim": forecast_horizon,
                "head_dropout": 0.1,
            },
        ),
        load=True,
    )
    end_time = timeit.default_timer()
    print(f"Time taken to load model: {end_time - start_time} seconds")

    print("Training...")
    P.train_eval(
        dataloader_train,
        dataloader_test,
        parts_to_train=["decoder"],
        train_cfg=train_config,
        inference_cfg=inference_config,
        path="etth1single_fore_momentbase_linear",
        metric_fn=get_mae,
        mlflow_cfg={
            "base_path": "/home/kgudipaty_umass_edu/FMTK/mlflow",
            "experiment_name": "etth1single_fore_momentbase_linear",
            "run_name": "etth1single_fore_momentbase_linear",
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
    model_id = "base"
    forecast_horizon = 96
    train_config = {
        "batch_size": 32,
        "shuffle": True,
        "epochs": 20,
        "lr": 1e-4,
        "scheduler": {"type": "cosine", "T_max": 10, "eta_min": 0},
        "use_cache": False,
    }
    inference_config = {"batch_size": 32, "shuffle": False}
    dataset_cfg = {
        "dataset_path": "../datasets/ETTh1",
        "model_id": "AutonLab/MOMENT-1-base",
    }
    model_cfg = {"return_all_tokens": False}

    train_data = ETTh1Dataset(
        dataset_cfg, task_cfg, split="train", forecast_horizon=forecast_horizon
    )
    test_data = ETTh1Dataset(
        dataset_cfg, task_cfg, split="test", forecast_horizon=forecast_horizon
    )

    print("Loading test dataloader...")
    dataloader_test = DataLoader(
        test_data,
        batch_size=inference_config["batch_size"],
        shuffle=inference_config["shuffle"],
        generator=generator,
    )
    print("Loading train dataloader...")
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
        n_channels=7,
        forecast_horizon=forecast_horizon,
    )
    print("MSE:", mse)
