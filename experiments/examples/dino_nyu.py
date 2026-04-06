import timeit
import torch
import gc
from torch.utils.data import DataLoader

start_time = timeit.default_timer()
from fmtk.pipeline import Pipeline

end_time = timeit.default_timer()
print(f"Time taken to import fmtk pipeline: {end_time - start_time} seconds")

from fmtk.components.backbones.dinov2 import DinoV2Model, get_dinov2_embed_dim
from fmtk.components.decoders.regression.monocular_depth import MonocularDepthDecoder
from fmtk.metrics import get_mae
from fmtk.datasetloaders.nyudepthv2 import NYUDepthV2Dataset
import traceback
from fmtk.logger import Logger

device = "cuda:0"
seed = 42
generator = torch.Generator()
generator.manual_seed(seed)


def train_model(
    dataloader_train,
    dataloader_test,
    model_id,
    model_cfg,
    decoder_cfg,
    train_config,
    inference_config,
    device,
):
    backbone = DinoV2Model(device, model_id, model_cfg)
    nyu_logger=Logger(device,'nyu_depth_v2')
    P = Pipeline(backbone,nyu_logger)

    depth_decoder = MonocularDepthDecoder(device, cfg=decoder_cfg)
    P.add_decoder(depth_decoder, load=True, train=False)

    end_time = timeit.default_timer()
    print(f"Time taken to load model: {end_time - start_time} seconds")

    # print("Training...")
    # P.train(
    #     dataloader_train,
    #     parts_to_train=["decoder"],
    #     cfg=train_config,
    #     path="nyudepth_dinolarge_monocular",
    # )

    y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
    result = get_mae(y_test, y_pred)
    print("MAE: ", result)

    gc.collect()
    del P, depth_decoder, backbone
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


if __name__ == "__main__":
    task_cfg = {"task_type": "regression"}
    train_config = {
        "batch_size": 16,
        "shuffle": True,
        "epochs": 10,
        "lr": 1e-3,
        "scheduler": {"type": "cosine", "T_max": 10, "eta_min": 0},
        "use_cache": True,
    }
    inference_config = {"batch_size": 16, "shuffle": False}

    target_size = 224
    dataset_cfg = {
        "dataset_path": "/work/pi_shenoy_umass_edu/kgudipaty/datasets/nyu-depth-v2",
        "target_size": target_size,
        "max_depth": 10.0,
        "normalize_depth": True,   # depth in [0, 1]; MAE will be in normalised units
    }

    model_id = "large"
    model_cfg = {"return_all_tokens": True}  # patch tokens required for spatial decoder

    embed_dim = get_dinov2_embed_dim(model_id)
    patch_size = 14                           # DINOv2 uses 14x14 patches
    grid_size = target_size // patch_size     # 224 / 14 = 16

    decoder_cfg = {
        "input_dim": embed_dim,
        "height": grid_size,
        "width": grid_size,
        "pixel_height": target_size,
        "pixel_width": target_size,
        "mode": "PATCH",
    }

    print("Loading datasets...")
    train_data = NYUDepthV2Dataset(dataset_cfg, task_cfg, split="train")
    test_data = NYUDepthV2Dataset(dataset_cfg, task_cfg, split="test")

    print(f"Train samples: {len(train_data)}  |  Test samples: {len(test_data)}")

    dataloader_train = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=train_config["shuffle"],
        generator=generator,
    )
    dataloader_test = DataLoader(
        test_data,
        batch_size=inference_config["batch_size"],
        shuffle=inference_config["shuffle"],
        generator=generator,
    )

    try:
        mae = train_model(
            dataloader_train,
            dataloader_test,
            model_id,
            model_cfg,
            decoder_cfg,
            train_config,
            inference_config,
            device,
        )
        print("Final MAE:", mae)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
