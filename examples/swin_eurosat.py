import timeit
import pandas as pd
import torch
import gc
from torch.utils.data import ConcatDataset

start_time = timeit.default_timer()
from fmtk.pipeline import Pipeline

end_time = timeit.default_timer()
print(f"Time taken to import fmtk pipeline: {end_time - start_time} seconds")

from fmtk.components.backbones.swin import SwinModel, EMBED_DIMS as SWIN_EMBED_DIMS
from fmtk.components.decoders.classification.linear import LinearDecoder
from fmtk.components.encoders.diff import LinearChannelCombiner
from fmtk.metrics import get_accuracy
from torch.utils.data import DataLoader, Subset
from peft import LoraConfig
from fmtk.datasets.EuroSAT import EuroSATDataset
import traceback

device = "cuda:0"
seed = 42
generator = torch.Generator()
generator.manual_seed(seed)


def train_model(
    dataloader_train,
    dataloader_test,
    model_id,
    model_cfg,
    train_config,
    inference_config,
    device,
):

    backbone = SwinModel(device, model_id, model_cfg)
    P = Pipeline(backbone)
    linear_decoder = P.add_decoder(
        LinearDecoder(
            device, cfg={"input_dim": SWIN_EMBED_DIMS[model_id], "output_dim": 10}
        ),
        load=True,
    )
    end_time = timeit.default_timer()
    print(f"Time taken to load model: {end_time - start_time} seconds")

    print("Training...")
    P.train(
        dataloader_train,
        parts_to_train=["decoder"],
        cfg=train_config,
        path="imgclass_swinsmall_eurosat",
    )

    y_test, y_pred = P.predict(dataloader_test, cfg=inference_config)
    result = get_accuracy(y_test, y_pred)
    print("Accuracy: ", result)
    gc.collect()
    del P, linear_decoder, backbone
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def run_multiple(
    epochs,
    samples_per_class,
    model_id,
    subsets,
    dataloader_test,
    model_cfg,
    train_config,
    inference_config,
    device,
):
    results = {"samples_per_class": [], "accuracy": []}
    for i in epochs:
        train_config["epochs"] = i
        for n_samples in samples_per_class:
            try:
                combined_data = [
                    Subset(subset, indices=list(range(n_samples))) for subset in subsets
                ]
                dataloader_train = DataLoader(
                    ConcatDataset(combined_data),
                    batch_size=train_config["batch_size"],
                    shuffle=train_config["shuffle"],
                    generator=generator,
                )
                print("Dataloader train length: ", len(dataloader_train))

                result = train_model(
                    dataloader_train,
                    dataloader_test,
                    model_id,
                    model_cfg,
                    train_config,
                    inference_config,
                    device,
                )
                results["samples_per_class"].append(n_samples)
                results["accuracy"].append(result)
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                results["samples_per_class"].append(n_samples)
                results["accuracy"].append(None)

        # df = pd.DataFrame(results)
        # df.to_csv(f"results/dino_small_eurosat_accuracy_epochs_{i}.csv", index=False)


if __name__ == "__main__":
    task_cfg = {"task_type": "classification"}
    train_config = {
        "batch_size": 32,
        "shuffle": False,
        "epochs": 20,
        "lr": 1e-3,
        "scheduler": {"type": "cosine", "T_max": 10, "eta_min": 0},
    }
    inference_config = {"batch_size": 32, "shuffle": False}
    dataset_cfg = {
        "dataset_path": "/work/pi_shenoy_umass_edu/kgudipaty/datasets/EuroSAT",
        "model_id": "microsoft/swin-small-patch4-window7-224",
    }
    model_cfg = {"return_all_tokens": False}

    model_id = "small"
    samples_per_class = [1000]
    train_data = EuroSATDataset(dataset_cfg, task_cfg, split="train")
    test_data = EuroSATDataset(dataset_cfg, task_cfg, split="test")

    print("Loading test dataloader...")
    dataloader_test = DataLoader(
        test_data,
        batch_size=inference_config["batch_size"],
        shuffle=inference_config["shuffle"],
        generator=generator,
    )
    print("Loading train dataloader...")
    subsets = []
    for label in range(train_data.num_classes):
        subsets.append(
            Subset(
                train_data,
                indices=train_data.indices[train_data.labels == label].tolist(),
            )
        )
    dataloader_train = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=train_config["shuffle"],
        generator=generator,
    )

    accuracy = train_model(dataloader_train, dataloader_test, model_id, model_cfg, train_config, inference_config, device)
    print("Accuracy: ", accuracy)
    # run_multiple(
    #     [10],
    #     samples_per_class,
    #     model_id,
    #     subsets,
    #     dataloader_test,
    #     model_cfg,
    #     train_config,
    #     inference_config,
    #     device,
    # )
