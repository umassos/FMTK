import timeit
import torch
import gc
from torch.utils.data import ConcatDataset

start_time = timeit.default_timer()
from fmtk.pipeline import Pipeline

end_time = timeit.default_timer()
print(f"Time taken to import fmtk pipeline: {end_time - start_time} seconds")

from fmtk.components.decoders.classification.linear import LinearDecoder
from fmtk.metrics import get_accuracy
from torch.utils.data import DataLoader, Subset
from fmtk.datasets.uwavegesture import UWaveGestureLibraryALLDataset
import traceback
from fmtk.components.backbones.moment import MomentModel
from fmtk.components.backbones.mantis import MantisModel
from fmtk.components.backbones.chronos import ChronosModel
from fmtk.components.decoders.classification.mlp import MLPDecoder

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

    backbone = MantisModel(device, model_id, model_cfg)
    P = Pipeline(backbone)
    # MLPDecoder(device,cfg={'input_dim':512,'output_dim':5,'hidden_dim':128})
    linear_decoder = P.add_decoder(
        LinearDecoder(device,cfg={'input_dim':256,'output_dim':8}),
        load=True,
        path='gestureclass_mantis8M_linear_v2',
        train=False,
    )
    end_time = timeit.default_timer()
    print(f"Time taken to load model: {end_time - start_time} seconds")

    print("Training...")
    # P.train_eval(
    #     dataloader_train,
    #     parts_to_train=["decoder"],
    #     train_cfg=train_config,
    #     path="gestureclass_mantis8M_linear_v2",
    #     test_loader=dataloader_test,
    #     metric_fn=get_accuracy,
    #     mlflow_cfg={
    #         "experiment_name": "gestureclass_mantis8M_linear_v2",
    #         "run_name": "gestureclass_mantis8M_linear_v2",
    #         "extra_params": {
    #             "model_id": model_id,
    #             "model_cfg": model_cfg,
    #             "train_config": train_config,
    #         },
    #     },
    # )

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

    model_id = "8M"
    samples_per_class = [1000]
    train_data = UWaveGestureLibraryALLDataset(dataset_cfg, task_cfg, split="train")
    test_data = UWaveGestureLibraryALLDataset(dataset_cfg, task_cfg, split="test")

    print("Loading test dataloader...")
    dataloader_test = DataLoader(
        test_data,
        batch_size=inference_config["batch_size"],
        shuffle=inference_config["shuffle"],
        generator=generator,
    )
    print("Loading train dataloader...")
    subsets = []
    # for label in range(train_data.num_classes):
    #     subsets.append(
    #         Subset(
    #             train_data,
    #             indices=train_data.indices[train_data.labels == label].tolist(),
    #         )
    #     )
    dataloader_train = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=train_config["shuffle"],
        generator=generator,
    )

    accuracy = train_model(
        dataloader_train,
        dataloader_test,
        model_id,
        model_cfg,
        train_config,
        inference_config,
        device,
    )
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
