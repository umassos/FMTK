"""Tests for MLLogger, MLFLOWLogger and the decoder training features of Pipeline.train."""
import glob
import os
import subprocess
import sys
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fmtk.components.decoders.classification.linear import LinearDecoder
from fmtk.logger import MLFLOWLogger, MLLogger
from fmtk.pipeline import Pipeline


class ToyDataset(Dataset):
    def __init__(self, n=16, dim=4):
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, dim, generator=generator)
        self.y = (self.x.sum(dim=1) > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


class StubBackbone:
    device = "cpu"
    model_category = "test"

    def __init__(self):
        self.model = nn.Identity()

    def forward(self, x, mask=None):
        return x


class RecordingLogger(MLLogger):
    def __init__(self, run_id=None):
        super().__init__(run_id=run_id)
        self.params = {}
        self.metrics = []

    def start_run(self):
        self.run_id = self.run_id or "recording-run"

    def end_run(self, failed=False):
        pass

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.append((step, dict(metrics)))


def make_pipeline(root, checkpoint_root=None):
    base_dir = os.path.join(root, "a", "b")
    os.makedirs(base_dir, exist_ok=True)
    P = Pipeline(StubBackbone(), checkpoint_root=checkpoint_root)
    P.base_dir = base_dir
    P.active_decoder = LinearDecoder("cpu", {"input_dim": 4, "output_dim": 2})
    return P


def make_loader():
    return DataLoader(ToyDataset(), batch_size=4)


def make_cfg(**overrides):
    return {"epochs": 3, "lr": 1e-2, "batch_size": 4, "shuffle": False, **overrides}


def finetuned_dir(root, path):
    return os.path.join(root, "a", "b", "..", "..", "models", "test", "finetuned", path)


class MLLoggerTrainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def test_recording_logger_gets_params_and_metrics(self):
        P = make_pipeline(self.root)
        with RecordingLogger() as ml:
            P.train(make_loader(), make_loader(), cfg=make_cfg(eval_metric="acc"), path="x", ml_logger=ml)
        self.assertEqual(ml.params["epochs"], 3)
        self.assertEqual(ml.params["num_train_samples"], 16)
        self.assertEqual(ml.params["parts_to_train"], "decoder")
        self.assertEqual([step for step, _ in ml.metrics], [0, 1, 2])
        for _, metrics in ml.metrics:
            self.assertEqual(
                set(metrics),
                {"train_loss", "lr", "epoch_time_sec", "update_time_sec_avg", "data_time_sec_avg", "acc"},
            )

    def test_log_interval_skips_epochs_but_keeps_the_last(self):
        P = make_pipeline(self.root)
        with RecordingLogger() as ml:
            P.train(make_loader(), cfg=make_cfg(epochs=5, log_interval=2), path="x", ml_logger=ml)
        self.assertEqual([step for step, _ in ml.metrics], [1, 3, 4])

    def test_best_checkpoint_restores_best_epoch(self):
        P = make_pipeline(self.root)
        scores = iter([0.1, 0.9, 0.5])
        P.train(make_loader(), make_loader(), cfg=make_cfg(), path="x", metric_fn=lambda y, p: next(scores))

        checkpoint_dir = finetuned_dir(self.root, "x")
        self.assertTrue(os.path.exists(f"{checkpoint_dir}/best_checkpoint.pt"))
        self.assertTrue(os.path.exists(f"{checkpoint_dir}/last_checkpoint.pt"))

        last_weight = P.active_decoder.model.weight.detach().clone()
        best = P.load_best_checkpoint("x")
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(best["best_metric"], 0.9)
        self.assertFalse(torch.equal(P.active_decoder.model.weight, last_weight))
        self.assertTrue(torch.equal(P.active_decoder.model.weight, best["decoder_state"]["weight"]))

    def test_metric_mode_min_picks_lowest(self):
        P = make_pipeline(self.root)
        scores = iter([0.5, 0.1, 0.9])
        P.train(make_loader(), make_loader(), cfg=make_cfg(metric_mode="min"), path="x", metric_fn=lambda y, p: next(scores))
        self.assertEqual(P.load_best_checkpoint("x")["epoch"], 1)

    def test_cosine_lr_continues_after_resume(self):
        cfg = make_cfg(epochs=4, scheduler={"type": "cosine", "T_max": 4})

        straight = RecordingLogger()
        make_pipeline(os.path.join(self.root, "straight")).train(make_loader(), cfg=cfg, path="x", ml_logger=straight)

        P = make_pipeline(os.path.join(self.root, "resumed"))
        P.train(make_loader(), cfg={**cfg, "epochs": 2}, path="x")
        resumed = RecordingLogger()
        P.train(make_loader(), cfg=cfg, path="x", ml_logger=resumed, resume_from="last")

        self.assertEqual([step for step, _ in resumed.metrics], [2, 3])
        self.assertEqual(resumed.metrics[0][1]["lr"], straight.metrics[2][1]["lr"])
        self.assertEqual(resumed.metrics[1][1]["lr"], straight.metrics[3][1]["lr"])

    def test_resume_with_adapter_is_rejected(self):
        P = make_pipeline(self.root)
        with self.assertRaises(ValueError):
            P.train(make_loader(), parts_to_train=["decoder", "adapter"], cfg=make_cfg(), path="x", resume_from="last")

    def test_adapter_checkpoint_holds_only_trainable_params(self):
        P = make_pipeline(self.root)
        P.model_instance.model = nn.Linear(2, 2)
        P.model_instance.model.weight.requires_grad = False
        optimizer = torch.optim.Adam([P.model_instance.model.bias])
        P._save_checkpoint("x", "best", 0, optimizer, None, 0.0, None, ["adapter"])

        saved = P._load_checkpoint("x", "best")["adapter_state"]
        self.assertEqual(set(saved), {"bias"})

        expected = saved["bias"].clone()
        frozen = P.model_instance.model.weight.detach().clone()
        with torch.no_grad():
            P.model_instance.model.bias.zero_()
        P.load_best_checkpoint("x", trains_decoder=False, trains_adapter=True)
        self.assertTrue(torch.equal(P.model_instance.model.bias, expected))
        self.assertTrue(torch.equal(P.model_instance.model.weight, frozen))

    def test_checkpoint_root_redirects_all_outputs(self):
        ckpt_root = os.path.join(self.root, "scratch")
        P = make_pipeline(self.root, checkpoint_root=ckpt_root)
        P.train(make_loader(), cfg=make_cfg(epochs=1), path="x")
        self.assertTrue(os.path.exists(f"{ckpt_root}/test/x/decoder.pth"))
        self.assertTrue(os.path.exists(f"{ckpt_root}/test/x/last_checkpoint.pt"))
        self.assertFalse(os.path.exists(finetuned_dir(self.root, "x")))

    def test_no_path_writes_no_checkpoints(self):
        P = make_pipeline(self.root)
        P.train(make_loader(), cfg=make_cfg(epochs=1))
        self.assertEqual(glob.glob(f"{finetuned_dir(self.root, 'None')}/*_checkpoint.pt"), [])

    def test_example_style_call_does_not_import_mlflow(self):
        script = (
            "import os, sys\n"
            f"sys.path.insert(0, {os.path.dirname(__file__)!r})\n"
            "from test_ml_logger import make_pipeline, make_loader, make_cfg, finetuned_dir\n"
            f"root = {self.root!r}\n"
            "P = make_pipeline(root)\n"
            "P.train(make_loader(), parts_to_train=['decoder'], cfg=make_cfg(epochs=1), path='x')\n"
            "assert 'mlflow' not in sys.modules\n"
            "assert os.path.exists(finetuned_dir(root, 'x') + '/decoder.pth')\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class MLFLOWLoggerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.uri = os.path.join(tmp.name, "mlruns")

    def client(self):
        from mlflow.tracking import MlflowClient
        return MlflowClient(tracking_uri=self.uri)

    def make_logger(self, **kwargs):
        return MLFLOWLogger("test-experiment", tracking_uri=self.uri, **kwargs)

    def test_round_trip_ends_finished(self):
        P = make_pipeline(self.root)
        with self.make_logger(run_name="round-trip") as ml:
            ml.log_params({"model_id": "stub"})
            P.train(make_loader(), make_loader(), cfg=make_cfg(eval_metric="acc"), path="x", ml_logger=ml)
            ml.log_metrics({"test_acc": 0.5})

        client = self.client()
        run = client.get_run(ml.run_id)
        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(run.info.run_name, "round-trip")
        self.assertEqual(run.data.params["model_id"], "stub")
        self.assertEqual(run.data.params["epochs"], "3")
        self.assertEqual(run.data.metrics["test_acc"], 0.5)
        self.assertEqual([m.step for m in client.get_metric_history(ml.run_id, "train_loss")], [0, 1, 2])
        self.assertEqual(len(client.get_metric_history(ml.run_id, "acc")), 3)

    def test_exception_ends_failed(self):
        with self.assertRaises(RuntimeError):
            with self.make_logger() as ml:
                raise RuntimeError("boom")
        self.assertEqual(self.client().get_run(ml.run_id).info.status, "FAILED")

    def test_resume_continues_same_run_with_more_epochs(self):
        P = make_pipeline(self.root)
        with self.make_logger() as ml:
            P.train(make_loader(), cfg=make_cfg(epochs=1), path="x", ml_logger=ml)
        first_id = ml.run_id
        self.assertEqual(P.checkpoint_run_id("x"), first_id)

        with self.make_logger(run_id=P.checkpoint_run_id("x")) as resumed:
            P.train(make_loader(), cfg=make_cfg(epochs=3), path="x", ml_logger=resumed, resume_from="last")

        self.assertEqual(resumed.run_id, first_id)
        client = self.client()
        self.assertEqual([m.step for m in client.get_metric_history(first_id, "train_loss")], [0, 1, 2])
        run = client.get_run(first_id)
        self.assertEqual(run.data.params["epochs"], "1")
        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(P.load_best_checkpoint("x")["run_id"], first_id)


if __name__ == "__main__":
    unittest.main()
