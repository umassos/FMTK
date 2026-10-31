"""Tests for model locations: the finetuned folder, decoder downloads and the backbone cache."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from fmtk.model_store import DECODER_REPO, decoder_file
from fmtk.pipeline import Pipeline


class Backbone(nn.Module):
    device = "cpu"
    model_category = "vision"


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)


class TestDecoderFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_disk_is_checked_before_download(self):
        (self.root / "head").mkdir()
        (self.root / "head" / "decoder.pth").touch()
        with mock.patch("fmtk.model_store.hf_hub_download") as download:
            self.assertEqual(decoder_file("head", self.root), self.root / "head" / "decoder.pth")
        download.assert_not_called()

    def test_missing_file_downloads_into_the_given_folder(self):
        with mock.patch("fmtk.model_store.hf_hub_download", return_value="/x/decoder.pth") as download, \
                mock.patch("fmtk.model_store._token", return_value=None):
            self.assertEqual(decoder_file("head", self.root), Path("/x/decoder.pth"))
        download.assert_called_once_with(repo_id=DECODER_REPO, filename="decoder.pth", subfolder="head",
                                         local_dir=self.root, token=None)

    def test_failed_download_raises_value_error(self):
        with mock.patch("fmtk.model_store.hf_hub_download", side_effect=OSError("offline")), \
                self.assertRaisesRegex(ValueError, "Decoder file not found"):
            decoder_file("head", self.root)


class TestPipelineLocation(unittest.TestCase):
    def test_default_is_fmtk_models_folder(self):
        pipeline = Pipeline(Backbone())
        expected = f"{pipeline.base_dir}/../../models/vision/finetuned"
        self.assertEqual(pipeline._finetuned_dir(), expected)

    def test_add_decoder_loads_from_the_given_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = nn.Linear(2, 2)
            os.makedirs(f"{tmp}/head")
            torch.save(saved.state_dict(), f"{tmp}/head/decoder.pth")
            pipeline = Pipeline(Backbone(), finetuned_dir=tmp)
            decoder = Decoder()
            pipeline.add_decoder(decoder, load=False, train=False, path="head")
            torch.testing.assert_close(decoder.model.weight, saved.weight)


class TestBackboneCache(unittest.TestCase):
    def test_cache_dir_reaches_from_pretrained(self):
        from fmtk.components.backbones import dinov2
        with mock.patch.object(dinov2, "AutoModel") as model, mock.patch.object(dinov2, "AutoImageProcessor") as proc:
            dinov2.DinoV2Model("cpu", "small", {"cache_dir": "/m/backbones"})
        model.from_pretrained.assert_called_once_with("facebook/dinov2-small", cache_dir="/m/backbones")
        proc.from_pretrained.assert_called_once_with("facebook/dinov2-small", cache_dir="/m/backbones")

    def test_no_cache_dir_keeps_the_default(self):
        from fmtk.components.backbones import dinov2
        with mock.patch.object(dinov2, "AutoModel") as model, mock.patch.object(dinov2, "AutoImageProcessor"):
            dinov2.DinoV2Model("cpu", "small")
        model.from_pretrained.assert_called_once_with("facebook/dinov2-small", cache_dir=None)


if __name__ == "__main__":
    unittest.main()
