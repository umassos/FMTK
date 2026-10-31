"""Finds model weights on disk and downloads the missing ones from FMTK's model repos."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

DECODER_REPO = "umass-lass/fmtk-decoder-zoo"
_TOKEN_FILE = Path(__file__).resolve().parents[2] / "hf-token.txt"


def _token() -> str | None:
    return _TOKEN_FILE.read_text().strip() if _TOKEN_FILE.exists() else None


def decoder_file(name: str, finetuned_dir: str | os.PathLike) -> Path:
    """Path of <finetuned_dir>/<name>/decoder.pth, downloading it if it isn't on disk."""
    local = Path(finetuned_dir) / name / "decoder.pth"
    if local.is_file():
        return local
    try:
        return Path(hf_hub_download(repo_id=DECODER_REPO, filename="decoder.pth", subfolder=name,
                                    local_dir=finetuned_dir, token=_token()))
    except Exception as err:
        raise ValueError(f"Decoder file not found at {local}") from err
