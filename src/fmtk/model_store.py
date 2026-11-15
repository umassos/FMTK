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


STITCH_REPO = "umass-lass/stitches"


def stitch_file(name: str, stitches_dir: str | os.PathLike) -> Path:
    """Path of <stitches_dir>/<name>/stitch.pt, downloading it if it isn't on disk."""
    local = Path(stitches_dir) / name / "stitch.pt"
    if local.is_file():
        return local
    try:
        return Path(hf_hub_download(repo_id=STITCH_REPO, filename=f"{name}/stitch.pt", local_dir=stitches_dir,
                                    token=_token()))
    except Exception as err:
        raise ValueError(f"Stitch file not found at {local}") from err


def upload_files(repo: str, files: list[tuple[str | os.PathLike, str]], message: str) -> None:
    """Upload (local path, path in repo) pairs to one of FMTK's model repos in a single commit (needs write access)."""
    from huggingface_hub import CommitOperationAdd, HfApi
    operations = [CommitOperationAdd(path_in_repo=target, path_or_fileobj=str(local)) for local, target in files]
    HfApi(token=_token()).create_commit(repo_id=repo, operations=operations, commit_message=message)
