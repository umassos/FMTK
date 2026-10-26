# Installation

FMTK is installed from source into a conda environment. The install script targets **Linux x86_64 with CUDA 12**
and **Python 3.10** (the bundled `flash-attn` wheel is built for that combination).

## Recommended: install script

```bash
git clone https://github.com/umassos/FMTK.git
cd FMTK
bash install.sh           # creates an env named 'fmtk'
conda activate fmtk
```

To use a different environment name:

```bash
bash install.sh my_env_name
conda activate my_env_name
```

The script:

1. creates the environment from `fmtk_environment.yml` (PyTorch with CUDA, transformers, vLLM, …),
2. installs packages whose version pins conflict, using `--no-deps`: `flash-attn`, `momentfm`, `pyPPG`,
3. installs Hugging Face `datasets`,
4. installs FMTK itself in editable mode (`pip install -e .`),
5. checks the install by importing torch, transformers, momentfm, chronos and vllm.

!!! tip "Running on a shared cluster"
    Creating the environment downloads several GB. On an HPC cluster, run `install.sh` inside an interactive or
    batch allocation rather than on a login node.

## Manual install (deprecated)

!!! warning
    This path does not install every dependency (for example `flash-attn`, or the vLLM backbones' dependencies).

```bash
cd FMTK
conda create -n fmtk python=3.10
conda activate fmtk
pip install -e .
pip install pyPPG==1.0.41  # may conflict with other packages
```

## Project layout after install

FMTK resolves several paths relative to the repository root, so keep these folders in place:

| Path | Purpose |
|---|---|
| `dataset/` | Raw datasets. Several time-series loaders look here directly (e.g. `dataset/ECG5000/`). |
| `models/<category>/pretrained/` | Download cache for backbone weights (`tsfm`, `vision`, `vlm`, `llm`). |
| `models/<category>/finetuned/<run>/` | Saved decoders, encoders and adapters (see [Training and inference](guides/training.md#saving-and-loading)). |
| `hf-token.txt` | Optional Hugging Face token, used to fetch pretrained decoders from the `umass-lass/fmtk-decoder-zoo` repo and gated models. Never commit it. |

## Building these docs locally

```bash
python -m venv .venv-docs && source .venv-docs/bin/activate
pip install -r docs/requirements.txt
mkdocs serve                  # http://127.0.0.1:8000, reloads on edits
mkdocs build --strict         # the same check CI runs
```

The docs tooling reads docstrings without importing FMTK, so you don't need PyTorch or a GPU to build the docs.
