#!/bin/bash
# FMTK Environment Setup
# Creates conda env and installs all dependencies including
# packages with conflicting version pins.
#
# Usage:
#   bash install.sh              # creates env named 'fmtk'
#   bash install.sh fmtk_vlm    # creates env with custom name

set -e

ENV_NAME="${1:-fmtk}"

echo "=== Creating conda environment: $ENV_NAME ==="
conda env create -f fmtk_environment.yml -n "$ENV_NAME"

echo ""
echo "=== Installing momentfm (--no-deps to bypass strict huggingface-hub pin) ==="
conda run -n "$ENV_NAME" pip install momentfm==0.1.4 --no-deps

echo ""
echo "=== Installing pyPPG (--no-deps to bypass strict scipy==1.9.1 pin) ==="
conda run -n "$ENV_NAME" pip install pyPPG==1.0.73 --no-deps

echo ""
echo "=== Installing fmtk in editable mode ==="
conda run -n "$ENV_NAME" pip install -e . --no-deps

echo ""
echo "=== Verifying installation ==="
conda run -n "$ENV_NAME" python -c "
import torch, transformers, momentfm, chronos
print(f'torch:        {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'CUDA:         {torch.cuda.is_available()}')
print('All imports OK')
"

echo ""
echo "=== Done! Activate with: conda activate $ENV_NAME ==="
