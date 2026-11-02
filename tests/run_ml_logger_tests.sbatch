#!/bin/bash
#SBATCH --job-name=fmtk_ml_logger_tests
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# Run the MLLogger and training tests. From the repo root: mkdir -p logs && sbatch tests/run_ml_logger_tests.sbatch
set -euo pipefail

hostname
date
module list 2>&1 || true

source /home/kgudipaty_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate fmtk
which python && python --version

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
python -c "import fmtk; print(fmtk.__file__)"
python -m unittest discover -s tests -p 'test_ml_logger.py' -v
