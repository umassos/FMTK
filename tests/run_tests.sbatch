#!/bin/bash
#SBATCH --job-name=fmtk_task_graph_tests
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# Run FMTK unit tests and the TaskGraph overhead benchmark. From the repo root: mkdir -p logs && sbatch tests/run_tests.sbatch
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
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/bench_task_graph.py
