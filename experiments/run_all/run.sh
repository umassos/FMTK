#!/bin/bash
#SBATCH --mem=48G  # Requested Memory
#SBATCH -p gpu,gpu-preempt # Partition
#SBATCH --gres=gpu:a16:1 # Number and type of GPUs
#SBATCH -t 08:00:00  # Job time limit
#SBATCH -o ./logs/slurm-a16.out  # %j = job ID

module load conda/latest
conda activate fmtk
python main.py combined_metrics_a16.csv