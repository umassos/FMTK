#!/bin/bash

#SBATCH --mem=32G  # Requested Memory
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 5:00:00  # Job time limit
#SBATCH -o train-eurosat-repa-dinov2-small-to-dinov2-base-%j.out  # %j = job ID
#SBATCH --constraint=vram32|vram40|vram48

eval "$(conda shell.bash hook)"
conda activate fmtk

cd /home/kgudipaty_umass_edu/FMTK/examples
python3 train_eurosat_repa.py -c config.yaml