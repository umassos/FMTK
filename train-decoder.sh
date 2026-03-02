#!/bin/bash

#SBATCH --mem=128G  # Requested Memory
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 12:00:00  # Job time limit
#SBATCH -o train-dino-coco-decoder-%j.out  # %j = job ID
#SBATCH --constraint=vram32|vram40|vram48

eval "$(conda shell.bash hook)"
conda activate fmtk

cd /home/kgudipaty_umass_edu/FMTK/examples
python3 dino_coco.py