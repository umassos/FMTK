#!/bin/bash

#SBATCH --mem=32G  # Requested Memory
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 2:00:00  # Job time limit
#SBATCH -o train-eurosat-repa-%j.out  # %j = job ID
#SBATCH --constraint=vram32|vram40|vram48

eval "$(conda shell.bash hook)"
conda activate fmtk

cd /home/kgudipaty_umass_edu/FMTK/examples
python3 train_uwave_repa.py -c config.yaml \
--num-samples-list 1,5,10,50,100,500,1000,3000 \
--model-from-name mantis \
--model-from-id  8M \
--model-to-name moment \
--model-to-id small \
--dataset-name uwave \
--epochs 50 \
--num-experiments 5 \
--name1 mantis-8M \
--name2 moment-small