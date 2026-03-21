#!/bin/bash

#SBATCH --mem=32G  # Requested Memory
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 5:00:00  # Job time limit
#SBATCH -o train-etth1-repa-%j.out  # %j = job ID
#SBATCH --constraint=vram32|vram40|vram48

eval "$(conda shell.bash hook)"
conda activate fmtk

cd /home/kgudipaty_umass_edu/FMTK/examples
python3 train_etth1_repa.py -c config.yaml \
--num-samples-list 1,5,10,50,100,500,1000,2000,3000 \
--model-from-name moment \
--model-from-id  large \
--model-to-name moment \
--model-to-id base \
--dataset-name etth1 \
--epochs 35 \
--num-experiments 5 \
--name1 moment-large \
--name2 moment-base