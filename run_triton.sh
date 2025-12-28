#!/bin/bash
#SBATCH --job-name=triton_server
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a16:1
#SBATCH --mem=200GB
#SBATCH --time=02:00:00
#SBATCH --output=triton_log_%j.out

# --- PATH DEFINITIONS ---
SIF_IMAGE="./tritonserver.sif"
MODEL_REPO="$(pwd)/model_repository"
# FMTK_CODE="$(pwd)/fmtk"     # Your live source code
MY_LIBS="$(pwd)/my_libs"    # Your installed dependencies

# --- RUN APPTAINER ---
# apptainer run --nv \
#  --bind "${MODEL_REPO}:/models" \
# #   --bind "${FMTK_CODE}:/packages/fmtk_code" \
#   --bind "${MY_LIBS}:/packages/libs" \
#   "${SIF_IMAGE}" \
#   tritonserver --model-repository=/models --http-port=8000 --grpc-port=8001 --metrics-port=8002

apptainer run --nv --env PYTHONPATH="/packages/libs" --bind "${MODEL_REPO}:/models" --bind "${MY_LIBS}:/packages/libs" \
  "${SIF_IMAGE}" \
  tritonserver --model-repository=/models --http-port=8000 --grpc-port=8001 --metrics-port=8002