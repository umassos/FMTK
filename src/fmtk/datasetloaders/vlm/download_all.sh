#!/bin/bash
# Run all VLM dataset download scripts sequentially.
# Usage: bash src/fmtk/datasetloaders/vlm/download_all.sh
#   from the FMTK project root.
#
# NOTE: Datasets are stored in FMTK/dataset/vlm/ (see experiments/run_all/config.py
#   and src/fmtk/tasks/vlm_utils.py). Run these download scripts only if the
#   data directory is missing or incomplete.
#
# Excluded (require special handling):
#   download_crowd.py              - needs ~1GB Google Drive download (gdown)
#   download_image_classification.py - needs imagenet-1k HF terms accepted

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use fmtk conda env Python where datasets/huggingface_hub are installed
PYTHON=/home/nesl/miniconda3/envs/fmtk/bin/python3

run() {
    echo ""
    echo "========== $1 =========="
    "$PYTHON" "$SCRIPT_DIR/$1"
}

run download_activity.py
run download_traffic.py
run download_ocr.py
run download_gesture.py
run download_scene.py
run download_object_detection.py
run download_vqa.py

echo ""
echo "All downloads complete."
