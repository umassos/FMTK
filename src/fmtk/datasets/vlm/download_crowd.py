# Download DroneCrowd dataset for categorical crowd counting
# Source: VisDrone2020-CC via Google Drive
# Mirrors FMaaS-motivation/vqa/updated/dataset/crowd_counting/download-data.py
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from tqdm import tqdm

FMTK_ROOT = Path(__file__).resolve().parents[4]
ROOT = FMTK_ROOT / "data" / "vlm" / "crowd_counting"
N = 100

GDRIVE_FILE_ID = "1HY3V4QObrVjzXUxL_J86oxn2bi7FMUgd"
DATASET_ZIP = ROOT / "DroneCrowd.zip"

CROWD_CATEGORIES = [
    (0, 20, "very_sparse"),
    (21, 50, "sparse"),
    (51, 100, "moderate"),
    (101, 300, "dense"),
    (301, float('inf'), "very_dense"),
]


def install_gdown():
    try:
        import gdown
        return True
    except ImportError:
        print("Installing gdown package for Google Drive downloads...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
            print("gdown installed successfully")
            return True
        except Exception as e:
            print(f"Failed to install gdown: {e}")
            return False


def download_dataset():
    if DATASET_ZIP.exists():
        print(f"Dataset ZIP already downloaded: {DATASET_ZIP}")
        return True

    if not install_gdown():
        print("Could not install gdown. Please install manually: pip install gdown")
        return False

    try:
        import gdown
        print(f"Downloading DroneCrowd dataset from Google Drive (~1.03 GB)...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(DATASET_ZIP), quiet=False)
        print(f"Download complete: {DATASET_ZIP}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Manual download: https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
        return False


def extract_dataset():
    dataset_path = ROOT / "DroneCrowd"
    if dataset_path.exists():
        print(f"Dataset already extracted: {dataset_path}")
        return True

    if not DATASET_ZIP.exists():
        print(f"ZIP file not found: {DATASET_ZIP}")
        return False

    try:
        print(f"Extracting dataset...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(ROOT)
        print(f"Extraction complete: {dataset_path}")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def get_category(count):
    for min_count, max_count, category in CROWD_CATEGORIES:
        if min_count <= count <= max_count:
            return category
    return "very_dense"


def count_people_in_frame(annotation_file, frame_number):
    if not os.path.exists(annotation_file):
        return None
    try:
        count = 0
        with open(annotation_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    frame_num = int(parts[0])
                    if frame_num == frame_number:
                        count += 1
        return count
    except Exception as e:
        print(f"Error reading {annotation_file}: {e}")
        return None


def main():
    print("=" * 60)
    print("DroneCrowd Dataset Download and Processing Script")
    print("=" * 60)

    dataset_path = ROOT / "VisDrone2020-CC"

    if not dataset_path.exists():
        print("Dataset not found. Starting automatic download...")

        if not download_dataset():
            print("Automatic download failed.")
            print("Manual download instructions:")
            print("  1. Download from: https://drive.google.com/file/d/1HY3V4QObrVjzXUxL_J86oxn2bi7FMUgd/view")
            print(f"  2. Save as: {DATASET_ZIP}")
            print("  3. Run this script again")
            return

        if not extract_dataset():
            print("Extraction failed. Please extract manually.")
            return

    print(f"Found VisDrone2020-CC dataset at: {dataset_path}")

    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    trainlist_file = dataset_path / "trainlist.txt"
    if not trainlist_file.exists():
        print(f"Error: trainlist.txt not found")
        return

    with open(trainlist_file, 'r') as f:
        train_sequences = [line.strip() for line in f if line.strip()]

    print(f"Found {len(train_sequences)} training sequences")

    records = []
    category_counts = Counter()
    processed_count = 0

    frames_per_seq = max(1, N // len(train_sequences) + 1)
    print(f"Sampling ~{frames_per_seq} frame(s) from each sequence...")

    for seq_id in tqdm(train_sequences, desc="Processing sequences"):
        if processed_count >= N:
            break

        seq_dir = dataset_path / "sequences" / seq_id
        ann_file = dataset_path / "annotations" / f"{seq_id}.txt"

        if not seq_dir.exists() or not ann_file.exists():
            continue

        frame_files = sorted(list(seq_dir.glob("*.jpg")))
        if len(frame_files) == 0:
            continue

        total_frames = len(frame_files)
        sample_indices = []

        if frames_per_seq == 1:
            sample_indices = [total_frames // 2]
        elif frames_per_seq == 2:
            sample_indices = [total_frames // 3, 2 * total_frames // 3]
        else:
            step = total_frames // frames_per_seq
            sample_indices = [i * step for i in range(frames_per_seq) if i * step < total_frames]

        for frame_idx in sample_indices[:frames_per_seq]:
            if processed_count >= N:
                break

            frame_path = frame_files[frame_idx]
            frame_number = frame_idx + 1

            person_count = count_people_in_frame(ann_file, frame_number)

            if person_count is None or person_count == 0:
                continue

            category = get_category(person_count)
            category_counts[category] += 1

            dest_img = img_dir / f"{processed_count:05d}.jpg"
            shutil.copy(frame_path, dest_img)

            records.append({
                "id": f"seq{seq_id}_frame{frame_number:05d}",
                "image_path": str(Path("images") / dest_img.name),
                "exact_count": person_count,
                "label": category,
            })

            processed_count += 1

    with open(ROOT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone! Exported {len(records)} images to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Labels: {ROOT / 'labels.json'}")

    print(f"\nCategory distribution:")
    for category in ["very_sparse", "sparse", "moderate", "dense", "very_dense"]:
        count = category_counts[category]
        if count > 0:
            print(f"  {category}: {count} images")

    print(f"\nCleaning up temporary files...")
    try:
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            print(f"  Removed: {dataset_path.name}/")
        if DATASET_ZIP.exists():
            DATASET_ZIP.unlink()
            print(f"  Removed: {DATASET_ZIP.name}")
    except Exception as e:
        print(f"  Cleanup warning: {e}")


if __name__ == "__main__":
    main()
