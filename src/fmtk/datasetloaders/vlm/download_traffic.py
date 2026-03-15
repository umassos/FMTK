# Download GTSRB (German Traffic Sign Recognition Benchmark) dataset
# Source: tanganke/gtsrb on HuggingFace
# Download script for traffic classification dataset.
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"  # fall back to standard HTTPS, avoid Xet/CAS CDN

import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

FMTK_ROOT = Path(__file__).resolve().parents[4]
ROOT = FMTK_ROOT / "data" / "vlm" / "traffic_classification"
N = 100

CLASS_NAMES = [
    "speed limit 20",
    "speed limit 30",
    "speed limit 50",
    "speed limit 60",
    "speed limit 70",
    "speed limit 80",
    "end speed limit 80",
    "speed limit 100",
    "speed limit 120",
    "no passing",
    "no passing trucks",
    "priority at next intersection",
    "priority road",
    "yield",
    "stop",
    "no entry",
    "no trucks",
    "no vehicles",
    "danger",
    "bend left",
    "bend right",
    "bend",
    "uneven road",
    "slippery road",
    "road narrows",
    "construction",
    "traffic signal",
    "pedestrian crossing",
    "school crossing",
    "cycles crossing",
    "snow",
    "animals",
    "end of restrictions",
    "go right",
    "go left",
    "go straight",
    "go right or straight",
    "go left or straight",
    "keep right",
    "keep left",
    "roundabout",
    "end no passing",
    "end no passing trucks",
]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    hf_token_path = Path(__file__).parent / "hf-token.txt"
    if hf_token_path.exists():
        token = hf_token_path.read_text().strip()
        if token:
            login(token=token, add_to_git_credential=False)
            print("Logged in to HuggingFace")

    print(f"Loading GTSRB test dataset (first {N} samples)...")

    try:
        ds = load_dataset("tanganke/gtsrb", split="test", trust_remote_code=True)
        subset = ds.select(range(min(N, len(ds))))
    except Exception as e:
        print(f"Error loading GTSRB: {e}")
        print("Trying alternative dataset...")
        try:
            ds = load_dataset("bazyl/GTSRB", split="test", trust_remote_code=True)
            subset = ds.select(range(min(N, len(ds))))
        except Exception as e2:
            print(f"Alternative dataset also failed: {e2}")
            return

    records = []

    for i, sample in tqdm(enumerate(subset), total=len(subset), desc="Downloading GTSRB images"):
        image = sample["image"].convert("RGB")
        label_id = sample["label"]

        if 0 <= label_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[label_id]
        else:
            print(f"Warning: Unknown label {label_id}, using 'unknown'")
            class_name = "unknown"

        img_filename = f"{i:05d}.jpg"
        img_path = img_dir / img_filename
        image.save(img_path)

        records.append({
            "id": f"test_{i:05d}",
            "image_path": str(Path("images") / img_filename),
            "label": class_name,
        })

    with open(ROOT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone! Exported {len(records)} images to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Labels: {ROOT / 'labels.json'}")

    from collections import Counter
    label_counts = Counter([r['label'] for r in records])
    print(f"\nClass distribution (top 10):")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
