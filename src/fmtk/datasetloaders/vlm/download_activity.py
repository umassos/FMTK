# Download Human Action Recognition dataset (balanced sampling)
# Source: Bingsu/Human_Action_Recognition on HuggingFace
# Mirrors FMaaS-motivation/vqa/updated/dataset/activity_recognition/download-data-fixed.py
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"  # fall back to standard HTTPS, avoid Xet/CAS CDN

import json
import random
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
from huggingface_hub import login, hf_hub_download, list_repo_files
import pyarrow.parquet as pq
from PIL import Image

FMTK_ROOT = Path(__file__).resolve().parents[4]
ROOT = FMTK_ROOT / "data" / "vlm" / "activity_recognition"
N = 100

ACTIVITY_NAMES = [
    "calling", "clapping", "cycling", "dancing", "drinking",
    "eating", "fighting", "hugging", "laughing", "listening_to_music",
    "running", "sitting", "sleeping", "texting", "using_laptop",
]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    hf_token_path = Path(__file__).parent / "hf-token.txt"
    token = None
    if hf_token_path.exists():
        token = hf_token_path.read_text().strip() or None
        if token:
            login(token=token, add_to_git_credential=False)
            print("Logged in to HuggingFace")

    repo_id = "Bingsu/Human_Action_Recognition"
    print(f"Fetching parquet files from {repo_id}...")

    # Use train split: the test split has a known label bug (all labels == 0)
    all_files = list(list_repo_files(repo_id, repo_type="dataset", token=token))
    test_files = sorted([f for f in all_files if "train" in f and f.endswith(".parquet")])
    print(f"Found {len(test_files)} parquet file(s): {test_files}")

    # Load all test rows into memory (dataset is small)
    rows = []
    for pq_file in test_files:
        local = hf_hub_download(repo_id=repo_id, filename=pq_file,
                                repo_type="dataset", token=token)
        table = pq.read_table(local)
        rows.extend(table.to_pylist())

    print(f"Total rows: {len(rows)}")

    # Determine label column name
    label_col = None
    if rows:
        for col in ("labels", "label"):
            if col in rows[0]:
                label_col = col
                break
    if label_col is None:
        print(f"Error: no label column found. Columns: {list(rows[0].keys())}")
        return

    # Group by label for balanced sampling
    by_label = {i: [] for i in range(len(ACTIVITY_NAMES))}
    for idx, row in enumerate(rows):
        lbl = row[label_col]
        if isinstance(lbl, int) and 0 <= lbl < len(ACTIVITY_NAMES):
            by_label[lbl].append(idx)

    samples_per_class = N // len(ACTIVITY_NAMES)
    remainder = N % len(ACTIVITY_NAMES)
    selected = []
    for label_id in range(len(ACTIVITY_NAMES)):
        n = samples_per_class + (1 if label_id < remainder else 0)
        available = by_label[label_id]
        selected.extend(random.sample(available, min(n, len(available))))

    random.shuffle(selected)
    selected = selected[:N]
    print(f"Selected {len(selected)} balanced samples")

    records = []
    for new_idx, orig_idx in tqdm(enumerate(selected), total=len(selected),
                                   desc="Saving activity images"):
        row = rows[orig_idx]
        label_id = row[label_col]
        activity_name = ACTIVITY_NAMES[label_id]

        # Image stored as dict with 'bytes' key (HF image feature)
        img_data = row.get("image", row.get("img", None))
        if img_data is None:
            continue
        if isinstance(img_data, dict):
            img_bytes = img_data.get("bytes") or img_data.get("path")
        else:
            img_bytes = img_data

        try:
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"Warning: could not decode image {orig_idx}: {e}")
            continue

        img_filename = f"{new_idx:05d}.jpg"
        image.save(img_dir / img_filename)

        records.append({
            "id": f"test_{new_idx:05d}",
            "image_path": str(Path("images") / img_filename),
            "label": activity_name,
        })

    with open(ROOT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone! Exported {len(records)} images to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Labels: {ROOT / 'labels.json'}")

    from collections import Counter
    label_counts = Counter([r['label'] for r in records])
    print(f"\nClass distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
