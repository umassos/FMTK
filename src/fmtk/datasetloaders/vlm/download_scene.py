# Download SUN397 dataset for scene classification
# Source: tanganke/sun397 on HuggingFace
# Download script for scene classification dataset.
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"  # fall back to standard HTTPS, avoid Xet/CAS CDN

import json
from collections import Counter
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
from huggingface_hub import login, hf_hub_download, list_repo_files
import pyarrow.parquet as pq
from PIL import Image

FMTK_ROOT = Path(__file__).resolve().parents[4]
ROOT = FMTK_ROOT / "data" / "vlm" / "scene_classification"
N = 100


def get_label_names(repo_id, token):
    """Fetch label names from dataset_info.json stored in the repo."""
    try:
        info_path = hf_hub_download(repo_id=repo_id, filename="dataset_infos.json",
                                    repo_type="dataset", token=token)
        with open(info_path) as f:
            info = json.load(f)
        # Navigate to ClassLabel names
        for split_info in info.values():
            features = split_info.get("features", {})
            label_feat = features.get("label", {})
            names = label_feat.get("names")
            if names:
                return names
    except Exception:
        pass
    return None


def main():
    print("=" * 60)
    print("SUN397 Scene Classification Dataset Download Script")
    print("=" * 60)

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

    repo_id = "tanganke/sun397"
    print(f"\nFetching parquet files from {repo_id}...")

    # Get label names from dataset metadata
    label_names = get_label_names(repo_id, token)
    if label_names:
        print(f"Found {len(label_names)} label names")
    else:
        print("Warning: could not fetch label names, will use raw label values")

    # List all train parquet shards
    all_files = list(list_repo_files(repo_id, repo_type="dataset", token=token))
    pq_files = sorted([f for f in all_files if f.endswith(".parquet") and "train" in f])
    print(f"Found {len(pq_files)} train parquet shard(s)")

    # Collect N rows, skipping any corrupted shards
    rows = []
    for pq_file in pq_files:
        if len(rows) >= N:
            break
        try:
            local = hf_hub_download(repo_id=repo_id, filename=pq_file,
                                    repo_type="dataset", token=token,
                                    force_download=True)  # bypass corrupted cache
            table = pq.read_table(local)
            rows.extend(table.to_pylist())
            print(f"  Loaded {len(table)} rows from {pq_file}")
        except Exception as e:
            print(f"  Warning: skipping {pq_file} ({e})")
            continue

    rows = rows[:N]
    print(f"Using {len(rows)} rows total")

    if not rows:
        print("Error: no rows loaded")
        return

    # Determine label column
    label_col = None
    for col in ("label", "labels"):
        if col in rows[0]:
            label_col = col
            break
    if label_col is None:
        print(f"Error: no label column found. Columns: {list(rows[0].keys())}")
        return

    records = []
    category_counts = Counter()

    for i, row in tqdm(enumerate(rows), total=len(rows), desc="Saving scene images"):
        raw_label = row[label_col]

        # Convert int index to name if we have the mapping
        if isinstance(raw_label, int) and label_names and 0 <= raw_label < len(label_names):
            label_name = label_names[raw_label]
        elif isinstance(raw_label, str):
            label_name = raw_label
        else:
            label_name = str(raw_label)

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
            print(f"Warning: could not decode image {i}: {e}")
            continue

        img_filename = f"{i:05d}.jpg"
        image.save(img_dir / img_filename)
        category_counts[label_name] += 1

        records.append({
            "id": f"sun397_{i:05d}",
            "image_path": str(Path("images") / img_filename),
            "label": label_name,
        })

    with open(ROOT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone! Exported {len(records)} images to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Labels: {ROOT / 'labels.json'}")
    print(f"\nCategory distribution (top 10):")
    for category, count in category_counts.most_common(10):
        print(f"  {category}: {count}")
    print(f"\nTotal unique categories: {len(category_counts)}")


if __name__ == "__main__":
    main()
