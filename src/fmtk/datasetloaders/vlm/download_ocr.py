# Download MNIST dataset for OCR (digit recognition)
# Source: ylecun/mnist on HuggingFace
# Mirrors FMaaS-motivation/vqa/updated/dataset/ocr/download-data.py
# Note: labels are stored as int (e.g. {"label": 5}); VLMDataset converts to str
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"  # fall back to standard HTTPS, avoid Xet/CAS CDN

import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

FMTK_ROOT = Path(__file__).resolve().parents[4]
ROOT = FMTK_ROOT / "data" / "vlm" / "ocr"
N = 1000  # unified_inference caps at 100 via MAX_SAMPLES


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ylecun/mnist")
    split = "test"
    subset = ds[split].select(range(min(N, len(ds[split]))))

    records = []
    for i, row in tqdm(enumerate(subset), total=len(subset), desc="Export MNIST"):
        img = row["image"].convert("RGB")
        fname = f"{i:05d}.png"
        path = img_dir / fname
        img.save(path)

        records.append({
            "id": f"{split}_{i:05d}",
            "image_path": str(Path("images") / fname),
            "label": int(row["label"]),  # stored as int; VLMDataset converts to str
        })

    with open(ROOT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Done. Wrote {len(records)} samples to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Labels: {ROOT / 'labels.json'}")


if __name__ == "__main__":
    main()
