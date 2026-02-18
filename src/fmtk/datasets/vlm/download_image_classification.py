# Download ImageNet-1k validation dataset for image classification
# Source: imagenet-1k on HuggingFace (requires accepting terms)
# Mirrors FMaaS-motivation/vqa/updated/dataset/image_classification/download-data.py
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"  # fall back to standard HTTPS, avoid Xet/CAS CDN

import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

FMTK_ROOT = Path(__file__).resolve().parents[4]
ROOT = FMTK_ROOT / "data" / "vlm" / "image_classification"
N = 1000  # number of images to export (unified_inference caps at 100 via MAX_SAMPLES)


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

    print(f"Loading ImageNet-1k validation dataset (streaming first {N} samples)...")
    print("Note: You may need to accept ImageNet terms at https://huggingface.co/datasets/imagenet-1k")

    try:
        ds_stream = load_dataset("imagenet-1k", split="validation", streaming=True)
        ds_meta = load_dataset("imagenet-1k", split="validation[:1]")
        label_names = ds_meta.features['label'].names if hasattr(ds_meta.features['label'], 'names') else None
    except Exception as e:
        print(f"Error loading ImageNet: {e}")
        print("Make sure you have:")
        print("  1. Accepted ImageNet terms at: https://huggingface.co/datasets/imagenet-1k")
        print("  2. Valid HuggingFace token in hf-token.txt at FMTK root")
        return

    if not label_names:
        print("Warning: Could not get label names, using label indices")

    records = []

    for i, sample in tqdm(enumerate(ds_stream.take(N)), total=N, desc="Downloading ImageNet images"):
        image = sample["image"].convert("RGB")
        label_id = sample["label"]

        if label_names:
            label = label_names[label_id]
        else:
            label = str(label_id)

        img_filename = f"{i:05d}.jpg"
        img_path = img_dir / img_filename
        image.save(img_path)

        records.append({
            "id": f"val_{i:05d}",
            "image_path": str(Path("images") / img_filename),
            "label": label,
        })

    with open(ROOT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone! Exported {len(records)} images to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Labels: {ROOT / 'labels.json'}")
    print(f"\nSample labels (first 5):")
    for rec in records[:5]:
        print(f"  {rec['id']}: {rec['label']}")


if __name__ == "__main__":
    main()
