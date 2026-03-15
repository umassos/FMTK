# Download MS COCO val2017 dataset for object detection
# Source: cocodataset.org
# Download script for object detection dataset.
import json
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

FMTK_ROOT = Path(__file__).resolve().parents[4]
ROOT = FMTK_ROOT / "data" / "vlm" / "object_detection"
N = 1000  # unified_inference caps at 100 via MAX_SAMPLES


def download_file(url, dest_path):
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Download annotations
    ann_zip_path = ROOT / "annotations_trainval2017.zip"
    if not ann_zip_path.exists():
        download_file(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            ann_zip_path
        )

    print("Extracting annotations...")
    with zipfile.ZipFile(ann_zip_path, 'r') as zip_ref:
        zip_ref.extractall(ROOT)

    instances_path = ROOT / "annotations" / "instances_val2017.json"
    with open(instances_path, 'r') as f:
        coco_data = json.load(f)

    cat_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    img_map = {img['id']: img for img in coco_data['images']}

    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    img_zip_path = ROOT / "val2017.zip"
    if not img_zip_path.exists():
        download_file(
            "http://images.cocodataset.org/zips/val2017.zip",
            img_zip_path
        )

    print(f"Extracting first {N} images from val2017.zip...")
    records = []
    extracted_count = 0

    with zipfile.ZipFile(img_zip_path, 'r') as zip_ref:
        all_files = [f for f in zip_ref.namelist() if f.endswith('.jpg')]

        for img_file in tqdm(all_files[:N], desc="Processing images"):
            if extracted_count >= N:
                break

            zip_ref.extract(img_file, ROOT)

            img_filename = Path(img_file).name
            img_id = int(img_filename.replace('.jpg', ''))

            if img_id not in img_map:
                continue

            src_path = ROOT / img_file
            dest_path = img_dir / img_filename
            src_path.rename(dest_path)

            anns = img_to_anns.get(img_id, [])
            unique_categories = sorted(set([cat_map[ann['category_id']] for ann in anns]))

            records.append({
                "id": f"val_{extracted_count:05d}",
                "image_path": str(Path("images") / img_filename),
                "categories": unique_categories,
            })

            extracted_count += 1

    val2017_dir = ROOT / "val2017"
    if val2017_dir.exists():
        shutil.rmtree(val2017_dir)

    with open(ROOT / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone! Exported {len(records)} images to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Annotations: {ROOT / 'annotations.json'}")

    print("\nCleaning up temporary files...")
    ann_zip_path.unlink()
    img_zip_path.unlink()
    shutil.rmtree(ROOT / "annotations")
    print("Cleanup complete!")


if __name__ == "__main__":
    main()
