# Download VQA v2 (COCO val2014) dataset
# Sources:
#   - Annotations: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
#   - Questions: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
#   - Images: http://images.cocodataset.org/zips/val2014.zip
# Output: val.json (dict keyed by composite id) + val2014/ directory of images
# Expected format for val.json.
import json
import urllib.request
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

FMTK_ROOT = Path(__file__).resolve().parents[4]
ROOT = FMTK_ROOT / "data" / "vlm" / "vqa"
N = 100  # number of QA pairs to export (mirrors MAX_SAMPLES cap)

VQA_ANN_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
VQA_Q_URL   = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
IMG_URL     = "http://images.cocodataset.org/zips/val2014.zip"


def download_file(url, dest_path):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "val2014"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ---- annotations ----
    ann_zip = ROOT / "v2_Annotations_Val_mscoco.zip"
    ann_json = ROOT / "v2_mscoco_val2014_annotations.json"

    if not ann_json.exists():
        if not ann_zip.exists():
            download_file(VQA_ANN_URL, ann_zip)
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as z:
            z.extractall(ROOT)
        ann_zip.unlink(missing_ok=True)

    # ---- questions ----
    q_zip  = ROOT / "v2_Questions_Val_mscoco.zip"
    q_json = ROOT / "v2_OpenEnded_mscoco_val2014_questions.json"

    if not q_json.exists():
        if not q_zip.exists():
            download_file(VQA_Q_URL, q_zip)
        print("Extracting questions...")
        with zipfile.ZipFile(q_zip, 'r') as z:
            z.extractall(ROOT)
        q_zip.unlink(missing_ok=True)

    # ---- build val.json (first N QA pairs) ----
    print("Loading annotations and questions...")
    with open(ann_json) as f:
        ann_data = json.load(f)
    with open(q_json) as f:
        q_data = json.load(f)

    # Build question_id -> question text map
    qid_to_question = {q['question_id']: q['question'] for q in q_data['questions']}

    # Select top N annotations; pick the most common answer for each
    records = {}
    for ann in ann_data['annotations'][:N]:
        qid     = ann['question_id']
        img_id  = ann['image_id']
        question = qid_to_question.get(qid, "")

        # Most common answer
        answers = [a['answer'] for a in ann['answers']]
        answer  = max(set(answers), key=answers.count)

        key = str(qid)
        records[key] = {
            "image_id": img_id,
            "question": question,
            "answer":   answer,
        }

    with open(ROOT / "val.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} QA pairs to {ROOT / 'val.json'}")

    # ---- download images for the selected image_ids ----
    needed_ids = set(r['image_id'] for r in records.values())
    print(f"Need {len(needed_ids)} unique images.")

    img_zip = ROOT / "val2014.zip"
    if not img_zip.exists():
        download_file(IMG_URL, img_zip)

    print("Extracting required val2014 images (streaming)...")
    found = 0
    with zipfile.ZipFile(img_zip, 'r') as z:
        all_files = [f for f in z.namelist() if f.endswith('.jpg')]
        for img_file in tqdm(all_files, desc="Scanning images"):
            img_name = Path(img_file).name
            # filename pattern: COCO_val2014_000000012345.jpg
            try:
                img_id = int(img_name.replace("COCO_val2014_", "").replace(".jpg", ""))
            except ValueError:
                continue

            if img_id in needed_ids:
                dest = img_dir / img_name
                if not dest.exists():
                    with z.open(img_file) as src, open(dest, 'wb') as dst:
                        dst.write(src.read())
                found += 1

            if found >= len(needed_ids):
                break

    img_zip.unlink(missing_ok=True)

    print(f"\nDone! {found} images saved to {img_dir}")
    print(f"  - Images:  {img_dir}")
    print(f"  - val.json: {ROOT / 'val.json'}")
    print(f"\nSample records (first 3):")
    for k, v in list(records.items())[:3]:
        print(f"  {k}: image_id={v['image_id']}, q={v['question']!r}, a={v['answer']!r}")


if __name__ == "__main__":
    main()
