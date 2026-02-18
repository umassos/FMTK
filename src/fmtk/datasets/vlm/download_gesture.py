# Download HaGRID (Hand Gesture Recognition) dataset
# Source: cj-mills/hagrid-sample-500k-384p on HuggingFace
# Uses HTTP range requests to stream only N images from the 13 GB zip without
# downloading the whole file. Mirrors FMaaS-motivation gesture download script.
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"  # fall back to standard HTTPS, avoid Xet/CAS CDN

import io
import json
import zipfile
import requests
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
from huggingface_hub import login
from PIL import Image

FMTK_ROOT = Path(__file__).resolve().parents[4]
ROOT = FMTK_ROOT / "data" / "vlm" / "gesture_recognition"
N = 100

GESTURE_NAMES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted", "no_gesture",
]


class HTTPRangeFile(io.RawIOBase):
    """Seekable file-like object that fetches byte ranges via HTTP."""

    def __init__(self, url, token=None):
        self.url = url
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._pos = 0
        # Get file size from HEAD
        r = requests.head(url, headers=self.headers, allow_redirects=True, timeout=30)
        r.raise_for_status()
        self._size = int(r.headers["Content-Length"])

    def readable(self):  return True
    def seekable(self):  return True
    def tell(self):      return self._pos

    def seek(self, offset, whence=0):
        if whence == 0:   self._pos = offset
        elif whence == 1: self._pos += offset
        elif whence == 2: self._pos = self._size + offset
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, n=-1):
        if self._pos >= self._size:
            return b""
        end = self._size - 1 if n < 0 else min(self._pos + n - 1, self._size - 1)
        r = requests.get(
            self.url,
            headers={**self.headers, "Range": f"bytes={self._pos}-{end}"},
            timeout=60,
        )
        r.raise_for_status()
        data = r.content
        self._pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def get_zip_url(repo_id, filename, token):
    """Resolve the direct download URL for a file in an HF dataset repo."""
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    # Follow redirects to get the actual CDN URL
    r = requests.head(url,
                      headers={"Authorization": f"Bearer {token}"} if token else {},
                      allow_redirects=True, timeout=30)
    return r.url


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

    repo_id = "cj-mills/hagrid-sample-500k-384p"
    zip_filename = "hagrid-sample-500k-384p.zip"

    print(f"Resolving download URL for {zip_filename}...")
    url = get_zip_url(repo_id, zip_filename, token)
    print(f"Streaming zip via HTTP range requests (avoids 13 GB full download)...")

    http_file = HTTPRangeFile(url, token)
    print(f"Zip size: {http_file._size / 1e9:.1f} GB")

    samples_per_class = max(1, N // len(GESTURE_NAMES))
    class_counts = {g: 0 for g in GESTURE_NAMES}
    records = []
    count = 0

    with zipfile.ZipFile(io.BufferedReader(http_file), 'r') as zf:
        all_entries = [e for e in zf.namelist() if e.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(all_entries)} image entries in zip")

        for entry in tqdm(all_entries, desc="Extracting gestures"):
            if count >= N:
                break

            parts = Path(entry).parts
            if len(parts) < 2:
                continue
            gesture_name = parts[-2]

            if gesture_name not in GESTURE_NAMES:
                continue
            if class_counts[gesture_name] >= samples_per_class:
                continue

            try:
                img_bytes = zf.read(entry)
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"Warning: could not decode {entry}: {e}")
                continue

            img_filename = f"{count:05d}.jpg"
            image.save(img_dir / img_filename)

            records.append({
                "id": f"train_{count:05d}",
                "image_path": str(Path("images") / img_filename),
                "label": gesture_name,
            })

            class_counts[gesture_name] += 1
            count += 1

    with open(ROOT / "labels.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone! Exported {len(records)} images to {ROOT}")
    print(f"  - Images: {img_dir}")
    print(f"  - Labels: {ROOT / 'labels.json'}")

    from collections import Counter
    label_counts = Counter([r['label'] for r in records])
    print(f"\nClass distribution:")
    for label, c in label_counts.most_common():
        print(f"  {label}: {c}")


if __name__ == "__main__":
    main()
