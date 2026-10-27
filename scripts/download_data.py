"""Download the Elliptic Bitcoin dataset (Weber et al., 2019) into data/raw/.

Primary source is the official Kaggle release by Elliptic; the PyTorch Geometric
mirror is used as a fallback. Both serve the same three CSV files.

Usage:
    python scripts/download_data.py
"""

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = RAW_DIR / "elliptic_bitcoin_dataset"

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set"
KAGGLE_SHA256 = "70ed052c52002392720615a03261039b62d1ad594f6081ad021cb174a8ad5717"
PYG_URL = "https://data.pyg.org/datasets/elliptic/{name}.zip"

FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
]


def fetch(url, dest):
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def from_kaggle():
    archive = RAW_DIR / "elliptic.zip"
    if not archive.exists():
        fetch(KAGGLE_URL, archive)
    digest = sha256(archive)
    if digest != KAGGLE_SHA256:
        raise ValueError(f"Checksum mismatch for {archive}: {digest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)


def from_pyg():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        archive = RAW_DIR / f"{name}.zip"
        fetch(PYG_URL.format(name=name), archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(OUT_DIR)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((OUT_DIR / name).exists() for name in FILES):
        print(f"Already present: {OUT_DIR}")
        return
    try:
        from_kaggle()
    except Exception as e:
        print(f"Kaggle download failed ({e}); falling back to PyG mirror", file=sys.stderr)
        from_pyg()
    missing = [name for name in FILES if not (OUT_DIR / name).exists()]
    if missing:
        sys.exit(f"Missing files after download: {missing}")
    print(f"Done: {OUT_DIR}")


if __name__ == "__main__":
    main()
