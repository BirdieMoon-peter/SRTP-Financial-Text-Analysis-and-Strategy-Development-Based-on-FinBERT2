"""
Download HuggingFace models via SOCKS5 proxy and save locally.
"""

import os
import sys
import requests
from pathlib import Path

PROXY = "socks5://127.0.0.1:10808"
MODELS_DIR = Path("/Users/peter/Desktop/026课程资料/SRTP/models")

MODELS = [
    {
        "name": "yiyanghkust/finbert-tone-chinese",
        "local_dir": "finbert-tone-chinese",
    },
    {
        "name": "bert-base-chinese",
        "local_dir": "bert-base-chinese",
    },
]


def download_model(model_name, local_dir):
    """Download all files for a HuggingFace model."""
    save_dir = MODELS_DIR / local_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.proxies = {"http": PROXY, "https": PROXY}

    print(f"\n{'='*50}")
    print(f"Model: {model_name} -> {save_dir}")

    # Get file list
    api_url = f"https://huggingface.co/api/models/{model_name}"
    r = session.get(api_url, timeout=30)
    r.raise_for_status()
    files = [s["rfilename"] for s in r.json()["siblings"]]
    print(f"Files: {len(files)}")

    base_url = f"https://huggingface.co/{model_name}/resolve/main"

    for fname in files:
        file_url = f"{base_url}/{fname}"
        file_path = save_dir / fname

        if file_path.exists():
            # Check size match
            r_head = session.head(file_url, timeout=30)
            expected_size = int(r_head.headers.get("content-length", 0))
            if file_path.stat().st_size == expected_size:
                print(f"  [skip] {fname} (exists)")
                continue

        print(f"  [downloading] {fname} ...", end="", flush=True)
        r = session.get(file_url, timeout=300, stream=True)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(file_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        size_mb = downloaded / 1024 / 1024
        print(f" done ({size_mb:.1f} MB)")

    print(f"  Model saved. Files: {os.listdir(save_dir)}")


def main():
    for m in MODELS:
        try:
            download_model(m["name"], m["local_dir"])
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print("\nAll models downloaded.")


if __name__ == "__main__":
    main()
