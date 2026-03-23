"""
upload_data.py
--------------
Uploads the ODIN runtime data to Hugging Face Hub (run this ONCE as the repo owner).

Uploads:
  data/processed/      — cleaned DDR / WITSML / EDM CSVs
  data/knowledge_base/ — Volve history ChromaDB vector store
  data/viking_context/ — OpenViking ChromaDB vector store

Usage:
    huggingface-cli login          # authenticate first
    python scripts/upload_data.py

Requirements:
    pip install huggingface_hub
"""
import sys
from pathlib import Path

HF_REPO_ID = "SPE-GCS-2026/odin-volve-data"   # <- your HF org/username + repo name
ROOT       = Path(__file__).parent.parent

UPLOAD_DIRS = [
    ROOT / "data" / "processed",
    ROOT / "data" / "knowledge_base",
    ROOT / "data" / "viking_context",
]

def main():
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Create dataset repo if it doesn't exist
    try:
        create_repo(HF_REPO_ID, repo_type="dataset", exist_ok=True, private=False)
        print(f"Dataset repo ready: https://huggingface.co/datasets/{HF_REPO_ID}\n")
    except Exception as e:
        print(f"Repo creation warning (may already exist): {e}")

    for folder in UPLOAD_DIRS:
        if not folder.exists():
            print(f"Skipping {folder} (not found)")
            continue
        hf_path = folder.relative_to(ROOT)   # e.g. data/processed
        print(f"Uploading {folder} → {hf_path} …")
        api.upload_folder(
            repo_id     = HF_REPO_ID,
            repo_type   = "dataset",
            folder_path = str(folder),
            path_in_repo= str(hf_path),
        )
        print(f"  ✓ {hf_path} uploaded\n")

    print("All done. Judges can now download with:")
    print("  python scripts/download_data.py")

if __name__ == "__main__":
    main()
