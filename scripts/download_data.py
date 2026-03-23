"""
download_data.py
----------------
Downloads the ODIN runtime data (processed CSVs + ChromaDB knowledge bases)
from Hugging Face Hub into the local data/ directory.

Usage:
    python scripts/download_data.py

Requirements:
    pip install huggingface_hub
"""
import os
import sys
from pathlib import Path

HF_REPO_ID  = "SPE-GCS-2026/odin-volve-data"   # <- update if repo is renamed
LOCAL_DIR   = Path(__file__).parent.parent / "data"

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading ODIN data from HuggingFace ({HF_REPO_ID}) …")
    print(f"Destination: {LOCAL_DIR.resolve()}")
    print("This may take a few minutes (~400 MB knowledge bases + processed CSVs).\n")

    snapshot_download(
        repo_id   = HF_REPO_ID,
        repo_type = "dataset",
        local_dir = str(LOCAL_DIR),
        ignore_patterns=["*.git*", "README.md"],
    )

    print("\nDone. You can now run the app:")
    print("  python src/agents/app.py")

if __name__ == "__main__":
    main()
