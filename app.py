"""
HuggingFace Spaces entry point for ODIN.
Downloads runtime data from KoopaK/OdinDB on first cold start, then launches the app.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Download data from HF Hub if not already present ──────────────────────────
_data_dir = ROOT / "data"
_marker   = _data_dir / "processed" / ".hf_downloaded"

if not _marker.exists():
    print("First run — downloading runtime data from KoopaK/OdinDB …")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id   = "KoopaK/OdinDB",
            repo_type = "dataset",
            local_dir = str(_data_dir),
            ignore_patterns=["*.git*"],
        )
        _marker.parent.mkdir(parents=True, exist_ok=True)
        _marker.touch()
        print("Data download complete.")
    except Exception as e:
        print(f"Warning: data download failed — {e}")
        print("App will start but data tools may return empty results.")

# ── Launch ────────────────────────────────────────────────────────────────────
from src.agents.app import demo, _figures_dir

demo.launch(
    server_name  = "0.0.0.0",
    server_port  = 7860,
    allowed_paths= [str(_figures_dir)],
)
