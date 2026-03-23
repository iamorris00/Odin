"""
parse_witsml_logs.py
--------------------
Parses WITSML realtime drilling log data from data/raw/WITSML Realtime drilling data/
into clean CSV files in data/processed/witsml/

WITSML tree structure:
  <well_dir>/
    1/                      <- wellbore
      log/
        MetaFileInfo.txt    <- "1  Depth\n2  DateTime"
        1/                  <- Depth-indexed logs
          MetaFileInfo.txt  <- log run names (e.g. "26in section MD Log")
          1/                <- log run 1
            1/              <- sequence chunk number
              00001.xml     <- actual data XML
              00002.xml
            ...
        2/                  <- Time-indexed logs
          ...
      trajectory/
      _wellboreInfo/
"""

import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import logging

from utils import normalize_well_name, safe_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_WITSML_DIR = BASE_DIR / "data" / "raw" / "WITSML Realtime drilling data"
OUT_DIR = BASE_DIR / "data" / "processed" / "witsml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# WITSML namespace (varies; we strip to handle any)
def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def read_meta(meta_path: Path) -> dict[str, str]:
    """Parse MetaFileInfo.txt: lines like '1  Log Name Here'"""
    result = {}
    if not meta_path.exists():
        return result
    for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1].strip()
    return result


def parse_log_xml(xml_path: Path) -> pd.DataFrame | None:
    """
    Parse a single WITSML log XML chunk file.
    Returns a DataFrame with columns = logCurveInfo mnemonics.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        log.warning(f"XML parse error in {xml_path}: {e}")
        return None

    # Find all 'log' elements (handle namespace)
    logs = [c for c in root.iter() if _strip_ns(c.tag) == "log"]
    if not logs:
        return None

    all_frames = []

    for log_elem in logs:
        # ── extract curve headers ──────────────────────────────
        curves = []
        for curve in log_elem:
            if _strip_ns(curve.tag) == "logCurveInfo":
                mnemonic = None
                unit = None
                for sub in curve:
                    tag = _strip_ns(sub.tag)
                    if tag == "mnemonic":
                        mnemonic = sub.text.strip() if sub.text else None
                    elif tag == "unit":
                        unit = (sub.text.strip() if sub.text else "")
                if mnemonic:
                    curves.append({"mnemonic": mnemonic, "unit": unit})

        if not curves:
            continue

        # ── extract data rows ─────────────────────────────────
        rows = []
        for elem in log_elem:
            if _strip_ns(elem.tag) == "logData":
                for data_elem in elem:
                    if _strip_ns(data_elem.tag) == "data" and data_elem.text:
                        values = [v.strip() for v in data_elem.text.split(",")]
                        # Align to curve count (some rows may be partial)
                        while len(values) < len(curves):
                            values.append("")
                        rows.append(values[:len(curves)])

        if not rows:
            continue

        col_names = [c["mnemonic"] for c in curves]
        units_map = {c["mnemonic"]: c["unit"] for c in curves}

        df = pd.DataFrame(rows, columns=col_names)

        # Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        # Tag unit metadata as attribute (not stored in CSV rows)
        df.attrs["units"] = units_map
        all_frames.append(df)

    if not all_frames:
        return None
    return pd.concat(all_frames, ignore_index=True)


def collect_well_log_data(well_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Walk a single well directory and collect all log data.
    Returns dict: {log_index_type + '_' + section_name → DataFrame}
    """
    wellbore_dir = well_dir / "1"
    log_dir = wellbore_dir / "log"

    if not log_dir.exists():
        log.warning(f"No log/ dir in {well_dir}")
        return {}

    # Top-level meta: "1  Depth", "2  DateTime"
    top_meta = read_meta(log_dir / "MetaFileInfo.txt")

    all_section_frames = {}

    for index_type_num, index_type_name in top_meta.items():
        index_subdir = log_dir / index_type_num
        if not index_subdir.is_dir():
            continue

        section_meta = read_meta(index_subdir / "MetaFileInfo.txt")

        for section_num, section_name in section_meta.items():
            section_dir = index_subdir / section_num
            if not section_dir.is_dir():
                continue

            frames = []
            # Data chunks live in numbered subdirs then 00001.xml etc.
            for chunk_dir in sorted(section_dir.iterdir()):
                if not chunk_dir.is_dir():
                    continue
                for xml_file in sorted(chunk_dir.glob("*.xml")):
                    df = parse_log_xml(xml_file)
                    if df is not None and not df.empty:
                        frames.append(df)

            if frames:
                combined = pd.concat(frames, ignore_index=True)
                label = f"{index_type_name}|{section_name}"
                all_section_frames[label] = combined
                log.info(f"  [{label}] → {len(combined)} rows, {combined.shape[1]} cols")

    return all_section_frames


def get_well_name_from_dir(well_dir: Path, meta_map: dict[str, str]) -> str:
    """Map folder name like 'Norway-Statoil-15_$47$_9-F-12' → well name."""
    folder = well_dir.name
    # Look up in MetaFileInfo mapping (folder → well name)
    for k, v in meta_map.items():
        if k.strip() == folder.strip():
            return v
    # Fallback: convert $47$ → /
    return folder.replace("_$47$_", "/").replace("$47$", "/")


def parse_all_wells():
    # Read global meta mapping
    global_meta_file = RAW_WITSML_DIR / "MetaFileInfo.txt"
    folder_to_well = {}
    if global_meta_file.exists():
        for line in global_meta_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split("  ", 1)
            if len(parts) == 2:
                folder_to_well[parts[0].strip()] = parts[1].strip()

    well_dirs = [d for d in RAW_WITSML_DIR.iterdir()
                 if d.is_dir() and d.name not in ("__pycache__",)]

    all_wells_summary = []

    for well_dir in sorted(well_dirs):
        well_name_raw = get_well_name_from_dir(well_dir, folder_to_well)
        well_name_canonical = normalize_well_name(well_name_raw)
        # Sanitize for filename
        well_name_safe = safe_filename(well_name_canonical)
        log.info(f"\n=== Processing well: {well_name_canonical} ({well_dir.name}) ===")

        section_frames = collect_well_log_data(well_dir)

        if not section_frames:
            log.warning(f"  No data found for {well_name_canonical}")
            continue

        # ── Strategy: prefer Depth-indexed data, pick the richest sections ──
        # Merge sections that share the first index column (depth) if possible
        depth_frames = {k: v for k, v in section_frames.items()
                        if k.startswith("Depth")}
        time_frames  = {k: v for k, v in section_frames.items()
                        if k.startswith("DateTime")}

        saved_files = []

        def save_frames(frames_dict: dict, suffix: str):
            for label, df in frames_dict.items():
                # Sanitize label for filename
                label_safe = label.replace("|", "_").replace("/", "-").replace(" ", "_")[:80]
                out_path = OUT_DIR / f"{well_name_safe}__{label_safe}.csv"
                df.to_csv(out_path, index=False)
                saved_files.append(str(out_path))
                log.info(f"  Saved: {out_path.name} ({len(df)} rows)")

        save_frames(depth_frames, "depth")
        save_frames(time_frames, "time")

        all_wells_summary.append({
            "well_name": well_name_canonical,
            "well_folder": well_dir.name,
            "n_depth_sections": len(depth_frames),
            "n_time_sections": len(time_frames),
            "total_sections": len(section_frames),
        })

    # Save summary
    if all_wells_summary:
        summary_df = pd.DataFrame(all_wells_summary)
        summary_path = OUT_DIR / "_witsml_extraction_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        log.info(f"\nSummary saved to {summary_path}")
        print(summary_df.to_string(index=False))
    else:
        log.warning("No data was extracted from any well.")


if __name__ == "__main__":
    parse_all_wells()
