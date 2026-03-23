"""
parse_ddr_xml.py
----------------
Parses Daily Drilling Report (DDR) XML files (WITSML 1.4 drillReport schema)
from data/raw/Well_technical_data/Daily Drilling Report - XML Version/
into structured CSV files in data/processed/ddr/

Produces two outputs per well:
  1. <well>_activities.csv  — timestamped activity log with depth, phase, code, comments
  2. <well>_daily_summary.csv — one row per daily report with high-level metadata

Also produces:
  - _ddr_all_activities.csv — consolidated across all wells (useful for agent queries)
"""

import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict

from utils import normalize_well_name, safe_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DDR_DIR = BASE_DIR / "data" / "raw" / "Well_technical_data" / "Daily Drilling Report - XML Version"
OUT_DIR = BASE_DIR / "data" / "processed" / "ddr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WITSML_NS = {
    "witsml": "http://www.witsml.org/schemas/1series"
}


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def find_text(elem: ET.Element, tag: str, ns: str = "witsml") -> str | None:
    """Find text of first matching child (namespace-aware and ns-stripped)."""
    # Try namespace-qualified
    child = elem.find(f"witsml:{tag}", WITSML_NS)
    if child is not None:
        return child.text.strip() if child.text else None
    # Fall back to strip-namespace search
    for c in elem:
        if _strip_ns(c.tag) == tag:
            return c.text.strip() if c.text else None
    return None


def parse_ddr_xml(xml_path: Path) -> dict:
    """
    Parse a single DDR XML file.
    Returns dict with keys:
      - 'daily':     dict of per-report metadata
      - 'activities': list of activity dicts
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        log.warning(f"Parse error {xml_path.name}: {e}")
        return {"daily": None, "activities": []}

    # drillReport elements can be at root level or nested
    reports = list(root.iter())
    dr_elems = [e for e in reports if _strip_ns(e.tag) == "drillReport"]

    if not dr_elems:
        return {"daily": None, "activities": []}

    all_daily = []
    all_activities = []

    for dr in dr_elems:
        # ── Daily header ─────────────────────────────────────────────────────
        well_name      = find_text(dr, "nameWell")
        wellbore_name  = find_text(dr, "nameWellbore")
        dtim_start     = find_text(dr, "dTimStart")
        dtim_end       = find_text(dr, "dTimEnd")
        create_date    = find_text(dr, "createDate")

        # wellboreInfo block
        wb_info = None
        for c in dr:
            if _strip_ns(c.tag) == "wellboreInfo":
                wb_info = c
                break

        spud_date       = find_text(wb_info, "dTimSpud")         if wb_info is not None else None
        drill_complete  = find_text(wb_info, "dateDrillComplete") if wb_info is not None else None
        operator        = find_text(wb_info, "operator")          if wb_info is not None else None
        drill_contractor= find_text(wb_info, "drillContractor")  if wb_info is not None else None

        daily_row = {
            "file":             xml_path.name,
            "well_name":        well_name,
            "wellbore_name":    wellbore_name,
            "report_start":     dtim_start,
            "report_end":       dtim_end,
            "create_date":      create_date,
            "spud_date":        spud_date,
            "drill_complete":   drill_complete,
            "operator":         operator,
            "drill_contractor": drill_contractor,
        }
        all_daily.append(daily_row)

        # ── Activities ───────────────────────────────────────────────────────
        for elem in dr.iter():
            if _strip_ns(elem.tag) == "activity":
                act_start    = find_text(elem, "dTimStart")
                act_end      = find_text(elem, "dTimEnd")
                phase        = find_text(elem, "phase")
                prop_code    = find_text(elem, "proprietaryCode")
                state        = find_text(elem, "state")
                state_detail = find_text(elem, "stateDetailActivity")
                comments     = find_text(elem, "comments")

                # Measured depth
                md_val = None
                md_uom = None
                for c in elem:
                    if _strip_ns(c.tag) == "md":
                        md_val = c.text.strip() if c.text else None
                        md_uom = c.attrib.get("uom", None)

                # Duration in hours if both timestamps available
                all_activities.append({
                    "file":             xml_path.name,
                    "well_name":        well_name,
                    "wellbore_name":    wellbore_name,
                    "report_start":     dtim_start,
                    "report_end":       dtim_end,
                    "act_start":        act_start,
                    "act_end":          act_end,
                    "md_m":             md_val,
                    "md_uom":           md_uom,
                    "phase":            phase,
                    "activity_code":    prop_code,
                    "state":            state,
                    "state_detail":     state_detail,
                    "comments":         comments,
                })

    return {"daily": all_daily, "activities": all_activities}


def extract_well_key(well_name: str | None) -> str:
    """Turn 'NO 15/9-F-12' → '15/9-F-12' (canonical) for consistent referencing."""
    return normalize_well_name(well_name or "UNKNOWN")


def parse_all_ddrs():
    xml_files = sorted([f for f in DDR_DIR.glob("*.xml")
                        if not f.name.endswith("Zone.Identifier")])

    log.info(f"Found {len(xml_files)} DDR XML files in {DDR_DIR}")

    all_daily_by_well: dict[str, list] = defaultdict(list)
    all_acts_by_well:  dict[str, list] = defaultdict(list)

    for xml_path in xml_files:
        result = parse_ddr_xml(xml_path)
        if result["daily"]:
            for row in result["daily"]:
                key = extract_well_key(row.get("well_name"))
                all_daily_by_well[key].append(row)
        for act in result["activities"]:
            key = extract_well_key(act.get("well_name"))
            all_acts_by_well[key].append(act)

    all_wells = sorted(set(list(all_daily_by_well.keys()) + list(all_acts_by_well.keys())))

    summary_rows = []
    all_acts_global = []

    for well_key in all_wells:
        # ── Daily summary CSV ────────────────────────────────────────────────
        daily_rows = all_daily_by_well.get(well_key, [])
        if daily_rows:
            df_daily = pd.DataFrame(daily_rows).drop_duplicates()
            df_daily["report_start"] = pd.to_datetime(df_daily["report_start"], errors="coerce", utc=True)
            df_daily = df_daily.sort_values("report_start")
            safe_key = safe_filename(well_key)
            out_daily = OUT_DIR / f"{safe_key}_daily_summary.csv"
            df_daily.to_csv(out_daily, index=False)
            log.info(f"  [{well_key}] {len(df_daily)} daily reports → {out_daily.name}")

        # ── Activities CSV ───────────────────────────────────────────────────
        act_rows = all_acts_by_well.get(well_key, [])
        if act_rows:
            df_acts = pd.DataFrame(act_rows)
            df_acts["act_start"] = pd.to_datetime(df_acts["act_start"], errors="coerce", utc=True)
            df_acts["act_end"]   = pd.to_datetime(df_acts["act_end"],   errors="coerce", utc=True)
            df_acts["md_m"]      = pd.to_numeric(df_acts["md_m"],       errors="coerce")
            df_acts = df_acts.sort_values("act_start")

            # Compute duration_hours
            mask = df_acts["act_start"].notna() & df_acts["act_end"].notna()
            df_acts.loc[mask, "duration_hours"] = (
                (df_acts.loc[mask, "act_end"] - df_acts.loc[mask, "act_start"])
                .dt.total_seconds() / 3600
            )

            safe_key = safe_filename(well_key)
            out_acts = OUT_DIR / f"{safe_key}_activities.csv"
            df_acts.to_csv(out_acts, index=False)
            log.info(f"  [{well_key}] {len(df_acts)} activities → {out_acts.name}")
            all_acts_global.append(df_acts)

        summary_rows.append({
            "well_key":        well_key,
            "n_daily_reports": len(daily_rows),
            "n_activities":    len(act_rows),
        })

    # ── Global consolidated activities file ───────────────────────────────────
    if all_acts_global:
        df_all = pd.concat(all_acts_global, ignore_index=True)
        df_all = df_all.sort_values(["well_name", "act_start"])
        df_all.to_csv(OUT_DIR / "_ddr_all_activities.csv", index=False)
        log.info(f"\nGlobal activities file: {len(df_all)} rows across {len(all_wells)} wells")

    # ── Extraction summary ────────────────────────────────────────────────────
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(OUT_DIR / "_ddr_extraction_summary.csv", index=False)
        print("\n" + df_summary.to_string(index=False))


if __name__ == "__main__":
    parse_all_ddrs()
