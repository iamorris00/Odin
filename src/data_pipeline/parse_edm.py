"""
parse_edm.py
------------
Parses the Volve F.edm.xml (Landmark Engineering Data Model) into
structured CSVs extracting well/wellbore metadata, casing configurations,
BHA (Bottom Hole Assembly) details, and daily cost records.

Outputs to data/processed/edm/
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
EDM_FILE = BASE_DIR / "data" / "raw" / "Well_technical_data" / "EDM.XML" / "Volve F.edm.xml"
OUT_DIR  = BASE_DIR / "data" / "processed" / "edm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def elem_to_dict(elem: ET.Element, prefix: str = "") -> dict:
    """
    Flatten an XML element into a flat dict by concatenating tag paths.
    Handles attributes and text content.
    """
    result = {}
    for attr_k, attr_v in elem.attrib.items():
        result[f"{prefix}{_strip_ns(attr_k)}"] = attr_v
    if elem.text and elem.text.strip():
        result[f"{prefix}value"] = elem.text.strip()
    for child in elem:
        tag = _strip_ns(child.tag)
        child_dict = elem_to_dict(child, prefix=f"{tag}_")
        result.update(child_dict)
    return result


def collect_elements(root: ET.Element, element_type: str) -> list[dict]:
    """Collect all elements of a given type into list of dicts."""
    rows = []
    for elem in root.iter():
        if _strip_ns(elem.tag).lower() == element_type.lower():
            rows.append(elem_to_dict(elem))
    return rows


def parse_edm():
    if not EDM_FILE.exists():
        log.error(f"EDM file not found: {EDM_FILE}")
        return

    log.info(f"Parsing EDM file: {EDM_FILE}")
    try:
        tree = ET.parse(EDM_FILE)
        root = tree.getroot()
    except ET.ParseError as e:
        log.error(f"XML parse error: {e}")
        return

    # Survey the top-level structure first
    tag_counts: dict[str, int] = {}
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    log.info("Top element types in EDM.XML:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:30]:
        log.info(f"  {tag}: {count}")

    # Save element inventory
    inv_df = pd.DataFrame(
        sorted(tag_counts.items(), key=lambda x: -x[1]),
        columns=["element_type", "count"]
    )
    inv_df.to_csv(OUT_DIR / "_edm_element_types.csv", index=False)

    # ── Extract key entities ──────────────────────────────────────────────────
    ENTITIES = [
        "CD_WELL",             # Well master data
        "CD_WELLBORE",         # Wellbore data
        "CD_ASSEMBLY",         # BHA assemblies
        "CD_ASSEMBLY_COMP",    # BHA component details
        "CD_HOLE_SECT",        # Hole sections (casing seats / section boundaries)
        "CD_HOLE_SECT_GROUP",  # Hole section groups
        "CD_WELLBORE_FORMATION",  # Formation tops
        "CD_BHA_COMP_MWD",    # MWD BHA components
        "CD_BHA_COMP_STAB",   # Stabilizer components
        "CD_BHA_COMP_NOZZLE", # Nozzle components
        "CD_BHA_COMP_DP_HW",  # Drill pipe / heavy weight
        "CD_SURVEY_STATION",  # Survey stations
        "CD_DEFINITIVE_SURVEY_STATION",  # Definitive survey stations
        "CD_PORE_PRESSURE",   # Pore pressure data
        "CD_FRAC_GRADIENT",   # Fracture gradient data
        "CD_CASE",            # Casing design cases
        "WP_TDA_DRAGCHART",   # Torque & drag charts
    ]

    for entity in ENTITIES:
        rows = collect_elements(root, entity)
        if rows:
            df = pd.DataFrame(rows)
            out_path = OUT_DIR / f"edm_{entity}.csv"
            df.to_csv(out_path, index=False)
            log.info(f"  Saved {entity}: {len(df)} rows → {out_path.name}")
        else:
            log.info(f"  {entity}: no rows found")


if __name__ == "__main__":
    parse_edm()
