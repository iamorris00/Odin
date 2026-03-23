"""
run_pipeline.py
---------------
Master runner for Phase 1 data extraction pipeline.
Runs in sequence:
  1. parse_witsml_logs   → data/processed/witsml/
  2. parse_ddr_xml       → data/processed/ddr/
  3. parse_edm           → data/processed/edm/
  4. well_registry       → data/processed/well_registry.csv

Run from project root: python src/data_pipeline/run_pipeline.py
"""
import sys
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).resolve().parents[2] / "data" / "processed" / "pipeline.log",
                            mode="w"),
    ]
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def step1_witsml():
    log.info("=" * 60)
    log.info("STEP 1: Parsing WITSML realtime logs")
    log.info("=" * 60)
    try:
        from parse_witsml_logs import parse_all_wells
        parse_all_wells()
        log.info("Step 1 COMPLETE")
    except Exception as e:
        log.error(f"Step 1 FAILED: {e}", exc_info=True)


def step2_ddr():
    log.info("=" * 60)
    log.info("STEP 2: Parsing Daily Drilling Reports (DDR)")
    log.info("=" * 60)
    try:
        from parse_ddr_xml import parse_all_ddrs
        parse_all_ddrs()
        log.info("Step 2 COMPLETE")
    except Exception as e:
        log.error(f"Step 2 FAILED: {e}", exc_info=True)


def step3_edm():
    log.info("=" * 60)
    log.info("STEP 3: Parsing EDM.XML (BHA/casing/well metadata)")
    log.info("=" * 60)
    try:
        from parse_edm import parse_edm
        parse_edm()
        log.info("Step 3 COMPLETE")
    except Exception as e:
        log.error(f"Step 3 FAILED: {e}", exc_info=True)


def step4_well_registry():
    log.info("=" * 60)
    log.info("STEP 4: Building well metadata registry")
    log.info("=" * 60)
    try:
        processed = BASE_DIR / "data" / "processed"
        rows = []

        # From WITSML summary
        witsml_summary = processed / "witsml" / "_witsml_extraction_summary.csv"
        if witsml_summary.exists():
            df_w = pd.read_csv(witsml_summary)
            for _, r in df_w.iterrows():
                rows.append({
                    "source":           "WITSML",
                    "well_name":        r.get("well_name", ""),
                    "well_folder":      r.get("well_folder", ""),
                    "n_depth_sections": r.get("n_depth_sections", 0),
                    "n_time_sections":  r.get("n_time_sections", 0),
                })

        # From DDR summary
        ddr_summary = processed / "ddr" / "_ddr_extraction_summary.csv"
        if ddr_summary.exists():
            df_d = pd.read_csv(ddr_summary)
            for _, r in df_d.iterrows():
                rows.append({
                    "source":           "DDR",
                    "well_name":        r.get("well_key", ""),
                    "n_daily_reports":  r.get("n_daily_reports", 0),
                    "n_activities":     r.get("n_activities", 0),
                })

        if rows:
            df_reg = pd.DataFrame(rows)
            out = processed / "well_registry.csv"
            df_reg.to_csv(out, index=False)
            log.info(f"Well registry saved: {out} ({len(df_reg)} records)")
            print(df_reg.to_string(index=False))
        else:
            log.warning("No data available for well registry")

        log.info("Step 4 COMPLETE")
    except Exception as e:
        log.error(f"Step 4 FAILED: {e}", exc_info=True)


if __name__ == "__main__":
    log.info("VOLVE FIELD ML CHALLENGE — PHASE 1 DATA PIPELINE")
    step1_witsml()
    step2_ddr()
    step3_edm()
    step4_well_registry()
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info("Outputs:")
    log.info("  data/processed/witsml/    — WITSML drilling parameter CSVs")
    log.info("  data/processed/ddr/       — DDR activity & daily summary CSVs")
    log.info("  data/processed/edm/       — EDM BHA/casing config CSVs")
    log.info("  data/processed/well_registry.csv — unified well catalog")
