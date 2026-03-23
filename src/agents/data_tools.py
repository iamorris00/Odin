"""
data_tools.py
-------------
Schema-aware, purpose-built tools for querying the Volve structured data.

These tools know the exact schema of each data source and use fuzzy matching
to handle typos or inconsistent well name formatting from users.

Available Tools:
    1. DataInventoryTool   - Lists all 23 wells and available data sources.
    2. DDRQueryTool        - Queries DDR activity logs for a named well with NPT focus.
    3. WITSMLAnalystTool   - Computes drilling stats (ROP/TQA/SPP/WOB) from WITSML CSVs.
    4. CrossWellCompareTool - Compares key statistics across two wells side by side.
    5. EDMTechnicalTool     - Queries Technical data (BHA, Casing, Formations) from EDM.
    6. PythonTool           - Allows the analyst to perform custom Pandas/Matplotlib analysis.
"""

import subprocess
import sys

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend - must be set before pyplot import
import matplotlib.pyplot as plt
from pathlib import Path
from crewai.tools import BaseTool
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DDR_DIR  = Path(os.environ.get("DDR_DIR",  str(BASE_DIR / "data" / "processed" / "ddr")))
WITSML_DIR = Path(os.environ.get("WITSML_DIR", str(BASE_DIR / "data" / "processed" / "witsml")))
EDM_DIR  = BASE_DIR / "data" / "processed" / "edm"
OUTPUTS_DIR = BASE_DIR / "outputs" / "figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_well(name: str) -> str:
    """
    Normalize a user-supplied well name to a canonical slug used in filenames.
    e.g. 'NO 15/9-19 A', '15/9-19A', '15-9-19a', '15 9 19 a' → '15_9_19_A'
    e.g. '15/9-F-1 C', '15/9 F 1C'                            → '15_9_F_1_C'
    """
    s = name.strip().upper()
    # Strip the 'NO ' prefix if present
    s = re.sub(r'^NO\s+', '', s)
    # Replace all separators (/, -, space) with single underscore
    s = re.sub(r'[\s/\-]+', '_', s)
    # Collapse multiple underscores
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s


def _fuzzy_find_well_file(desired: str, suffix: str = "_activities.csv") -> Path | None:
    """
    Find the best-matching file in DDR_DIR for a given well name.
    Uses normalised string similarity: exact match first, then longest common subsequence.
    """
    target_slug = _normalize_well(desired)
    
    candidates = list(DDR_DIR.glob(f"*{suffix}"))
    
    # Step 1: Try exact slug match
    for c in candidates:
        stem_slug = _normalize_well(c.stem.replace(suffix.replace('.csv',''), ''))
        if c.stem.upper() == (target_slug + suffix.replace('.csv', '')).upper():
            return c

    # Step 2: Exact slug prefix match (file stem starts with the target slug)
    for c in candidates:
        if c.stem.upper().startswith(target_slug.upper()):
            return c

    # Step 3: Target slug is contained in filename slug
    for c in candidates:
        if target_slug.upper() in c.stem.upper():
            return c

    # Step 4: Fuzzy token overlap - find file with most shared tokens
    target_tokens = set(target_slug.split('_'))
    best_score = 0
    best_match = None
    for c in candidates:
        file_tokens = set(re.sub(r'_+', '_', c.stem.upper()).split('_'))
        score = len(target_tokens & file_tokens)
        if score > best_score:
            best_score = score
            best_match = c

    return best_match if best_score >= 2 else None


_PHASE_MAP = {
    # (keyword in activity_code) → phase label
    "drilling -- drill": "Rotary/Sliding Drilling",
    "drilling -- trip":  "Tripping (POOH/TIH)",
    "drilling -- wiper": "Wiper Trip",
    "drilling -- circulate": "Circulation/Conditioning",
    "drilling -- circ":  "Circulation/Conditioning",
    "drilling -- condition": "Circulation/Conditioning",
    "casing":            "Casing/Liner Running",
    "liner":             "Casing/Liner Running",
    "cement":            "Cementing",
    "logging":           "Logging/Survey",
    "wireline":          "Logging/Survey",
    "sidetrack":         "Sidetrack/Remedial",
    "whipstock":         "Sidetrack/Remedial",
    "milling":           "Sidetrack/Remedial",
    "fishing":           "NPT – Fishing",
    "stuck":             "NPT – Stuck Pipe",
    "repair":            "NPT – Equipment Repair",
    "wait":              "NPT – Waiting/Weather",
    "weather":           "NPT – Waiting/Weather",
    "npt":               "NPT – General",
    "bha":               "BHA Change/Rig-Up",
    "bit change":        "BHA Change/Rig-Up",
    "washout":           "NPT – Washout/Losses",
    "loss":              "NPT – Washout/Losses",
    "lcm":               "NPT – Washout/Losses",
    "trip":              "Tripping (POOH/TIH)",  # catch-all trip at end
    "drill":             "Rotary/Sliding Drilling",  # catch-all drill at end
}


def _classify_phase(activity_code: str) -> str:
    """Map an activity_code string to a drilling phase label."""
    if not isinstance(activity_code, str):
        return "Other"
    ac = activity_code.lower().strip()
    for keyword, phase in _PHASE_MAP.items():
        if keyword in ac:
            return phase
    return "Other"


def _list_all_wells() -> list[str]:
    """Return sorted list of canonical well names from DDR file stems."""
    wells = []
    for f in DDR_DIR.glob("*_activities.csv"):
        if f.stem.startswith('_'):
            continue  # skip aggregate files
        # Convert slug back to readable form e.g. 15_9_19_A → 15/9-19 A
        stem = f.stem.replace('_activities', '')
        # Only the last letter token is a well variant (A, B, C …)
        readable = stem.replace('_', '/')
        wells.append(readable)
    return sorted(wells)


# ── Tool 1: Data Inventory ─────────────────────────────────────────────────────

class DataInventoryTool(BaseTool):
    name: str = "data_inventory_inspector"
    description: str = (
        "Use this tool FIRST when the user asks what wells or datasets are available, "
        "or before any data query to confirm a well name exists. "
        "Returns a structured inventory of all 23 Volve wells and the types of data "
        "available (DDR activities, WITSML sensor logs, EDM metadata)."
    )

    def _run(self, query: str = "") -> str:
        lines = ["## 📋 Volve Field – Available Data Inventory\n"]

        # DDR wells
        wells = _list_all_wells()
        lines.append(f"### Daily Drilling Reports (DDR) — {len(wells)} Wells")
        lines.append("Each well has: `_activities.csv` (activity time-log) and `_daily_summary.csv` (per-day totals).")
        lines.append("**Available Wells:**")
        for w in wells:
            lines.append(f"  - `{w}`")

        # Global aggregate files
        if (DDR_DIR / "_ddr_all_activities.csv").exists():
            lines.append("\n**Global Aggregate File:** `_ddr_all_activities.csv` — all 23 wells merged (~32,000 rows)")
        if (DDR_DIR / "_ddr_extraction_summary.csv").exists():
            lines.append("**Summary File:** `_ddr_extraction_summary.csv` — one row per well with spud/completion dates")

        # WITSML
        witsml_files = list(WITSML_DIR.glob("*.csv"))
        lines.append(f"\n### WITSML Sensor Logs — {len(witsml_files)} CSV files")
        lines.append("Fields include: `ROP`, `RPM`, `WOB`, `SPPA` (standpipe pressure), `HKLD` (hookload), `TQA` (torque), depth, and more.")
        witsml_wells = sorted(set(f.name.split('__')[0] for f in witsml_files if '__' in f.name))
        lines.append(f"Wells with WITSML data: {', '.join(witsml_wells)}")

        # EDM
        edm_files = list(EDM_DIR.glob("*.csv")) if EDM_DIR.exists() else []
        lines.append(f"\n### EDM (Engineering Data Model) — {len(edm_files)} tables")
        lines.append("Includes: wellbore geometry, BHA components, survey stations, pore pressure, casing data.")

        lines.append("\n---")
        lines.append("💡 **Tip:** Use `DDR_Query` with a well name to get activity logs, or `WITSML_Analyst` for sensor-level stats.")
        return "\n".join(lines)


# ── Tool 2: DDR Activity Query ─────────────────────────────────────────────────

class DDRQueryTool(BaseTool):
    name: str = "DDR_Query"
    description: str = (
        "Query the Daily Drilling Report (DDR) activity log for a specific well. "
        "Accepts any well name variant (e.g. '15/9-19 A', '15/9-F-1C', '15-9-F-1 C', typos OK). "
        "Returns: a Markdown table of activities with dates, depths, activity codes, duration, and comments. "
        "Also returns NPT (Non-Productive Time) summary and total drilled depth. "
        "Input: well name as a string."
    )

    def _run(self, well_name: str) -> str:
        path = _fuzzy_find_well_file(well_name, "_activities.csv")
        if path is None:
            available = ", ".join(_list_all_wells()[:10]) + "..."
            return (f"❌ Could not find DDR data for well `{well_name}`. "
                    f"Try: {available}")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            return f"Error reading file {path}: {e}"

        total_rows = len(df)
        matched_well = path.stem.replace('_activities', '')

        # ── Basic stats ──
        lines = [f"## DDR Activity Report — Well: `{matched_well}` (matched from `{well_name}`)\n"]
        lines.append(f"**Total activity records:** {total_rows}")

        # Duration totals
        if 'duration_hours' in df.columns:
            df['duration_hours'] = pd.to_numeric(df['duration_hours'], errors='coerce')
            total_h = df['duration_hours'].sum()
            lines.append(f"**Total logged time:** {total_h:.1f} hours ({total_h/24:.1f} days)")

        # Depth range
        if 'md_m' in df.columns:
            df['md_m'] = pd.to_numeric(df['md_m'], errors='coerce')
            lines.append(f"**Depth range:** {df['md_m'].min():.0f} m — {df['md_m'].max():.0f} m MD")

        # ── Activity code breakdown ──
        if 'activity_code' in df.columns:
            act_counts = df.groupby('activity_code')['duration_hours'].sum().sort_values(ascending=False).head(12)
            lines.append("\n### Top Activities by Time (hours)\n")
            lines.append(act_counts.reset_index().rename(columns={'activity_code': 'Activity', 'duration_hours': 'Hours'}).to_markdown(index=False, floatfmt=".1f"))

        # ── Drilling Phase Breakdown ──
        if 'activity_code' in df.columns and 'duration_hours' in df.columns:
            df['_phase'] = df['activity_code'].apply(_classify_phase)
            phase_totals = (
                df.groupby('_phase')['duration_hours']
                .sum()
                .sort_values(ascending=False)
            )
            total_phase_h = phase_totals.sum()
            if total_phase_h > 0:
                phase_df = phase_totals.reset_index()
                phase_df.columns = ['Phase', 'Hours']
                phase_df['%'] = (phase_df['Hours'] / total_phase_h * 100).round(1)
                lines.append("\n### 🔄 Drilling Phase Distribution\n")
                lines.append(phase_df.to_markdown(index=False, floatfmt=".1f"))

        # ── NPT summary ──
        if 'activity_code' in df.columns:
            # Broadened NPT keywords for stricter classification
            npt_keywords = ['npt', 'fishing', 'stuck', 'repair', 'wait', 'sidetrack', 'washout', 'twist off', 'leak', 'loss', 'plug']
            npt_mask = df['activity_code'].str.lower().str.contains('|'.join(npt_keywords), na=False)
            
            # Also catch comments mentioning problems
            problem_keywords = ['problem', 'failure', 'broken', 'damage', 'stuck', 'overpull', 'tight']
            # Safeguard if comments column is missing or all null
            if 'comments' in df.columns:
                comment_mask = df['comments'].str.lower().str.contains('|'.join(problem_keywords), na=False)
            else:
                comment_mask = False
            
            combined_npt_mask = npt_mask | (comment_mask if isinstance(comment_mask, pd.Series) else False)
            npt_df = df[combined_npt_mask]
            
            if not npt_df.empty:
                npt_total = npt_df['duration_hours'].sum() if 'duration_hours' in npt_df.columns else len(npt_df)
                lines.append(f"\n### ⚠️ NPT & Operational Events Summary")
                lines.append(f"**Total NPT/Event hours:** {npt_total:.1f} h ({npt_total/total_h*100:.1f}% of total logged time)")
                lines.append(npt_df[['act_start', 'activity_code', 'state_detail', 'duration_hours', 'comments']].head(20).fillna('').to_markdown(index=False))

        # ── Depth Samples ──
        if 'md_m' in df.columns:
             lines.append("\n### 📏 Depth Progression Sample")
             lines.append(df[['act_start', 'md_m', 'activity_code', 'comments']].dropna(subset=['md_m']).tail(10).to_markdown(index=False))

        # ── Recent activities sample ──
        cols = [c for c in ['act_start', 'md_m', 'activity_code', 'state', 'duration_hours', 'comments'] if c in df.columns]
        lines.append(f"\n### Recent Activity Sample (last 10 records)\n")
        lines.append(df[cols].tail(10).fillna('').to_markdown(index=False))

        result = "\n".join(lines)
        if len(result) > 14000:
            return result[:14000] + "\n\n...[TRUNCATED — use more specific queries for details]"
        return result


# ── Tool 3: WITSML Sensor Analyst ─────────────────────────────────────────────

class WITSMLAnalystTool(BaseTool):
    name: str = "WITSML_Analyst"
    description: str = (
        "Compute drilling performance statistics from WITSML sensor logs for a specific well. "
        "Accepts any well name variant (typos OK). "
        "Returns: average/max/min ROP (rate of penetration), WOB (weight on bit), RPM, torque, "
        "standpipe pressure, hookload, and available depth range. "
        "Can also save a time-series plot of ROP vs depth if 'plot=true' is in the input. "
        "Input: well name (optionally append ' plot=true' to generate a chart)."
    )

    def _run(self, query: str) -> str:
        # Parse plot flag and filters
        plot = 'plot=true' in query.lower()
        query = query.lower().replace('plot=true', '').strip()
        
        # Extract depth=X-Y
        depth_range = None
        depth_match = re.search(r'depth=([\d\.]+)-([\d\.]+)', query)
        if depth_match:
            depth_range = (float(depth_match.group(1)), float(depth_match.group(2)))
            query = query.replace(depth_match.group(0), '')
            
        # Extract section=X
        section_filter = None
        sec_match = re.search(r'section=([\d\.]+)', query)
        if sec_match:
            section_filter = sec_match.group(1)
            query = query.replace(sec_match.group(0), '')
            
        well_name = query.replace(',', '').strip()
        
        well_slug = _normalize_well(well_name)
        
        # Find all WITSML files for this well
        all_files = list(WITSML_DIR.glob("*.csv"))
        matching = [f for f in all_files if f.name.upper().startswith(well_slug.upper() + '__')]
        
        if not matching:
            # Fuzzy: find files containing max token overlap with the slug
            tokens = set(well_slug.split('_'))
            scored = []
            for f in all_files:
                file_tokens = set(re.sub(r'_+', '_', f.name.upper()).split('_'))
                score = len(tokens & file_tokens)
                scored.append((score, f))
            scored.sort(reverse=True)
            if scored and scored[0][0] >= 2:
                top_score = scored[0][0]
                matching = [f for s, f in scored if s == top_score]
            
        if not matching:
            return (f"❌ No WITSML data found for well `{well_name}` (slug: `{well_slug}`). "
                    f"Use the data_inventory_inspector tool to see what wells have WITSML data.")

        # Prefer Depth-log files (more useful for drilling analysis)
        depth_files = [f for f in matching if 'DEPTH' in f.name.upper() and 'MD_LOG' in f.name.upper()]
        target_files = depth_files if depth_files else matching

        # If section filter is specified, only use files matching that section
        if section_filter:
            sec_files = [f for f in target_files if section_filter in f.name]
            if sec_files:
                target_files = sec_files
            else:
                return f"❌ No WITSML logs found for section {section_filter} in well {well_name}."

        # Load and concatenate all matching files
        dfs = []
        for f in target_files:
            try:
                dfs.append(pd.read_csv(f, low_memory=False))
            except Exception:
                pass

        if not dfs:
            return f"Found {len(matching)} WITSML file(s) but could not read any of them."

        df = pd.concat(dfs, ignore_index=True)
        matched_well = matching[0].name.split('__')[0]

        # ── Column mapping: handle alternate column names ──
        COL_MAP = {
            'ROP':    ['ROP', 'GS_ROP', 'ROP5', 'ROPIH', 'ROPH'],
            'WOB':    ['CWOB', 'WOB'],
            'RPM':    ['RPM', 'GS_RPM', 'DRPM', 'TRPM_RT'],
            'TORQUE': ['TQA', 'GS_TQA'],
            'SPP':    ['SPPA', 'GS_SPPA'],
            'HOOKLD': ['HKLD', 'GS_HKLD', 'HKLO', 'HKLI'],
            'DEPTH':  ['DMEA', 'DEPTH', 'DEPT', 'TVDE'],
        }

        found_cols = {}
        for key, alts in COL_MAP.items():
            for alt in alts:
                if alt in df.columns:
                    if pd.to_numeric(df[alt], errors='coerce').notnull().any():
                        found_cols[key] = alt
                        break

        # ── Depth sanity: sentinel removal, feet→meters, cap at 5500m ──
        _NULLS = {-999.25, -999.0, -9999.0, 9999.0, 9999.25}
        if 'DEPTH' in found_cols:
            d_col = found_cols['DEPTH']
            df[d_col] = pd.to_numeric(df[d_col], errors='coerce')
            df[d_col] = df[d_col].where(~df[d_col].isin(_NULLS))
            med = df[d_col].median()
            if pd.notna(med) and med > 5000:   # likely feet → convert
                df[d_col] = df[d_col] * 0.3048
            df[d_col] = df[d_col].clip(upper=5500)

        # Filter by depth if specified and available
        if depth_range and 'DEPTH' in found_cols:
            d_col = found_cols['DEPTH']
            df_filtered = df[(df[d_col] >= depth_range[0]) & (df[d_col] <= depth_range[1])]
            if not df_filtered.empty:
                df = df_filtered

        title_suffix = ""
        if section_filter: title_suffix += f" | Section: {section_filter}\""
        if depth_range: title_suffix += f" | Depth: {depth_range[0]}-{depth_range[1]}m"

        lines = [f"## WITSML Sensor Analysis — Well: `{matched_well}`{title_suffix}\n"]
        lines.append(f"**Source files:** {len(target_files)} | **Total rows:** {len(df):,}")

        lines.append(f"\n**Mapped columns:** {found_cols}\n")

        stats_rows = []
        for param, col in found_cols.items():
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            
            # Robust filtering for ROP (Rate of Penetration)
            if param == 'ROP':
                # Only include data where ROP is physically plausible (e.g., 0.1 to 300 m/hr)
                # This excludes noise and non-drilling time (zeros)
                s = s[(s > 0.5) & (s < 500)]
            
            if len(s) == 0:
                continue
            
            stats_rows.append({
                'Parameter': param,
                'Column': col,
                'Mean': round(s.mean(), 2),
                'Median': round(s.median(), 2),
                'Max': round(s.max(), 2),
                'Min': round(s.min(), 2),
                'StdDev': round(s.std(), 2),
                'N': len(s)
            })

        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            lines.append("### Drilling Performance Statistics\n")
            lines.append(stats_df.to_markdown(index=False))

        # ── Depth summary ──
        if 'DEPTH' in found_cols:
            depth_col = found_cols['DEPTH']
            depth_s = pd.to_numeric(df[depth_col], errors='coerce').dropna()
            if len(depth_s) > 0:
                lines.append(f"\n**Total drilled depth range:** {depth_s.min():.0f} m — {depth_s.max():.0f} m MD")
                lines.append(f"**Net drilled footage:** {depth_s.max() - depth_s.min():.0f} m")

        # ── Optional: generate ROP vs Depth plot ──
        if plot and 'ROP' in found_cols and 'DEPTH' in found_cols:
            try:
                rop_col = found_cols['ROP']
                dep_col = found_cols['DEPTH']
                plot_df = df[[dep_col, rop_col]].copy()
                plot_df[rop_col] = pd.to_numeric(plot_df[rop_col], errors='coerce')
                plot_df[dep_col] = pd.to_numeric(plot_df[dep_col], errors='coerce')
                plot_df = plot_df.dropna()
                plot_df = plot_df[plot_df[rop_col] > 0]  # Only while drilling

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(plot_df[rop_col], plot_df[dep_col], alpha=0.3, s=5, color='steelblue')
                ax.invert_yaxis()
                ax.set_xlabel('ROP (m/hr)')
                ax.set_ylabel('Depth (m MD)')
                ax.set_title(f'ROP vs Depth — {matched_well}')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                out_path = OUTPUTS_DIR / f"{well_slug}_rop_profile.png"
                plt.savefig(out_path, dpi=100)
                plt.close()
                lines.append(f"\n📊 **Chart saved:** `{out_path}`")
            except Exception as e:
                lines.append(f"\n⚠️ Could not generate chart: {e}")

        return "\n".join(lines)


# ── Tool 4: Cross-Well Comparison ─────────────────────────────────────────────

class CrossWellCompareTool(BaseTool):
    name: str = "CrossWell_Comparison"
    description: str = (
        "Compare DDR activity statistics AND WITSML drilling performance between multiple wells side by side. "
        "Generates a comparison bar chart saved to outputs/figures/. "
        "Input: well names separated by ' vs ' or ' and ', e.g. 'Well A vs Well B vs Well C'."
        "Accepts typos and different name formats."
    )

    def _run(self, query: str) -> str:
        # Parse multiple well names (separated by vs, and, or commas)
        well_names = []
        # Normalise separators to ' vs ' then split
        norm_query = re.sub(r'(\s+and\s+|,)', ' vs ', query, flags=re.IGNORECASE)
        parts = [p.strip() for p in re.split(r'\s+vs\.?\s+', norm_query, flags=re.IGNORECASE) if p.strip()]
        
        if len(parts) < 2:
            return "❌ Please provide at least two well names, e.g. '15/9-19 A vs 15/9-19 B vs 15/9-F-1 C'"

        results = []
        for wname in parts:
            slug = _normalize_well(wname)
            wresult = {
                'user_name': wname, 
                'slug': slug, 
                'matched_name': wname,
                'total_hours': 0,
                'max_depth_m': 0,
                'npt_hours': 0,
                'avg_rop': 0,
                'bha_summary': 'N/A'
            }

            # DDR stats
            ddr_path = _fuzzy_find_well_file(wname, "_activities.csv")
            if ddr_path:
                try:
                    df = pd.read_csv(ddr_path)
                    df['duration_hours'] = pd.to_numeric(df.get('duration_hours', pd.Series()), errors='coerce')
                    df['md_m'] = pd.to_numeric(df.get('md_m', pd.Series()), errors='coerce')
                    wresult['total_hours'] = df['duration_hours'].sum()
                    wresult['max_depth_m'] = df['md_m'].max()
                    wresult['matched_name'] = ddr_path.name.replace('_activities.csv', '').replace('_', '/')

                    # NPT
                    npt_kw = ['npt', 'fishing', 'stuck', 'repair', 'wait', 'sidetrack', 'washout']
                    if 'activity_code' in df.columns:
                        npt_mask = df['activity_code'].str.lower().str.contains('|'.join(npt_kw), na=False)
                        wresult['npt_hours'] = df.loc[npt_mask, 'duration_hours'].sum()
                except Exception as e:
                    wresult['ddr_error'] = str(e)

            # WITSML ROP
            witsml_files = list(WITSML_DIR.glob(f"{slug}__*MD_Log*.csv"))
            if not witsml_files:
                witsml_files = list(WITSML_DIR.glob(f"{slug}__*.csv"))
            if witsml_files:
                try:
                    dfs = []
                    for f in witsml_files[:5]:  # limit files loaded
                        dfs.append(pd.read_csv(f, low_memory=False))
                    wdf = pd.concat(dfs, ignore_index=True)
                    for rop_col in ['ROP', 'GS_ROP', 'ROP5', 'ROPIH']:
                        if rop_col in wdf.columns:
                            s = pd.to_numeric(wdf[rop_col], errors='coerce').dropna()
                            s = s[s > 0]
                            if len(s) > 0:
                                wresult['avg_rop'] = round(s.mean(), 2)
                                break
                except Exception as e:
                    wresult['witsml_error'] = str(e)

            # --- Attempt to pull basic BHA info from EDM ---
            try:
                well_f = EDM_DIR / "edm_CD_WELL.csv"
                comp_f = EDM_DIR / "edm_CD_ASSEMBLY_COMP.csv"
                if well_f.exists() and comp_f.exists():
                    df_well = pd.read_csv(well_f)
                    df_comp = pd.read_csv(comp_f, low_memory=False)
                    # Find well id using startswith for flexibility
                    if 'well_common_name' in df_well.columns:
                        df_well['slug'] = df_well['well_common_name'].apply(lambda x: _normalize_well(str(x)))
                    else:
                        df_well['slug'] = df_well['well_legal_name'].apply(lambda x: _normalize_well(str(x)))

                    match_mask = df_well['slug'].apply(
                        lambda x: isinstance(x, str) and (x in slug or slug in x)
                    )
                    if match_mask.any():
                        # Use shortest valid match
                        matches = df_well[match_mask].copy()
                        matches['slug_len'] = matches['slug'].apply(len)
                        w_id = matches.sort_values('slug_len')['well_id'].iloc[0]
                        # Find assemblies for this well
                        w_comps = df_comp[df_comp['well_id'] == w_id]
                        if not w_comps.empty:
                            bits_df = w_comps[w_comps['comp_type_code'].str.upper() == 'BIT']
                            motors_df = w_comps[w_comps['comp_type_code'].str.upper() == 'STM']
                            
                            def _format_comp(cdf):
                                items = []
                                for _, row in cdf.iterrows():
                                    name = str(row.get('comp_name', '')).strip()
                                    od = str(row.get('outer_diameter', '')).strip()
                                    
                                    if name and name.lower() != 'nan':
                                        items.append(name)
                                    elif od and od.lower() != 'nan':
                                        items.append(f"{od}\" OD")
                                    else:
                                        items.append("Present")
                                return list(set(items))

                            bits = _format_comp(bits_df)
                            motors = _format_comp(motors_df)
                            
                            summary_parts = []
                            if len(bits) > 0:
                                summary_parts.append(f"Bits: {', '.join(bits[:2])}")
                            if len(motors) > 0:
                                summary_parts.append(f"Motors: {', '.join(motors[:2])}")
                            
                            if summary_parts:
                                wresult['bha_summary'] = ' | '.join(summary_parts)
            except Exception as e:
                pass # Non-fatal if BHA can't be found
                
            results.append(wresult)

        # ── Format text comparison ──
        lines = [f"## ⚔️ Multi-Well Comparison\n"]
        metric_rows = []
        for wr in results:
            row = {
                'Well': wr['matched_name'],
                'Max Depth (m)': f"{wr.get('max_depth_m', 0):.0f}" if wr['max_depth_m'] > 0 else 'N/A',
                'Total Hours': f"{wr.get('total_hours', 0):.1f}",
                'NPT Hours': f"{wr.get('npt_hours', 0):.1f}",
                'Avg ROP (m/hr)': f"{wr.get('avg_rop', 0):.2f}" if wr['avg_rop'] > 0 else 'N/A',
                'BHA Focus': wr.get('bha_summary', 'N/A')
            }
            metric_rows.append(row)

        lines.append(pd.DataFrame(metric_rows).to_markdown(index=False))

        # ── Generate chart ──
        try:
            labels = [wr['matched_name'] for wr in results]
            depths = [wr.get('max_depth_m', 0) for wr in results]
            avg_rops = [wr.get('avg_rop', 0) for wr in results]
            npt_hours = [wr.get('npt_hours', 0) for wr in results]

            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            fig.suptitle(f"Drilling Performance Comparison", fontsize=14, fontweight='bold')
            
            # Dynamic colors
            cmap = plt.cm.get_cmap('viridis', len(labels))
            colors = [cmap(i) for i in range(len(labels))]

            axes[0].bar(labels, depths, color=colors)
            axes[0].set_title('Max Depth (m)')
            axes[0].tick_params(axis='x', rotation=45)

            axes[1].bar(labels, avg_rops, color=colors)
            axes[1].set_title('Avg ROP (m/hr)')
            axes[1].tick_params(axis='x', rotation=45)

            axes[2].bar(labels, npt_hours, color=colors)
            axes[2].set_title('Total NPT Hours')
            axes[2].tick_params(axis='x', rotation=45)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            chart_path = OUTPUTS_DIR / "comparison.png"
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            lines.append(f"\n📊 **Comparison chart saved:** `{chart_path}`")
        except Exception as e:
            lines.append(f"\n⚠️ Could not generate chart: {e}")

        return "\n".join(lines)


# ── Tool 5: Python Interpreter ────────────────────────────────────────────────

class EDMTechnicalTool(BaseTool):
    name: str = "EDM_Technical_Query"
    description: str = (
        "Queries technical data for a well: Formation Tops, Casing strings, and BHA (Assembly). "
        "Use this for 'complete' well comparisons or when asked about specific depths/geology."
    )

    def _run(self, well_name: str) -> str:
        slug = _normalize_well(well_name)
        
        # 1. Formation Tops
        formation_f = EDM_DIR / "edm_CD_WELLBORE_FORMATION.csv"
        well_f = EDM_DIR / "edm_CD_WELL.csv"
        
        output = [f"## Technical Specification: `{well_name}`"]
        
        try:
            if well_f.exists():
                df_well = pd.read_csv(well_f)
                # EDM well names are sometimes just 'F-1' or '15/9-19'
                # Check for slugs in well_common_name or well_legal_name
                df_well['slug'] = df_well['well_common_name'].apply(lambda x: _normalize_well(str(x)))
                
                # If query is '15/9-19 A', slug is '15_9_19_A'. But EDM might just have '15/9-19'.
                # So we check if the EDM slug is a prefix of the requested slug.
                # Sort by length descending so we match the most specific well base first.
                df_well['slug_len'] = df_well['slug'].str.len()
                df_well = df_well.sort_values('slug_len', ascending=False)
                
                row = pd.DataFrame()
                for _, r in df_well.iterrows():
                    if r['slug'] in slug or slug in r['slug']:
                        row = pd.DataFrame([r])
                        break
                
                if row.empty:
                    # Try partial match on common name as fallback
                    row = df_well[df_well['well_common_name'].str.contains(well_name.replace('_','/').split()[0], na=False)]

                if not row.empty:
                    w_id = row.iloc[0]['well_id']
                    
                    # Resolve wellbore if possible
                    wb_id = None
                    wb_f = EDM_DIR / "edm_CD_WELLBORE.csv"
                    if wb_f.exists():
                        df_wb = pd.read_csv(wb_f)
                        df_wb_w = df_wb[df_wb['well_id'] == w_id]
                        
                        # Try exact match on legal name first
                        wb_exact = df_wb_w[df_wb_w['well_legal_name'].str.contains(well_name, na=False, case=False)]
                        if not wb_exact.empty:
                            wb_id = wb_exact.iloc[0]['wellbore_id']
                        elif not df_wb_w.empty:
                            # Fallback to the first wellbore
                            wb_id = df_wb_w.iloc[0]['wellbore_id']

                    output.append(f"**Well ID:** {w_id} | **Wellbore ID:** {wb_id or 'N/A'} | **Water Depth:** {row.iloc[0].get('water_depth','N/A')} m")
                    
                    # BHA (Assembly) Data
                    assembly_f = EDM_DIR / "edm_CD_ASSEMBLY.csv"
                    assembly_comp_f = EDM_DIR / "edm_CD_ASSEMBLY_COMP.csv"
                    
                    if assembly_f.exists() and assembly_comp_f.exists():
                        df_assy = pd.read_csv(assembly_f, low_memory=False)
                        df_comp = pd.read_csv(assembly_comp_f, low_memory=False)
                        
                        # Find assemblies for this well and wellbore
                        w_assy = df_assy[df_assy['well_id'] == w_id]
                        if wb_id and 'wellbore_id' in df_assy.columns:
                            # Prioritize assembly linked to wellbore, but some might just be linked to well.
                            wb_assy = w_assy[w_assy['wellbore_id'] == wb_id]
                            if not wb_assy.empty:
                                w_assy = wb_assy
                        if not w_assy.empty:
                            output.append("\n### Bottom Hole Assemblies (BHA)")
                            bha_list = []
                            
                            # Critical components for drilling optimization
                            focus_comps = ['BIT', 'MWD', 'LWD', 'STM', 'IBS', 'NBS', 'DC', 'HW']
                            
                            # Merge and group assemblies
                            for _, assy in w_assy.iterrows():
                                a_id = assy['assembly_id']
                                a_name = assy.get('assembly_name', 'Unknown Assembly')
                                h_size = assy.get('hole_size', 'Unknown')
                                
                                comps = df_comp[df_comp['assembly_id'] == a_id]
                                if not comps.empty:
                                    # Filter to just the important drilling components
                                    focus_mask = comps['comp_type_code'].isin(focus_comps)
                                    focus_c = comps[focus_mask].sort_values(by='sequence_no', ascending=False) if 'sequence_no' in comps.columns else comps[focus_mask]
                                    
                                    if not focus_c.empty:
                                        # Summarize components
                                        comp_summary = []
                                        for _, c in focus_c.iterrows():
                                            c_type = c['comp_type_code']
                                            c_desc = str(c.get('description', '')).split(',')[0] # keep it short
                                            c_od = c.get('od_body', 'N/A')
                                            comp_summary.append(f"{c_type} ({c_od}\" OD): {c_desc}")
                                            
                                        bha_list.append({
                                            'Assembly Name': a_name,
                                            'Hole Size': h_size,
                                            'Key Components': ' | '.join(comp_summary)
                                        })
                            
                            if bha_list:
                                output.append(pd.DataFrame(bha_list).to_markdown(index=False))

                    # Casing
                    case_f = EDM_DIR / "edm_CD_CASE.csv"
                    if case_f.exists():
                        df_case = pd.read_csv(case_f)
                        w_case = df_case[df_case['well_id'] == w_id]
                        if wb_id and 'wellbore_id' in df_case.columns:
                            wb_case = w_case[w_case['wellbore_id'] == wb_id]
                            if not wb_case.empty:
                                w_case = wb_case
                                
                        if not w_case.empty:
                            output.append("\n### Casing / Liners")
                            # Filter to strings and get basic details
                            str_case = w_case[w_case['case_name'].str.contains("Casing|Liner", na=False, case=False)]
                            if str_case.empty:
                                str_case = w_case
                                
                            cols_to_show = [c for c in ['case_name', 'phase', 'job_pipe_size'] if c in str_case.columns]
                            if cols_to_show:
                                output.append(str_case[cols_to_show].head(10).to_markdown(index=False))

                    # Formations
                    formation_f = EDM_DIR / "edm_CD_WELLBORE_FORMATION.csv"
                    if formation_f.exists():
                        df_form = pd.read_csv(formation_f)
                        w_form = df_form[df_form['well_id'] == w_id]
                        if wb_id and 'wellbore_id' in df_form.columns:
                            wb_form = w_form[w_form['wellbore_id'] == wb_id]
                            if not wb_form.empty:
                                w_form = wb_form
                        
                        if not w_form.empty:
                            output.append("\n### Formation Tops")
                            # Sort by depth if available
                            sort_col = 'prognosed_md' if 'prognosed_md' in w_form.columns else w_form.columns[0]
                            w_form = w_form.sort_values(by=sort_col)
                            
                            cols_to_show = [c for c in ['formation_name', 'prognosed_md', 'prognosed_tvd'] if c in w_form.columns]
                            if cols_to_show:
                                output.append(w_form[cols_to_show].head(10).to_markdown(index=False))
            
            if len(output) <= 1:
                return f"No EDM records found for {well_name}."
            
            return "\n".join(output)
        except Exception as e:
            return f"Error querying EDM: {e}"

class PythonTool(BaseTool):
    name: str = "python_interpreter"
    description: str = (
        "Execute Python code (Pandas, Plotly, Numpy) for custom data analysis. "
        "Use for Days-vs-Depth charts, ROP correlations, NPT analysis, statistical filtering, or multi-signal plots.\n"
        "**MANDATORY RULES — violations cause FileNotFoundError or wrong charts:**\n"
        "1. NEVER construct file paths manually. NEVER use pd.read_csv('/data/...') or ANY hardcoded path.\n"
        "   The paths /data/ddr/, /data/processed/, etc. DO NOT EXIST. Use ONLY load_ddr() and load_witsml().\n"
        "2. Load DDR with: `df = load_ddr('15/9-F-12')` — columns: md_m (metres), activity_code, duration_hours, act_start, comments.\n"
        "3. days_vs_depth() is ONLY for explicit 'days vs depth' or 'drilling timeline' requests. "
        "NEVER call days_vs_depth() for NPT analysis, phase distribution, ROP charts, or any other chart type — "
        "use load_ddr() + load_witsml() directly for those. "
        "When called: `dvd = days_vs_depth('15/9-F-12')` — already cleaned, monotonic depth, correct time axis. "
        "Plot dvd['days_from_spud'] (x) vs dvd['max_depth_m'] (y, inverted). DO NOT use raw df['md_m'] for D-vs-D.\n"
        "4. Load WITSML with: `df, cols = load_witsml('15/9-F-12')` — depth is in metres (auto-converted). "
        "Always use `df[cols['ROP']]` not `df['ROP']`. Filter ROP > 0 to exclude non-drilling rows.\n"
        "5. Save charts with: `save_plotly_html(fig, 'chart_name')` — automatically saves HTML + PNG.\n"
        "6. Depth sanity: all depths are in metres MD, max ~3500m for F-12, ~5200m for deepest Volve well.\n"
        "Pre-injected: DDR_DIR, WITSML_DIR, EDM_DIR, px, go, load_ddr(), load_witsml(), days_vs_depth(), save_plotly_html().\n"
        "EDM tables: edm_CD_WELL.csv, edm_CD_HOLE_SECT.csv, edm_CD_ASSEMBLY_COMP.csv, edm_CD_WELLBORE_FORMATION.csv.\n"
        "Always print() results. Input: direct Python code string."
    )

    def _run(self, code: str) -> str:
        # Hard-code absolute paths at injection time — agent must NOT construct paths manually
        ddr_abs     = str(BASE_DIR / "data" / "processed" / "ddr")
        witsml_abs  = str(BASE_DIR / "data" / "processed" / "witsml")
        edm_abs     = str(BASE_DIR / "data" / "processed" / "edm")
        outputs_abs = str(BASE_DIR / "outputs" / "figures")

        full_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re as _re

# ── Pre-resolved absolute paths (DO NOT reconstruct these) ──
DDR_DIR    = Path(r"{ddr_abs}")
WITSML_DIR = Path(r"{witsml_abs}")
EDM_DIR    = Path(r"{edm_abs}")
OUTPUTS_DIR = Path(r"{outputs_abs}")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── WITSML column aliases (DMEA first = most reliable measured depth) ──
_WITSML_COL_MAP = {{
    'ROP':    ['ROP5', 'GS_ROP', 'ROPIH', 'ROPH', 'ROP'],
    'WOB':    ['CWOB', 'WOB'],
    'RPM':    ['RPM', 'GS_RPM', 'DRPM', 'TRPM_RT'],
    'TORQUE': ['TQA', 'GS_TQA'],
    'SPP':    ['SPPA', 'GS_SPPA'],
    'HOOKLD': ['HKLD', 'GS_HKLD', 'HKLO', 'HKLI'],
    'DEPTH':  ['DMEA', 'DEPTH', 'DEPT', 'TVDE'],  # DMEA first — DEPT can be a row index
}}

# Maximum realistic depth for any Volve well (deepest is ~5200m MD)
_VOLVE_MAX_DEPTH_M = 5500
# Sentinel / null values used in WITSML/LAS exports
_WITSML_NULLS = {{-999.25, -999.0, -9999.0, 9999.0, 9999.25, 99999.0}}

def _well_slug(name):
    s = _re.sub(r'[\\s/\\-]+', '_', name.strip().upper())
    s = _re.sub(r'^NO_', '', s)
    return _re.sub(r'_+', '_', s).strip('_')

def _clean_depth(series):
    \"\"\"Coerce to numeric, remove nulls/sentinels, convert ft→m if median > 5000.\"\"\"
    s = pd.to_numeric(series, errors='coerce')
    # Remove WITSML sentinel values
    s = s[~s.isin(_WITSML_NULLS) & s.notna() & (s >= 0)]
    if s.empty:
        return s
    # Auto-detect feet: if median depth exceeds 5000 it cannot be metres for Volve
    if s.median() > 5000:
        s = s / 3.28084   # convert ft → m
    # Cap at maximum realistic well depth
    s = s[s <= _VOLVE_MAX_DEPTH_M]
    return s

def load_ddr(well_name, drilling_only=False):
    \"\"\"
    Load DDR activities CSV for a well.
    Columns: act_start, md_m (meters MD), activity_code, state, duration_hours, comments.
    drilling_only=True keeps only depth-advancing activities (drill/slide) for D-vs-D charts.
    IMPORTANT: For Days vs Depth charts use cummax() on md_m — do NOT plot raw md_m directly
    (depth can repeat/jump during POOH). See days_vs_depth() helper below.
    \"\"\"
    slug = _well_slug(well_name)
    candidates = list(DDR_DIR.glob("*_activities.csv"))
    match = None
    for c in candidates:
        if c.stem.upper().startswith(slug):
            match = c; break
    if not match:
        for c in candidates:
            if slug in c.stem.upper():
                match = c; break
    if not match:
        toks = set(slug.split('_'))
        best, best_f = 0, None
        for c in candidates:
            s = len(toks & set(c.stem.upper().split('_')))
            if s > best: best, best_f = s, c
        if best >= 2: match = best_f
    if not match:
        print(f"ERROR: No DDR file found for '{{well_name}}' (slug: {{slug}})")
        print(f"Available: {{[c.name for c in candidates[:8]]}}")
        return pd.DataFrame()
    print(f"Loading DDR: {{match.name}}")
    df = pd.read_csv(match)
    if 'duration_hours' in df.columns:
        df['duration_hours'] = pd.to_numeric(df['duration_hours'], errors='coerce')
    if 'md_m' in df.columns:
        df['md_m'] = pd.to_numeric(df['md_m'], errors='coerce').clip(lower=0, upper=_VOLVE_MAX_DEPTH_M)
    if drilling_only and 'activity_code' in df.columns:
        mask = df['activity_code'].str.lower().str.contains('drill', na=False)
        df = df[mask]
    return df

def days_vs_depth(well_name):
    \"\"\"
    Build a clean Days-vs-Depth DataFrame for plotting.
    Returns df with columns: days_from_spud (float), max_depth_m (float), activity_code.
    Only includes the DRILLING CAMPAIGN (stops at max depth — no completion/workover extension).
    The max_depth_m column is monotonically non-decreasing (industry standard D-vs-D).
    \"\"\"
    df = load_ddr(well_name)
    if df.empty or 'act_start' not in df.columns:
        return pd.DataFrame()
    df = df[df['md_m'] > 0].copy()
    df['act_start'] = pd.to_datetime(df['act_start'], errors='coerce')
    df = df.dropna(subset=['act_start']).sort_values('act_start').reset_index(drop=True)
    t0 = df['act_start'].min()
    df['days_from_spud'] = (df['act_start'] - t0).dt.total_seconds() / 86400
    df['max_depth_m'] = df['md_m'].cummax()
    # Trim to drilling campaign: stop when depth stops increasing for >3 days
    td_idx = df['max_depth_m'].idxmax()
    post_td = df.loc[td_idx:, 'activity_code'].str.lower()
    # Find first completion/workover row after TD
    completion_mask = post_td.str.contains('complet|workover|abandon', na=False)
    if completion_mask.any():
        cut = completion_mask.idxmax()
        df = df.loc[:cut]
    else:
        df = df.loc[:td_idx + 5]  # allow a small buffer past TD
    print(f"Days-vs-Depth for {{well_name}}: {{len(df)}} points, "
          f"TD={{df['max_depth_m'].max():.0f}}m, total={{df['days_from_spud'].max():.1f}} days")
    return df[['days_from_spud', 'max_depth_m', 'activity_code', 'duration_hours']].copy()

def load_witsml(well_name):
    \"\"\"
    Load WITSML Depth/MD_Log files for a well.
    Returns (df, cols). Always access columns via df[cols['ROP']] — NEVER df['ROP'].
    Depth is in metres MD (auto-converts from feet if needed, removes sentinels).
    Available keys: 'ROP', 'WOB', 'RPM', 'TORQUE', 'SPP', 'HOOKLD', 'DEPTH'.
    \"\"\"
    slug = _well_slug(well_name)
    all_files = list(WITSML_DIR.glob("*.csv"))
    matching = [f for f in all_files if f.name.upper().startswith(slug + '__') and 'MD_LOG' in f.name.upper()]
    if not matching:
        matching = [f for f in all_files if slug in f.name.upper() and 'MD_LOG' in f.name.upper()]
    if not matching:
        matching = [f for f in all_files if slug in f.name.upper()]
    dfs = []
    for f in matching[:6]:
        try:
            _df = pd.read_csv(f, low_memory=False)
            # Per-file: clean any depth-like columns before concat to avoid unit mixing
            for dc in ['DMEA', 'DEPTH', 'DEPT', 'TVDE']:
                if dc in _df.columns:
                    cleaned = _clean_depth(_df[dc])
                    # If the cleaned series has < 20% valid rows, this column is not a depth
                    if len(cleaned) < 0.2 * len(_df):
                        _df.drop(columns=[dc], inplace=True, errors='ignore')
                    else:
                        _df[dc] = pd.to_numeric(_df[dc], errors='coerce')
                        # Replace sentinel/out-of-range with NaN
                        _df.loc[~_df[dc].isin(cleaned.index.map(lambda i: _df[dc].iloc[i] if i < len(_df) else None)), dc] = float('nan')
            dfs.append(_df)
        except Exception:
            pass
    if not dfs:
        print(f"WARNING: No WITSML files found for '{{well_name}}' (slug: {{slug}})")
        return pd.DataFrame(), {{}}
    df = pd.concat(dfs, ignore_index=True)
    # Resolve column map: pick first alias that has valid data in realistic range
    cols = {{}}
    for key, alts in _WITSML_COL_MAP.items():
        for alt in alts:
            if alt not in df.columns:
                continue
            v = pd.to_numeric(df[alt], errors='coerce')
            v = v[v.notna() & ~v.isin(_WITSML_NULLS) & (v >= 0)]
            if key == 'DEPTH':
                # Extra validation: must have median in realistic drilling depth range
                if v.empty or v.median() > _VOLVE_MAX_DEPTH_M:
                    continue
                # Convert feet if needed
                if v.median() > 5000:
                    df[alt] = df[alt].apply(lambda x: float(x)/3.28084 if pd.notna(x) else x)
            if len(v) > 10:
                cols[key] = alt; break
    # ── Physical-range guard: remove impossible values per parameter ──────────
    # Wide enough to accept both metric and imperial units; catches 10 000+ garbage.
    _PHYS = {{
        'ROP':    (0.01, 300),    # m/hr or ft/hr — max practical ~200
        'WOB':    (0,    500),    # klbs or kN    — 500 klbs ≈ 2 225 kN
        'RPM':    (0,    400),    # rpm
        'TORQUE': (0,    150000), # Nm or ft-lbs  — wide range
        'SPP':    (0,    10000),  # PSI or bar    — 10 000 PSI ≈ 690 bar
        'HOOKLD': (0,    10000),  # klbs or kN
    }}
    for _param, (_lo, _hi) in _PHYS.items():
        if _param in cols:
            _col = cols[_param]
            df[_col] = pd.to_numeric(df[_col], errors='coerce')
            # Replace sentinel nulls with NaN
            df.loc[df[_col].isin(_WITSML_NULLS), _col] = float('nan')
            # Null out physically impossible values (not clamp — keeps data honest)
            df.loc[~df[_col].between(_lo, _hi, inclusive='both') & df[_col].notna(), _col] = float('nan')
    print(f"WITSML for {{well_name}}: {{len(df)}} rows | params: {{list(cols.keys())}}")
    if 'DEPTH' in cols:
        dep = pd.to_numeric(df[cols['DEPTH']], errors='coerce').dropna()
        if not dep.empty:
            print(f"  Depth range: {{dep.min():.0f}}–{{dep.max():.0f}} m MD")
    for _p, _c in cols.items():
        if _p != 'DEPTH':
            _s = pd.to_numeric(df[_c], errors='coerce').dropna()
            if not _s.empty:
                print(f"  {{_p}} ({{_c}}): mean={{_s.mean():.1f}}, p5={{_s.quantile(0.05):.1f}}, p95={{_s.quantile(0.95):.1f}}")
    return df, cols

_VOLVE_MAX_DEPTH_M = 5500
_WITSML_NULLS = {{-999.25, -999.0, -9999.0, 9999.0, 9999.25}}

def _clean_depth_series(s):
    s = pd.to_numeric(s, errors='coerce')
    s = s[~s.isin(_WITSML_NULLS) & s.notna() & (s >= 0)]
    if s.empty: return s
    if s.median() > 5000: s = s / 3.28084
    return s[s <= _VOLVE_MAX_DEPTH_M]

def days_vs_depth(well_name):
    \"\"\"
    Return clean Days-vs-Depth DataFrame:
      days_from_spud (float), max_depth_m (monotonically increasing), activity_code.
    Automatically trims post-TD completion operations.
    ALWAYS use this helper for D-vs-D charts — never build from raw DDR.
    \"\"\"
    df = load_ddr(well_name)
    if df.empty or 'act_start' not in df.columns: return pd.DataFrame()
    df = df[df['md_m'] > 0].copy()
    df['act_start'] = pd.to_datetime(df['act_start'], errors='coerce')
    df = df.dropna(subset=['act_start']).sort_values('act_start').reset_index(drop=True)
    t0 = df['act_start'].min()
    df['days_from_spud'] = (df['act_start'] - t0).dt.total_seconds() / 86400
    df['max_depth_m'] = df['md_m'].cummax()
    td_idx = int(df['max_depth_m'].idxmax())
    # Cut off post-TD completion/workover
    post = df.loc[td_idx:, 'activity_code'].str.lower()
    comp_mask = post.str.contains('complet|workover|abandon', na=False)
    cut = int(comp_mask.idxmax()) if comp_mask.any() else td_idx + 10
    df = df.loc[:cut].copy()
    print(f"days_vs_depth({{well_name}}): {{len(df)}} pts | TD={{df['max_depth_m'].max():.0f}}m | {{df['days_from_spud'].max():.1f}} days")
    return df[['days_from_spud','max_depth_m','activity_code','duration_hours']].reset_index(drop=True)

def save_plotly_html(fig, filename_without_ext):
    \"\"\"Save interactive HTML + PNG snapshot for inline display.\"\"\"
    html_path = str(OUTPUTS_DIR / f"{{filename_without_ext}}.html")
    png_path  = str(OUTPUTS_DIR / f"{{filename_without_ext}}.png")
    fig.write_html(html_path, include_plotlyjs='cdn')
    try:
        fig.write_image(png_path, width=1000, height=520, scale=1.5)
        print(f"Chart PNG saved to: {{png_path}}")
    except Exception as _e:
        print(f"PNG export skipped: {{_e}}")
    print(f"Interactive chart saved to: {{html_path}}")

{code}
"""
        # Save to temp file
        tmp_script = "/tmp/analyst_script.py"
        with open(tmp_script, "w") as f:
            f.write(full_code)
        
        try:
            result = subprocess.run(
                [sys.executable, tmp_script],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout
            if result.stderr:
                output += f"\\nError: {result.stderr}"
            return output if output.strip() else "Success (No output returned)."
        except Exception as e:
            return f"Execution Error: {e}"
