---
title: ODIN — Operational Drilling Intelligence Network
emoji: 🛢️
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: true
license: mit
---

# ODIN — Operational Drilling Intelligence Network

> Multi-agent AI system for subsurface and drilling engineering analysis
> Built on the public Equinor Volve Field dataset · SPE GCS 2026 ML Challenge

---

## Overview

ODIN is a CrewAI-powered multi-agent system that answers complex drilling engineering questions by reasoning over structured data (WITSML, EDM) and unstructured reports (Daily Drilling Reports). It combines real-time data retrieval, RAG over domain knowledge, and a Gradio chat interface with inline Plotly visualizations.

**Key capabilities:**
- Drill phase distribution & NPT breakdown analysis
- ROP / WOB / RPM performance profiling
- Cross-well KPI comparison
- BHA configuration review and handover summaries
- Stuck-pipe and wellbore stability root-cause analysis
- Evidence-cited answers with confidence levels

---

## Architecture

```
User Query
    │
    ▼
Orchestrator (orchestrator.py)
    │  Classifies query → lean or full crew
    │
    ├── LEAN (chart / compare queries, ~40s)
    │     Analyst  ──► Lead (Odin)
    │
    └── FULL (deep analysis, ~80s)
          Lead  ──► Analyst  ──► Historian  ──► Lead (Odin)
```

**Agents:**
| Agent | Role |
|---|---|
| **Odin (Lead)** | Synthesizes findings, grounds in Volve KB |
| **Data Analyst** | Runs DDR / WITSML / EDM queries & Python charts |
| **Historian** | Searches operational history, validates stats |

**Tools available to agents:**
- `DDR_Query` — Daily Drilling Report search
- `WITSML_Analyst` — Realtime drilling log analysis
- `EDM_Technical_Query` — Casing, BHA, formation data
- `CrossWell_Comparison` — Multi-well KPI comparison
- `VolveHistory_SearchTool` — RAG over Volve campaign history
- `python_interpreter` — Pandas + Plotly for custom charts

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash (via `google-generativeai`) |
| Agent framework | CrewAI 1.10 |
| RAG / Vector store | ChromaDB + `sentence-transformers` |
| Data processing | Pandas, NumPy, PDFPlumber |
| Visualisation | Plotly (HTML) + Kaleido (PNG) |
| UI | Gradio 6 |

---

## Data

This project uses the **Equinor Volve Field open dataset** (released under the Volve Data Sharing Agreement).

> Download from: [https://www.equinor.com/energy/volve-data-sharing](https://www.equinor.com/energy/volve-data-sharing)

After downloading, extract to `data/raw/` and run the ETL pipeline:

```bash
python src/data_pipeline/run_pipeline.py
```

Then build the knowledge base:

```bash
python src/rag/build_volve_db.py
python src/rag/build_openviking_db.py
```

---

## Quickstart (judges)

```bash
# 1. Clone & install
git clone <repo-url>
cd odin
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download runtime data (~400 MB knowledge bases + processed CSVs)
python scripts/download_data.py

# 3. Add your Gemini API key
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY=<your key>
# Free key at: https://aistudio.google.com/app/apikey

# 4. Run
python src/agents/app.py
```

Open `http://localhost:7860` in your browser.

---

## Project Structure

```
odin/
├── src/
│   ├── agents/           # Main application
│   │   ├── app.py        # Gradio UI (entry point)
│   │   ├── orchestrator.py  # Query routing & streaming
│   │   ├── crew.py       # CrewAI agent definitions & tasks
│   │   ├── tools.py      # DDR / WITSML / EDM / RAG tools
│   │   └── data_tools.py # Python interpreter tool + data helpers
│   │
│   ├── data_pipeline/    # ETL: raw Volve data → processed CSV
│   │   ├── run_pipeline.py
│   │   ├── parse_witsml_logs.py
│   │   ├── parse_ddr_xml.py
│   │   └── parse_edm.py
│   │
│   └── rag/              # Knowledge base builders
│       ├── build_volve_db.py
│       └── build_openviking_db.py
│
├── tests/
│   └── prompts/          # Agent prompt test cases
│
├── data/                 # ← NOT in git (download separately)
│   ├── raw/              # Original Volve dataset
│   ├── processed/        # ETL output (CSV / Parquet)
│   └── knowledge_base/   # ChromaDB vector stores
│
├── outputs/              # ← NOT in git (generated at runtime)
│   └── figures/          # Plotly charts (HTML + PNG)
│
├── requirements.txt
├── .env.example
└── promptfooconfig.yaml  # Evaluation harness (PromptFoo)
```

---

## Rate Limits

The system is tuned for the Gemini free tier (15 RPM):

| Crew mode | LLM calls | Target time |
|---|---|---|
| Lean (chart / compare) | ~6 calls | ~40s |
| Full (deep analysis) | ~10 calls | ~80s |

Automatic 429 retry with exponential back-off (10 → 20 → 40 → 60s) is built in.

---

## License

Source code: MIT
Volve dataset: [Volve Data Sharing Agreement](https://www.equinor.com/energy/volve-data-sharing) (not included in this repo)
