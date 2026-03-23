"""
tools.py
--------
Custom Tools for the SPE GCS 2026 ML Challenge Agents.

1. StatefulPythonExecutionTool: Safely executes generated Pandas code, keeping state.
2. IADC_SearchTool: Queries the local IADC ChromaDB for drilling concepts.
3. VolveHistory_SearchTool: Queries the Volve DDR ChromaDB for historical events.
"""
import os
import io
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from crewai.tools import BaseTool
from pydantic import Field
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
DDR_DIR = str(DATA_DIR / "ddr")
WITSML_DIR = str(DATA_DIR / "witsml")
OUTPUTS_DIR = BASE_DIR / "outputs" / "figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Stateful Python Execution Tool ─────────────────────────────────────────

class DataInventoryTool(BaseTool):
    name: str = "data_inventory_inspector"
    description: str = "Use this tool to see what data (CSVs, WITSML, Files) are available across DDR_DIR and WITSML_DIR. Returns a summary of wells and datasets."

    def _run(self, query: str = "") -> str:
        summary = ["### Project Data Inventory"]
        
        # DDR Directory
        ddr_path = os.environ.get('DDR_DIR', DDR_DIR)
        if os.path.exists(ddr_path):
            files = os.listdir(ddr_path)
            summary.append(f"\n**DDR Directory ({ddr_path}):**")
            summary.append(f"- Total Files: {len(files)}")
            csvs = [f for f in files if f.endswith('.csv')]
            wells = set([f.split('_activities')[0].split('_daily')[0] for f in csvs if '_' in f])
            summary.append(f"- Detected Wells: {', '.join(sorted(list(wells))[:10])}...")
            if '_ddr_extraction_summary.csv' in files:
                summary.append("- [Key File]: `_ddr_extraction_summary.csv` (High-level well metadata)")
            if '_ddr_all_activities.csv' in files:
                summary.append("- [Key File]: `_ddr_all_activities.csv` (Granular time-log across all wells)")
        
        # WITSML Directory
        witsml_path = os.environ.get('WITSML_DIR', WITSML_DIR)
        if os.path.exists(witsml_path):
            wells_witsml = [d for d in os.listdir(witsml_path) if os.path.isdir(os.path.join(witsml_path, d))]
            summary.append(f"\n**WITSML Directory ({witsml_path}):**")
            summary.append(f"- Well Folders: {', '.join(wells_witsml)}")
            
        # PDF Reports
        pdf_path = "data/raw/Reports"
        if os.path.exists(pdf_path):
            pdfs = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
            summary.append(f"\n**PDF Knowledge Source:**")
            summary.append(f"- Reports: {', '.join(pdfs)}")

        return "\n".join(summary)

def save_plotly_html(fig, filename_without_ext):
    """Helper to be passed to the agent REPL so it can easily save html."""
    path = os.path.join(str(OUTPUTS_DIR), f"{filename_without_ext}.html")
    fig.write_html(path)
    print(f"Interactive Plotly chart saved to: {path}")

# Global REPL state so variables persist between tool calls in the same run
_repl_globals = {
    "pd": pd,
    "plt": plt,
    "np": np,
    "px": px,
    "go": go,
    "os": os,
    "DDR_DIR": DDR_DIR,
    "WITSML_DIR": WITSML_DIR,
    "OUTPUTS_DIR": str(OUTPUTS_DIR),
    "DataInventory": DataInventoryTool(),
    "save_plotly_html": save_plotly_html
}

class StatefulPythonExecutionTool(BaseTool):
    name: str = "Python REPL Data Analyst"
    description: str = (
        "Execute Python code (especially Pandas, and Plotly) to analyze data. "
        "Variables defined here PERSIST between calls. "
        "You have access to Plotly via `px` (plotly.express) and `go` (plotly.graph_objects). "
        "IMPORTANT FOR VISUALIZATIONS: Use Plotly instead of Matplotlib whenever possible. "
        "After creating a Plotly figure `fig`, save it using the provided helper: `save_plotly_html(fig, 'my_chart_name')`. "
        "Always use `print()` or `print(df.to_markdown())` to output the results so you can read them. "
        "Truncate massive outputs; do not print DataFrames with >50 rows."
    )
    
    def _run(self, code: str) -> str:
        # Strip markdown code block formatting if present
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        
        try:
            exec(code, getattr(sys.modules[__name__], '_repl_globals'))
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error executing code:\n{e}"
        finally:
            sys.stdout = old_stdout
            
        output = redirected_output.getvalue()
        
        # Hard limits on output size to protect the LLM context window
        if not output.strip():
            return "Code executed successfully, but nothing was printed. Please `print()` the result to see it."
        
        if len(output) > 8000:
            return output[:8000] + "\n\n... [OUTPUT TRUNCATED: Result exceeded 8000 characters. Please refine your code to print smaller summaries.]"
            
        return output

# ── 2. Vector Search Tools ─────────────────────────────────────────────────────

# Lazy singletons for the two vector databases
_iadc_db = None
_volve_db = None
_embeddings = None
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key
        )
    return _embeddings

def get_iadc_db():
    global _iadc_db
    if _iadc_db is None:
        db_path = BASE_DIR / "data" / "viking_context" / "chroma_fallback"
        _iadc_db = Chroma(persist_directory=str(db_path), embedding_function=get_embeddings())
    return _iadc_db

def get_volve_db():
    global _volve_db
    if _volve_db is None:
        db_path = BASE_DIR / "data" / "viking_context" / "chroma_fallback"
        _volve_db = Chroma(persist_directory=str(db_path), embedding_function=get_embeddings())
    return _volve_db

class IADC_SearchTool(BaseTool):
    name: str = "Drilling Knowledge (IADC) DB Search"
    description: str = (
        "Search the IADC drilling glossary and general Wikipedia technical articles. "
        "Use this for DEFINITIONS and THEORY (e.g. 'What is a BHA?', 'What causes stuck pipe?'). "
        "Do NOT use this for specific Volve well events."
    )
    
    def _run(self, query: str) -> str:
        try:
            db = get_iadc_db()
            # OpenViking namespace filter
            viking_filter = {"viking_namespace": "resources/iadc/"} 
            results = db.similarity_search(query, k=3, filter=viking_filter)
            if not results:
                return "No relevant IADC information found in OpenViking context."
            output = []
            for i, doc in enumerate(results):
                source = doc.metadata.get('source', 'Unknown')
                # Clean up path to just file name
                if isinstance(source, str) and '/' in source:
                    source = source.split('/')[-1]
                output.append(f"[Source: {source}]: {doc.page_content}")
            return "\n\n".join(output)
        except Exception as e:
            return f"Error searching IADC DB: {e}"

class VolveHistory_SearchTool(BaseTool):
    name: str = "Volve Campaign History DB Search"
    description: str = (
        "Search the historical Daily Drilling Reports (DDR) from the Volve campaign. "
        "Use this for HISTORICAL EVENTS and EQUIPMENTS (e.g. 'What BHA components failed on well 15/9-F-1 C?', 'Find instances of stuck pipe', 'Motor performance'). "
        "Do NOT use this for general definitions."
    )
    
    def _run(self, query: str) -> str:
        try:
            # 1. Semantic Search (OpenViking L2 Overview via Gemini 2)
            db = get_volve_db()
            viking_filter = {"viking_namespace": "resources/volve/"} 
            results = db.similarity_search(query, k=10, filter=viking_filter)
            
            output = []
            seen_content = set()
            
            # Identify high-value keywords for fallback (OpenViking L0 Hybrid Glob logic)
            keywords = ["whipstock", "milling", "stuck", "fishing", "loss", "kick", "cement", "casing", "liner", "window", "weather", "heave", "bha", "assembly", "motor", "mwd", "lwd", "bit", "failure", "twist off"]
            query_keywords = [k for k in keywords if k in query.lower()]

            # 2. Keyword Fallback: If no results or if specific keywords were missed
            found_keywords = False
            for doc in results:
                for k in query_keywords:
                    if k.upper() in doc.page_content.upper():
                        found_keywords = True
                        break
            
            # If we didn't find specific matches, try a literal scan of the narratives CSV
            if not found_keywords and query_keywords:
                csv_path = BASE_DIR / "data" / "processed" / "serialized_text" / "ddr_narratives.csv"
                if csv_path.exists():
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    # Simple keyword filter
                    mask = df['text'].str.lower().str.contains('|'.join(query_keywords), na=False)
                    kw_results = df[mask].tail(10) # Get latest 10 matches
                    if not kw_results.empty:
                        for idx, row in kw_results.iterrows():
                            content = row['text']
                            if content not in seen_content:
                                output.append(f"[Volve-KeywordMatch]:\n{content}")
                                seen_content.add(content)

            # Add semantic results (avoiding duplicates)
            for i, doc in enumerate(results):
                if doc.page_content not in seen_content:
                    source = doc.metadata.get('source', 'Unknown source')
                    if isinstance(source, str) and '/' in source:
                        source = source.split('/')[-1]
                    output.append(f"[Source: {source}]:\n{doc.page_content}")
                    seen_content.add(doc.page_content)

            if not output:
                return "No historical Volve events found matching this query."
            
            result_str = "\n\n---\n\n".join(output)
            if len(result_str) > 12000:
                return result_str[:12000] + "\n...[TRUNCATED]"
            return result_str
            
        except Exception as e:
            return f"Error searching Volve History DB: {e}"

