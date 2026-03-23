"""
orchestrator.py
---------------
Hybrid Orchestrator for the Drilling Intelligence System (Phase 6).
Supports streaming "Thinking" logs and real-time responses.
"""
import os
import re
import time
import logging
from pathlib import Path
from typing import Generator, Dict, Any
from dotenv import load_dotenv
from google import genai

# Tools
from src.agents.tools import get_iadc_db, get_volve_db
# The deep reasoning loop
from src.agents.crew import run_aggregation_loop

load_dotenv()
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
_genai_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# ── Router Tags ──────────────────────────────────────────────────────────────
ROUTING_IADC = "IADC_Definition"
ROUTING_VOLVE_HISTORY = "Volve_History" 
ROUTING_DEEP_ANALYST = "Data_Analysis"
ROUTING_AGGREGATE = "Extrapolation"
ROUTING_DUAL = "Dual_Search" # New in Phase 6: Multi-source for ambiguous terms

# ── 1. Classification Engine ──────────────────────────────────────────────────

def classify_question(question: str) -> str:
    """Heuristic router with Phase 6 'Dual Search' and 'Geophysics' awareness."""
    q_lower = question.lower()
    
    # 1. Macro / Lessons
    agg_kw = ["lessons learned", "extrapolate", "summarize", "overall", "compare across"]
    if any(kw in q_lower for kw in agg_kw): return ROUTING_AGGREGATE
    
    # 2. Tech Terms that need Dual Search (Theory + Volve Context)
    # Give 65% weight to Volve later in prompt.
    dual_kw = ["wow", "waiting on weather", "npt", "stuck pipe", "milling", "kicks", "losses"]
    if any(kw == q_lower.strip() or f" {kw} " in f" {q_lower} " for kw in dual_kw):
        return ROUTING_DUAL

    # 3. Geophysics (Formation Tops)
    geo_kw = ["formation", "top", "stratigraphy", "geology", "lithology", "hugin", "shetland", "skagerrak"]
    if any(kw in q_lower for kw in geo_kw): return ROUTING_VOLVE_HISTORY

    # 4. Numerical / Analytics
    math_kw = ["average", "mean", "max", "min", "trend", "calc", "rop", "rpm", "chart", "table", "plot", "compare"]
    if any(kw in q_lower for kw in math_kw): return ROUTING_DEEP_ANALYST

    # 5. Volve Historical
    history_kw = ["what happened", "records", "incident", "daily log", "instance"]
    well_pattern = r"(\d{1,2}/\d+-[A-Za-z]+-?\d+(?:\s*[A-Z])?)"
    if "instance" in q_lower or "record" in q_lower or re.search(well_pattern, q_lower):
        return ROUTING_VOLVE_HISTORY

    return ROUTING_IADC

# ── 2. Unified RAG Execution ──────────────────────────────────────────────────

def run_fast_rag(question: str, routes: list, persona="Technical Assistant") -> str:
    """Supports single OR multi-source RAG (Dual Search)."""
    context_blocks = []
    
    for route in routes:
        if route == ROUTING_IADC:
            db = get_iadc_db()
            label = "IADC Drilling Glossary (Theory)"
            results = db.similarity_search(
                question, k=4,
                filter={"viking_namespace": "resources/iadc/"}
            )
            # Fallback: unfiltered search if namespace yields nothing
            if not results:
                results = db.similarity_search(question, k=4)
        else:
            db = get_volve_db()
            label = "Volve Field records (Operational History & Formation Picks)"
            results = db.similarity_search(
                question, k=25,
                filter={"viking_namespace": "resources/volve/"}
            )
            if not results:
                results = db.similarity_search(question, k=25)
        
        for i, doc in enumerate(results):
            source = doc.metadata.get('source', 'Unknown source')
            if isinstance(source, str) and '/' in source:
                source = source.split('/')[-1]
            context_blocks.append(f"[{label} - Source: {source}]: {doc.page_content}")

    if not context_blocks:
        return "I couldn't find relevant technical or historical records for this query."

    context_str = "\n\n".join(context_blocks)
    
    # User Request: Technical Chat tone, weight Volve (65%).
    # Align with SPE Challenge grading requirements.
    system_prompt = f"""You are Odin, a strictly professional, highly technical, and analytical engineering AI system.
TONE: Maintain a serious, formal, and precise engineering tone. Provide logically structured, evidence-based answers.
DO NOT use casual language.

PRIORITY: When answering about operational concepts (like WOW or NPT),
give 65% more weight and detail to the Volve Field historical examples provided
over general definitions.

LANGUAGE: The Volve source documents may contain Norwegian text (from the Volve PUD and field reports).
If retrieved context contains Norwegian, translate it to English and present ONLY the English translation.
Never output Norwegian text to the user. Key translations: foringsrør=casing, borevæske=drilling fluid,
boreslam=drilling mud, brønn=well, hullseksjon=hole section, borekaks=drill cuttings.

EVIDENCE & ASSUMPTIONS: Always clearly state your evidence (e.g., "According to Volve DDR...") and declare any assumptions or confidence levels.

ONLY IF the user explicitly asks for a formal report, analysis, or structured breakdown, should you use rigorous sections like ## Evidence, ## Assumptions, etc. Otherwise, maintain a concise but highly professional technical summary.

CONTEXT:
{context_str}

QUESTION: {question}"""

    try:
        response = _genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=system_prompt
        )
        return response.text
    except Exception as e:
        return f"LLM Error: {e}"

# ── 3. Streaming Orchestrator ─────────────────────────────────────────────────

def run_pipeline(question: str, chat_history=None) -> Generator[Dict[str, Any], None, None]:
    """
    Generator that yields incremental status logs and the final answer.
    """
    t0 = time.time()
    
    def log_evt(icon, name, status, detail=""):
        return {"event": "log", "icon": icon, "name": name, "status": status, "detail": detail, "time": time.time()}

    # 1. Memory Analysis
    if chat_history:
        yield log_evt("🧠", "Memory", f"Analyzing {len(chat_history)} messages...", "Restoring context.")

    # 2. Routing
    yield log_evt("🔍", "Classifier", "Analyzing intent...", f"'{question[:50]}...'")
    route = classify_question(question)
    yield log_evt("🔀", "Router", f"Path: Agentic Loop", "Delegating to Multi-Agent Crew.")

    # 3. Execution
    answer = ""
    charts = []
    
    # CrewAI Path (100% routing to allow dynamic tool discovery)
    yield log_evt("🤖", "Rig Crew", "Waking up Agents...", "Initializing reasoning loop.")
    try:
        # run_aggregation_loop is now a generator yielding log/answer events
        for event in run_aggregation_loop(question):
            if event["event"] == "log":
                yield log_evt(event["icon"], event["name"], event["status"], event["detail"])
            elif event["event"] == "final_answer":
                answer = event["answer"]
            elif event["event"] == "verbose_log":
                yield {"event": "verbose_log", "content": event.get("content", "")}
            elif event["event"] == "error":
                answer = f"CrewAI Error: {event['message']}"
        
        # Check for charts in outputs/figures
        fig_dir = BASE_DIR / "outputs" / "figures"
        if fig_dir.exists():
            for ext in ["*.png", "*.html"]:
                for p in fig_dir.glob(ext):
                    # Only append charts created in the last 2 minutes to avoid old charts
                    if time.time() - p.stat().st_mtime < 120:
                        if str(p.absolute()) not in charts:
                            charts.append(str(p.absolute()))
    except Exception as e:
        answer = f"Agent Error: {e}"

    elapsed = time.time() - t0
    yield log_evt("✅", "Complete", f"Done in {elapsed:.1f}s", "Finalizing response.")
    
    yield {"event": "final_answer", "answer": str(answer), "route": route, "charts": charts}
