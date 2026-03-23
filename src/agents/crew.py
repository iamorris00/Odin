"""
crew.py
-------
Defines the multi-agent CrewAI Team for the SPE GCS 2026 ML Challenge.
The Crew is triggered ONLY when deep reasoning or data aggregation is required.
"""
import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# ── Transparent 429 retry patch for the native Gemini provider ────────────────
# CrewAI 1.10 uses GeminiCompletion (google-genai SDK) when litellm is absent.
# The provider has NO rate-limit retry — a 429 kills the task immediately.
# We patch _call_api once at import time so every LLM call auto-retries on 429.
def _patch_gemini_retry():
    """
    Monkey-patch GeminiCompletion._handle_completion to transparently sleep
    and retry on 429 / RESOURCE_EXHAUSTED without surfacing failures to CrewAI.
    Delays: 10s → 20s → 40s → 60s (4 retries, max ~130s total wait).
    """
    try:
        from crewai.llms.providers.gemini.completion import GeminiCompletion
        from google.genai.errors import APIError as _GeminiAPIError

        _orig_handle = GeminiCompletion._handle_completion
        _patch_log   = logging.getLogger(__name__)

        def _retrying_handle_completion(self, *args, **kwargs):
            _delays = [10, 20, 40, 60]
            last_exc = None
            for attempt, wait in enumerate([0] + _delays):
                if wait:
                    _patch_log.warning(
                        f"[Gemini 429] Rate limit — sleeping {wait}s "
                        f"(attempt {attempt+1}/{len(_delays)+1})"
                    )
                    time.sleep(wait)
                try:
                    return _orig_handle(self, *args, **kwargs)
                except _GeminiAPIError as e:
                    if e.code in (429, 503) or "RESOURCE_EXHAUSTED" in str(e):
                        last_exc = e
                        continue
                    raise
                except Exception:
                    raise
            raise last_exc

        GeminiCompletion._handle_completion = _retrying_handle_completion
        logging.getLogger(__name__).info(
            "GeminiCompletion._handle_completion patched — 429 auto-retry active."
        )
    except Exception as _patch_err:
        logging.getLogger(__name__).warning(
            f"Could not patch GeminiCompletion for 429 retry: {_patch_err}"
        )

_patch_gemini_retry()


def _patch_max_iter_fallback():
    """
    Patch handle_max_iterations_exceeded so that when the forced-summary LLM
    call returns None/empty (often due to oversized context after many tool
    calls), we return a graceful fallback string instead of raising ValueError.
    Without this patch a max_iter breach always crashes the entire crew.
    """
    try:
        import crewai.agents.crew_agent_executor as _exec_mod
        from crewai.utilities.agent_utils import handle_max_iterations_exceeded as _orig_hmie
        _patch_log = logging.getLogger(__name__)

        def _safe_hmie(*args, **kwargs):
            try:
                result = _orig_hmie(*args, **kwargs)
                return result
            except ValueError as e:
                if "None or empty" in str(e):
                    _patch_log.warning(
                        "[CrewAI] handle_max_iterations_exceeded returned empty "
                        "— substituting graceful fallback to prevent crew crash."
                    )
                    return (
                        "I retrieved the data from the available datasets but reached the "
                        "iteration limit while correlating the findings. "
                        "The tool outputs above contain the raw numerical results. "
                        "Please ask a more focused question (e.g., one specific metric or one well) "
                        "for a complete synthesized answer."
                    )
                raise

        # Patch both the module reference and the executor's local import
        import crewai.utilities.agent_utils as _au
        _au.handle_max_iterations_exceeded = _safe_hmie
        # The executor imports it at module level — patch the executor's namespace too
        if hasattr(_exec_mod, 'handle_max_iterations_exceeded'):
            _exec_mod.handle_max_iterations_exceeded = _safe_hmie
        _patch_log.info(
            "handle_max_iterations_exceeded patched — empty-response fallback active."
        )
    except Exception as _e:
        logging.getLogger(__name__).warning(
            f"Could not patch handle_max_iterations_exceeded: {_e}"
        )


_patch_max_iter_fallback()


# Schema-aware structured data tools (replace fragile Python REPL)
from src.agents.data_tools import (
    DataInventoryTool,
    DDRQueryTool,
    WITSMLAnalystTool,
    CrossWellCompareTool,
    EDMTechnicalTool,
    PythonTool,
)
# Vector search tools for qualitative knowledge
from src.agents.tools import IADC_SearchTool, VolveHistory_SearchTool

load_dotenv()
log = logging.getLogger(__name__)

# ── Dynamic Model Selection ───────────────────────────────────────────────────
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini/gemini-3.1-flash-lite-preview")
API_KEY = os.environ.get("GOOGLE_API_KEY")
os.environ["GEMINI_API_KEY"] = API_KEY  # Required for liteLLM underlying CrewAI

# ── Rate limit constants (Gemini flash-lite-preview free tier) ────────────────
# 15 RPM / 250K TPM / 500 RPD  (TPM is never hit; RPM is the binding constraint)
# Lean (2-task): ~6 LLM calls. Full (4-task): ~10 calls.
_INTER_TASK_DELAY_S = 2   # seconds between task completions (was 4)
_TASK_RETRY_DELAYS  = [10, 20, 40]  # exponential back-off on 429 (s)

# ── Safe LLM Configuration ───────────────────────────────────────────────────
secure_llm = LLM(
    model=MODEL_NAME,
    api_key=API_KEY,
    max_tokens=8192,      # restored — 4096 caused empty responses on complex summaries
    temperature=0.2,
    num_retries=5,
    timeout=180
)

# ── Agent Factories ───────────────────────────────────────────────────────────

def get_prompt(filename: str) -> str:
    path = Path(__file__).resolve().parents[2] / "tests" / "prompts" / filename
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        log.warning(f"Prompt file {filename} not found, using generic fallback.")
        return "You are an AI assistant."


def create_data_analyst():
    return Agent(
        role="Drilling Data Analyst",
        goal="Retrieve, correlate, and analyze exact numerical data from DDR and WITSML datasets.",
        backstory=get_prompt("analyst_prompt.txt"),
        tools=[DataInventoryTool(), DDRQueryTool(), WITSMLAnalystTool(), CrossWellCompareTool(), EDMTechnicalTool(), PythonTool()],
        llm=secure_llm,
        allow_delegation=True,
        max_iter=10  # headroom for multi-well queries; 6 was too low when agent makes 4+ tool calls
    )


def create_history_agent():
    return Agent(
        role="Volve Campaign Historian",
        goal="Find qualitative context from the Daily Drilling Report text for events found by the Data Analyst.",
        backstory=get_prompt("historian_prompt.txt"),
        tools=[VolveHistory_SearchTool()],
        llm=secure_llm,
        allow_delegation=True,
        max_iter=3
    )


def create_engineer_lead():
    return Agent(
        role="Lead Drilling Engineer",
        goal="Synthesize the Analyst's data and Historian's context into a professional Markdown report.",
        backstory=get_prompt("lead_prompt.txt"),
        tools=[IADC_SearchTool()],
        llm=secure_llm,
        allow_delegation=True,
        max_iter=3
    )


def create_auditor_agent():
    return Agent(
        role="Rig Operations Auditor",
        goal="Audit the findings of the Analyst and Historian for technical consistency and hidden statistical patterns.",
        backstory=get_prompt("auditor_prompt.txt"),
        tools=[DataInventoryTool(), IADC_SearchTool(), VolveHistory_SearchTool(), PythonTool()],
        llm=secure_llm,
        allow_delegation=True,
        max_iter=3
    )


# ── Request classifier ────────────────────────────────────────────────────────

def _is_lean_request(question: str) -> bool:
    """
    Returns True for chart/visualization and simple single-source queries.
    These go through a 2-task crew (analysis→synthesis only), skipping
    KB grounding, Historian, and Auditor to stay well within the 15 RPM budget.

    Always returns False (full crew) for questions requiring historical narrative,
    cross-well comparison, lessons learned, root cause, or risk assessment.
    """
    q = question.lower()
    # Full crew only for questions that genuinely need narrative context or cross-well synthesis.
    # Data questions (even NPT) are lean — DDRQueryTool already returns activity codes + comments.
    full_kw = [
        'lessons learned', 'lessons from', 'campaign summary', 'what happened',
        'explain why', 'root cause', 'why did', 'compare across', 'comparison between wells',
        'recommend', 'recommendation', 'predict', 'risk assessment',
        'handover', 'handoff summary', 'give me a summary of the campaign',
    ]
    if any(kw in q for kw in full_kw):
        return False
    lean_kw = [
        'chart', 'plot', 'graph', 'visualize', 'days vs depth', 'generate a',
        'draw', 'how many', 'what is the average', 'list the', 'show me the',
        'compar',  # catches compare/comparison → uses CrossWellCompareTool (1 call vs 6+)
    ]
    return any(kw in q for kw in lean_kw)


# ── Shared crew infrastructure ────────────────────────────────────────────────

def _build_shared(question: str, event_queue):
    """Create shared callbacks and agent instances."""
    def step_callback(step):
        agent_name = "Agent"
        thought = ""
        tool = None
        tool_input = ""
        try:
            if hasattr(step, 'agent'): agent_name = step.agent
            if hasattr(step, 'tool'): tool = step.tool
            if hasattr(step, 'tool_input'): tool_input = step.tool_input
            if hasattr(step, 'thought'): thought = step.thought
            elif hasattr(step, 'text'): thought = step.text
            if isinstance(step, dict):
                agent_name = step.get('agent', agent_name)
                thought = step.get('thought', step.get('text', ''))
                tool = step.get('tool')
                tool_input = step.get('tool_input', '')
            if thought and len(thought) > 5:
                event_queue.put({"event": "log", "icon": "🧠", "name": agent_name,
                                 "status": "Thought", "detail": thought[:200],
                                 "detail_full": thought, "is_dialogue": False})
            if tool:
                if tool in ["Ask question to co-worker", "Delegate work to co-worker"]:
                    event_queue.put({"event": "log", "icon": "💬", "name": agent_name,
                                     "status": f"🗣️ Interaction: {tool}",
                                     "detail": f"Message: {tool_input}" if tool_input else "",
                                     "is_dialogue": True})
                else:
                    ti_str = str(tool_input) if tool_input else ""
                    event_queue.put({"event": "log", "icon": "🔧", "name": agent_name,
                                     "status": f"Action: {tool}",
                                     "detail": f"Input: {ti_str[:120]}" if ti_str else "",
                                     "detail_full": f"Tool: {tool}\nInput:\n{ti_str}" if ti_str else f"Tool: {tool}",
                                     "is_dialogue": False})
        except Exception as e:
            event_queue.put({"event": "log", "icon": "⚠️", "name": "System",
                             "status": "Callback Error", "detail": str(e), "is_dialogue": False})

    def task_callback(task_output):
        agent_role = getattr(task_output, 'agent', 'Agent')
        summary = ""
        raw_output = ""
        if hasattr(task_output, 'raw') and task_output.raw:
            raw_output = str(task_output.raw)
            summary = raw_output.replace('\n', ' ')[:120] + "..."
        else:
            summary = "Passing analysis to the next step..."
        event_queue.put({"event": "log", "icon": "📋", "name": agent_role,
                         "status": "🗣️ Interaction: Handoff Complete",
                         "detail": summary, "detail_full": raw_output or summary, "is_dialogue": True})
        # The Data Analyst is the heaviest RPM consumer (up to 4 tool calls × LLM).
        # Give a longer cooling window specifically after it to protect the next agent.
        role_str = str(agent_role)
        delay = 6 if "Analyst" in role_str else _INTER_TASK_DELAY_S
        event_queue.put({"event": "log", "icon": "⏳", "name": "Rate Limiter",
                         "status": f"Cooling {delay}s after {role_str.split()[-1]} task…",
                         "detail": "Respecting Gemini 15 RPM budget", "is_dialogue": False})
        time.sleep(delay)

    analyst  = create_data_analyst()
    historian = create_history_agent()
    auditor  = create_auditor_agent()
    lead     = create_engineer_lead()

    for agent in [analyst, historian, auditor, lead]:
        agent.step_callback = step_callback

    return analyst, historian, auditor, lead, step_callback, task_callback


def _run_crew_thread(crew, event_queue):
    """Retry-aware crew kickoff with exponential back-off on 429.
    NOTE: stdout is already redirected by run_aggregation_loop before Crew construction
    so that the Rich Console (created at Crew.__init__ time) writes to the capture buffer.
    """
    import traceback
    last_exc = None
    for attempt, delay in enumerate([0] + _TASK_RETRY_DELAYS):
        if delay:
            event_queue.put({"event": "log", "icon": "⏳", "name": "Rate Limiter",
                             "status": f"429 back-off — waiting {delay}s (attempt {attempt+1}/4)…",
                             "detail": "Gemini RPM limit hit, retrying shortly", "is_dialogue": False})
            time.sleep(delay)
        try:
            res = crew.kickoff()
            event_queue.put({"event": "final_answer", "answer": res.raw})
            event_queue.put(None)
            return
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            tb = traceback.format_exc()
            log.error(f"Crew attempt {attempt+1} failed: {type(e).__name__}: {e}\n{tb}")
            # Surface the exception detail to the UI as a log event
            event_queue.put({"event": "log", "icon": "🔴", "name": "Crew Error",
                             "status": f"{type(e).__name__}: {str(e)[:120]}",
                             "detail": tb.splitlines()[-3] if tb else "",
                             "is_dialogue": False})
            if "429" not in err_str and "rate" not in err_str and "quota" not in err_str:
                break
    event_queue.put({"event": "error", "message": f"{type(last_exc).__name__}: {last_exc}"})
    event_queue.put(None)


# ── Aggregation Loop ──────────────────────────────────────────────────────────

def run_aggregation_loop(question: str):
    """
    Generator yielding status logs then a final_answer event.
    Routes to a lean 2-task crew (chart/simple) or full 4-task crew (deep analysis).
    Lean crew: ~6 LLM calls, ~35-50s.  Full crew: ~10 calls, ~75-90s.
    """
    from queue import Queue
    import threading

    event_queue = Queue()
    lean = _is_lean_request(question)

    mode_label = "LEAN (2-task)" if lean else "FULL (4-task)"
    event_queue.put({"event": "log", "icon": "🔀", "name": "Router",
                     "status": f"Crew mode: {mode_label}",
                     "detail": "Lean = analysis+synthesis | Full adds grounding+context",
                     "is_dialogue": False})

    analyst, historian, auditor, lead, step_callback, task_callback = \
        _build_shared(question, event_queue)

    # ── Task definitions ──────────────────────────────────────────────────────

    # Comparison-specific vs general analysis description
    _is_comparison = 'compar' in question.lower()
    if _is_comparison:
        _analyze_desc = (
            f"The user asked: '{question}'\n\n"
            "MANDATORY TOOL SEQUENCE — follow exactly, no deviations:\n"
            "Step 1 (ONLY step): Call `CrossWell_Comparison` ONCE with all wells mentioned.\n"
            "  → This single call returns DDR + WITSML data for every well. NO other data tools are needed.\n"
            "Step 2: Write your markdown answer immediately after receiving the CrossWell_Comparison result.\n"
            "  → Include a comparison table (ROP, NPT %, BHA runs) per well and per hole section.\n"
            "PROHIBITED: Do NOT call data_inventory_inspector, DDR_Query, WITSML_Analyst, or python_interpreter.\n"
            "NOTE: Translate any Norwegian text in tool output to English."
        )
    else:
        _analyze_desc = (
            f"The user asked: '{question}'\n\n"
            "Retrieve and analyze data with the MINIMUM set of tools needed:\n"
            "  • Single-well data (phases, ROP, NPT)? → Use `DDR_Query` and/or `WITSML_Analyst`\n"
            "  • BHA / casing / formations? → Use `EDM_Technical_Query`\n"
            "  • Chart/visualization? → Use `python_interpreter` with load_ddr() / load_witsml() / days_vs_depth() helpers\n"
            "    NPT identification: always call df['activity_code'].value_counts().head(30) FIRST to see available codes,\n"
            "    then filter with df['activity_code'].str.upper().str.contains('NPT|WOW|WAIT|STUCK|PACK|FISH|CIRC|TEST|DELAY|BREAK', na=False)\n"
            "  • Skip data_inventory_inspector unless you genuinely don't know which wells exist.\n"
            "Return tables, stats, and any chart file paths. Translate Norwegian text to English."
        )

    # ── LEAN: 2-task crew (analysis + synthesis only — no KB grounding step) ──
    task_analyze_lean = Task(
        description=_analyze_desc,
        expected_output=(
            "Markdown summary with exact numbers from tools. "
            "Activity/stats table required. If a chart was generated, include the full file path."
        ),
        agent=analyst,
        context=[]
    )

    task_synth_lean = Task(
        description=(
            f"The user asked: '{question}'\n"
            "Synthesize the Analyst's findings into a direct Odin response. "
            "DO NOT call any tools — use only the context you already have. "
            "CRITICAL: Do NOT mention crew members. Present findings natively as Odin. "
            "CRITICAL: ABSOLUTELY NO email headers, no To/From/Subject, no memorandum structure."
        ),
        expected_output="A direct, highly technical engineering response. No email headers.",
        agent=lead,
        context=[task_analyze_lean]
    )

    # ── FULL: 4-task crew (grounding + analysis + context + synthesis) ────────
    task_ground = Task(
        description=(
            f"Question: '{question}'\n"
            "Search the Volve Campaign History DB for relevant background context on this topic. "
            "Use `VolveHistory_SearchTool` ONLY (one call). "
            "Provide a brief 'Contextual Brief' — key events, problems, or precedents relevant to the question."
        ),
        expected_output="A concise contextual brief from the Volve operational history database.",
        agent=lead
    )

    task_analyze_full = Task(
        description=_analyze_desc,
        expected_output=(
            "Markdown summary with exact numbers from tools. "
            "Activity/stats table required. If a chart was generated, include the full file path."
        ),
        agent=analyst,
        context=[task_ground]
    )

    task_context = Task(
        description=(
            f"The user asked: '{question}'\n"
            "The Analyst found quantitative results (see context above). Do two things in ONE pass:\n"
            "1. HISTORY: Use `VolveHistory_SearchTool` to find narrative context — events, incidents, or decisions "
            "that explain the Analyst's numbers. Cite sources as [Volve-Hist-N].\n"
            "2. STATS AUDIT: Using only the numbers already in context (no new tool calls), check Mean vs Median "
            "for ROP/NPT. Note whether performance was consistent or outlier-dominated.\n"
            "Combine both into a single 'Context & Verification' response."
        ),
        expected_output=(
            "Combined: (a) relevant historical events with source citations, "
            "(b) quick statistical consistency note on the Analyst's key numbers."
        ),
        agent=historian,
        context=[task_analyze_full]
    )

    task_synth_full = Task(
        description=(
            f"The user asked: '{question}'\n"
            "Synthesize all findings into a comprehensive Odin response. "
            "DO NOT call any tools — use only the context you already have. "
            "Weave in the quantitative results, historical context, and statistical insights naturally. "
            "Include Evidence, Assumptions, and Confidence Level inline (not as separate sections unless asked). "
            "CRITICAL: Do NOT mention crew members. Present all data natively as Odin. "
            "CRITICAL: ABSOLUTELY NO email headers, no To/From/Subject, no formal memorandum structure."
        ),
        expected_output="A direct, conversational yet highly technical engineering response. No email headers.",
        agent=lead,
        context=[task_analyze_full, task_context]
    )

    # ── Redirect stdout BEFORE Crew construction so the Rich Console writes to buffer ──
    # CrewAI's verbose output uses a Rich Console created at Crew.__init__ time.
    # If we redirect after construction, Console keeps the original stdout reference.
    from io import StringIO
    import sys as _sys
    import re as _re_ansi
    _stdout_buf = StringIO()
    _orig_stdout = _sys.stdout
    _sys.stdout = _stdout_buf

    # ── Route to lean (2-task) or full (4-task) crew ──────────────────────────
    try:
        if lean:
            crew = Crew(
                agents=[analyst, lead],
                tasks=[task_analyze_lean, task_synth_lean],
                process=Process.sequential,
                max_rpm=14,
                verbose=True,
                task_callback=task_callback,
                step_callback=step_callback
            )
        else:
            crew = Crew(
                agents=[lead, analyst, historian],
                tasks=[task_ground, task_analyze_full, task_context, task_synth_full],
                process=Process.sequential,
                max_rpm=10,
                verbose=True,
                task_callback=task_callback,
                step_callback=step_callback
            )
    except Exception:
        _sys.stdout = _orig_stdout
        raise

    def run_crew():
        _run_crew_thread(crew, event_queue)

    thread = threading.Thread(target=run_crew)
    thread.start()

    while True:
        event = event_queue.get()
        if event is None:
            break
        yield event

    thread.join()

    # ── Restore stdout and emit captured transcript ────────────────────────────
    _sys.stdout = _orig_stdout
    _raw_transcript = _stdout_buf.getvalue()
    if _raw_transcript.strip():
        # Strip ANSI escape codes (Rich colour markup)
        _clean = _re_ansi.sub(r'\x1b\[[0-9;]*[mGKHF]', '', _raw_transcript)
        yield {"event": "verbose_log", "content": _clean}
