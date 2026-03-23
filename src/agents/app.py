"""
app.py
------
Odin Drilling Intelligence System — Competition UI v3.0
SPE GCS 2026 ML Challenge · Full Redesign

Changes from v2.3:
  - Tabbed right panel: Challenge Questions | Agent HUD | Charts
  - 24 challenge-aligned question buttons covering all rubric categories
  - Vertical pipeline HUD with telemetry (tools used, elapsed time, action count)
  - Well selector dropdown (all 23 Volve wells) with auto-injection into queries
  - Answer metadata chips: sources used + confidence badge + elapsed time
  - Dedicated chart panel (no more iframes inside chat)
  - Export to Markdown button
  - Clear session button
  - Clean brand header (no internal version/phase strings)
"""

import time
import os
import re
import tempfile
import gradio as gr
from pathlib import Path
from src.agents.orchestrator import run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# DATA: Wells + Challenge Questions
# ─────────────────────────────────────────────────────────────────────────────

SUGGESTED_PROMPTS = [
    "Analyze and provide a chart of the drilling phase distribution and NPT breakdown for 15/9-F-12, with evidence from DDR and WITSML.",
    "What were the main stuck pipe and wellbore stability events across the Volve campaign, and what formation was responsible?",
    "Produce an operational handover summary for 15/9-F-14 and recommend a BHA configuration for the next 12.25-inch section.",
    "Do an in-depth analysis of the drilling performance of three Volve wells and compare their key KPIs.",
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;500;600;700;900&display=swap');

/* ── Base ── */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    font-family: 'Inter', sans-serif;
    background: #030712 !important;
    min-height: 100vh;
}
footer { display: none !important; }

/* Custom scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #10b981; }

/* ── Header ── */
#odin-header {
    background: #020617 !important;
    border-bottom: 1px solid #0d2a1f !important;
    box-shadow: 0 1px 0 #10b98122, 0 4px 24px #00000066 !important;
    padding: 0 20px !important;
    height: 54px;
    align-items: center !important;
    flex-wrap: nowrap !important;
    gap: 12px !important;
}
.odin-logo-wrap {
    display: flex; align-items: center; gap: 10px; text-decoration: none;
}
.odin-rune {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6em; font-weight: 900;
    color: #10b981;
    text-shadow: 0 0 12px #10b98166, 0 0 24px #10b98133;
    letter-spacing: 4px;
    line-height: 1;
}
.odin-divider {
    width: 1px; height: 26px; background: #1e293b; flex-shrink: 0;
}
.odin-wordmark {
    font-size: 0.68em; color: #475569; line-height: 1.3;
    font-family: 'Share Tech Mono', monospace; letter-spacing: 0.5px;
}
.odin-wordmark strong { color: #94a3b8; font-weight: 600; }
.odin-stats {
    margin-left: auto;
    display: flex; gap: 16px; align-items: center;
}
.odin-stat {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.66em; color: #334155; line-height: 1.3; text-align: center;
}
.odin-stat span { display: block; color: #10b981; font-weight: 700; font-size: 1.15em; }

/* ── Chat column ── */
#chat-col {
    background: #030712 !important;
    border-right: 1px solid #0f172a !important;
}
.chatbot-wrap {
    background: #030712 !important;
    border: none !important;
}

/* User bubbles */
.message.user {
    background: linear-gradient(135deg, #0f2a1e 0%, #0d2234 100%) !important;
    color: #e2e8f0 !important;
    border: 1px solid #1a3a2a !important;
    border-radius: 10px 10px 2px 10px !important;
}
/* Bot bubbles */
.message.bot {
    background: #0a0f1e !important;
    color: #cbd5e1 !important;
    border: 1px solid #0f172a !important;
    border-left: 2px solid #10b98133 !important;
    border-radius: 2px 10px 10px 10px !important;
}
/* Code blocks in responses */
.message.bot code { background: #0f172a !important; color: #6ee7b7 !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.88em !important; }
.message.bot pre  { background: #0a0f1e !important; border: 1px solid #1e293b !important; border-left: 3px solid #10b981 !important; }
/* Tables */
.message.bot table { font-size: 0.83em !important; border-collapse: collapse !important; }
.message.bot th { background: #0f172a !important; color: #10b981 !important; border: 1px solid #1e293b !important; padding: 4px 8px !important; font-family: 'Share Tech Mono', monospace; }
.message.bot td { border: 1px solid #1e293b !important; padding: 3px 8px !important; color: #94a3b8 !important; }
.message.bot tr:nth-child(even) td { background: #0a0f1e !important; }

/* ── Input zone ── */
#input-zone {
    padding: 10px 16px 12px !important;
    background: #030712 !important;
    border-top: 1px solid #0f172a !important;
    align-items: flex-end !important;
    gap: 8px !important;
}
#msg-input textarea {
    background: #0a0f1e !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-size: 0.9em !important;
    font-family: 'Inter', sans-serif !important;
    resize: none !important;
}
#msg-input textarea:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px #10b98122 !important;
}
#msg-input textarea::placeholder { color: #334155 !important; }
#send-btn {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    border: 1px solid #065f46 !important;
    font-weight: 700 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px !important;
    box-shadow: 0 2px 8px #10b98133 !important;
    transition: all 0.2s !important;
}
#send-btn:hover {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    box-shadow: 0 4px 16px #10b98144 !important;
    transform: translateY(-1px) !important;
}

/* ── Meta chips bar ── */
#meta-bar { padding: 5px 16px 2px; background: #030712; min-height: 28px; }

/* ── Chart area ── */
#chart-area { padding: 0 4px; }
/* export-file is always in the DOM (hidden via size, not display:none)
   so JS getElementById works even before the user clicks Export */
#export-file { height: 0 !important; overflow: hidden !important;
               padding: 0 !important; margin: 0 !important; }

/* ── Right panel ── */
#right-panel {
    background: #020617 !important;
    border-left: 1px solid #0f172a !important;
}

/* ── Tabs ── */
.tabs { background: transparent !important; }
.tab-nav {
    background: #020617 !important;
    border-bottom: 1px solid #0f172a !important;
    padding: 0 10px !important;
}
.tab-nav button {
    color: #334155 !important;
    font-size: 0.75em !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.5px !important;
    padding: 10px 10px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button:hover { color: #64748b !important; }
.tab-nav button.selected { color: #10b981 !important; border-bottom-color: #10b981 !important; }

/* ── Suggested prompts ── */
.prompts-scroll { max-height: calc(100vh - 130px); overflow-y: auto; padding: 12px 14px; }
.prompt-hint {
    font-size: 0.68em; color: #1e3a2a;
    padding: 6px 10px 12px; line-height: 1.6;
    font-family: 'Share Tech Mono', monospace;
    border-left: 2px solid #10b98133; margin-bottom: 8px;
}
.p-btn {
    display: block !important; width: 100% !important; text-align: left !important;
    padding: 10px 12px !important; margin: 6px 0 !important;
    background: #0a0f1e !important;
    border: 1px solid #1e293b !important;
    border-left: 3px solid #1e3a2a !important;
    border-radius: 6px !important; cursor: pointer !important;
    color: #64748b !important; font-size: 0.77em !important; line-height: 1.55 !important;
    white-space: normal !important; height: auto !important;
    transition: all 0.2s !important;
    font-family: 'Inter', sans-serif !important;
}
.p-btn:hover {
    background: #0d1f18 !important;
    border-color: #1e3a2a !important;
    border-left-color: #10b981 !important;
    color: #a7f3d0 !important;
    transform: translateX(3px) !important;
    box-shadow: -3px 0 12px #10b98122 !important;
}

/* ── Pipeline HUD tab ── */
.hud-scroll { overflow-y: auto; padding: 10px 12px; display:flex; flex-direction:column; gap:10px; }
.pipe-title {
    color: #10b981; font-weight: 700; text-transform: uppercase;
    letter-spacing: 2px; font-size: 0.65em; margin-bottom: 10px;
    font-family: 'Share Tech Mono', monospace;
    display: flex; align-items: center; gap: 6px;
}
.pipe-title::after {
    content: ''; flex: 1; height: 1px; background: linear-gradient(to right, #1e293b, transparent);
}
.pipe-track { border-left: 2px solid #0f172a; margin-left: 8px; padding-left: 14px; }
.pipe-step {
    position: relative; display: flex; align-items: center; gap: 8px;
    padding: 6px 8px; margin-bottom: 6px;
    border-radius: 6px; background: #0a0f1e; border: 1px solid #0f172a;
    transition: all 0.3s ease; opacity: 0.25; filter: grayscale(1); font-size: 0.79em;
}
.pipe-step.active    { opacity:1; filter:none; background:#051a11; border-color:#10b981; animation:pipeGlow 2s infinite; }
.pipe-step.complete  { opacity:0.8; filter:none; background:#0a0f1e; border-color:#1e3a5f; }
.pipe-step.delegating{ opacity:1; filter:none; background:#150d2a; border-color:#8b5cf6; animation:pipeDel 1.5s ease infinite; }
.pipe-dot { width:7px; height:7px; border-radius:50%; background:#1e293b; flex-shrink:0; position:absolute; left:-18px; top:11px; }
.pipe-step.active    .pipe-dot { background:#10b981; box-shadow:0 0 6px #10b981; }
.pipe-step.complete  .pipe-dot { background:#3b82f6; }
.pipe-step.delegating .pipe-dot{ background:#8b5cf6; }
.pipe-icon { font-size:0.95em; flex-shrink:0; }
.pipe-name { font-weight:600; color:#64748b; white-space:nowrap; font-size:0.95em; }
.pipe-sub  { font-size:0.82em; color:#334155; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:140px; }
.pipe-step.active    .pipe-name { color:#a7f3d0; }
.pipe-step.active    .pipe-sub  { color:#6ee7b7; }
.pipe-step.complete  .pipe-name { color:#7dd3fc; }
.pipe-step.complete  .pipe-sub  { color:#334155; }
.pipe-step.delegating .pipe-name{ color:#c4b5fd; }
/* KB mini-nodes */
.pipe-kb-row { display:flex; gap:5px; margin-bottom:8px; }
.pipe-kb-node { flex:1; display:flex; align-items:center; gap:5px; padding:5px 7px; border-radius:6px; font-size:0.74em; background:#0a0f1e; border:1px solid #0f172a; opacity:0.25; filter:grayscale(1); transition:all 0.3s; }
.pipe-kb-node.active   { opacity:1; filter:none; background:#051a11; border-color:#10b981; animation:pipeGlow 2s infinite; }
.pipe-kb-node.complete { opacity:0.8; filter:none; background:#0a0f1e; border-color:#1e3a5f; }
.pipe-kb-name { font-weight:600; color:#475569; display:block; font-size:0.9em; }
.pipe-kb-sub  { color:#334155; display:block; font-size:0.82em; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:90px; }
.pipe-kb-node.active .pipe-kb-name  { color:#a7f3d0; }
.pipe-kb-node.active .pipe-kb-sub   { color:#6ee7b7; }
.pipe-kb-node.complete .pipe-kb-name{ color:#7dd3fc; }
/* Telemetry */
.pipe-telemetry { padding:8px 10px; border-radius:6px; background:#04080f; border:1px solid #0f172a; font-size:0.75em; }
.telem-title { color:#1e293b; text-transform:uppercase; letter-spacing:1.5px; font-size:0.78em; margin-bottom:5px; font-family:'Share Tech Mono',monospace; }
.telem-chip  { display:inline-block; padding:2px 7px; border-radius:4px; margin:2px 2px 2px 0; font-size:0.82em; font-weight:700; font-family:'Share Tech Mono',monospace; }
.telem-footer{ color:#1e293b; margin-top:5px; padding-top:5px; border-top:1px solid #0f172a; font-family:'Share Tech Mono',monospace; font-size:0.9em; }
/* Live Feed */
.feed-wrap { border-radius:7px; background:#04080f; border:1px solid #0f172a; overflow:hidden; }
.feed-header{ padding:5px 10px; background:#020617; border-bottom:1px solid #0f172a; font-size:0.65em; font-weight:700; color:#10b981; text-transform:uppercase; letter-spacing:2px; font-family:'Share Tech Mono',monospace; }
.feed-body  { max-height:240px; overflow-y:auto; padding:4px 0; }
.feed-entry { display:flex; align-items:flex-start; gap:6px; padding:4px 10px; border-bottom:1px solid #04080f; font-size:0.75em; }
.feed-entry:last-child { border-bottom:none; }
.feed-entry.thought  { background:#0a0f1e33; }
.feed-entry.tool     { background:#051a1133; }
.feed-entry.handoff  { background:#0c1a3333; border-left:2px solid #1e3a5f; }
.feed-entry.system   { opacity:0.45; }
.feed-badge { flex-shrink:0; padding:1px 5px; border-radius:3px; font-size:0.77em; font-weight:700; white-space:nowrap; font-family:'Share Tech Mono',monospace; letter-spacing:0.3px; }
.feed-badge.analyst   { background:#051a11; color:#6ee7b7; border:1px solid #064e3b; }
.feed-badge.historian { background:#1c0a04; color:#fed7aa; border:1px solid #7c2d12; }
.feed-badge.auditor   { background:#060d1e; color:#bfdbfe; border:1px solid #1e3a8a; }
.feed-badge.engineer  { background:#0f0a1e; color:#ddd6fe; border:1px solid #4c1d95; }
.feed-badge.system    { background:#080c12; color:#475569; border:1px solid #1e293b; }
.feed-badge.tool-badge{ background:#04080f; color:#64748b; border:1px solid #0f172a; }
.feed-text  { color:#334155; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
.feed-text b{ color:#64748b; font-weight:600; }

/* ── Animations ── */
@keyframes pipeGlow {
    0%   { box-shadow: 0 0 0 0 rgba(16,185,129,.35); }
    70%  { box-shadow: 0 0 0 5px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}
@keyframes pipeDel {
    0%,100% { box-shadow: 0 0 0 0 rgba(139,92,246,.35); }
    50%     { box-shadow: 0 0 8px 2px rgba(139,92,246,.25); }
}

/* ── Responsive ── */
@media (max-width: 860px) {
    #right-panel { border-left: none !important; border-top: 1px solid #0f172a !important; }
    .odin-stats  { display: none; }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY_HUD_STATE = {
    "q_status": "", "q_detail": "",
    "iadc_status": "", "iadc_detail": "IADC Glossary · 2,400 chunks",
    "volve_status": "", "volve_detail": "Volve DDR/EDM · 23K chunks",
    "analyst_status": "", "analyst_detail": "Waiting",
    "historian_status": "", "historian_detail": "Waiting",
    "auditor_status": "", "auditor_detail": "Waiting",
    "engineer_status": "", "engineer_detail": "Waiting",
    "s_status": "", "s_detail": "Queued",
    "tools_used": set(), "action_count": 0, "elapsed": 0.0,
    "live_feed": [],  # list of {icon, badge_class, badge, type, text}
}

_TOOL_COLORS = {
    "DDR":         ("#1e3a8a", "#bfdbfe"),
    "WITSML":      ("#064e3b", "#a7f3d0"),
    "EDM":         ("#7c2d12", "#fed7aa"),
    "IADC":        ("#4c1d95", "#ddd6fe"),
    "Volve DB":    ("#0c4a6e", "#bae6fd"),
    "Python REPL": ("#1f2937", "#d1d5db"),
}


_AGENT_BADGE = {
    "Drilling Data Analyst":    ("analyst",   "📊"),
    "Volve Campaign Historian":  ("historian", "📜"),
    "Rig Operations Auditor":   ("auditor",   "📋"),
    "Lead Drilling Engineer":   ("engineer",  "👷"),
    "Rate Limiter":             ("system",    "⏳"),
    "Router":                   ("system",    "🔀"),
}


def render_hud(state: dict) -> str:
    state = {**_EMPTY_HUD_STATE, **state}

    def _step(s_key, icon, label, d_key):
        st  = state.get(s_key, "")
        det = (state.get(d_key, "") or "")[:36]
        return f"""<div style="position:relative">
  <div class="pipe-dot"></div>
  <div class="pipe-step {st}">
    <span class="pipe-icon">{icon}</span>
    <div style="min-width:0;overflow:hidden">
      <span class="pipe-name">{label}</span>
      <span class="pipe-sub">{det}</span>
    </div>
  </div>
</div>"""

    # KB dual-node row
    iadc_sub  = (state['iadc_detail']  or "IADC Glossary · 2,400 chunks")[:22]
    volve_sub = (state['volve_detail'] or "Volve DDR/EDM · 23K chunks")[:22]
    kb_row = f"""<div class="pipe-kb-row">
  <div class="pipe-kb-node {state['iadc_status']}">
    <span>📚</span>
    <div><span class="pipe-kb-name">IADC DB</span><span class="pipe-kb-sub">{iadc_sub}</span></div>
  </div>
  <div class="pipe-kb-node {state['volve_status']}">
    <span>🗂️</span>
    <div><span class="pipe-kb-name">Volve DB</span><span class="pipe-kb-sub">{volve_sub}</span></div>
  </div>
</div>"""

    # Compact telemetry
    tools = state.get("tools_used", set())
    chips = "".join(
        f'<span class="telem-chip" style="background:{bg};color:{fg}">{t}</span>'
        for t, (bg, fg) in _TOOL_COLORS.items() if t in tools
    ) or '<span style="color:#334155">No tools yet</span>'
    elapsed = state.get("elapsed", 0.0)
    telemetry = f"""<div class="pipe-telemetry">
  <div class="telem-title">Tools Used</div>
  <div>{chips}</div>
  <div class="telem-footer">⏱ {f"{elapsed:.0f}s" if elapsed else "--"} &nbsp;|&nbsp; 🔧 {state.get("action_count", 0)} actions</div>
</div>"""

    # Live Feed — flat entries
    feed_entries = ""
    for entry in state.get("live_feed", []):
        bclass = entry.get("badge_class", "system")
        badge  = entry.get("badge", "SYS")
        text   = entry.get("text", "")[:90]
        etype  = entry.get("type", "system")
        feed_entries += (
            f'<div class="feed-entry {etype}">'
            f'<span class="feed-badge {bclass}">{badge}</span>'
            f'<span class="feed-text">{text}</span>'
            f'</div>'
        )
    if not feed_entries:
        feed_entries = '<div style="padding:12px 10px;color:#334155;font-size:0.75em">Waiting for agent activity…</div>'

    live_feed = f"""<div class="feed-wrap">
  <div class="feed-header">// LIVE AGENT FEED</div>
  <div class="feed-body">{feed_entries}</div>
</div>"""

    return f"""<div class="hud-scroll">
  <div>
    <div class="pipe-title">▶ PIPELINE</div>
    {_step("q_status", "❓", "Query", "q_detail")}
    <div class="pipe-track">
      {kb_row}
      {_step("analyst_status",   "📊", "Data Analyst", "analyst_detail")}
      {_step("historian_status", "📜", "Historian",    "historian_detail")}
      {_step("auditor_status",   "📋", "Auditor",      "auditor_detail")}
      {_step("engineer_status",  "👷", "Odin",         "engineer_detail")}
      {_step("s_status",         "✅", "Synthesis",    "s_detail")}
    </div>
    {telemetry}
  </div>
  {live_feed}
</div>"""


def extract_confidence_with_reason(text: str) -> tuple:
    """Returns (level: str|None, reason: str)."""
    for pat in [
        r'confidence[:\s*]+\**\s*(high|medium|low)\**',
        r'\**(high|medium|low)\*\*\s+confidence',
        r'(high|medium|low)\s+confidence',
    ]:
        m = re.search(pat, text.lower())
        if m:
            level = m.group(1).upper()
            # Extract a window of text around the match as the reasoning snippet
            start = max(0, m.start() - 80)
            end   = min(len(text), m.end() + 250)
            reason = text[start:end].strip().replace("\n", " ")
            return level, reason
    return None, ""


# Keep backward-compatible alias
def extract_confidence(text: str) -> str | None:
    level, _ = extract_confidence_with_reason(text)
    return level


_CONF_EXPLAIN = {
    "HIGH":   "Multiple independent data sources agree (DDR + WITSML ± EDM). No contradictions detected.",
    "MEDIUM": "Primary data source used. Minor ambiguities or single-source validation.",
    "LOW":    "Limited data coverage, significant assumptions required, or conflicting signals.",
}

def render_metadata(tools: set, confidence: str | None, elapsed: float,
                    confidence_reason: str = "") -> str:
    """Compact one-line footer HTML to embed directly inside a bot chat message."""
    if not tools and not confidence:
        return ""
    _conf_col = {"HIGH": ("#064e3b", "#6ee7b7"), "MEDIUM": ("#78350f", "#fde68a"), "LOW": ("#7f1d1d", "#fca5a5")}
    _tool_labels = {"DDR": "DDR", "WITSML": "WITSML", "EDM": "EDM",
                    "IADC": "IADC", "Volve DB": "Volve", "Python REPL": "Python"}
    parts = []
    for t, (bg, fg) in _TOOL_COLORS.items():
        if t in tools and t in _tool_labels:
            parts.append(
                f'<span style="background:{bg};color:{fg};padding:1px 6px;border-radius:3px;'
                f'font-size:0.7em;font-weight:700;font-family:\'Share Tech Mono\',monospace">'
                f'{_tool_labels[t]}</span>'
            )
    if confidence:
        bg, fg = _conf_col.get(confidence, ("#1f2937", "#d1d5db"))
        tip = (confidence_reason[:200] + "…") if confidence_reason else _CONF_EXPLAIN.get(confidence, "")
        parts.append(
            f'<span style="background:{bg};color:{fg};padding:1px 7px;border-radius:3px;'
            f'font-size:0.7em;font-weight:700;cursor:default;font-family:\'Share Tech Mono\',monospace"'
            f' title="{tip}">● {confidence}</span>'
        )
    if elapsed > 0:
        parts.append(f'<span style="color:#1e3a2a;font-size:0.68em;font-family:\'Share Tech Mono\',monospace">⏱ {elapsed:.0f}s</span>')
    inner = ' '.join(parts)
    return (
        f'<div style="margin-top:10px;padding-top:7px;border-top:1px solid #0d1a24;'
        f'display:flex;gap:5px;align-items:center;flex-wrap:wrap">{inner}</div>'
    )


def _chart_embed(p: str) -> str:
    """Return an embed snippet for a chart file — no file-serving required."""
    import base64 as _b64
    path = Path(p)
    if not path.exists():
        return f'<div style="color:#ef4444;padding:8px;font-size:0.8em">Missing: {path.name}</div>'
    wrap = 'style="border-radius:8px;border:1px solid #1e293b;overflow:hidden;margin-bottom:14px"'
    if p.endswith(".png"):
        data = _b64.b64encode(path.read_bytes()).decode()
        return f'<div {wrap}><img src="data:image/png;base64,{data}" style="width:100%;display:block"/></div>'
    # HTML chart — base64 data URI avoids all srcdoc escaping issues
    b64_html = _b64.b64encode(path.read_bytes()).decode()
    return (f'<div {wrap}><iframe src="data:text/html;base64,{b64_html}" width="100%" height="480" '
            f'frameborder="0" style="display:block" sandbox="allow-scripts"></iframe></div>')


def render_charts(chart_paths: list) -> str:
    if not chart_paths:
        return """<div class="charts-scroll">
  <div class="chart-empty">
    <div style="font-size:2.5em">📊</div>
    <div style="color:#475569;font-weight:600">No charts yet</div>
    <div style="color:#334155;max-width:200px">
      Ask about ROP, NPT, Days vs Depth, or well comparisons to trigger visualizations.
    </div>
  </div>
</div>"""
    # Prefer PNG over HTML for the same chart stem
    stems_with_png = {Path(p).stem for p in chart_paths if p.endswith(".png") and Path(p).exists()}
    items = []
    for p in chart_paths:
        stem = Path(p).stem
        if p.endswith(".html") and stem in stems_with_png:
            continue  # PNG version covers this chart
        if not Path(p).exists():
            continue
        name = stem.replace("_", " ").title()
        label = (f'<div style="color:#475569;font-size:0.7em;text-transform:uppercase;'
                 f'letter-spacing:1px;margin-bottom:4px">{name}</div>')
        items.append(label + _chart_embed(p))
    if not items:
        return render_charts([])  # all paths missing → empty state
    return f'<div class="charts-scroll">{"".join(items)}</div>'




# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_KEYWORDS = [
    ("DDR_Query", "DDR"), ("DDR", "DDR"),
    ("WITSML_Analyst", "WITSML"), ("WITSML", "WITSML"),
    ("EDM_Technical", "EDM"), ("EDM", "EDM"),
    ("IADC_SearchTool", "IADC"), ("IADC", "IADC"),
    ("VolveHistory_SearchTool", "Volve DB"), ("VolveHistory", "Volve DB"), ("Volve", "Volve DB"),
    ("python_interpreter", "Python REPL"), ("Python REPL", "Python REPL"),
]

_AGENT_MAP = {
    "Drilling Data Analyst":   "analyst",
    "Volve Campaign Historian": "historian",
    "Rig Operations Auditor":  "auditor",
    "Lead Drilling Engineer":  "engineer",
    "Rig Crew":                "analyst",
}


def chat_response(message, history):
    if not message.strip():
        yield history, gr.update(), render_hud(_EMPTY_HUD_STATE), gr.update(), "", gr.update()
        return

    query = message.strip()
    t0 = time.time()
    hud = {**_EMPTY_HUD_STATE, "q_status": "active", "q_detail": query[:40],
           "iadc_status": "active", "iadc_detail": "Mandatory search…",
           "tools_used": set()}

    history = list(history) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "⟳ Initializing Odin…"},
    ]
    chart_paths = []
    base_figures = Path(__file__).resolve().parents[2] / "outputs" / "figures"

    yield history, gr.update(value=""), render_hud(hud), gr.update(), "", gr.update(value="", visible=True)

    logs = ("<details open><summary style='cursor:pointer;color:#64748b;font-size:0.82em;"
            "user-select:none;padding:4px 0'>⚙️ Thinking Process</summary>"
            "<ul style='list-style:none;padding:2px 0 0;margin:0;font-family:monospace;font-size:0.79em;color:#475569'>")
    step_log = []      # High-level event log (always captured)
    verbose_log = ""   # Full CrewAI stdout transcript (set before final_answer by orchestrator)

    context_text = [f"{m['role'].upper()}: {m['content']}" for m in history[:-2]]

    for event in run_pipeline(query, chat_history=context_text):
        hud["elapsed"] = time.time() - t0

        if event["event"] == "log":
            name   = event.get("name", "")
            status = event.get("status", "")
            detail = event.get("detail", "")
            icon   = event.get("icon", "•")
            is_dia = event.get("is_dialogue", False)
            ts     = time.strftime("%H:%M:%S", time.localtime(event["time"]))

            # ── Parse chart paths from tool output in real-time ──
            for line in (detail + " " + status).split("\n"):
                if "chart saved to:" in line.lower() or "interactive chart saved to:" in line.lower():
                    for part in line.split():
                        if part.endswith((".html", ".png")) and "/" in part:
                            if part not in chart_paths:
                                chart_paths.append(part)

            # Tool tracking
            for kw, label in _TOOL_KEYWORDS:
                if kw in status or kw in detail:
                    hud["tools_used"].add(label)

            if "Action:" in status:
                hud["action_count"] = hud.get("action_count", 0) + 1

            # HUD state machine
            if name == "Classifier":
                hud["q_status"] = "complete"
            elif "IADC" in status or "IADC" in detail:
                hud["iadc_status"] = "active"
                hud["iadc_detail"] = "Searching definitions…"
            elif "Volve" in status or "VolveHistory" in status or "Volve" in detail:
                hud["volve_status"] = "active"
                hud["volve_detail"] = "Searching 23K chunks…"
            elif name == "Complete":
                for k in ["q", "iadc", "volve", "analyst", "historian", "auditor", "engineer"]:
                    hud[f"{k}_status"] = "complete"
                hud["s_status"] = "active"; hud["s_detail"] = "Synthesizing…"

            if name in _AGENT_MAP:
                pfx = _AGENT_MAP[name]
                if pfx == "analyst":
                    if hud["iadc_status"] == "active":  hud["iadc_status"]  = "complete"
                    if hud["volve_status"] == "active": hud["volve_status"] = "complete"
                if "Handoff Complete" in status:
                    hud[f"{pfx}_status"] = "complete"; hud[f"{pfx}_detail"] = "Done ✓"
                else:
                    hud[f"{pfx}_status"] = "delegating" if is_dia else "active"
                    hud[f"{pfx}_detail"] = status[:36]

            # ── Live feed entry ──
            bclass, _ = _AGENT_BADGE.get(name, ("system", "•"))
            badge_short = {"Drilling Data Analyst": "ANALYST", "Volve Campaign Historian": "HIST",
                           "Rig Operations Auditor": "AUDIT", "Lead Drilling Engineer": "ODIN",
                           "Rate Limiter": "RATE", "Router": "ROUTE"}.get(name, name[:6].upper())
            if "Action:" in status:
                tool_name = status.replace("Action:", "").strip()
                inp = detail.replace("Input:", "").strip()[:50]
                feed_text = f"<b>{tool_name}</b> ← {inp}" if inp else f"<b>{tool_name}</b>"
                feed_type = "tool"
                badge_short = tool_name[:12]
                bclass = "tool-badge"
            elif "Thought" in status:
                feed_text = detail[:85]
                feed_type = "thought"
            elif "Handoff" in status or is_dia:
                feed_text = detail[:85]
                feed_type = "handoff"
            elif name in ("Rate Limiter", "Router"):
                feed_text = status[:85]
                feed_type = "system"
            else:
                feed_text = None  # skip low-signal events

            if feed_text:
                full_text = detail if "Thought" in status else (detail or status)
                hud["live_feed"] = (hud.get("live_feed", []) + [
                    {"badge_class": bclass, "badge": badge_short, "type": feed_type,
                     "text": feed_text[:80], "full_text": full_text}
                ])[-12:]

            # Collapsible log in chat
            if is_dia:
                logs += (f"<li style='margin:5px 0;padding:6px;background:#1e3a8a22;border-left:3px solid #3b82f6;"
                         f"border-radius:4px'>[{ts}] {icon} <b style='color:#93c5fd'>{name}</b>: "
                         f"<span style='color:#64748b'>{status}</span><br/>"
                         f"<span style='color:#475569;font-style:italic'>{detail[:120]}</span></li>")
            else:
                det = f" <i style='color:#334155'>{detail[:80]}</i>" if detail else ""
                logs += f"<li style='margin:2px 0'>[{ts}] {icon} <b style='color:#64748b'>{name}</b>: <span style='color:#475569'>{status}</span>{det}</li>"

            # Accumulate rich step log for MD export — use detail_full if available
            detail_full = event.get("detail_full", detail)
            step_log.append(
                f"[{ts}] **{icon} {name}** — {status}" +
                (f"\n\n```\n{detail_full}\n```" if detail_full else "")
            )

            history[-1]["content"] = logs + "</ul></details>"
            yield history, gr.update(), render_hud(hud), gr.update(), "", gr.update()

        elif event["event"] == "verbose_log":
            # Full CrewAI terminal transcript — forwarded by orchestrator before final_answer
            # Storing here so it's available when export_payload is built in final_answer handler
            verbose_log = event.get("content", "")

        elif event["event"] == "final_answer":
            elapsed = time.time() - t0
            hud["elapsed"] = elapsed
            hud["s_status"] = "complete"
            hud["s_detail"] = f"Done in {elapsed:.1f}s"

            # Collect charts: sweep figures dir for files created during THIS query (since t0)
            # Using t0 as cutoff prevents old charts from previous queries bleeding in
            if base_figures.exists():
                for ext in ["*.html", "*.png"]:
                    for p in sorted(base_figures.glob(ext), key=lambda x: x.stat().st_mtime, reverse=True):
                        if p.stat().st_mtime >= t0 - 5:  # 5s grace for slow saves
                            sp = str(p.absolute())
                            if sp not in chart_paths:
                                chart_paths.append(sp)

            answer = event.get("answer", "")
            confidence, conf_reason = extract_confidence_with_reason(answer)
            # Fallback: infer confidence from data sources used if LLM didn't state it
            if not confidence:
                data_tools = hud["tools_used"] & {"DDR", "WITSML", "EDM"}
                if len(data_tools) >= 3:
                    confidence, conf_reason = "HIGH", "DDR + WITSML + EDM all queried and correlated."
                elif len(data_tools) == 2:
                    confidence, conf_reason = "MEDIUM", f"Two sources used: {', '.join(sorted(data_tools))}."
                elif data_tools:
                    confidence, conf_reason = "MEDIUM", f"Single data source: {list(data_tools)[0]}."
                else:
                    confidence, conf_reason = "MEDIUM", "Knowledge base (IADC / Volve corpus) consulted."
            meta_html = render_metadata(hud["tools_used"], confidence, elapsed, conf_reason)

            # Embed charts inline in the chat message
            chart_md, chart_html_fb = _embed_charts_inline(chart_paths)

            closed_logs = logs.replace("<details open>", "<details>") + "</ul></details>"
            # Meta chips embedded directly at bottom of bot message — no separate bar
            history[-1]["content"] = closed_logs + "\n\n" + answer + chart_md + meta_html

            # Pack export state with full step_log (not HTML-stripped log)
            tools_list = sorted(hud["tools_used"])
            export_payload = {
                "answer": answer, "confidence": confidence or "",
                "confidence_reason": conf_reason,
                "tools": tools_list, "elapsed": elapsed,
                "step_log": step_log,       # high-level event log
                "verbose_log": verbose_log, # full CrewAI stdout transcript
                "chart_paths": chart_paths,
            }

            # Pre-compute download link so the Export button fires instantly (no queue wait)
            _export_html_update = gr.update()
            try:
                import urllib.parse as _ul
                _ep = export_answer(export_payload)
                if _ep:
                    _enc = _ul.quote(open(_ep, encoding="utf-8").read(), safe="")
                    # Hidden anchor — Export MD button JS clicks it, no visible link shown
                    _export_html_update = gr.update(visible=True, value=(
                        f'<a id="odin-dl" href="data:text/markdown;charset=utf-8,{_enc}" download="odin_report.md"></a>'
                    ))
            except Exception:
                pass

            yield history, gr.update(value=""), render_hud(hud), gr.update(value=chart_html_fb), export_payload, _export_html_update

        elif event["event"] == "error":
            elapsed = time.time() - t0
            err_msg = event.get("message", "Unknown error")
            hud["s_status"] = "complete"
            hud["s_detail"] = "Failed"
            # Still sweep for charts — some may have been generated before the failure
            if base_figures.exists():
                for ext in ["*.html", "*.png"]:
                    for p in sorted(base_figures.glob(ext), key=lambda x: x.stat().st_mtime, reverse=True):
                        if time.time() - p.stat().st_mtime < 600:
                            sp = str(p.absolute())
                            if sp not in chart_paths:
                                chart_paths.append(sp)
            closed_logs = logs.replace("<details open>", "<details>") + "</ul></details>"
            error_block = (
                f"\n\n> ⚠️ **Agent Error** — `{err_msg[:200]}`\n\n"
                "_The crew encountered an error. This is usually a Gemini rate limit (429) "
                "or max_iter exceeded — please wait 30–60 seconds and try again._"
            )
            # Any charts generated before the failure — show as HTML fallback
            _, chart_html_fb = _embed_charts_inline(chart_paths)
            history[-1]["content"] = closed_logs + error_block
            yield history, gr.update(value=""), render_hud(hud), gr.update(value=chart_html_fb), None, gr.update(value="", visible=True)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """Minimal HTML → plain text for .md export."""
    import re as _re
    text = _re.sub(r'<br\s*/?>', '\n', html)
    text = _re.sub(r'<li[^>]*>', '• ', text)
    text = _re.sub(r'<[^>]+>', '', text)
    return text.strip()


def export_answer(payload):
    """Generate a rich .md report from the export payload dict."""
    if not payload:
        return None
    if isinstance(payload, str):
        # Legacy fallback: just the answer string
        payload = {"answer": payload, "confidence": "", "tools": [], "elapsed": 0,
                   "confidence_reason": "", "log_html": "", "chart_paths": []}

    answer      = payload.get("answer", "")
    confidence  = payload.get("confidence", "")
    conf_reason = payload.get("confidence_reason", "")
    tools       = payload.get("tools", [])
    elapsed     = payload.get("elapsed", 0)
    step_log    = payload.get("step_log", [])
    verbose_log = payload.get("verbose_log", "")
    chart_paths = payload.get("chart_paths", [])

    if not answer.strip():
        return None

    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# ⚡ Odin — Drilling Intelligence Report",
        "_SPE GCS 2026 ML Challenge · Volve Field Dataset_",
        f"\n**Generated:** {ts}",
    ]
    if elapsed:
        lines.append(f"**Response time:** {elapsed:.0f}s")
    if tools:
        lines.append(f"**Data sources:** {', '.join(tools)}")
    if confidence:
        lines.append(f"**Confidence:** {confidence}")
        if conf_reason:
            lines.append(f"> {conf_reason[:300]}")

    lines += ["", "---", "", "## Analysis", "", answer]

    if chart_paths:
        lines += ["", "---", "", "## Charts Generated", ""]
        for p in chart_paths:
            lines.append(f"- `{p}`")

    # Full agent transcript: prefer verbose_log (complete stdout) over step_log (event summaries)
    if verbose_log.strip():
        # Strip ANSI colour codes that CrewAI/Rich outputs
        import re as _re2
        clean = _re2.sub(r'\x1b\[[0-9;]*m', '', verbose_log)
        lines += ["", "---", "", "## Full Agent Transcript", "", "```", clean.strip(), "```"]
    elif step_log:
        lines += ["", "---", "", "## Agent Interaction Log", ""]
        lines += step_log

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".md", mode="w",
        encoding="utf-8", prefix="odin_report_"
    )
    tmp.write("\n".join(lines))
    tmp.close()
    return tmp.name


def _embed_charts_inline(chart_paths: list):
    """
    Embed all charts directly in the chat message as HTML.
    Priority: interactive HTML srcdoc iframe > static PNG base64.
    Returns (inline_html: str, "")  — second value kept for API compat.
    """
    import base64 as _b64
    parts = []
    stems_done = set()

    def _chart_label(stem):
        return stem.replace("_", " ").title()

    def _wrap(name, inner):
        return (
            f'<div style="margin:18px 0 10px">'
            f'<div style="color:#10b981;font-size:0.66em;font-family:\'Share Tech Mono\',monospace;'
            f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;'
            f'display:flex;align-items:center;gap:6px">'
            f'<span style="opacity:.5">▬</span> {name}</div>'
            f'{inner}</div>'
        )

    # Build a stem → {html, png} map so we can pick HTML first
    by_stem: dict = {}
    for cp in chart_paths:
        p = Path(cp)
        if p.exists():
            by_stem.setdefault(p.stem, {})[p.suffix] = p

    for stem, files in by_stem.items():
        if stem in stems_done:
            continue
        name = _chart_label(stem)
        if ".html" in files:
            stems_done.add(stem)
            try:
                # Use base64 data URI — avoids ALL newline/quote escaping issues with srcdoc
                raw = files[".html"].read_bytes()
                b64_html = _b64.b64encode(raw).decode()
                inner = (
                    f'<div style="border-radius:6px;border:1px solid #1e293b;overflow:hidden">'
                    f'<iframe src="data:text/html;base64,{b64_html}" width="100%" height="480" '
                    f'frameborder="0" style="display:block;background:#030712" sandbox="allow-scripts"></iframe></div>'
                )
                parts.append(_wrap(name, inner))
            except Exception:
                pass
        elif ".png" in files:
            stems_done.add(stem)
            try:
                b64 = _b64.b64encode(files[".png"].read_bytes()).decode()
                inner = (
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="width:100%;border-radius:6px;border:1px solid #1e293b;display:block"/>'
                )
                parts.append(_wrap(name, inner))
            except Exception:
                pass

    return "".join(parts), ""  # second value empty — all charts are now inline


def clear_session():
    return ([], gr.update(value=""),
            render_hud(_EMPTY_HUD_STATE), gr.update(value=""),
            gr.update(value=""), None)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD APP
# ─────────────────────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="Odin — Drilling Intelligence") as app:

        answer_state = gr.State(None)  # holds export payload dict

        # ── Header ──────────────────────────────────────────────────────────
        with gr.Row(elem_id="odin-header"):
            gr.HTML(
                '<div class="odin-logo-wrap">'
                  '<span class="odin-rune">ODIN</span>'
                  '<div class="odin-divider"></div>'
                  '<div class="odin-wordmark">'
                    '<strong>Drilling Intelligence System</strong><br>'
                    'SPE GCS 2026 · Volve Field'
                  '</div>'
                '</div>'
                '<div class="odin-stats">'
                  '<div class="odin-stat"><span>23</span>Wells</div>'
                  '<div class="odin-stat"><span>32K+</span>DDR Records</div>'
                  '<div class="odin-stat"><span>55K+</span>WITSML Rows</div>'
                '</div>'
            )
            clear_btn  = gr.Button("Clear",     size="sm", variant="secondary", min_width=70)
            export_btn = gr.Button("Export MD", size="sm", variant="primary",   min_width=100)

        # ── Main Content ─────────────────────────────────────────────────────
        with gr.Row():
            # ── LEFT: Chat ───────────────────────────────────────────────────
            with gr.Column(scale=7, elem_id="chat-col"):
                chatbot = gr.Chatbot(
                    value=[],
                    show_label=False,
                    elem_classes=["chatbot-wrap"],
                    height=560,
                    render_markdown=True,
                    buttons=["copy"],
                    sanitize_html=False,
                )
                # Inline chart area: HTML-only charts (no PNG) fall back here
                chart_area  = gr.HTML(value="", elem_id="chart-area")
                export_file = gr.HTML(value="", visible=True, elem_id="export-file")
                with gr.Row(elem_id="input-zone"):
                    msg_input = gr.Textbox(
                        show_label=False,
                        placeholder="Ask about drilling phases, NPT, ROP, BHA performance, or well comparisons…",
                        scale=9, lines=1, max_lines=4, elem_id="msg-input",
                    )
                    send_btn = gr.Button("Send ⚡", variant="primary", scale=1,
                                         min_width=90, elem_id="send-btn")

            # ── RIGHT: Tabs ───────────────────────────────────────────────────
            with gr.Column(scale=3, elem_id="right-panel"):
                with gr.Tabs():

                    # ── Tab 1: Suggested Prompts ──────────────────────────────
                    with gr.TabItem("💡 Prompts", id="tab-prompts"):
                        p_buttons = []
                        with gr.Column(elem_classes=["prompts-scroll"]):
                            gr.HTML('<div class="prompt-hint">// SELECT QUERY · PRESS SEND ⚡</div>')
                            for p in SUGGESTED_PROMPTS:
                                btn = gr.Button(
                                    value=p, size="sm",
                                    variant="secondary",
                                    elem_classes=["p-btn"],
                                )
                                p_buttons.append((btn, p))

                    # ── Tab 2: Agent HUD ──────────────────────────────────────
                    with gr.TabItem("🛰️ HUD", id="tab-hud"):
                        hud_html = gr.HTML(value=render_hud(_EMPTY_HUD_STATE))

        # ── Outputs list (order must match generator yields) ─────────────────
        _outs = [chatbot, msg_input, hud_html, chart_area, answer_state, export_file]

        # ── Event Wiring ──────────────────────────────────────────────────────
        send_btn.click(fn=chat_response, inputs=[msg_input, chatbot], outputs=_outs)
        msg_input.submit(fn=chat_response, inputs=[msg_input, chatbot], outputs=_outs)

        # Prompt buttons: click → fill textbox
        for btn, p_text in p_buttons:
            btn.click(fn=lambda pt=p_text: pt, inputs=[], outputs=[msg_input])

        # Clear — also wipe chart area and export link
        def _clear():
            return ([], gr.update(value=""), render_hud(_EMPTY_HUD_STATE),
                    gr.update(value=""),
                    gr.update(value="", visible=True), None)
        clear_btn.click(fn=_clear, inputs=[],
                        outputs=[chatbot, msg_input, hud_html, chart_area, export_file, answer_state])

        # Export — JS-only click: the download link is pre-rendered when the answer arrives.
        # No Python fn needed, no queue, fires instantly.
        export_btn.click(
            fn=None, inputs=[], outputs=[],
            js="() => { const a = document.getElementById('odin-dl'); if(a) a.click(); else alert('Run a query first to generate the report.'); }"
        )

    return app


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_proj_dir = Path(__file__).resolve().parents[2]
    figures_dir   = base_proj_dir / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[str(figures_dir)],
        theme=theme,
        css=CUSTOM_CSS,
    )
