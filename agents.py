"""
Orchestrator and data-gathering agents.

The orchestrator dispatches three one-shot agents (quant, news, filings)
in parallel, then passes their outputs to a synthesis agent. Tools are
called directly in Python — the LLM is only used for formatting raw data
and for the final synthesis step.
"""

import asyncio
import time
import uuid
import json
from datetime import datetime, timezone

from llm import chat, get_text

import config
from tools import execute
from db import log_tool_call


# ── Formatting prompts ───────────────────────────────────────────────────

QUANT_FMT = """You receive raw market data for a stock. Structure it as JSON:
{"quant_data": {"current_price":..., "week_52_high":..., "week_52_low":...,
  "pe_ratio":..., "forward_pe":..., "ev_ebitda":..., "revenue_ttm_b":...,
  "revenue_growth_pct":..., "analyst_target":..., "analyst_rating":...,
  "market_cap_b":..., "company_name":...},
 "sources": ["Yahoo Finance"]}
Ignore recent_history if present — it is handled separately.
Only use data provided. Do not invent figures."""

NEWS_FMT = """You receive raw news search results. Analyse sentiment and structure as JSON:
{"news_data": {"articles": [{"title":..., "snippet":..., "source":..., "date":...}],
  "overall_sentiment": "bullish|neutral|bearish",
  "key_development": "one sentence"},
 "sources": [...]}
Only use data provided."""

FILINGS_FMT = """You receive raw SEC filing data. Structure as JSON:
{"filings_data": {"revenue":..., "net_income":..., "operating_income":...,
  "total_assets":..., "cash":..., "long_term_debt":..., "company_name":...},
 "sources": ["SEC EDGAR"]}
Only use data provided. All figures in billions USD."""

SYNTH_SYS = """You are a senior equity analyst writing a short research brief.
RULES:
- Every claim must cite a specific source + figure. Never invent numbers.
- Flag contradictions between sources.
- Give each claim a confidence 0.0-1.0 based on source quality.
Output ONLY this JSON:
{
 "ticker":"...", "company_name":"...",
 "summary":"2-3 sentences",
 "sections":[{"title":"...","claims":[{"text":"...","citation":{"source":"...","detail":"..."},"confidence":0.95}]}],
 "overall_sentiment":"bullish|neutral|bearish",
 "key_risks":["...","...","..."],
 "all_sources":["..."]
}"""


# ── Tool caching ─────────────────────────────────────────────────────────

_cache = {}

def _cached_tool(tool_name, inputs, report_id, agent_name):
    """Execute a tool with TTL caching. Logs every call (cached or not)."""
    cache_key = f"{tool_name}:{json.dumps(inputs, sort_keys=True)}"
    now = time.monotonic()

    if cache_key in _cache:
        result, cached_at = _cache[cache_key]
        if now - cached_at < config.CACHE_TTL_SECONDS:
            log_tool_call(report_id, agent_name, tool_name, inputs, result, 0, None)
            return result

    t0 = time.monotonic()
    result = execute(tool_name, inputs)
    ms = int((time.monotonic() - t0) * 1000)
    _cache[cache_key] = (result, now)
    log_tool_call(report_id, agent_name, tool_name, inputs, result, ms,
                  result.get("error") if isinstance(result, dict) else None)
    return result


def _extract_json(text):
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return {"raw_output": text}


def _trim(data, max_chars=3000):
    """Truncate tool output to stay within token budget."""
    s = json.dumps(data, indent=1)
    return s[:max_chars] + "\n...(trimmed)" if len(s) > max_chars else s


# ── One-shot agents ──────────────────────────────────────────────────────

async def _one_shot(agent_name, tool_name, tool_input, format_prompt,
                    report_id, emit=None):
    """Call a tool directly, then pass the result through a cheap model for formatting."""
    if emit:
        await emit({"event": "tool_call", "agent": agent_name,
                     "tool": tool_name, "input": tool_input})

    result = await asyncio.to_thread(
        _cached_tool, tool_name, tool_input, report_id, agent_name
    )

    if emit:
        await emit({"event": "tool_result", "agent": agent_name,
                     "tool": tool_name, "latency_ms": 0,
                     "ok": not (isinstance(result, dict) and result.get("error"))})

    resp = await chat(
        model=config.EVAL_MODEL,
        system=format_prompt,
        messages=[{"role": "user", "content": _trim(result)}],
        max_tokens=1024,
    )
    return _extract_json(get_text(resp))


async def run_quant(ticker, report_id, emit=None):
    """Quant agent. Strips recent_history before the LLM call (raw table data
    doesn't need formatting) and reattaches it for the frontend."""
    tool_input = {"ticker": ticker, "period": "6mo"}

    if emit:
        await emit({"event": "tool_call", "agent": "quant",
                     "tool": "get_price_data", "input": tool_input})

    raw = await asyncio.to_thread(
        _cached_tool, "get_price_data", tool_input, report_id, "quant"
    )

    if emit:
        await emit({"event": "tool_result", "agent": "quant",
                     "tool": "get_price_data", "latency_ms": 0,
                     "ok": not (isinstance(raw, dict) and raw.get("error"))})

    history = raw.pop("recent_history", []) if isinstance(raw, dict) else []

    resp = await chat(
        model=config.EVAL_MODEL,
        system=QUANT_FMT,
        messages=[{"role": "user", "content": _trim(raw)}],
        max_tokens=1024,
    )
    result = _extract_json(get_text(resp))

    if "quant_data" in result:
        result["quant_data"]["recent_history"] = history
    else:
        result["recent_history"] = history
    return result

async def run_news(ticker, report_id, emit=None):
    return await _one_shot(
        "news", "search_news", {"query": f"{ticker} stock news", "max_results": 5},
        NEWS_FMT, report_id, emit,
    )

async def run_filings(ticker, report_id, emit=None):
    return await _one_shot(
        "filings", "search_filings", {"ticker": ticker, "filing_type": "10-Q"},
        FILINGS_FMT, report_id, emit,
    )


# ── Synthesis ─────────────────────────────────────────────────────────────

async def synthesize(ticker, query, quant, news, filings):
    ctx = (f"Ticker: {ticker}\nQuery: {query}\n\n"
           f"QUANT:\n{_trim(quant)}\n\n"
           f"NEWS:\n{_trim(news)}\n\n"
           f"FILINGS:\n{_trim(filings)}\n\n"
           "Write the cited research brief as specified.")
    resp = await chat(model=config.AGENT_MODEL, system=SYNTH_SYS,
                      messages=[{"role": "user", "content": ctx}])
    return _extract_json(get_text(resp))


# ── Orchestrator ──────────────────────────────────────────────────────────

def _ticker_from(query):
    common = {"THE", "AND", "FOR", "ANALYSE", "ANALYZE", "A", "AN", "IS", "OF"}
    for m in __import__("re").findall(r"\b([A-Z]{1,5})\b", query):
        if m not in common:
            return m
    return query.strip().upper().split()[0] if query.strip() else "UNKNOWN"


async def run_research(query, ticker=None, emit=None):
    """Run the full pipeline: 3 agents in parallel, then synthesis + eval."""
    ticker = (ticker or _ticker_from(query)).upper()
    report_id = str(uuid.uuid4())

    if emit:
        await emit({"event": "start", "ticker": ticker})

    quant, news, filings = await asyncio.gather(
        run_quant(ticker, report_id, emit),
        run_news(ticker, report_id, emit),
        run_filings(ticker, report_id, emit),
        return_exceptions=True,
    )

    def safe(r, name):
        return {"error": str(r), "agent": name} if isinstance(r, Exception) else r

    quant, news, filings = safe(quant, "quant"), safe(news, "news"), safe(filings, "filings")

    if emit:
        await emit({"event": "synthesizing"})

    report = await synthesize(ticker, query, quant, news, filings)
    report["ticker"] = report.get("ticker", ticker)
    report["report_id"] = report_id
    report["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Attach raw price history directly to the report for the frontend
    qd = quant.get("quant_data", quant) if isinstance(quant, dict) else {}
    report["recent_history"] = qd.get("recent_history", [])

    return report_id, report
