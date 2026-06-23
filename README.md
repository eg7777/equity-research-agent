# Research Terminal

**Automated equity research with built-in trust scoring.**

Type a ticker. Three specialist agents — quantitative, news, and filings —
gather data in parallel from Yahoo Finance, Tavily, and SEC EDGAR. A synthesis
agent reconciles their outputs into a structured research brief where every
factual claim carries an inline citation and a confidence score. An evaluation
pipeline then scores the entire report on factual grounding, citation quality,
and numeric accuracy — automatically, with no human in the loop.

Every report ships with its own quality scorecard.

## What it demonstrates

- **Orchestrated multi-agent workflow** — an orchestrator dispatches three
  data-gathering agents in parallel, with fault isolation so one failure
  degrades the report rather than killing it
- **Cost-conscious architecture** — tools run in Python, not through the LLM;
  the expensive model runs once (synthesis), everything else hits the cheap
  tier; prompt caching cuts repeated input-token cost
- **Automated evaluation** — LLM-as-judge grounding scores, citation
  precision/recall, and numeric diff against raw tool outputs, all computed
  post-generation with zero human labelling
- **Full audit trail** — every tool call, every agent decision, every latency
  measurement logged to SQLite and replayable from the UI
- **Provider-agnostic** — swap between Claude, GPT, or local models (Ollama)
  by changing one line in config

## Architecture

```
            ┌──────────────┐
   query →  │ Orchestrator │
            └──────┬───────┘
        ┌──────────┼───────────┐        (parallel via asyncio.gather)
   ┌────▼───┐ ┌────▼────┐ ┌─────▼─────┐
   │ Quant  │ │  News   │ │  Filings  │   tool call + LLM formatting
   └────┬───┘ └────┬────┘ └─────┬─────┘
        └──────────┼────────────┘
            ┌──────▼───────┐
            │  Synthesis   │  → cited report (claims + confidence)
            └──────┬───────┘
            ┌──────▼───────┐
            │  Evaluator   │  → grounding · citation · numeric scores
            └──────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Provider selection, model strings, thresholds |
| `llm.py` | Provider adapter — exposes `chat()` and `get_text()` |
| `tools.py` | Tool implementations and schemas |
| `agents.py` | One-shot agents, orchestrator, synthesis |
| `evaluator.py` | Grounding, citation, and numeric eval metrics |
| `db.py` | SQLite persistence (reports, tool_calls, evals) |
| `main.py` | FastAPI server — API, SSE streaming, serves frontend |
| `static/index.html` | Single-file frontend (precompiled, no build step) |
| `seed.py` | Pre-populate the dashboard with sample reports |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env              # add your ANTHROPIC_API_KEY
python main.py                     # http://localhost:8000
```

To pre-populate the dashboard before a demo:

```bash
python seed.py
```

The Tavily key is optional — without it, news search returns mock data
and the rest of the pipeline still runs.

## Evaluation

After each report, the evaluator runs three checks:

1. **Grounding** — a fast model judges whether each claim's citation
   actually supports it, scored 0–1. All claims are batched into a
   single API call.
2. **Citation precision / recall** — precision is verified citations
   over total; recall is cited claims over all claims.
3. **Numeric accuracy** — numbers in the report are extracted and
   matched against raw tool outputs with ±1% tolerance.

Claims below the confidence threshold are flagged and highlighted in the UI.

## Deploy

Single service. `Procfile` included for Railway/Render:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set `ANTHROPIC_API_KEY` (and optionally `TAVILY_API_KEY`) in the host environment.

## Roadmap

### In progress

**Prompt refinement pipeline** — Extract recurring failure patterns from
flagged claims across reports and use them to iteratively update the
synthesis prompt. High-scoring reports become few-shot examples; common
failure modes become explicit constraints. The eval database already
stores the signal this needs.

**Human review queue** — A review interface where an analyst marks
claims as correct, incorrect, or needs-context. Each judgment is stored
as a preference pair (original vs. corrected), building a dataset that
feeds back into prompt construction over time.

### Planned

**Full-text filing retrieval (RAG)** — The filings agent currently pulls
structured XBRL fields. This misses qualitative context (risk factors,
MD&A, forward guidance). Planned: chunk and embed full 10-Q/10-K text
into a vector store, adding a semantic search tool to the filings agent.

**Multi-user support** — Per-user report history and review tracking.

## Notes

- Not investment advice. A research aid that shows its sources.
- Model strings live in `config.py` — update if the API returns a model-not-found error.
