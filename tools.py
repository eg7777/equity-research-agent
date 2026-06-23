"""Tool implementations and schemas. Each tool returns structured JSON."""

import re
import json
import httpx
from datetime import datetime, timezone, timedelta

import config

# ── Schemas (passed to the Claude/OpenAI API for tool-use if needed) ─────

TOOL_SCHEMAS = [
    {
        "name": "get_price_data",
        "description": "Current price, 52-week range, P/E, revenue, margins, and analyst consensus for a ticker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "e.g. NVDA"},
                "period": {"type": "string", "description": "1mo,3mo,6mo,1y", "default": "6mo"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "search_news",
        "description": "Recent financial news for a company. Returns titles, snippets, sources, dates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_filings",
        "description": "Key financials from the company's latest SEC filing (revenue, income, assets, debt).",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "filing_type": {"type": "string", "default": "10-Q"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression for ratios/growth. e.g. '(150-120)/120*100'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "description": {"type": "string", "description": "what it represents"},
            },
            "required": ["expression", "description"],
        },
    },
]


# ── Implementations ──────────────────────────────────────────────────────

def get_price_data(ticker, period="6mo"):
    try:
        import yfinance as yf
        s = yf.Ticker(ticker)
        info, hist = s.info, s.history(period=period)
        if hist.empty:
            return {"error": f"no price data for {ticker}", "ticker": ticker}
        cur = float(hist["Close"].iloc[-1])
        first = float(hist["Close"].iloc[0])

        recent = hist.tail(10).iloc[::-1]
        recent_history = []
        for date, row in recent.iterrows():
            o, h, l, c, v = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), int(row["Volume"])
            recent_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": v,
                "daily_range": round(h - l, 2),
                "daily_change": round(c - o, 2),
            })

        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "current_price": round(cur, 2),
            "price_change_pct": round((cur - first) / first * 100, 2),
            "recent_history": recent_history,
            "week_52_high": round(info.get("fiftyTwoWeekHigh", 0) or 0, 2),
            "week_52_low": round(info.get("fiftyTwoWeekLow", 0) or 0, 2),
            "market_cap_b": round((info.get("marketCap") or 0) / 1e9, 2),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "revenue_ttm_b": round((info.get("totalRevenue") or 0) / 1e9, 2),
            "revenue_growth_pct": round((info.get("revenueGrowth") or 0) * 100, 2),
            "gross_margin_pct": round((info.get("grossMargins") or 0) * 100, 2),
            "analyst_target": info.get("targetMeanPrice"),
            "analyst_rating": info.get("recommendationKey"),
            "sector": info.get("sector"),
            "source": "Yahoo Finance",
        }
    except ImportError:
        return {"error": "pip install yfinance"}
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def search_news(query, max_results=5):
    if not config.TAVILY_API_KEY:
        return {
            "query": query,
            "results": [{
                "title": f"[mock] news for {query}",
                "snippet": "Add TAVILY_API_KEY to .env for live news.",
                "source": "mock", "date": "", "relevance": 0.5,
            }],
            "source": "mock_fallback",
        }
    try:
        with httpx.Client(timeout=10) as c:
            r = c.post("https://api.tavily.com/search", json={
                "api_key": config.TAVILY_API_KEY,
                "query": query, "search_depth": "basic", "max_results": max_results,
                "include_domains": ["reuters.com", "bloomberg.com", "cnbc.com",
                                    "wsj.com", "ft.com", "marketwatch.com"],
            })
        data = r.json()
        return {
            "query": query,
            "results": [{
                "title": x.get("title", ""),
                "snippet": x.get("content", "")[:350],
                "url": x.get("url", ""),
                "date": x.get("published_date", ""),
                "relevance": round(x.get("score", 0), 3),
            } for x in data.get("results", [])[:max_results]],
            "source": "Tavily",
        }
    except Exception as e:
        return {"error": str(e), "query": query}


def search_filings(ticker, filing_type="10-Q"):
    headers = {"User-Agent": "personal-research-project contact@example.com"}
    try:
        with httpx.Client(timeout=15, headers=headers) as c:
            tk = c.get("https://www.sec.gov/files/company_tickers.json").json()
            cik = None
            for v in tk.values():
                if v["ticker"].upper() == ticker.upper():
                    cik = str(v["cik_str"]).zfill(10)
                    break
            if not cik:
                return {"error": f"no CIK for {ticker}", "ticker": ticker, "source": "SEC EDGAR"}
            facts = c.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").json()

        gaap = facts.get("facts", {}).get("us-gaap", {})

        def latest(concept):
            data = gaap.get(concept, {}).get("units", {}).get("USD", [])
            data = [d for d in data if d.get("val") is not None]
            if not data:
                return None
            d = sorted(data, key=lambda x: x.get("end", ""))[-1]
            return {"value_b": round(d["val"] / 1e9, 3), "period_end": d.get("end")}

        return {
            "ticker": ticker.upper(),
            "cik": cik,
            "company_name": facts.get("entityName", ticker),
            "financials": {
                "revenue": latest("Revenues") or latest("RevenueFromContractWithCustomerExcludingAssessedTax"),
                "net_income": latest("NetIncomeLoss"),
                "operating_income": latest("OperatingIncomeLoss"),
                "total_assets": latest("Assets"),
                "long_term_debt": latest("LongTermDebtNoncurrent") or latest("LongTermDebt"),
                "cash": latest("CashAndCashEquivalentsAtCarryingValue"),
            },
            "source": "SEC EDGAR",
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker, "source": "SEC EDGAR"}


def calculate(expression, description):
    allowed = set("0123456789 +-*/().,%")
    if not all(ch in allowed for ch in expression):
        return {"error": "only basic math allowed", "expression": expression}
    try:
        val = eval(expression, {"__builtins__": {}}, {})
        return {
            "expression": expression,
            "result": round(float(val), 4) if isinstance(val, (int, float)) else val,
            "description": description,
        }
    except Exception as e:
        return {"error": str(e), "expression": expression}


TOOLS = {
    "get_price_data": get_price_data,
    "search_news": search_news,
    "search_filings": search_filings,
    "calculate": calculate,
}


def execute(name, inputs):
    fn = TOOLS.get(name)
    return fn(**inputs) if fn else {"error": f"unknown tool {name}"}
