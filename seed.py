"""
Seed the dashboard with a few reports so the demo isn't empty.
Run once after setup:  python seed.py
"""

import asyncio
from db import init_db, save_report, get_report
from agents import run_research
from evaluator import evaluate

TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]


async def main():
    init_db()
    for t in TICKERS:
        print(f"\nSeeding {t}…")
        try:
            rid, rep = await run_research(f"Analyse {t}", t)
            save_report(rid, t, f"Analyse {t}", rep)
            data = get_report(rid)
            ev = await evaluate(rid, rep, data["tool_calls"])
            print(f"  done — overall {ev.get('overall')}, flags {len(ev.get('flagged', []))}")
        except Exception as e:
            print(f"  skipped {t}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
