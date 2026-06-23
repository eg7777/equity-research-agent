"""
FastAPI app — serves the API and the single-page frontend.

Run:  python main.py   (or: uvicorn main:app --reload)
Open: http://localhost:8000
"""

import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import init_db, save_report, list_reports, get_report
from agents import run_research
from evaluator import evaluate

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(title="Investment Research", lifespan=lifespan)


class Query(BaseModel):
    query: str
    ticker: str | None = None


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/reports")
async def reports():
    return {"reports": list_reports()}


@app.get("/api/reports/{report_id}")
async def report(report_id: str):
    data = get_report(report_id)
    if not data:
        raise HTTPException(404, "not found")
    return data


@app.get("/api/stream")
async def stream(query: str, ticker: str | None = None):
    """SSE — emits live agent activity, then the final report + eval."""
    q: asyncio.Queue = asyncio.Queue()

    async def emit(event):
        await q.put(event)

    async def runner():
        try:
            report_id, rep = await run_research(query, ticker, emit=emit)
            t = rep.get("ticker", ticker or "?")
            save_report(report_id, t, query, rep)
            await q.put({"event": "report", "report_id": report_id, "report": rep})

            await q.put({"event": "evaluating"})
            data = get_report(report_id)
            ev = await evaluate(report_id, rep, data["tool_calls"])
            await q.put({"event": "eval", "eval": ev})
        except Exception as e:
            await q.put({"event": "error", "message": str(e)})
        finally:
            await q.put(None)

    asyncio.create_task(runner())

    async def gen():
        while True:
            ev = await q.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/research")
async def research(req: Query):
    """Non-streaming version — runs the pipeline and returns everything at once."""
    report_id, rep = await run_research(req.query, req.ticker)
    t = rep.get("ticker", req.ticker or "?")
    save_report(report_id, t, req.query, rep)
    data = get_report(report_id)
    ev = await evaluate(report_id, rep, data["tool_calls"])
    return {"report_id": report_id, "report": rep, "eval": ev}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
