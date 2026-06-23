"""SQLite persistence layer. Three tables: reports, tool_calls, evals."""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "data.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS reports (
                id          TEXT PRIMARY KEY,
                ticker      TEXT NOT NULL,
                query       TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tool_calls (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id   TEXT NOT NULL,
                agent       TEXT NOT NULL,
                tool        TEXT NOT NULL,
                input_json  TEXT NOT NULL,
                output_json TEXT,
                latency_ms  INTEGER,
                error       TEXT,
                called_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evals (
                report_id           TEXT PRIMARY KEY,
                grounding           REAL,
                citation_precision  REAL,
                citation_recall     REAL,
                numeric_accuracy    REAL,
                overall             REAL,
                flagged_json        TEXT,
                scored_at           TEXT NOT NULL
            );
        """)
    print(f"DB ready at {DB_PATH}")


def save_report(report_id, ticker, query, report):
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO reports VALUES (?,?,?,?,?)",
            (report_id, ticker, query, json.dumps(report), _now()),
        )


def log_tool_call(report_id, agent, tool, inp, out, latency_ms, error=None):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO tool_calls
               (report_id, agent, tool, input_json, output_json, latency_ms, error, called_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (report_id, agent, tool, json.dumps(inp),
             json.dumps(out) if out is not None else None,
             latency_ms, error, _now()),
        )


def save_eval(report_id, e):
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO evals
               (report_id, grounding, citation_precision, citation_recall,
                numeric_accuracy, overall, flagged_json, scored_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (report_id, e.get("grounding"), e.get("citation_precision"),
             e.get("citation_recall"), e.get("numeric_accuracy"),
             e.get("overall"), json.dumps(e.get("flagged", [])), _now()),
        )


def list_reports():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT r.id, r.ticker, r.query, r.created_at,
                   e.grounding, e.citation_precision, e.numeric_accuracy, e.overall
            FROM reports r
            LEFT JOIN evals e ON r.id = e.report_id
            ORDER BY r.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_report(report_id):
    with get_db() as conn:
        report = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        if not report:
            return None
        calls = conn.execute(
            "SELECT * FROM tool_calls WHERE report_id=? ORDER BY called_at", (report_id,)
        ).fetchall()
        ev = conn.execute("SELECT * FROM evals WHERE report_id=?", (report_id,)).fetchone()

    def parse(d, *fields):
        d = dict(d)
        for f in fields:
            if d.get(f):
                try:
                    d[f] = json.loads(d[f])
                except Exception:
                    pass
        return d

    return {
        "report": parse(report, "report_json"),
        "tool_calls": [parse(c, "input_json", "output_json") for c in calls],
        "eval": parse(ev, "flagged_json") if ev else None,
    }
