"""
Post-generation evaluation pipeline.

Scores each report on three metrics:
  - Grounding:          LLM-as-judge support score per claim (batched, one API call)
  - Citation precision: verified citations / total citations
  - Citation recall:    cited claims / all claims
  - Numeric accuracy:   report figures matched against raw tool outputs
"""

import re
import json

from llm import chat, get_text
import config
from db import save_eval

JUDGE_SYS = """You fact-check investment research claims against source data.
You will receive multiple claims with their citations. Score EACH ONE.

Return ONLY a JSON array — one object per claim, in the same order:
[
  {"claim_index": 0, "support_score": 0.95, "flag": false},
  {"claim_index": 1, "support_score": 0.3, "flag": true},
  ...
]
support_score: 1.0 = directly supported, 0.0 = unsupported/contradicted.
flag: true if numbers don't match or claim isn't supported."""


async def _judge_batch(claims_with_citations, source_text):
    """Score all claims in a single API call."""
    if not claims_with_citations:
        return []

    lines = []
    for i, c in enumerate(claims_with_citations):
        lines.append(f"[{i}] Claim: \"{c['text']}\"\n    Citation: {json.dumps(c.get('citation', {}))}")

    prompt = (f"Source data:\n{source_text[:3000]}\n\n"
              f"Claims to fact-check:\n" + "\n".join(lines))

    try:
        resp = await chat(model=config.EVAL_MODEL, max_tokens=1024,
                          system=JUDGE_SYS,
                          messages=[{"role": "user", "content": prompt}])
        text = get_text(resp)
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return [{"support_score": 0.5, "flag": False} for _ in claims_with_citations]
    except Exception:
        return [{"support_score": 0.5, "flag": False} for _ in claims_with_citations]


def _numbers(text):
    """Extract all numeric values from a string."""
    out = []
    for m in re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+", text):
        try:
            out.append(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def _numeric_accuracy(report, tool_calls):
    """Compare numbers in the report against raw tool outputs (±1% tolerance)."""
    src = set()
    for tc in tool_calls:
        if tc.get("output_json"):
            src.update(_numbers(json.dumps(tc["output_json"])))
    if not src:
        return None
    rep = set(_numbers(json.dumps(report)))
    if not rep:
        return 1.0
    matched = sum(1 for r in rep if any(abs(r - s) / max(abs(s), 0.01) < 0.01 for s in src))
    return round(matched / len(rep), 3)


async def evaluate(report_id, report, tool_calls):
    """Run all eval metrics and persist results."""
    source_text = "\n".join(json.dumps(tc.get("output_json", ""))[:400] for tc in tool_calls)

    all_claims, cited_claims = [], []
    for section in report.get("sections", []):
        for claim in section.get("claims", []):
            all_claims.append(claim)
            if claim.get("citation"):
                cited_claims.append(claim)

    scores = await _judge_batch(cited_claims, source_text)

    flagged = []
    for claim, score in zip(cited_claims, scores):
        s = score.get("support_score", 0.5)
        if score.get("flag") or s < config.CONFIDENCE_FLAG_THRESHOLD:
            flagged.append({"text": claim["text"], "support_score": s})

    grounding = (round(sum(s.get("support_score", 0) for s in scores) / len(scores), 3)
                 if scores else None)
    verified = sum(1 for s in scores if s.get("support_score", 0) >= config.GROUNDING_PASS_THRESHOLD)
    precision = round(verified / len(scores), 3) if scores else None
    recall = round(len(cited_claims) / len(all_claims), 3) if all_claims else None
    numeric = _numeric_accuracy(report, tool_calls)

    parts = [x for x in (grounding, precision, numeric) if x is not None]
    overall = round(sum(parts) / len(parts), 3) if parts else None

    result = {
        "report_id": report_id, "grounding": grounding,
        "citation_precision": precision, "citation_recall": recall,
        "numeric_accuracy": numeric, "overall": overall, "flagged": flagged,
        "total_claims": len(all_claims),
    }
    save_eval(report_id, result)
    return result
