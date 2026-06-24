"""FastAPI service.

Serves the precomputed artifact so the frontend is a thin consumer, not the brain.
Endpoints:
  GET /health        liveness + artifact freshness
  GET /nowcast       headline summary (composite, regime, GDP, recession odds)
  GET /series        composite + recession-probability time series
  GET /contributions latest indicator contributions
  GET /drift         monitoring scan
  POST /memo         generate a research memo from the current state
"""
from __future__ import annotations

import datetime as dt
import math

from fastapi import FastAPI, HTTPException

from ..llm.memo_agent import MemoContext, generate_memo
from ..pipeline import build_artifact, load_artifact

app = FastAPI(title="Macro Nowcaster", version="0.1.0")


def _f(v, nd: int = 4):
    """JSON-safe float: NaN/inf become None."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _artifact():
    art = load_artifact()
    if art is None:
        art = build_artifact()
    return art


@app.get("/health")
def health():
    art = load_artifact()
    return {
        "status": "ok",
        "has_artifact": art is not None,
        "as_of": art.as_of if art else None,
        "server_time": dt.datetime.utcnow().isoformat(),
    }


@app.get("/nowcast")
def nowcast():
    return _artifact().summary()


@app.get("/series")
def series():
    art = _artifact()
    comp = art.activity.factor
    out = {"dates": [d.strftime("%Y-%m-%d") for d in comp.index]}
    out["composite"] = [_f(v) for v in comp.values]
    out["nowcast_recprob"] = [_f(v) for v in art.nowcast.prob.reindex(comp.index).values]
    out["lead_recprob"] = [_f(v) for v in art.leading.prob.reindex(comp.index).values]
    return out


@app.get("/contributions")
def contributions():
    c = _artifact().contributions
    return {"indicator": list(c.index), "contribution": [_f(v) for v in c.values]}


@app.get("/drift")
def drift():
    d = _artifact().drift.copy()
    d["psi"] = d["psi"].map(lambda v: _f(v))
    return d.to_dict(orient="records")


@app.post("/memo")
def memo():
    art = _artifact()
    s = art.summary()
    ctx = MemoContext(
        as_of=s["as_of"],
        composite=s["composite"],
        regime=s["regime"],
        nowcast_recprob=s["nowcast_recprob"],
        lead_recprob=s["lead_recprob"],
        gdp_nowcast=s["gdp_nowcast"],
        top_tailwinds=s["top_tailwinds"],
        top_drags=s["top_drags"],
    )
    text, used_llm = generate_memo(ctx)
    return {"memo": text, "used_llm": used_llm}


@app.post("/refresh")
def refresh():
    try:
        art = build_artifact()
        return {"status": "rebuilt", "as_of": art.as_of}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:200]) from exc
