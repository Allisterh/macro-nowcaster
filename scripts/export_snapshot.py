"""Export a precomputed snapshot to app/snapshot.json for fast, free hosting.

Run this locally. With FRED_API_KEY set it captures the real US economy; with
ANTHROPIC_API_KEY set it also bakes in a written research memo. The deployed
Streamlit app reads this file and renders instantly, with no model build or
network call at visit time, so a free host stays fast and reliable.

    python scripts/export_snapshot.py
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

from macro_nowcaster.pipeline import build_artifact
from macro_nowcaster.llm.memo_agent import MemoContext, generate_memo
from macro_nowcaster.data.fred_client import get_client
from macro_nowcaster.benchmarks import fetch_benchmarks, compare

OUT = Path(__file__).resolve().parents[1] / "app" / "snapshot.json"


def _clean(values):
    """JSON-safe list: NaN/inf become None."""
    out = []
    for v in values:
        try:
            f = float(v)
            out.append(f if pd.notna(f) and abs(f) != float("inf") else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _read_performance() -> dict:
    """Pull the headline backtest numbers out of RESULTS.md so the dashboard can
    show them without re-running the slow replay."""
    p = Path(__file__).resolve().parents[1] / "RESULTS.md"
    if not p.exists():
        return {}
    txt = p.read_text()

    def grab(label):
        m = re.search(rf"{re.escape(label)}\s*:\s*([0-9.]+)", txt)
        return round(float(m.group(1)), 3) if m else None

    out = {
        "oos_auc": grab("out-of-sample recession AUC"),
        "is_auc": grab("in-sample recession AUC"),
        "brier": grab("OOS Brier score"),
        "rt_final_corr": grab("real-time vs final corr"),
    }
    wm = re.search(r"replay window:\s*(\S+)\s+to\s+(\S+)", txt)
    if wm:
        out["window"] = f"{wm.group(1)} to {wm.group(2)}"
    return {k: v for k, v in out.items() if v is not None}


def main() -> None:
    art = build_artifact(persist=False)
    s = art.summary()
    comp = art.activity.factor

    bench = fetch_benchmarks(get_client(), comp.index)
    bench_stats = compare(
        comp, art.nowcast.prob.reindex(comp.index), s["gdp_nowcast"], bench
    )

    series = {
        "dates": [d.strftime("%Y-%m-%d") for d in comp.index],
        "composite": _clean(comp.values),
        "nowcast_recprob": _clean(art.nowcast.prob.reindex(comp.index).values),
        "lead_recprob": _clean(art.leading.prob.reindex(comp.index).values),
        "cfnai": _clean(bench["cfnai"].values),
        "gdpnow": _clean(bench["gdpnow"].values),
        "recprob_bench": _clean(bench["recprob"].values),
    }
    contrib = {
        "indicator": list(art.contributions.index),
        "contribution": _clean(art.contributions.values),
    }
    drift = art.drift.copy()
    drift["psi"] = drift["psi"].map(lambda v: None if pd.isna(v) else round(float(v), 4))
    drift_records = drift.to_dict(orient="records")

    memo, used_llm = generate_memo(MemoContext(
        as_of=s["as_of"], composite=s["composite"], regime=s["regime"],
        nowcast_recprob=s["nowcast_recprob"], lead_recprob=s["lead_recprob"],
        gdp_nowcast=s["gdp_nowcast"], top_tailwinds=s["top_tailwinds"],
        top_drags=s["top_drags"]))

    payload = {
        "summary": s,
        "series": series,
        "contrib": contrib,
        "drift": drift_records,
        "benchmark_stats": bench_stats,
        "performance": _read_performance(),
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "memo": memo,
        "memo_used_llm": used_llm,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"snapshot written to {OUT}")
    print(f"as of {s['as_of']}, memo written by LLM: {used_llm}")


if __name__ == "__main__":
    main()
