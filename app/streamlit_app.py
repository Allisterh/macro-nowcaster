"""Streamlit frontend.

Data source priority:
  1. MN_API_URL set      -> read live from the FastAPI service
  2. app/snapshot.json   -> read a precomputed snapshot (used for free hosting,
                            so the page loads instantly with no model build)
  3. neither             -> build the artifact locally (full standalone demo)

Renders the gauge, composite index, recession probabilities, contributions,
drift table, and a research memo.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Macro Nowcaster", layout="wide")
API = os.environ.get("MN_API_URL", "").rstrip("/")
SNAPSHOT = Path(__file__).parent / "snapshot.json"


@st.cache_data(ttl=1800, show_spinner=True)
def load():
    # 1. live API
    if API:
        s = requests.get(f"{API}/nowcast", timeout=30).json()
        series = requests.get(f"{API}/series", timeout=30).json()
        contrib = requests.get(f"{API}/contributions", timeout=30).json()
        drift = requests.get(f"{API}/drift", timeout=30).json()
        return s, series, contrib, drift, None

    # 2. precomputed snapshot (fast path for free hosting)
    if SNAPSHOT.exists():
        data = json.loads(SNAPSHOT.read_text())
        return (data["summary"], data["series"], data["contrib"],
                data["drift"], data.get("memo"))

    # 3. build locally
    from macro_nowcaster.pipeline import build_artifact

    art = build_artifact(persist=False)
    s = art.summary()
    comp = art.activity.factor
    series = {
        "dates": [d.strftime("%Y-%m-%d") for d in comp.index],
        "composite": [float(v) for v in comp.values],
        "nowcast_recprob": [None if pd.isna(v) else float(v)
                            for v in art.nowcast.prob.reindex(comp.index).values],
        "lead_recprob": [None if pd.isna(v) else float(v)
                         for v in art.leading.prob.reindex(comp.index).values],
    }
    contrib = {"indicator": list(art.contributions.index),
               "contribution": [float(v) for v in art.contributions.values]}
    drift = art.drift.to_dict(orient="records")
    return s, series, contrib, drift, None


def pct(values):
    """Scale a list to percent, treating missing values as 0 for plotting."""
    return [(v or 0) * 100 for v in (values or [])]


s, series, contrib, drift, snapshot_memo = load()
dates = pd.to_datetime(series["dates"])

bench_stats, perf, generated_at = {}, {}, None
if SNAPSHOT.exists() and not API:
    try:
        _snap = json.loads(SNAPSHOT.read_text())
        bench_stats = _snap.get("benchmark_stats", {}) or {}
        perf = _snap.get("performance", {}) or {}
        generated_at = _snap.get("generated_at")
    except Exception:
        pass

st.title("Macro Nowcasting System")
st.caption(f"As of {s['as_of']}  |  factor method: {s['factor_method']}  |  "
           f"variance explained: {s['var_explained']:.0%}")

st.markdown(
    "Official GDP prints four to eight weeks after a quarter closes. This system reads "
    "higher-frequency data (payrolls, jobless claims, the yield curve, credit spreads) to "
    "estimate current-quarter conditions in real time, the same class of problem the Atlanta "
    "Fed's GDPNow and the Chicago Fed's CFNAI are built to solve."
)

c1, c2, c3 = st.columns([1, 1, 1])
c1.metric("Composite activity", f"{s['composite']:+.2f} sd", s["regime"])
c2.metric("Recession prob (now)", f"{s['nowcast_recprob']:.0%}")
c3.metric("GDP nowcast", f"{s['gdp_nowcast']:+.1f}%", f"+/- {s.get('gdp_nowcast_std', 0):.1f}")

if perf:
    st.markdown("**Model performance - out-of-sample backtest**")
    p1, p2, p3 = st.columns(3)
    p1.metric("Recession AUC (OOS)", perf.get("oos_auc", "n/a"))
    p1.caption("0.50 is a coin flip; this separates recession from expansion months well.")
    p2.metric("Brier score (OOS)", perf.get("brier", "n/a"))
    p2.caption("Mean squared probability error; lower is better, 0 is perfect.")
    p3.metric("Real-time vs final", perf.get("rt_final_corr", "n/a"))
    p3.caption(f"Live signal vs revised data. Backtest window: {perf.get('window', 'n/a')}.")

_alerts = sum(1 for r in drift if str(r.get("status")) == "ALERT")
st.info(
    f"Data drift: {_alerts} of {len(drift)} indicators show distributional drift "
    f"(PSI above the 1.0 alert threshold) versus their long history. PSI = Population "
    f"Stability Index, a model-monitoring signal that flags when live data diverges from "
    f"the distribution the model was calibrated on. A persistently elevated count reflects "
    f"genuine post-2020 macro shifts, not a broken model; a sudden jump would be the real "
    f"early warning."
)

if generated_at:
    st.caption(
        f"Data as of {s['as_of']}  ·  snapshot built {generated_at}  ·  "
        f"refreshes daily via GitHub Actions."
    )

with st.expander("Methodology & model details"):
    st.markdown(
        "- **No look-ahead bias.** The backtest reconstructs each month using only data that "
        "had published by then: true ALFRED vintages for revised monthly series, and a "
        "publication-lag proxy for non-revised daily series.\n"
        "- **Composite.** A single common factor from a mixed-frequency dynamic factor model "
        "(statsmodels DynamicFactorMQ, Kalman filter and EM), with a PCA fallback if the DFM "
        "fails to converge. It explains about 41% of the variance across 29 indicators, which "
        "is typical for one factor on a broad macro panel.\n"
        "- **Mixed frequencies.** Daily series (yield curve, VIX, spreads), weekly claims, and "
        "monthly activity series are aligned to month-end; the Kalman filter handles the ragged "
        "edge where the most recent months have not all reported.\n"
        "- **Recession probability.** A coincident probit on the factor and the yield curve, "
        "plus a separate 12-month-ahead probit on the 10y-3m spread.\n"
        "- **Benchmarks.** Pulled live from FRED and compared to CFNAI-MA3, GDPNow, and the "
        "Chauvet-Piger model. Full write-up in METHODOLOGY.md in the repo."
    )

with st.expander("Limitations - what this model is not"):
    st.markdown(
        "- It is a **coincident, short-horizon nowcast**, not a structural or long-range "
        "forecasting model.\n"
        "- The panel is **29 series**, far smaller than CFNAI's 85, so it diverges from the "
        "official index and is noisier.\n"
        "- A real-time recession call is **noisier than a single point estimate suggests**; "
        "the probability path matters more than any one month's number.\n"
        "- The GDP nowcast carries a **wide uncertainty band**; treat it as a direction-and-"
        "magnitude read, not a precise figure."
    )

g = go.Figure(go.Indicator(
    mode="gauge+number", value=s["nowcast_recprob"] * 100, number={"suffix": "%"},
    title={"text": "Recession probability (nowcast)"},
    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2166ac"},
           "steps": [{"range": [0, 33], "color": "#d9f0d3"},
                     {"range": [33, 66], "color": "#fee08b"},
                     {"range": [66, 100], "color": "#fdae61"}],
           "threshold": {"line": {"color": "red", "width": 4}, "value": 50}}))
g.update_layout(height=280, margin=dict(t=50, b=10))
st.plotly_chart(g, use_container_width=True)

fc = go.Figure()
fc.add_trace(go.Scatter(x=dates, y=series["composite"], name="My composite",
                        line=dict(color="#2166ac", width=2)))
if series.get("cfnai"):
    fc.add_trace(go.Scatter(x=dates, y=series["cfnai"], name="CFNAI-MA3 (Chicago Fed)",
                            line=dict(color="#999999", width=1.5, dash="dot")))
fc.add_hline(y=0, line_dash="dash", line_color="gray")
_cfc = bench_stats.get("composite_vs_cfnai_corr")
fc.update_layout(title="Composite Activity Index"
                 + (f"  (corr vs CFNAI-MA3: {_cfc})" if _cfc is not None else ""),
                 height=300, legend=dict(orientation="h", y=1.12),
                 yaxis_title="standard deviations")
st.plotly_chart(fc, use_container_width=True)

fp = go.Figure()
fp.add_trace(go.Scatter(x=dates, y=pct(series["nowcast_recprob"]),
                        name="Nowcast", line=dict(color="#b2182b", width=2)))
fp.add_trace(go.Scatter(x=dates, y=pct(series["lead_recprob"]),
                        name="12m ahead", line=dict(color="#ef8a62", width=2, dash="dot")))
if series.get("recprob_bench"):
    fp.add_trace(go.Scatter(x=dates, y=series["recprob_bench"],
                            name="Chauvet-Piger (FRED)",
                            line=dict(color="#7f7f7f", width=1.5, dash="dot")))
fp.add_hline(y=50, line_dash="dash", line_color="gray")
fp.update_layout(title="Recession Probability", yaxis_range=[0, 100], height=320,
                 legend=dict(orientation="h", y=1.12))
st.plotly_chart(fp, use_container_width=True)

if bench_stats:
    st.subheader("Benchmark comparison (vs public gold standards)")

    def _fmt(v, suffix=""):
        return "n/a" if v is None else f"{v}{suffix}"

    rows = [
        {"Benchmark": "CFNAI-MA3 (Chicago Fed)",
         "Comparison": "correlation with my composite",
         "Value": _fmt(bench_stats.get("composite_vs_cfnai_corr"))},
        {"Benchmark": "Chauvet-Piger smoothed prob (FRED)",
         "Comparison": "correlation with my recession prob",
         "Value": _fmt(bench_stats.get("recprob_vs_chauvetpiger_corr"))},
        {"Benchmark": "GDPNow (Atlanta Fed)",
         "Comparison": "latest reading",
         "Value": _fmt(bench_stats.get("gdpnow_latest"), "%")},
        {"Benchmark": "GDPNow (Atlanta Fed)",
         "Comparison": "my GDP nowcast minus GDPNow",
         "Value": _fmt(bench_stats.get("gdp_nowcast_vs_gdpnow_gap"), " pp")},
    ]
    st.table(pd.DataFrame(rows).set_index("Benchmark"))
    st.caption("My composite is a 29-series dynamic factor model; CFNAI-MA3 is the "
               "Chicago Fed's 85-series PCA, so moderate correlation is expected. "
               "Benchmarks are pulled live from FRED at snapshot build time.")

col_a, col_b = st.columns(2)
with col_a:
    cdf = pd.DataFrame(contrib)
    fb = go.Figure(go.Bar(x=cdf["contribution"], y=cdf["indicator"], orientation="h",
                          marker_color=["#1b7837" if v >= 0 else "#b2182b" for v in cdf["contribution"]]))
    fb.update_layout(title="Indicator Contributions", height=520)
    st.plotly_chart(fb, use_container_width=True)
with col_b:
    st.subheader("Data drift monitor")
    st.dataframe(pd.DataFrame(drift), use_container_width=True, height=460)

if st.button("Generate research memo"):
    if snapshot_memo is not None:
        memo = snapshot_memo
    elif API:
        memo = requests.post(f"{API}/memo", timeout=60).json()["memo"]
    else:
        from macro_nowcaster.llm.memo_agent import MemoContext, generate_memo
        memo, _ = generate_memo(MemoContext(
            as_of=s["as_of"], composite=s["composite"], regime=s["regime"],
            nowcast_recprob=s["nowcast_recprob"], lead_recprob=s["lead_recprob"],
            gdp_nowcast=s["gdp_nowcast"], top_tailwinds=s["top_tailwinds"],
            top_drags=s["top_drags"]))
    st.code(memo)
