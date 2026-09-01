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

# --- author / attribution (edit AUTHOR_LINKEDIN with your real URL) ---
AUTHOR_LINKEDIN = "https://www.linkedin.com/in/owenfelaris/"
AUTHOR_GITHUB = "https://github.com/felariop-jpg"
AUTHOR_REPO = "https://github.com/felariop-jpg/macro-nowcaster"


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
st.caption(f"Built by [Owen Felaris]({AUTHOR_LINKEDIN})  ·  finance and "
           f"entrepreneurship, Miami University  ·  [GitHub]({AUTHOR_REPO})")

simple_mode = st.toggle(
    "🔤 Simple explanations",
    value=False,
    help="Switch every label and explanation on this page to plain language, no finance jargon.",
)


def txt(simple: str, technical: str) -> str:
    """Pick the simple or technical copy for a piece of UI text."""
    return simple if simple_mode else technical


st.markdown(
    txt(
        "Official economic growth numbers (GDP) come out weeks after each quarter ends. "
        "This tool reads faster-moving data, like hiring, jobless claims, and interest rates, "
        "to estimate how the economy is doing right now instead of waiting.",
        "Official GDP prints four to eight weeks after a quarter closes. This system reads "
        "higher-frequency data (payrolls, jobless claims, the yield curve, credit spreads) to "
        "estimate current-quarter conditions in real time, the same class of problem the Atlanta "
        "Fed's GDPNow and the Chicago Fed's CFNAI are built to solve.",
    )
)

c1, c2, c3 = st.columns([1, 1, 1])
c1.metric(txt("Economic health score", "Composite activity"), f"{s['composite']:+.2f} sd", s["regime"])
c2.metric(txt("Chance of recession right now", "Recession prob (now)"), f"{s['nowcast_recprob']:.0%}")
c3.metric(txt("Estimated economic growth", "GDP nowcast"),
          f"{s['gdp_nowcast']:+.1f}%", f"+/- {s.get('gdp_nowcast_std', 0):.1f}")

if perf:
    st.markdown(txt("**How accurate has this been?**", "**Model performance - out-of-sample backtest**"))
    p1, p2, p3 = st.columns(3)
    p1.metric(txt("Accuracy score", "Recession AUC (OOS)"), perf.get("oos_auc", "n/a"))
    p1.caption(txt(
        "How well this tells recessions apart from normal times, tested only on data it "
        "hadn't seen. 0.50 is a random guess; higher is better.",
        "0.50 is a coin flip; this separates recession from expansion months well.",
    ))
    p2.metric(txt("Error score", "Brier score (OOS)"), perf.get("brier", "n/a"))
    p2.caption(txt(
        "How far off the predictions were on average. Lower is better, 0 would be perfect.",
        "Mean squared probability error; lower is better, 0 is perfect.",
    ))
    p3.metric(txt("Consistency score", "Real-time vs final"), perf.get("rt_final_corr", "n/a"))
    p3.caption(txt(
        f"How closely the live estimate matches the number once all the data is finalized. "
        f"Tested over: {perf.get('window', 'n/a')}.",
        f"Live signal vs revised data. Backtest window: {perf.get('window', 'n/a')}.",
    ))

_alerts = sum(1 for r in drift if str(r.get("status")) == "ALERT")
st.info(
    txt(
        f"Data check: {_alerts} of {len(drift)} data sources are behaving noticeably "
        f"differently than their own history. That's normal during unusual economic "
        f"periods and doesn't mean the model is broken; a sudden new jump would be the "
        f"real warning sign to watch for.",
        f"Data drift: {_alerts} of {len(drift)} indicators show distributional drift "
        f"(PSI above the 1.0 alert threshold) versus their long history. PSI = Population "
        f"Stability Index, a model-monitoring signal that flags when live data diverges from "
        f"the distribution the model was calibrated on. A persistently elevated count reflects "
        f"genuine post-2020 macro shifts, not a broken model; a sudden jump would be the real "
        f"early warning.",
    )
)

if generated_at:
    st.caption(
        f"Data as of {s['as_of']}  ·  snapshot built {generated_at}  ·  "
        f"this snapshot is frozen; the maintained dashboard is the static site "
        f"at felariop-jpg.github.io/macro-nowcaster, rebuilt daily."
    )

with st.expander(txt("How this works", "Methodology & model details")):
    st.markdown(
        txt(
            "- **No peeking at the future.** Every historical estimate only uses data that "
            "was actually available at that point in time, the same way an investor living "
            "through it would have seen it.\n"
            "- **One score from many indicators.** 30 economic indicators (jobs, spending, "
            "credit, and more) are blended into a single number that captures the overall "
            "trend.\n"
            "- **Handles data that arrives on different schedules.** Some data comes in "
            "daily, some monthly. The model combines it all without waiting for the "
            "slowest source to catch up.\n"
            "- **Recession odds.** A statistical model estimates the chance a recession is "
            "happening right now, plus a separate estimate for the next 12 months.\n"
            "- **Checked against official sources.** Results are compared live against "
            "well-known Federal Reserve indexes, so it's held to a real standard, not "
            "graded on its own curve.",
            "- **No look-ahead bias.** The backtest reconstructs each month using only data that "
            "had published by then: true ALFRED vintages for revised monthly series, and a "
            "publication-lag proxy for non-revised daily series.\n"
            "- **Composite.** A single common factor from a mixed-frequency dynamic factor model "
            "(statsmodels DynamicFactorMQ, Kalman filter and EM), with a PCA fallback if the DFM "
            "fails to converge. It explains about 41% of the variance across 30 indicators, which "
            "is typical for one factor on a broad macro panel.\n"
            "- **Mixed frequencies.** Daily series (yield curve, VIX, spreads), weekly claims, and "
            "monthly activity series are aligned to month-end; the Kalman filter handles the ragged "
            "edge where the most recent months have not all reported.\n"
            "- **Recession probability.** A coincident probit on the factor and the yield curve, "
            "plus a separate 12-month-ahead probit on the 10y-3m spread.\n"
            "- **Benchmarks.** Pulled live from FRED and compared to CFNAI-MA3, GDPNow, and the "
            "Chauvet-Piger model. Full write-up in METHODOLOGY.md in the repo.",
        )
    )

with st.expander(txt("What this can't do", "Limitations - what this model is not")):
    st.markdown(
        txt(
            "- It reads **conditions right now**, not a long-range forecast years out.\n"
            "- It uses **fewer data sources** than the big official indexes, so it can look "
            "noisier and jump around more.\n"
            "- **Any single month can be misleading**; the overall trend across several "
            "months matters more than one reading.\n"
            "- The growth estimate has a **real margin of error**; treat it as a rough "
            "range, not an exact number.",
            "- It is a **coincident, short-horizon nowcast**, not a structural or long-range "
            "forecasting model.\n"
            "- The panel is **30 series**, far smaller than CFNAI's 85, so it diverges from the "
            "official index and is noisier.\n"
            "- A real-time recession call is **noisier than a single point estimate suggests**; "
            "the probability path matters more than any one month's number.\n"
            "- The GDP nowcast carries a **wide uncertainty band**; treat it as a direction-and-"
            "magnitude read, not a precise figure.",
        )
    )

g = go.Figure(go.Indicator(
    mode="gauge+number", value=s["nowcast_recprob"] * 100, number={"suffix": "%"},
    title={"text": txt("Chance of a recession right now", "Recession probability (nowcast)")},
    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2166ac"},
           "steps": [{"range": [0, 33], "color": "#d9f0d3"},
                     {"range": [33, 66], "color": "#fee08b"},
                     {"range": [66, 100], "color": "#fdae61"}],
           "threshold": {"line": {"color": "red", "width": 4}, "value": 50}}))
g.update_layout(height=280, margin=dict(t=50, b=10))
st.plotly_chart(g, use_container_width=True)

fc = go.Figure()
fc.add_trace(go.Scatter(x=dates, y=series["composite"], name=txt("This tool's score", "My composite"),
                        line=dict(color="#2166ac", width=2)))
if series.get("cfnai"):
    fc.add_trace(go.Scatter(x=dates, y=series["cfnai"], name="CFNAI-MA3 (Chicago Fed)",
                            line=dict(color="#999999", width=1.5, dash="dot")))
fc.add_hline(y=0, line_dash="dash", line_color="gray")
_cfc = bench_stats.get("composite_vs_cfnai_corr")
fc.update_layout(
    title=txt("Economic Health Over Time", "Composite Activity Index")
    + (f"  (corr vs CFNAI-MA3: {_cfc})" if _cfc is not None and not simple_mode else ""),
    height=300, legend=dict(orientation="h", y=1.12),
    yaxis_title=txt("above 0 = growing, below 0 = shrinking", "standard deviations"),
)
st.plotly_chart(fc, use_container_width=True)

fp = go.Figure()
fp.add_trace(go.Scatter(x=dates, y=pct(series["nowcast_recprob"]),
                        name=txt("Right now", "Nowcast"), line=dict(color="#b2182b", width=2)))
fp.add_trace(go.Scatter(x=dates, y=pct(series["lead_recprob"]),
                        name=txt("In 12 months", "12m ahead"), line=dict(color="#ef8a62", width=2, dash="dot")))
if series.get("recprob_bench"):
    fp.add_trace(go.Scatter(x=dates, y=series["recprob_bench"],
                            name="Chauvet-Piger (FRED)",
                            line=dict(color="#7f7f7f", width=1.5, dash="dot")))
fp.add_hline(y=50, line_dash="dash", line_color="gray")
fp.update_layout(title=txt("Chance of a Recession", "Recession Probability"),
                 yaxis_range=[0, 100], height=320,
                 legend=dict(orientation="h", y=1.12))
st.plotly_chart(fp, use_container_width=True)

if bench_stats:
    st.subheader(txt("How this compares to official sources", "Benchmark comparison (vs public gold standards)"))

    def _fmt(v, suffix=""):
        return "n/a" if v is None else f"{v}{suffix}"

    rows = [
        {"Benchmark": "CFNAI-MA3 (Chicago Fed)",
         "Comparison": txt("how closely it matches this tool's score", "correlation with my composite"),
         "Value": _fmt(bench_stats.get("composite_vs_cfnai_corr"))},
        {"Benchmark": "Chauvet-Piger smoothed prob (FRED)",
         "Comparison": txt("how closely it matches this tool's recession odds",
                            "correlation with my recession prob"),
         "Value": _fmt(bench_stats.get("recprob_vs_chauvetpiger_corr"))},
        {"Benchmark": "GDPNow (Atlanta Fed)",
         "Comparison": txt("latest official reading", "latest reading"),
         "Value": _fmt(bench_stats.get("gdpnow_latest"), "%")},
        {"Benchmark": "GDPNow (Atlanta Fed)",
         "Comparison": txt("this tool's estimate minus the official one",
                            "my GDP nowcast minus GDPNow"),
         "Value": _fmt(bench_stats.get("gdp_nowcast_vs_gdpnow_gap"), " pp")},
    ]
    st.table(pd.DataFrame(rows).set_index("Benchmark"))
    st.caption(txt(
        "This tool uses fewer data sources than the official Chicago Fed index, so an "
        "exact match isn't expected, just a similar overall trend. Benchmarks are "
        "pulled live from FRED each time the page refreshes.",
        "My composite is a 30-series dynamic factor model; CFNAI-MA3 is the "
        "Chicago Fed's 85-series PCA, so moderate correlation is expected. "
        "Benchmarks are pulled live from FRED at snapshot build time.",
    ))

if series.get("cfnai"):
    _bc = pd.DataFrame(
        {"composite": series["composite"], "cfnai": series["cfnai"]}, index=dates
    ).apply(pd.to_numeric, errors="coerce")

    st.markdown(txt(
        "**Is it tracking well lately?**  This tool's score vs the Chicago Fed's index, last 24 months",
        "**Does it track recently?**  Composite vs CFNAI-MA3, last 24 months",
    ))
    _recent = _bc.tail(24)
    fr = go.Figure()
    fr.add_trace(go.Scatter(x=_recent.index, y=_recent["composite"], name=txt("This tool's score", "My composite"),
                            line=dict(color="#2166ac", width=2)))
    fr.add_trace(go.Scatter(x=_recent.index, y=_recent["cfnai"],
                            name="CFNAI-MA3 (Chicago Fed)",
                            line=dict(color="#999999", width=2, dash="dot")))
    fr.add_hline(y=0, line_dash="dash", line_color="gray")
    fr.update_layout(height=280, yaxis_title=txt("above 0 = growing, below 0 = shrinking", "standard deviations"),
                     legend=dict(orientation="h", y=1.15), margin=dict(t=30))
    st.plotly_chart(fr, use_container_width=True)

    st.markdown(txt(
        "**Is this reliable over time, or just one lucky stretch?**  3-year rolling match with the Chicago Fed's index",
        "**Stable relationship, or one crisis?**  Rolling 36-month correlation",
    ))
    _roll = _bc["composite"].rolling(36).corr(_bc["cfnai"])
    frc = go.Figure()
    frc.add_trace(go.Scatter(x=dates, y=_roll, name=txt("3-year rolling match", "36m rolling corr"),
                             line=dict(color="#1b7837", width=2)))
    _full = bench_stats.get("composite_vs_cfnai_corr")
    if _full is not None:
        frc.add_hline(y=_full, line_dash="dash", line_color="gray",
                      annotation_text=f"full-sample {_full}", annotation_position="top left")
    frc.update_layout(height=260, yaxis_range=[-1, 1],
                      yaxis_title=txt("how closely they match", "correlation"),
                      margin=dict(t=30))
    st.plotly_chart(frc, use_container_width=True)
    st.caption(txt(
        "A steady band above zero means this tool tracks the Chicago Fed's index across "
        "different economic conditions, not just during one crisis.",
        "A stable band above zero means the composite tracks CFNAI-MA3 across "
        "regimes, not only during one recession.",
    ))

col_a, col_b = st.columns(2)
with col_a:
    cdf = pd.DataFrame(contrib)
    fb = go.Figure(go.Bar(x=cdf["contribution"], y=cdf["indicator"], orientation="h",
                          marker_color=["#1b7837" if v >= 0 else "#b2182b" for v in cdf["contribution"]],
                          text=[f"{v:+.2f}" for v in cdf["contribution"]],
                          textposition="outside", cliponaxis=False, textfont=dict(size=10)))
    fb.update_layout(title=txt("What's Driving the Score", "Indicator Contributions"), height=520,
                     xaxis_title=txt("impact on the score", "share of latest move"),
                     uniformtext=dict(mode="show", minsize=8))
    st.plotly_chart(fb, use_container_width=True)
with col_b:
    st.subheader(txt("Data reliability check", "Data drift monitor"))
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

st.divider()
st.markdown(
    f"**Built by Owen Felaris.** Finance and entrepreneurship co-major at Miami "
    f"University's Farmer School of Business, Class of 2028. I built and validated "
    f"this system end to end: data sourcing and point-in-time vintage handling, the "
    f"mixed-frequency dynamic factor model, the out-of-sample backtest, and "
    f"benchmarking against the Chicago and Atlanta Fed indices. Open to finance and "
    f"consulting internships."
)
st.markdown(
    f"[LinkedIn]({AUTHOR_LINKEDIN})  ·  [GitHub]({AUTHOR_GITHUB})  ·  "
    f"[Source code]({AUTHOR_REPO})"
)
