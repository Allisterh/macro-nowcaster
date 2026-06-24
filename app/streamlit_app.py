"""Streamlit frontend.

Consumes the FastAPI service when MN_API_URL is set, otherwise builds the
artifact locally (so the demo works standalone). Renders the gauge, composite
index, recession probabilities, contributions, drift table, and a memo button.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Macro Nowcaster", layout="wide")
API = os.environ.get("MN_API_URL", "").rstrip("/")


@st.cache_data(ttl=1800, show_spinner=True)
def load():
    if API:
        s = requests.get(f"{API}/nowcast", timeout=30).json()
        series = requests.get(f"{API}/series", timeout=30).json()
        contrib = requests.get(f"{API}/contributions", timeout=30).json()
        drift = requests.get(f"{API}/drift", timeout=30).json()
        return s, series, contrib, drift
    # local fallback
    from macro_nowcaster.pipeline import build_artifact

    art = build_artifact(persist=False)
    s = art.summary()
    comp = art.activity.factor
    series = {
        "dates": [d.strftime("%Y-%m-%d") for d in comp.index],
        "composite": [float(v) for v in comp.values],
        "nowcast_recprob": [float(v) for v in art.nowcast.prob.reindex(comp.index).values],
        "lead_recprob": [float(v) for v in art.leading.prob.reindex(comp.index).values],
    }
    contrib = {"indicator": list(art.contributions.index),
               "contribution": [float(v) for v in art.contributions.values]}
    drift = art.drift.to_dict(orient="records")
    return s, series, contrib, drift


s, series, contrib, drift = load()
dates = pd.to_datetime(series["dates"])

st.title("Macro Nowcasting System")
st.caption(f"As of {s['as_of']}  |  factor method: {s['factor_method']}  |  "
           f"variance explained: {s['var_explained']:.0%}")

c1, c2, c3 = st.columns([1, 1, 1])
c1.metric("Composite activity", f"{s['composite']:+.2f} sd", s["regime"])
c2.metric("Recession prob (now)", f"{s['nowcast_recprob']:.0%}")
c3.metric("GDP nowcast", f"{s['gdp_nowcast']:+.1f}%", f"+/- {s.get('gdp_nowcast_std', 0):.1f}")

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
fc.add_trace(go.Scatter(x=dates, y=series["composite"], line=dict(color="#2166ac", width=2)))
fc.add_hline(y=0, line_dash="dash", line_color="gray")
fc.update_layout(title="Composite Activity Index", height=300)
st.plotly_chart(fc, use_container_width=True)

fp = go.Figure()
fp.add_trace(go.Scatter(x=dates, y=[(v or 0) * 100 for v in (series["nowcast_recprob"] or [])],
                        name="Nowcast", line=dict(color="#b2182b", width=2)))
fp.add_trace(go.Scatter(x=dates, y=[(v or 0) * 100 for v in (series["lead_recprob"] or [])],
                        name="12m ahead", line=dict(color="#ef8a62", width=2, dash="dot")))
fp.add_hline(y=50, line_dash="dash", line_color="gray")
fp.update_layout(title="Recession Probability", yaxis_range=[0, 100], height=320)
st.plotly_chart(fp, use_container_width=True)

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
    if API:
        memo = requests.post(f"{API}/memo", timeout=60).json()["memo"]
    else:
        from macro_nowcaster.llm.memo_agent import MemoContext, generate_memo
        memo, _ = generate_memo(MemoContext(
            as_of=s["as_of"], composite=s["composite"], regime=s["regime"],
            nowcast_recprob=s["nowcast_recprob"], lead_recprob=s["lead_recprob"],
            gdp_nowcast=s["gdp_nowcast"], top_tailwinds=s["top_tailwinds"],
            top_drags=s["top_drags"]))
    st.code(memo)
