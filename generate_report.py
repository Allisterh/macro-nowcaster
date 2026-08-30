"""Build the static site in docs/ from a full nowcast run.

This is the entry point the scheduled GitHub Action calls. It runs the whole
system end to end - pull the current FRED vintage, fit the mixed-frequency
dynamic factor model, run the recession probits and the GDP nowcast - and writes
plain static files that GitHub Pages serves with no server and no cold start:

    docs/latest.json          headline numbers + the series behind the charts
    docs/chart_activity.html  composite activity index vs CFNAI-MA3
    docs/chart_gdp.html       GDP nowcast history vs realized GDP growth
    docs/chart_recession.html recession probability, coincident and 12m ahead

The FRED key is read from the FRED_API_KEY environment variable (via
``macro_nowcaster.config``); with no key the pipeline falls back to its
deterministic synthetic client, and the page is labelled as such.

    python generate_report.py
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # run from a clone with no editable install

from macro_nowcaster.benchmarks import compare, fetch_benchmarks
from macro_nowcaster.config import get_settings
from macro_nowcaster.data.fred_client import FredClient, get_client
from macro_nowcaster.pipeline import build_artifact

log = logging.getLogger("generate_report")

DOCS = ROOT / "docs"

# Palette: categorical slots 1 and 2 of the validated default (blue, orange),
# which clear the colour-vision-deficiency separation gates as an adjacent pair.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#ffffff"
INK = "#1b1b1a"
MUTED = "#6f6e69"
GRID = "#e8e7e3"
BAND = "rgba(111,110,105,0.12)"  # NBER recession shading

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clean(values) -> list[float | None]:
    """JSON-safe list: NaN and inf become None."""
    out: list[float | None] = []
    for v in values:
        try:
            f = float(v)
            out.append(f if pd.notna(f) and abs(f) != float("inf") else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _read_performance() -> dict:
    """Headline backtest numbers, parsed out of RESULTS.md.

    Reuses the snapshot exporter's parser so the two front ends can never report
    different accuracy figures for the same replay.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from export_snapshot import _read_performance as read

        return read()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read RESULTS.md performance block: %s", str(exc)[:80])
        return {}


def _gdp_growth(client, settings) -> pd.Series | None:
    """Realized GDP as annualized q/q growth (same convention as the pipeline)."""
    g = client.get_series(settings.target_gdp)
    if g is None or g.dropna().empty:
        return None
    g = g.dropna()
    if float(g.iloc[-1]) > 1000:  # a level series, not already a growth rate
        g = (g.pct_change() * 400).dropna()
    return g


def _recession_bands(client, settings, index: pd.DatetimeIndex) -> list[dict]:
    """Shaded shapes for NBER recession episodes, for context behind the lines."""
    try:
        rec = client.get_series(settings.recession_flag)
    except Exception:  # noqa: BLE001
        return []
    if rec is None or rec.empty:
        return []
    rec = (rec.resample("ME").mean() > 0.5).astype(int).reindex(index).fillna(0)
    shapes, start = [], None
    for date, flag in rec.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            shapes.append(_band(start, date))
            start = None
    if start is not None:
        shapes.append(_band(start, rec.index[-1]))
    return shapes


def _band(x0, x1) -> dict:
    return {
        "type": "rect", "xref": "x", "yref": "paper",
        "x0": x0, "x1": x1, "y0": 0, "y1": 1,
        "fillcolor": BAND, "line": {"width": 0}, "layer": "below",
    }


def _end_label(fig: go.Figure, series: pd.Series, fmt: str) -> None:
    """Mark and label the latest point, anchored to the right edge.

    A text trace would stretch the x range to make room for the glyphs; a paper-
    anchored annotation labels the point without moving the axis.
    """
    s = series.dropna()
    if s.empty:
        return
    fig.add_trace(go.Scatter(
        x=[s.index[-1]], y=[s.iloc[-1]], mode="markers", marker={"color": BLUE, "size": 9},
        showlegend=False, hoverinfo="skip"))
    fig.add_annotation(
        xref="paper", x=1, y=float(s.iloc[-1]), text=fmt.format(s.iloc[-1]),
        xanchor="left", xshift=7, showarrow=False, font={"color": INK, "size": 12})


def _xrange(fig: go.Figure) -> list | None:
    """Data-bounded x range: plotly's autorange leaves years of empty space here."""
    xs = [x for t in fig.data if t.x is not None and len(t.x) for x in (min(t.x), max(t.x))]
    if not xs:
        return None
    lo, hi = min(xs), max(xs)
    pad = (hi - lo) / 80
    return [lo - pad, hi + pad]


def _style(fig: go.Figure, title: str, ytitle: str, height: int = 360) -> go.Figure:
    named = sum(1 for t in fig.data if t.showlegend is not False)
    fig.update_layout(
        showlegend=named > 1,
        title={"text": title, "font": {"size": 15, "color": INK}, "x": 0, "xanchor": "left"},
        template="simple_white",
        font={"family": FONT, "size": 12, "color": MUTED},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=height,
        margin={"l": 58, "r": 58, "t": 74, "b": 40},
        hovermode="x unified",
        hoverlabel={"font": {"family": FONT, "size": 12}, "bgcolor": SURFACE},
        legend={"orientation": "h", "y": 1.02, "x": 0, "yanchor": "bottom",
                "bgcolor": "rgba(0,0,0,0)", "font": {"color": MUTED}},
        yaxis_title=ytitle,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID,
                     range=_xrange(fig))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _write(fig: go.Figure, name: str) -> str:
    """Write one self-contained chart page (plotly.js from the CDN)."""
    html = fig.to_html(include_plotlyjs="cdn", full_html=True, config=PLOT_CONFIG,
                       default_width="100%")
    style = f'<body style="margin:0;background:{SURFACE}">'
    html = html.replace("<body>", style) if "<body>" in html else style + html
    (DOCS / name).write_text(html)
    log.info("wrote %s", name)
    return name


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def chart_activity(factor: pd.Series, cfnai: pd.Series | None, bands: list[dict]) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=factor.index, y=factor.values, name="Composite activity index",
        line={"color": BLUE, "width": 2}, hovertemplate="%{y:+.2f} sd<extra></extra>"))
    if cfnai is not None and cfnai.notna().any():
        fig.add_trace(go.Scatter(
            x=cfnai.index, y=cfnai.values, name="CFNAI-MA3 (Chicago Fed)",
            line={"color": ORANGE, "width": 1.6, "dash": "dot"},
            hovertemplate="%{y:+.2f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED, line_width=1)
    fig.update_layout(shapes=bands)
    _end_label(fig, factor, "{:+.2f}")
    return _write(_style(fig, "Composite activity index",
                         "standard deviations from trend"), "chart_activity.html")


def chart_gdp(nowcast_hist: pd.Series, actual: pd.Series | None) -> str:
    fig = go.Figure()
    if actual is not None and not actual.empty:
        fig.add_trace(go.Bar(
            x=actual.index, y=actual.values, name="Realized GDP growth",
            marker={"color": ORANGE, "line": {"color": SURFACE, "width": 1}},
            opacity=0.55, hovertemplate="%{y:+.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=nowcast_hist.index, y=nowcast_hist.values, name="Model nowcast",
        line={"color": BLUE, "width": 2}, hovertemplate="%{y:+.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    _end_label(fig, nowcast_hist, "{:+.1f}%")
    return _write(_style(fig, "GDP nowcast history vs realized growth",
                         "annualized q/q %"), "chart_gdp.html")


def chart_recession(nowcast: pd.Series, leading: pd.Series, bands: list[dict]) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nowcast.index, y=(nowcast * 100).values, name="Recession now",
        line={"color": BLUE, "width": 2}, hovertemplate="%{y:.0f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=leading.index, y=(leading * 100).values, name="Recession within 12 months",
        line={"color": ORANGE, "width": 2, "dash": "dot"},
        hovertemplate="%{y:.0f}%<extra></extra>"))
    fig.add_hline(y=50, line_dash="dash", line_color=MUTED, line_width=1)
    fig.update_layout(shapes=bands, yaxis_range=[0, 100])
    return _write(_style(fig, "Recession probability (shaded: NBER recessions)",
                         "probability, %"), "chart_recession.html")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    DOCS.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    client = get_client(settings)
    source = "fred" if isinstance(client, FredClient) else "synthetic"
    log.info("data source: %s", source)

    art = build_artifact(settings=settings, persist=False)
    s = art.summary()
    comp = art.activity.factor

    bench = fetch_benchmarks(client, comp.index)
    bench_stats = compare(comp, art.nowcast.prob.reindex(comp.index), s["gdp_nowcast"], bench)

    bands = _recession_bands(client, settings, comp.index)
    gdp_hist = art.gdp_midas.fitted.copy()
    gdp_hist.index = gdp_hist.index.to_timestamp(how="end")  # PeriodIndex -> dates
    actual = _gdp_growth(client, settings)

    charts = [
        chart_activity(comp, bench.get("cfnai"), bands),
        chart_gdp(gdp_hist, actual),
        chart_recession(art.nowcast.prob.reindex(comp.index),
                        art.leading.prob.reindex(comp.index), bands),
    ]

    payload = {
        "gdp_nowcast": s["gdp_nowcast"],
        "gdp_nowcast_std": s["gdp_nowcast_std"],
        "recession_prob": s["nowcast_recprob"],
        "recession_prob_12m": s["lead_recprob"],
        "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of": s["as_of"],
        "composite": s["composite"],
        "regime": s["regime"],
        "factor_method": s["factor_method"],
        "var_explained": s["var_explained"],
        "nowcast_auc": s["nowcast_auc"],
        "leading_auc": s["leading_auc"],
        "top_tailwinds": s["top_tailwinds"],
        "top_drags": s["top_drags"],
        "data_source": source,
        "benchmarks": bench_stats,
        "performance": _read_performance(),
        "charts": charts,
        "contributions": {
            "indicator": list(art.contributions.index),
            "contribution": _clean(art.contributions.values),
        },
    }
    (DOCS / "latest.json").write_text(json.dumps(payload, indent=2) + "\n")
    log.info("wrote latest.json (as of %s, source %s)", s["as_of"], source)
    print(json.dumps({k: payload[k] for k in
                      ("gdp_nowcast", "recession_prob", "updated", "data_source")}, indent=2))


if __name__ == "__main__":
    main()
