"""External benchmarks: compare the homemade indices to public gold standards.

Pulls three FRED series that are the official versions of what this project
estimates, so the dashboard can show how closely the homemade lines track them
and, just as important, where they diverge:

  * CFNAI          Chicago Fed National Activity Index    -> composite activity
  * GDPNOW         Atlanta Fed GDPNow                      -> GDP nowcast
  * RECPROUSM156N  Chauvet-Piger smoothed recession prob  -> recession probability

Everything degrades gracefully: a benchmark that can't be fetched becomes an
empty aligned series, and the stats for it become None, so the snapshot and
dashboard never break on a missing benchmark.
"""
from __future__ import annotations

import pandas as pd

BENCHMARK_CODES = {
    "cfnai": "CFNAIMA3",
    "gdpnow": "GDPNOW",
    "recprob": "RECPROUSM156N",
}


def fetch_benchmarks(client, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Fetch each benchmark and align it to the given monthly index."""
    out: dict[str, pd.Series] = {}
    for name, code in BENCHMARK_CODES.items():
        try:
            s = client.get_series(code)
        except Exception:  # noqa: BLE001
            s = None
        if s is None or len(s) == 0:
            out[name] = pd.Series(index=index, dtype=float)
            continue
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        out[name] = s.resample("ME").mean().reindex(index)
    return out


def _corr(a: pd.Series, b: pd.Series) -> float | None:
    if a is None or b is None:
        return None
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 24:
        return None
    return round(float(df.iloc[:, 0].corr(df.iloc[:, 1])), 3)


def compare(composite: pd.Series, recprob: pd.Series, gdp_point: float,
            benchmarks: dict[str, pd.Series]) -> dict:
    """Correlations vs the public benchmarks and the current GDP-nowcast gap."""
    gn = benchmarks.get("gdpnow")
    latest_gn = None
    if gn is not None and gn.dropna().size:
        latest_gn = round(float(gn.dropna().iloc[-1]), 2)
    return {
        "composite_vs_cfnai_corr": _corr(composite, benchmarks.get("cfnai")),
        "recprob_vs_chauvetpiger_corr": _corr(recprob, benchmarks.get("recprob")),
        "gdpnow_latest": latest_gn,
        "gdp_nowcast_vs_gdpnow_gap": (
            round(float(gdp_point) - latest_gn, 2) if latest_gn is not None else None
        ),
    }
