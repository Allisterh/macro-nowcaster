"""Frequency alignment, stationarity transforms, and standardization.

The z-score function supports two modes:
  * ``full``      standardize over the whole sample (fine for a live snapshot)
  * ``expanding`` standardize using only data up to each point (no look-ahead),
                  which is what the pseudo-real-time backtest must use.
"""
from __future__ import annotations

import pandas as pd

from ..config import Settings


def to_monthly(s: pd.Series) -> pd.Series:
    """Resample any frequency to month-end, averaging within the month."""
    return s.resample("ME").mean()


def apply_transform(s: pd.Series, how: str) -> pd.Series:
    if how == "yoy":
        return s.pct_change(12) * 100.0
    if how == "yoy_diff":
        return s.diff(12)
    if how == "level":
        return s
    raise ValueError(f"unknown transform {how!r}")


def build_feature_panel(raw: dict[str, pd.Series], settings: Settings) -> pd.DataFrame:
    """Raw native-frequency series -> monthly, transformed feature panel."""
    cols = {}
    for code, s in raw.items():
        if code not in settings.indicators:
            continue
        ind = settings.indicators[code]
        cols[code] = apply_transform(to_monthly(s), ind.transform)
    panel = pd.DataFrame(cols)
    panel = panel[panel.index >= settings.start_date]
    return panel


def zscore(panel: pd.DataFrame, mode: str = "full", min_periods: int = 36) -> pd.DataFrame:
    """Standardize each column.

    ``expanding`` uses an expanding mean/std so each row only reflects data
    available up to that date, removing the look-ahead bias that full-sample
    standardization introduces.
    """
    if mode == "full":
        return (panel - panel.mean()) / panel.std(ddof=0)
    if mode == "expanding":
        mean = panel.expanding(min_periods=min_periods).mean()
        std = panel.expanding(min_periods=min_periods).std(ddof=0)
        return (panel - mean) / std
    raise ValueError(f"unknown zscore mode {mode!r}")


def sign_align(z: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Flip signs so that a positive value always means a stronger economy."""
    out = z.copy()
    for code in out.columns:
        out[code] = out[code] * settings.indicators[code].sign
    return out


def standardized_panel(
    raw: dict[str, pd.Series], settings: Settings, mode: str = "full"
) -> pd.DataFrame:
    """Convenience: raw -> transformed -> z-scored -> sign-aligned, ordered."""
    panel = build_feature_panel(raw, settings)
    z = sign_align(zscore(panel, mode=mode), settings)
    ordered = [c for c in settings.ordered_codes() if c in z.columns]
    return z[ordered]
