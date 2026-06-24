"""Data quality gates.

These run after ingestion and before modelling so a malformed FRED response
fails loudly instead of silently poisoning the nowcast. This is the kind of
defensive check that separates a production pipeline from a notebook.
"""
from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd


class DataQualityError(Exception):
    """Raised when the panel fails a hard validation gate."""


def validate_panel(panel: pd.DataFrame, min_coverage: float = 0.3) -> list[str]:
    """Hard checks on the feature panel. Returns a list of soft warnings.

    Hard failures (raise): empty panel, non-monotonic or duplicated index,
    a column that is entirely missing.
    Soft warnings (collect): low coverage columns, all-constant columns.
    """
    if panel is None or panel.empty:
        raise DataQualityError("panel is empty")
    if not panel.index.is_monotonic_increasing:
        raise DataQualityError("panel index is not monotonic increasing")
    if panel.index.has_duplicates:
        raise DataQualityError("panel index has duplicate dates")

    warnings_out: list[str] = []
    for col in panel.columns:
        s = panel[col]
        if s.notna().sum() == 0:
            raise DataQualityError(f"column {col} is entirely missing")
        coverage = s.notna().mean()
        if coverage < min_coverage:
            warnings_out.append(f"{col}: low coverage {coverage:.0%}")
        if s.dropna().nunique() <= 1:
            warnings_out.append(f"{col}: constant series")
    return warnings_out


def check_freshness(panel: pd.DataFrame, max_staleness_days: int = 75) -> None:
    """Warn if the most recent observation is older than expected.

    Monthly data plus publication lag means roughly two months of staleness is
    normal; beyond that something in the pipeline is likely stuck.
    """
    last = panel.dropna(how="all").index.max()
    if pd.isna(last):
        raise DataQualityError("panel has no non-empty rows")
    age = (pd.Timestamp(dt.date.today()) - pd.Timestamp(last)).days
    if age > max_staleness_days:
        warnings.warn(
            f"panel is stale: last observation {last.date()} is {age} days old",
            stacklevel=2,
        )
