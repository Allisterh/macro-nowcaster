"""Model and data monitoring: make degradation visible.

  * ``population_stability_index`` flags when an indicator's distribution shifts
    away from its training distribution.
  * ``calibration_report`` checks whether the recession probabilities are well
    calibrated (a 30% forecast should verify ~30% of the time).

A senior pipeline surfaces failure modes, not just successes; these emit alerts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    """PSI between a reference window and a recent window. >0.25 is a big shift."""
    ref, cur = reference.dropna(), current.dropna()
    if len(ref) < bins or len(cur) < bins:
        return float("nan")
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    eps = 1e-6
    ref_pct, cur_pct = ref_pct + eps, cur_pct + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_scan(panel: pd.DataFrame, recent_months: int = 12) -> pd.DataFrame:
    """PSI per indicator: recent window vs the rest of history."""
    rows = []
    for col in panel.columns:
        s = panel[col].dropna()
        if len(s) < recent_months * 3:
            continue
        ref, cur = s.iloc[:-recent_months], s.iloc[-recent_months:]
        psi = population_stability_index(ref, cur)
        flag = "ALERT" if (psi == psi and psi > 0.25) else ("watch" if psi > 0.1 else "ok")
        rows.append({"indicator": col, "psi": psi, "status": flag})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


@dataclass
class CalibrationReport:
    brier: float
    table: pd.DataFrame
    well_calibrated: bool


def calibration_report(prob: pd.Series, actual: pd.Series, bins: int = 10) -> CalibrationReport:
    """Reliability table comparing predicted probability to realized frequency."""
    df = pd.concat([prob.rename("p"), actual.rename("y")], axis=1).dropna()
    df["bucket"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    table = df.groupby("bucket", observed=True).agg(
        predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")
    ).dropna()
    brier = float(((df["p"] - df["y"]) ** 2).mean())
    gap = float((table["predicted"] - table["actual"]).abs().mean()) if not table.empty else 1.0
    return CalibrationReport(brier=brier, table=table, well_calibrated=gap < 0.1)
