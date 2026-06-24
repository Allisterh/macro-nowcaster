"""Pseudo-real-time evaluation: the honesty proof.

We replay history month by month. At each reference date we reconstruct the panel
using only data knowable then (vintages or publication lags), standardize on an
expanding window, and generate the nowcast as it would have been produced live.
We then compare the real-time nowcast to the final-vintage estimate (revision
cost) and score the real-time recession probability against what actually
happened (honest out-of-sample AUC).

The factor here uses PCA rather than the DFM purely for speed across hundreds of
refits; swapping in the DFM is a cost-vs-fidelity choice, not a correctness one.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from ..config import Settings
from ..features.transforms import standardized_panel
from ..models.dfm import fit_pca_factor
from ..models.recession import fit_nowcast
from ..models.recession import _score as score_clf

log = logging.getLogger(__name__)


def replay(
    client,
    settings: Settings,
    start: str = "1995-01-01",
    end: str | None = None,
    recognition_lag_months: int = 4,
) -> pd.DataFrame:
    """Generate the real-time nowcast at each month-end in the window.

    ``recognition_lag_months`` reflects that recession status is only confirmed
    with a delay, so the real-time recession model is trained on labels lagged by
    this amount to avoid using knowledge that did not yet exist.
    """
    end = end or dt.date.today().isoformat()
    dates = pd.date_range(start, end, freq="ME")
    records = []
    for asof in dates:
        raw = {c: client.get_series_as_of(c, asof.date()) for c in settings.codes}
        raw = {k: v for k, v in raw.items() if v is not None and not v.empty}
        if len(raw) < 8:
            continue
        z = standardized_panel(raw, settings, mode="expanding")
        z = z.dropna(how="all")
        if len(z) < 48:
            continue
        af = fit_pca_factor(z)
        composite_now = float(af.factor.iloc[-1])

        usrec = client.get_series_as_of(settings.recession_flag, asof.date())
        prob_now = np.nan
        if usrec is not None and len(usrec) > 60:
            usrec_m = usrec.resample("ME").mean()
            usrec_m = (usrec_m > 0.5).astype(int).shift(0)
            usrec_lagged = usrec_m.iloc[: max(0, len(usrec_m) - recognition_lag_months)]
            slope = z["T10Y3M"] if "T10Y3M" in z else None
            try:
                rm = fit_nowcast(af.factor, slope, usrec_lagged)
                prob_now = float(rm.prob.iloc[-1])
            except Exception as exc:  # noqa: BLE001
                log.debug("rt recession fit failed at %s: %s", asof.date(), str(exc)[:50])
        records.append(
            {"asof": asof, "rt_composite": composite_now, "rt_recprob": prob_now}
        )
    return pd.DataFrame(records).set_index("asof")


def evaluate(realtime: pd.DataFrame, final_factor: pd.Series, final_usrec: pd.Series) -> dict:
    """Score the real-time series against final data."""
    joined = realtime.join(final_factor.rename("final_composite"))
    joined = joined.dropna(subset=["rt_composite", "final_composite"])
    rev_corr = float(joined["rt_composite"].corr(joined["final_composite"]))
    rev_mae = float((joined["rt_composite"] - joined["final_composite"]).abs().mean())

    out = {
        "n_periods": int(len(joined)),
        "composite_realtime_vs_final_corr": rev_corr,
        "composite_revision_mae": rev_mae,
    }
    rp = realtime["rt_recprob"].dropna()
    if len(rp) > 24:
        auc, brier = score_clf(rp, final_usrec.astype(float))
        out["recession_oos_auc"] = auc
        out["recession_oos_brier"] = brier
    return out
