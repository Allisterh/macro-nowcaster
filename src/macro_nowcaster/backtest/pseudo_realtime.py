"""Pseudo-real-time evaluation: the honesty proof.

We replay history month by month. At each reference date we reconstruct the panel
using only data knowable then (vintages or publication lags), standardize on an
expanding window, and generate the nowcast as it would have been produced live.
We then compare the real-time nowcast to the final-vintage estimate (revision
cost) and score the real-time recession probability against what actually
happened (honest out-of-sample AUC).

The factor defaults to PCA for speed across hundreds of refits, but ``factor="dfm"``
replays the estimator the live site actually ships (roughly two seconds per month
against milliseconds for PCA). A DFM refit that fails to converge falls back to PCA
for that month and is counted, so the report can say how much of the replay really
was the DFM.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from ..config import Settings
from ..features.transforms import standardized_panel
from ..models.dfm import fit_activity_factor, fit_pca_factor
from ..models.recession import _score as score_clf
from ..models.recession import fit_nowcast

log = logging.getLogger(__name__)


def replay(
    client,
    settings: Settings,
    start: str = "1995-01-01",
    end: str | None = None,
    recognition_lag_months: int = 4,
    factor: str = "pca",
) -> pd.DataFrame:
    """Generate the real-time nowcast at each month-end in the window.

    ``recognition_lag_months`` reflects that recession status is only confirmed
    with a delay, so the real-time recession model is trained on labels lagged by
    this amount to avoid using knowledge that did not yet exist.

    ``factor`` selects the estimator: ``"pca"`` (fast) or ``"dfm"`` (what the live
    pipeline runs, with a per-month PCA fallback on non-convergence).
    """
    end = end or dt.datetime.now(dt.timezone.utc).date().isoformat()
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
        af = fit_activity_factor(z, prefer="dfm") if factor == "dfm" else fit_pca_factor(z)
        composite_now = float(af.factor.iloc[-1])

        usrec = client.get_series_as_of(settings.recession_flag, asof.date())
        prob_now = np.nan
        if usrec is not None and len(usrec) > 60:
            usrec_m = usrec.resample("ME").mean()
            usrec_m = (usrec_m > 0.5).astype(int).shift(0)
            usrec_lagged = usrec_m.iloc[: max(0, len(usrec_m) - recognition_lag_months)]
            slope = z.get("T10Y3M")
            try:
                rm = fit_nowcast(af.factor, slope, usrec_lagged)
                # fit_nowcast re-applies the model to the feature history, so this
                # is the probability for pred_month, not for the last labelled row.
                prob_now = float(rm.prob.iloc[-1])
            except Exception as exc:  # noqa: BLE001
                log.debug("rt recession fit failed at %s: %s", asof.date(), str(exc)[:50])
        records.append(
            {
                "asof": asof,
                "pred_month": af.factor.index[-1],  # month the nowcast describes
                "factor_method": af.method,  # "dfm", or "pca" where the DFM fell back
                "rt_composite": composite_now,
                "rt_recprob": prob_now,
            }
        )
    return pd.DataFrame(records).set_index("asof")


def evaluate(realtime: pd.DataFrame, final_factor: pd.Series, final_usrec: pd.Series) -> dict:
    """Score the real-time series against final data."""
    joined = realtime.join(final_factor.rename("final_composite"))
    joined = joined.dropna(subset=["rt_composite", "final_composite"])
    rev_corr = float(joined["rt_composite"].corr(joined["final_composite"]))
    rev_mae = float((joined["rt_composite"] - joined["final_composite"]).abs().mean())

    out = {
        "n_periods": len(joined),
        "composite_realtime_vs_final_corr": rev_corr,
        "composite_revision_mae": rev_mae,
    }
    rp = realtime["rt_recprob"].dropna()
    if "pred_month" in realtime:
        # score each probability against the month it actually describes
        rp.index = realtime.loc[rp.index, "pred_month"]
        rp = rp[~rp.index.duplicated(keep="last")]
    if len(rp) > 24:
        auc, brier = score_clf(rp, final_usrec.astype(float))
        out["recession_oos_auc"] = auc
        out["recession_oos_brier"] = brier
    return out
