"""Composite activity index via a mixed-frequency dynamic factor model.

This replaces the prototype's PCA-on-zero-filled-data with a proper state space
model. ``DynamicFactorMQ`` (Banbura-Modugno EM) handles ragged edges and missing
data natively through the Kalman filter, which is the academic state of the art
for nowcasting. A PCA estimate is kept as a robust fallback and cross-check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)


@dataclass
class ActivityFactor:
    factor: pd.Series          # standardized composite activity index
    loadings: pd.Series        # per-indicator loading on the factor
    var_explained: float       # mean share of indicator variance from the factor
    method: str                # "dfm" or "pca"
    means: pd.Series           # column means used (for contribution decomposition)


def _orient_and_standardize(factor: pd.Series, panel: pd.DataFrame) -> pd.Series:
    simple = panel.mean(axis=1)
    if factor.corr(simple) < 0:
        factor = -factor
    return (factor - factor.mean()) / factor.std(ddof=0)


def _loadings(panel: pd.DataFrame, factor: pd.Series) -> tuple[pd.Series, float]:
    """Regress each indicator on the factor; slope is the loading, mean R2 the fit."""
    loads, r2s = {}, []
    f = factor.reindex(panel.index)
    for col in panel.columns:
        df = pd.concat([panel[col], f], axis=1).dropna()
        if len(df) < 24:
            loads[col], _r2 = 0.0, 0.0
        else:
            X = sm.add_constant(df.iloc[:, 1])
            res = sm.OLS(df.iloc[:, 0], X).fit()
            loads[col] = float(res.params.iloc[1])
            r2s.append(float(res.rsquared))
    return pd.Series(loads), float(np.mean(r2s)) if r2s else 0.0


def fit_pca_factor(panel_z: pd.DataFrame) -> ActivityFactor:
    """PCA first component on mean-imputed data. Robust fallback path."""
    means = panel_z.mean()
    filled = panel_z.fillna(0.0)
    coverage = panel_z.notna().mean(axis=1)
    mask = coverage >= 0.70
    pca = PCA(n_components=1).fit(filled[mask] if mask.any() else filled)
    raw = pd.Series(pca.transform(filled)[:, 0], index=panel_z.index)
    factor = _orient_and_standardize(raw, panel_z)
    loads, var = _loadings(panel_z, factor)
    return ActivityFactor(factor, loads, var, "pca", means)


def fit_dfm_factor(panel_z: pd.DataFrame, factor_order: int = 2) -> ActivityFactor:
    """Single-factor dynamic factor model via the Kalman filter and EM."""
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    endog = panel_z.copy()
    mod = DynamicFactorMQ(
        endog,
        factors=1,
        factor_orders=factor_order,
        idiosyncratic_ar1=True,
        standardize=True,
    )
    res = mod.fit(disp=False, maxiter=200)
    smoothed = res.factors.smoothed
    raw = smoothed.iloc[:, 0]
    factor = _orient_and_standardize(raw, panel_z)
    loads, var = _loadings(panel_z, factor)
    return ActivityFactor(factor, loads, var, "dfm", panel_z.mean())


def fit_activity_factor(panel_z: pd.DataFrame, prefer: str = "dfm") -> ActivityFactor:
    """Fit the composite. Try the DFM, fall back to PCA on any failure."""
    if prefer == "dfm":
        try:
            return fit_dfm_factor(panel_z)
        except Exception as exc:  # noqa: BLE001
            log.warning("DFM failed (%s); falling back to PCA", str(exc)[:80])
    return fit_pca_factor(panel_z)


def contributions(af: ActivityFactor, panel_z: pd.DataFrame) -> pd.Series:
    """Signed contribution of each indicator to the latest activity reading."""
    latest = panel_z.iloc[-1].fillna(0.0)
    contrib = (latest - af.means.fillna(0.0)) * af.loadings
    denom = contrib.abs().sum()
    return (contrib / denom).sort_values() if denom else contrib.sort_values()
