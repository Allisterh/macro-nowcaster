"""Recession probability models.

  * ``fit_nowcast``  coincident probit: P(in recession now | activity factor, slope)
  * ``fit_leading``  yield-curve probit: P(recession within H months | 10Y-3M spread)

Probit is the standard binary tool; a regularized logistic fallback prevents the
pipeline dying on perfect separation (likely because the activity factor is a very
strong coincident classifier).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

log = logging.getLogger(__name__)


@dataclass
class RecessionModel:
    kind: str
    model: object
    prob: pd.Series
    auc: float
    brier: float
    features: list[str]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.kind == "probit":
            return pd.Series(self.model.predict(sm.add_constant(X, has_constant="add")), index=X.index)
        return pd.Series(self.model.predict_proba(X.values)[:, 1], index=X.index)


def _fit(X: pd.DataFrame, y: pd.Series) -> tuple[str, object, pd.Series]:
    Xc = sm.add_constant(X, has_constant="add")
    try:
        m = sm.Probit(y, Xc).fit(disp=0, maxiter=200)
        return "probit", m, pd.Series(m.predict(Xc), index=X.index)
    except Exception as exc:  # noqa: BLE001
        log.info("probit -> logistic fallback: %s", str(exc)[:60])
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X.values, y.values)
        return "logit", lr, pd.Series(lr.predict_proba(X.values)[:, 1], index=X.index)


def _score(prob: pd.Series, y: pd.Series) -> tuple[float, float]:
    y = y.reindex(prob.index).dropna()
    p = prob.reindex(y.index)
    if y.nunique() < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(brier_score_loss(y, p))


def fit_nowcast(factor: pd.Series, slope: pd.Series | None, usrec: pd.Series) -> RecessionModel:
    parts = {"composite": factor}
    if slope is not None:
        parts["slope"] = slope
    df = pd.concat(list(parts.values()) + [usrec.rename("rec")], axis=1).dropna()
    df.columns = list(parts.keys()) + ["rec"]
    feats = list(parts.keys())
    kind, model, prob = _fit(df[feats], df["rec"])
    auc, brier = _score(prob, df["rec"])
    return RecessionModel(kind, model, prob, auc, brier, feats)


def fit_leading(slope: pd.Series, usrec: pd.Series, lead_months: int = 12) -> RecessionModel:
    y = usrec.shift(-lead_months).rename("rec_fwd")
    df = pd.concat([slope.rename("slope"), y], axis=1).dropna()
    kind, model, prob_fit = _fit(df[["slope"]], df["rec_fwd"])
    auc, brier = _score(prob_fit, df["rec_fwd"])
    rm = RecessionModel(kind, model, prob_fit, auc, brier, ["slope"])
    # full-history applied probability (today's reading included)
    rm.prob = rm.predict(slope.dropna().to_frame("slope"))
    return rm
