"""GDP nowcast from the monthly activity factor.

Two estimators:
  * ``BridgeGDP``  aggregate the monthly factor to quarterly and regress GDP on
                   it (the standard, transparent bridge equation).
  * ``MidasGDP``   unrestricted MIDAS: regress quarterly GDP on the three monthly
                   factor values within the quarter, so an incomplete quarter
                   still produces a nowcast from whatever months have printed.

The bridge gives a clean point estimate; MIDAS exploits within-quarter timing.
Both report an in-sample standard error used for the nowcast uncertainty band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _quarter_index(idx: pd.DatetimeIndex) -> pd.PeriodIndex:
    return idx.to_period("Q")


@dataclass
class GDPNowcast:
    point: float
    std: float
    fitted: pd.Series
    method: str
    r2: float


class BridgeGDP:
    """GDP growth ~ quarterly-averaged activity factor."""

    def __init__(self) -> None:
        self.res = None

    def fit(self, factor: pd.Series, gdp: pd.Series) -> "BridgeGDP":
        fq = factor.groupby(_quarter_index(factor.index)).mean()
        g = gdp.copy()
        g.index = _quarter_index(g.index)
        df = pd.concat([g.rename("gdp"), fq.rename("f")], axis=1).dropna()
        self.res = sm.OLS(df["gdp"], sm.add_constant(df["f"])).fit()
        self._fq = fq
        self._sigma = float(np.sqrt(self.res.mse_resid))
        return self

    def nowcast(self, factor: pd.Series) -> GDPNowcast:
        fq = factor.groupby(_quarter_index(factor.index)).mean()
        x = sm.add_constant(fq.rename("f"), has_constant="add")
        fitted = pd.Series(self.res.predict(x), index=fq.index)
        return GDPNowcast(
            point=float(fitted.iloc[-1]),
            std=self._sigma,
            fitted=fitted,
            method="bridge",
            r2=float(self.res.rsquared),
        )


class MidasGDP:
    """Unrestricted MIDAS: GDP ~ factor month1, month2, month3 of the quarter."""

    def __init__(self) -> None:
        self.res = None

    @staticmethod
    def _monthly_matrix(factor: pd.Series) -> pd.DataFrame:
        q = _quarter_index(factor.index)
        pos = factor.groupby(q).cumcount() + 1  # 1,2,3 within quarter
        df = pd.DataFrame({"q": q, "pos": pos, "val": factor.values})
        df = df[df["pos"] <= 3]
        wide = df.pivot_table(index="q", columns="pos", values="val")
        wide.columns = [f"m{c}" for c in wide.columns]
        return wide

    def fit(self, factor: pd.Series, gdp: pd.Series) -> "MidasGDP":
        X = self._monthly_matrix(factor)
        g = gdp.copy()
        g.index = _quarter_index(g.index)
        df = pd.concat([g.rename("gdp"), X], axis=1).dropna()
        self._cols = [c for c in X.columns]
        self.res = sm.OLS(df["gdp"], sm.add_constant(df[self._cols])).fit()
        self._sigma = float(np.sqrt(self.res.mse_resid))
        return self

    def nowcast(self, factor: pd.Series) -> GDPNowcast:
        X = self._monthly_matrix(factor)
        # forward-fill missing within-quarter months so a partial quarter works
        X = X.ffill(axis=1).bfill(axis=1)
        Xc = sm.add_constant(X[self._cols], has_constant="add")
        fitted = pd.Series(self.res.predict(Xc), index=X.index)
        return GDPNowcast(
            point=float(fitted.iloc[-1]),
            std=self._sigma,
            fitted=fitted,
            method="midas",
            r2=float(self.res.rsquared),
        )
