"""End-to-end orchestration.

Builds the full macro artifact from data to memo and pickles it for the API and
frontend to consume. This is the single entry point a scheduled job calls.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field

import pandas as pd

from .config import ARTIFACT_PATH, Settings, get_settings
from .data.fred_client import get_client
from .data.validation import check_freshness, validate_panel
from .features.transforms import build_feature_panel, standardized_panel
from .models.dfm import ActivityFactor, contributions, fit_activity_factor
from .models.midas import BridgeGDP, MidasGDP
from .models.recession import RecessionModel, fit_leading, fit_nowcast
from .models.regime import RegimeResult, fit_regimes
from .monitoring.drift import drift_scan

log = logging.getLogger(__name__)


@dataclass
class Artifact:
    as_of: str
    z_panel: pd.DataFrame
    activity: ActivityFactor
    contributions: pd.Series
    nowcast: RecessionModel
    leading: RecessionModel
    regimes: RegimeResult
    gdp_bridge: object
    gdp_midas: object
    drift: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        comp = float(self.activity.factor.iloc[-1])
        state = self.regimes.labels.get(int(self.regimes.states.iloc[-1]), "n/a")
        contrib_named = self.contributions.copy()
        return {
            "as_of": self.as_of,
            "composite": round(comp, 3),
            "regime": state,
            "factor_method": self.activity.method,
            "var_explained": round(self.activity.var_explained, 3),
            "nowcast_recprob": round(float(self.nowcast.prob.iloc[-1]), 3),
            "lead_recprob": round(float(self.leading.prob.iloc[-1]), 3),
            "gdp_nowcast": round(self.gdp_midas.point, 2),
            "gdp_nowcast_std": round(self.gdp_midas.std, 2),
            "top_tailwinds": list(contrib_named.tail(3).index[::-1]),
            "top_drags": list(contrib_named.head(3).index),
            "nowcast_auc": round(self.nowcast.auc, 3),
            "leading_auc": round(self.leading.auc, 3),
        }


def build_artifact(settings: Settings | None = None, persist: bool = True) -> Artifact:
    settings = settings or get_settings()
    client = get_client(settings)

    # 1. Collect current-vintage data and validate.
    raw = {c: client.get_series(c) for c in settings.codes}
    raw = {k: v for k, v in raw.items() if v is not None and not v.empty}
    feat = build_feature_panel(raw, settings)
    warns = validate_panel(feat)
    check_freshness(feat)

    # 2. Standardize (full-sample for the live snapshot) and fit the factor.
    z = standardized_panel(raw, settings, mode="full")
    af = fit_activity_factor(z, prefer="dfm")
    contrib = contributions(af, z).rename(index={c: settings.indicators[c].name for c in z.columns})

    # 3. Recession models.
    usrec = client.get_series(settings.recession_flag).resample("ME").mean()
    usrec = (usrec > 0.5).astype(int)
    slope = feat["T10Y3M"] if "T10Y3M" in feat else None
    nowcast = fit_nowcast(af.factor, slope, usrec)
    leading = fit_leading(slope, usrec, settings.recession_lead_months) if slope is not None else nowcast

    # 4. Regimes.
    regimes = fit_regimes(af.factor)

    # 5. GDP nowcast (bridge + MIDAS).
    gdp = client.get_series(settings.target_gdp)
    if gdp is not None and float(gdp.dropna().iloc[-1]) > 1000:
        gdp = (gdp.pct_change() * 400).dropna()  # annualized q/q % growth

    bridge = BridgeGDP().fit(af.factor, gdp)
    midas = MidasGDP().fit(af.factor, gdp)
    gdp_bridge = bridge.nowcast(af.factor)
    gdp_midas = midas.nowcast(af.factor)

    # 6. Monitoring.
    drift = drift_scan(z)

    art = Artifact(
        as_of=str(z.index[-1].date()),
        z_panel=z,
        activity=af,
        contributions=contrib,
        nowcast=nowcast,
        leading=leading,
        regimes=regimes,
        gdp_bridge=gdp_bridge,
        gdp_midas=gdp_midas,
        drift=drift,
        warnings=warns,
    )
    if persist:
        with open(ARTIFACT_PATH, "wb") as fh:
            pickle.dump(art, fh)
        log.info("artifact written to %s", ARTIFACT_PATH)
    return art


def load_artifact() -> Artifact | None:
    try:
        with open(ARTIFACT_PATH, "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    a = build_artifact()
    import json

    print(json.dumps(a.summary(), indent=2))
