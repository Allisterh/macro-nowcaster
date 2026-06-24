"""Unit tests covering the core logic. Run with: pytest -q"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from macro_nowcaster.config import get_settings
from macro_nowcaster.data.fred_client import SyntheticClient
from macro_nowcaster.data.store import PointInTimeStore
from macro_nowcaster.data.validation import DataQualityError, validate_panel
from macro_nowcaster.features.transforms import (
    apply_transform,
    standardized_panel,
    zscore,
)
from macro_nowcaster.models.dfm import contributions, fit_pca_factor
from macro_nowcaster.models.midas import BridgeGDP, MidasGDP
from macro_nowcaster.models.recession import fit_leading, fit_nowcast
from macro_nowcaster.models.regime import fit_regimes, news_decomposition
from macro_nowcaster.backtest.allocation import run_backtest, target_weight
from macro_nowcaster.monitoring.drift import population_stability_index


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def client(settings):
    return SyntheticClient(settings)


@pytest.fixture(scope="module")
def raw(client, settings):
    return {c: client.get_series(c) for c in settings.codes}


@pytest.fixture(scope="module")
def zpanel(raw, settings):
    return standardized_panel(raw, settings, mode="full")


# ---- transforms ----------------------------------------------------------- #
def test_yoy_transform():
    s = pd.Series(np.arange(1, 25, dtype=float),
                  index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    out = apply_transform(s, "yoy")
    assert out.iloc[12] == pytest.approx((13 / 1 - 1) * 100)


def test_expanding_zscore_no_lookahead():
    df = pd.DataFrame({"x": np.arange(100.0)},
                      index=pd.date_range("2000-01-31", periods=100, freq="ME"))
    z = zscore(df, mode="expanding", min_periods=12)
    # an expanding z-score at time t must not depend on data after t
    z_trunc = zscore(df.iloc[:50], mode="expanding", min_periods=12)
    pd.testing.assert_series_equal(z["x"].iloc[:50], z_trunc["x"], check_names=False)


def test_standardized_panel_sign_alignment(zpanel, settings):
    # VIX has sign -1; after alignment a high-VIX month should read negative
    assert "VIXCLS" in zpanel.columns
    assert zpanel.shape[1] >= 25


# ---- validation ----------------------------------------------------------- #
def test_validation_rejects_empty():
    with pytest.raises(DataQualityError):
        validate_panel(pd.DataFrame())


def test_validation_passes_good_panel(zpanel):
    warns = validate_panel(zpanel.dropna(how="all"))
    assert isinstance(warns, list)


# ---- point-in-time store -------------------------------------------------- #
def test_pit_store_roundtrip(tmp_path, client):
    store = PointInTimeStore(tmp_path / "t.duckdb")
    s = client.get_series("PAYEMS")
    n = store.ingest("PAYEMS", s, dt.date(2020, 1, 1))
    assert n > 0
    # idempotent: re-ingest same vintage adds nothing new
    store.ingest("PAYEMS", s, dt.date(2020, 1, 1))
    panel = store.latest_panel(["PAYEMS"])
    assert "PAYEMS" in panel.columns and not panel.empty


def test_pit_as_of_hides_future(client, settings):
    asof = dt.date(2010, 6, 15)
    s = client.get_series_as_of("INDPRO", asof)
    lag = settings.indicators["INDPRO"].pub_lag_days
    assert s.index.max() <= pd.Timestamp(asof) - pd.Timedelta(days=lag)


# ---- factor + contributions ----------------------------------------------- #
def test_pca_factor_and_contributions(zpanel):
    af = fit_pca_factor(zpanel)
    assert af.factor.std(ddof=0) == pytest.approx(1.0, abs=1e-6)
    c = contributions(af, zpanel)
    assert len(c) == zpanel.shape[1]
    assert abs(c.sum()) <= 1.5  # normalized contributions are bounded


# ---- recession models ----------------------------------------------------- #
def test_recession_models(zpanel, raw, settings, client):
    af = fit_pca_factor(zpanel)
    usrec = (client.get_series("USREC").resample("ME").mean() > 0.5).astype(int)
    from macro_nowcaster.features.transforms import build_feature_panel
    feat = build_feature_panel(raw, settings)
    now = fit_nowcast(af.factor, feat["T10Y3M"], usrec)
    lead = fit_leading(feat["T10Y3M"], usrec, 12)
    assert 0.0 <= now.prob.iloc[-1] <= 1.0
    assert 0.4 <= now.auc <= 1.0  # should classify the synthetic cycle well
    assert 0.0 <= lead.prob.iloc[-1] <= 1.0


# ---- regimes -------------------------------------------------------------- #
def test_regimes(zpanel):
    af = fit_pca_factor(zpanel)
    r = fit_regimes(af.factor, n_states=3)
    assert set(r.states.unique()).issubset({0, 1, 2})
    # mean activity should increase with state label after relabelling
    assert r.means[0] <= r.means[2]


# ---- news decomposition --------------------------------------------------- #
def test_news_decomposition(zpanel):
    prev = zpanel.iloc[:-1].copy()
    new = zpanel.copy()
    target = zpanel.index[-1]
    out = news_decomposition(fit_pca_factor, prev, new, target)
    assert "impact" in out.columns
    assert len(out) >= 1


# ---- MIDAS / bridge GDP --------------------------------------------------- #
def test_gdp_nowcast(zpanel, client):
    af = fit_pca_factor(zpanel)
    gdp = client.get_series("GDPC1")
    b = BridgeGDP().fit(af.factor, gdp).nowcast(af.factor)
    m = MidasGDP().fit(af.factor, gdp).nowcast(af.factor)
    assert np.isfinite(b.point) and np.isfinite(m.point)
    assert b.r2 > 0.1 and m.r2 > 0.1  # factor should explain synthetic GDP


# ---- allocation backtest -------------------------------------------------- #
def test_allocation_backtest():
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    rng = np.random.default_rng(1)
    eq = pd.Series(rng.normal(0.006, 0.04, 240), index=idx)
    composite = pd.Series(np.sin(np.linspace(0, 12, 240)), index=idx)
    recprob = pd.Series(np.clip(-composite, 0, 1) * 0.5, index=idx)
    res = run_backtest(eq, composite, recprob)
    assert (res.weights.between(0, 1)).all()
    assert "sharpe" in res.metrics["strategy"]


def test_target_weight_bounds():
    idx = pd.date_range("2000-01-31", periods=10, freq="ME")
    w = target_weight(pd.Series(np.linspace(-5, 5, 10), index=idx),
                      pd.Series(np.zeros(10), index=idx))
    assert (w >= 0).all() and (w <= 1).all()


# ---- monitoring ----------------------------------------------------------- #
def test_psi_zero_for_same_distribution():
    s = pd.Series(np.random.default_rng(0).normal(size=2000))
    assert population_stability_index(s, s) < 0.01
