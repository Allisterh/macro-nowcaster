"""Data clients for FRED and its archival (ALFRED) vintage database.

The point of this module is point-in-time integrity. ``get_series_as_of`` returns
the data *as it was known* on a given date, including publication lag and pre-
revision values. That is what makes the historical nowcast honest rather than
secretly aware of the future.

Two backends:
  * ``FredClient``      live FRED/ALFRED via the ``fredapi`` wrapper (needs a key).
  * ``SyntheticClient`` deterministic synthetic data so the full system runs,
                        tests, and demos with no key and no network.

``get_client`` picks the live backend when ``FRED_API_KEY`` is set, else synthetic.
"""
from __future__ import annotations

import os

import datetime as dt
import logging
import time
from typing import Protocol

import numpy as np
import pandas as pd

from ..config import Settings, get_settings

log = logging.getLogger(__name__)


class DataClient(Protocol):
    def get_series(self, code: str) -> pd.Series | None: ...
    def get_series_as_of(self, code: str, as_of: dt.date) -> pd.Series | None: ...


# --------------------------------------------------------------------------- #
# Live FRED / ALFRED
# --------------------------------------------------------------------------- #
class FredClient:
    """Live client. Uses ALFRED vintages when available, else a lag-based proxy."""

    def __init__(self, settings: Settings | None = None):
        from fredapi import Fred  # imported lazily so synthetic runs need no dep

        self.s = settings or get_settings()
        if not self.s.fred_api_key:
            raise RuntimeError("FRED_API_KEY not set; use SyntheticClient instead.")
        self.fred = Fred(api_key=self.s.fred_api_key)
        self._cache: dict = {}
        self._vintage_cache: dict = {}

    def _retry(self, fn, *args, retries: int = 3, pause: float = 1.0, **kw):
        for attempt in range(1, retries + 1):
            try:
                return fn(*args, **kw)
            except Exception as exc:  # noqa: BLE001
                log.warning("attempt %d failed: %s", attempt, str(exc)[:80])
                time.sleep(pause * attempt)
        return None

    def get_series(self, code: str) -> pd.Series | None:
        if code in self._cache:
            return self._cache[code]
        s = self._retry(self.fred.get_series, code, observation_start=self.s.start_date)
        if s is None:
            return None
        s = s.dropna()
        s.name = code
        result = s if not s.empty else None
        self._cache[code] = result
        return result

    def _vintage_history(self, code: str):
        """Full ALFRED release history for a series, downloaded once and cached."""
        if code not in self._vintage_cache:
            df = self._retry(self.fred.get_series_all_releases, code, retries=2)
            if df is not None and not df.empty:
                df = df.copy()
                df["_rt"] = pd.to_datetime(df["realtime_start"], errors="coerce")
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
            self._vintage_cache[code] = df
        return self._vintage_cache[code]

    def get_series_as_of(self, code: str, as_of: dt.date) -> pd.Series | None:
        """Vintage known on ``as_of``. Falls back to a publication-lag proxy.

        The full revision history is downloaded once per series and cached, then
        sliced in memory for each as-of date (taking the latest revision known by
        that date). Daily/weekly series, anything not in the panel, and any
        series whose history can't be fetched use the publication-lag proxy.
        Set MN_NO_VINTAGE=1 to skip ALFRED and run a faster release-lag backtest.
        """
        freq = self.s.indicators[code].freq if code in self.s.indicators else "m"
        use_vintage = freq == "m" and os.environ.get("MN_NO_VINTAGE") != "1"
        if use_vintage:
            try:
                allrel = self._vintage_history(code)
                if allrel is not None and not allrel.empty:
                    known = allrel[allrel["_rt"] <= pd.Timestamp(as_of)]
                    if not known.empty:
                        s = known.sort_values("_rt").groupby("date")["value"].last().dropna()
                        s.name = code
                        if not s.empty:
                            return s
            except Exception as exc:  # noqa: BLE001
                log.info("no ALFRED vintage for %s (%s); using lag proxy", code, str(exc)[:60])

        latest = self.get_series(code)
        if latest is None:
            return None
        if code in self.s.indicators:
            lag = pd.Timedelta(days=self.s.indicators[code].pub_lag_days)
        else:
            lag = pd.Timedelta(days=30)  # recession flag / GDP: not in the panel
        cutoff = pd.Timestamp(as_of) - lag
        return latest[latest.index <= cutoff]


# --------------------------------------------------------------------------- #
# Synthetic backend (no key, fully deterministic, vintage-aware)
# --------------------------------------------------------------------------- #
class SyntheticClient:
    """Generates a shared latent business cycle and indicators driven by it.

    Recessions are derived from the latent cycle so the recession flag, the
    factor, and GDP are mutually consistent. Vintages respect publication lag.
    """

    def __init__(self, settings: Settings | None = None, seed: int = 7):
        self.s = settings or get_settings()
        self.rng = np.random.default_rng(seed)
        self._build()

    def _build(self) -> None:
        idx = pd.date_range(self.s.start_date, dt.date.today(), freq="ME")
        n = len(idx)
        t = np.linspace(0, 22 * np.pi, n)
        cycle = (
            np.sin(t)
            + 0.4 * np.sin(0.37 * t + 1.0)
            + np.cumsum(self.rng.normal(0, 0.05, n))
        )
        cycle = (cycle - cycle.mean()) / cycle.std()
        self.cycle = pd.Series(cycle, index=idx, name="cycle")

        rec = np.array((self.cycle < self.cycle.quantile(0.16)).astype(int).values, copy=True)
        # clean one-month blips so episodes look realistic
        for i in range(1, n - 1):
            if rec[i] == 1 and rec[i - 1] == 0 and rec[i + 1] == 0:
                rec[i] = 0
        self.usrec = pd.Series(rec, index=idx, name="USREC")

        self.raw: dict[str, pd.Series] = {}
        for code, ind in self.s.indicators.items():
            self.raw[code] = self._make_indicator(code, ind, idx)
        # GDP: quarterly, driven by the cycle
        q_idx = pd.date_range(self.s.start_date, idx[-1], freq="QE")
        cyc_q = self.cycle.reindex(q_idx, method="nearest").values
        gdp = 2.0 + 1.6 * cyc_q + self.rng.normal(0, 0.6, len(q_idx))
        self.gdp = pd.Series(gdp, index=q_idx, name=self.s.target_gdp)

    def _make_indicator(self, code, ind, idx) -> pd.Series:
        noise = self.rng.normal(0, 0.35, len(idx))
        signal = ind.sign * self.cycle.values
        if ind.transform == "level":
            base = 1.5 * signal + noise
            if code == "VIXCLS":
                base = 18 - 6 * signal + 3 * np.abs(noise)
            series = pd.Series(base, index=idx)
        else:
            growth = 0.002 + 0.012 * signal / 12 + noise / 400
            series = pd.Series(100 * np.exp(np.cumsum(growth)), index=idx)
        # expand monthly to native frequency for daily/weekly series
        if ind.freq == "d":
            d = pd.date_range(idx[0], idx[-1], freq="B")
            series = pd.Series(np.interp(d.asi8, idx.asi8, series.values), index=d)
        elif ind.freq == "w":
            w = pd.date_range(idx[0], idx[-1], freq="W-FRI")
            series = pd.Series(np.interp(w.asi8, idx.asi8, series.values), index=w)
        series.name = code
        return series.dropna()

    def get_series(self, code: str) -> pd.Series | None:
        if code == self.s.recession_flag:
            return self.usrec
        if code == self.s.target_gdp:
            return self.gdp
        return self.raw.get(code)

    def get_series_as_of(self, code: str, as_of: dt.date) -> pd.Series | None:
        s = self.get_series(code)
        if s is None:
            return None
        if code in (self.s.recession_flag, self.s.target_gdp):
            lag = pd.Timedelta(days=30)
        else:
            lag = pd.Timedelta(days=self.s.indicators[code].pub_lag_days)
        cutoff = pd.Timestamp(as_of) - lag
        return s[s.index <= cutoff]


def get_client(settings: Settings | None = None) -> DataClient:
    settings = settings or get_settings()
    if settings.fred_api_key:
        try:
            return FredClient(settings)
        except Exception as exc:  # noqa: BLE001
            log.warning("falling back to synthetic client: %s", str(exc)[:80])
    log.info("using SyntheticClient (set FRED_API_KEY for live data)")
    return SyntheticClient(settings)
