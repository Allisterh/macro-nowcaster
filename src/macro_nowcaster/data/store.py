"""DuckDB-backed point-in-time observation store.

Every observation is stored with three timestamps:
  * ``ref_date``     the period the value describes (e.g. month-end)
  * ``vintage_date`` when that value became known (publication date)
  * ``value``        the number

This lets a query reconstruct exactly what was knowable as of any date, which is
the foundation of an honest backtest. The store is idempotent: re-ingesting the
same vintage is a no-op.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd

from ..config import DATA_DIR

DB_PATH = DATA_DIR / "pit_store.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    code         VARCHAR,
    ref_date     DATE,
    vintage_date DATE,
    value        DOUBLE,
    PRIMARY KEY (code, ref_date, vintage_date)
);
"""


class PointInTimeStore:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(self.path) as con:
            con.execute(_SCHEMA)

    def ingest(self, code: str, series: pd.Series, vintage_date: dt.date) -> int:
        """Store a vintage of a series. Returns rows written."""
        if series is None or series.empty:
            return 0
        df = pd.DataFrame(
            {
                "code": code,
                "ref_date": pd.to_datetime(series.index).date,
                "vintage_date": vintage_date,
                "value": series.values.astype(float),
            }
        ).dropna()
        with duckdb.connect(self.path) as con:
            con.execute(
                "INSERT OR IGNORE INTO observations SELECT * FROM df"
            )
        return len(df)

    def latest_panel(self, codes: list[str]) -> pd.DataFrame:
        """Most recent vintage of each (code, ref_date): the current best data."""
        q = """
            WITH ranked AS (
                SELECT code, ref_date, value,
                       ROW_NUMBER() OVER (
                           PARTITION BY code, ref_date ORDER BY vintage_date DESC
                       ) AS rn
                FROM observations WHERE code IN ?
            )
            SELECT code, ref_date, value FROM ranked WHERE rn = 1
            ORDER BY ref_date
        """
        with duckdb.connect(self.path) as con:
            df = con.execute(q, [codes]).df()
        return self._pivot(df)

    def panel_as_of(self, codes: list[str], as_of: dt.date) -> pd.DataFrame:
        """Reconstruct the panel using only vintages published on or before as_of."""
        q = """
            WITH visible AS (
                SELECT code, ref_date, value, vintage_date
                FROM observations
                WHERE code IN ? AND vintage_date <= ?
            ), ranked AS (
                SELECT code, ref_date, value,
                       ROW_NUMBER() OVER (
                           PARTITION BY code, ref_date ORDER BY vintage_date DESC
                       ) AS rn
                FROM visible
            )
            SELECT code, ref_date, value FROM ranked WHERE rn = 1
            ORDER BY ref_date
        """
        with duckdb.connect(self.path) as con:
            df = con.execute(q, [codes, as_of]).df()
        return self._pivot(df)

    @staticmethod
    def _pivot(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        out = df.pivot(index="ref_date", columns="code", values="value")
        out.index = pd.to_datetime(out.index)
        return out.sort_index()
