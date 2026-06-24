"""Configuration loading for the macro nowcaster.

Centralizes the indicator universe and runtime settings so every layer reads
from one source of truth (config/indicators.yaml) plus environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("MN_CONFIG", PACKAGE_ROOT / "config" / "indicators.yaml"))
DATA_DIR = Path(os.environ.get("MN_DATA_DIR", PACKAGE_ROOT / "data"))
ARTIFACT_PATH = DATA_DIR / "artifact.pkl"


@dataclass(frozen=True)
class Indicator:
    code: str
    name: str
    category: str
    transform: str
    sign: int
    freq: str
    pub_lag_days: int


@dataclass(frozen=True)
class Settings:
    indicators: dict[str, Indicator]
    target_gdp: str
    recession_flag: str
    start_date: str
    category_order: list[str]
    dfm_factors: int
    dfm_factor_order: int
    recession_lead_months: int
    fred_api_key: str = field(default="")
    anthropic_api_key: str = field(default="")

    @property
    def codes(self) -> list[str]:
        return list(self.indicators.keys())

    def ordered_codes(self) -> list[str]:
        return sorted(
            self.indicators,
            key=lambda c: (
                self.category_order.index(self.indicators[c].category),
                self.indicators[c].name,
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings from YAML plus environment variables."""
    with open(CONFIG_PATH) as fh:
        raw = yaml.safe_load(fh)

    indicators = {
        code: Indicator(code=code, **spec) for code, spec in raw["indicators"].items()
    }
    s = raw["settings"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return Settings(
        indicators=indicators,
        target_gdp=raw["target_gdp"],
        recession_flag=raw["recession_flag"],
        start_date=s["start_date"],
        category_order=s["category_order"],
        dfm_factors=s["dfm_factors"],
        dfm_factor_order=s["dfm_factor_order"],
        recession_lead_months=s["recession_lead_months"],
        fred_api_key=os.environ.get("FRED_API_KEY", "").strip(),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
    )
