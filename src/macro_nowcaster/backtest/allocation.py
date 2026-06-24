"""Macro-state asset allocation overlay: the bridge from signal to money.

A nowcast that does not change a portfolio is trivia. This module turns the
activity factor and recession probability into a risk-on/risk-off equity weight,
backtests it against buy-and-hold, and reports risk-adjusted performance net of a
simple turnover cost. The signal is lagged one month so the backtest only acts on
information available before the return is earned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    weights: pd.Series
    strategy_returns: pd.Series
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    metrics: dict


def target_weight(composite: pd.Series, recprob: pd.Series) -> pd.Series:
    """Map macro state to an equity weight in [0, 1].

    Base allocation tilts up with activity momentum (tanh keeps it bounded) and
    down with recession probability. Both inputs are standardized signals, so the
    rule is scale-stable.
    """
    base = 0.5 + 0.5 * np.tanh(composite)
    w = base - recprob.reindex(composite.index).fillna(0.0)
    return w.clip(0.0, 1.0)


def annualized(returns: pd.Series, periods: int = 12) -> dict:
    r = returns.dropna()
    if r.std() == 0 or r.empty:
        return {"cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    cagr = (1 + r).prod() ** (periods / len(r)) - 1
    vol = r.std() * np.sqrt(periods)
    sharpe = (r.mean() * periods) / vol if vol else 0.0
    curve = (1 + r).cumprod()
    dd = (curve / curve.cummax() - 1).min()
    return {
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
    }


def run_backtest(
    equity_returns: pd.Series,
    composite: pd.Series,
    recprob: pd.Series,
    rf_returns: pd.Series | None = None,
    cost_per_turn: float = 0.0005,
) -> BacktestResult:
    """Backtest the macro overlay. ``equity_returns`` are monthly simple returns."""
    eq = equity_returns.dropna()
    rf = (rf_returns if rf_returns is not None else pd.Series(0.0, index=eq.index)).reindex(eq.index).fillna(0.0)

    w = target_weight(composite, recprob).reindex(eq.index).ffill().shift(1).fillna(0.5)
    turnover = w.diff().abs().fillna(0.0)
    strat = w * eq + (1 - w) * rf - turnover * cost_per_turn

    res = BacktestResult(
        weights=w,
        strategy_returns=strat,
        equity_curve=(1 + strat).cumprod(),
        benchmark_curve=(1 + eq).cumprod(),
        metrics={
            "strategy": annualized(strat),
            "buy_and_hold": annualized(eq),
            "avg_equity_weight": float(w.mean()),
            "annual_turnover": float(turnover.sum() / (len(turnover) / 12)),
        },
    )
    return res
