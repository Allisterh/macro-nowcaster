"""Regime labelling (HMM) and nowcast news decomposition."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    states: pd.Series          # integer regime label per month
    labels: dict[int, str]     # state -> human label (Expansion/Slowdown/Contraction)
    means: dict[int, float]    # state -> mean activity level


def fit_regimes(factor: pd.Series, n_states: int = 3, seed: int = 0) -> RegimeResult:
    """Label the activity factor into regimes with a Gaussian HMM.

    States are relabelled by their mean activity so 0=Contraction .. high=Expansion,
    which keeps the output interpretable regardless of HMM initialization order.
    """
    from hmmlearn.hmm import GaussianHMM

    x = factor.dropna().values.reshape(-1, 1)
    hmm = GaussianHMM(n_components=n_states, covariance_type="full",
                      n_iter=200, random_state=seed)
    hmm.fit(x)
    raw_states = hmm.predict(x)

    order = np.argsort(hmm.means_.ravel())  # ascending activity
    remap = {old: new for new, old in enumerate(order)}
    states = pd.Series([remap[s] for s in raw_states], index=factor.dropna().index)

    names = {0: "Contraction", n_states - 1: "Expansion"}
    for k in range(1, n_states - 1):
        names[k] = "Slowdown"
    means = {new: float(hmm.means_.ravel()[old]) for old, new in remap.items()}
    return RegimeResult(states, names, means)


def news_decomposition(
    fit_fn,
    panel_prev: pd.DataFrame,
    panel_new: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Attribute the revision in the activity nowcast to newly arrived data.

    ``fit_fn`` maps a panel to an activity factor (Series). We compare the factor
    at ``target_date`` under the previous vintage versus the new vintage, then use
    leave-one-out: for each cell that changed or newly arrived, refit without that
    cell and measure how much of the revision it accounts for. This is the
    interpretable "which release moved the nowcast" view a desk wants.
    """
    f_prev = fit_fn(panel_prev).factor.reindex([target_date]).iloc[0]
    f_new = fit_fn(panel_new).factor.reindex([target_date]).iloc[0]
    total_revision = float(f_new - f_prev)

    # cells present (non-null) in new but absent/different in prev
    changed: list[tuple[pd.Timestamp, str]] = []
    aligned_prev = panel_prev.reindex_like(panel_new)
    for col in panel_new.columns:
        diff = panel_new[col].notna() & (
            aligned_prev[col].isna() | (aligned_prev[col] != panel_new[col])
        )
        for d in panel_new.index[diff]:
            changed.append((d, col))

    rows = []
    for d, col in changed:
        knocked = panel_new.copy()
        knocked.loc[d, col] = np.nan
        f_knock = fit_fn(knocked).factor.reindex([target_date]).iloc[0]
        impact = float(f_new - f_knock)  # contribution of this release to the new level
        rows.append({"date": d, "indicator": col, "impact": impact})

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("impact", key=lambda s: s.abs(), ascending=False)
        out.attrs["total_revision"] = total_revision
    return out
