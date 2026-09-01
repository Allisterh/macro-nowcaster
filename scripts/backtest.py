"""Honest evaluation: in-sample vs out-of-sample on one consistent dataset.

Run with ``make backtest`` (set FRED_API_KEY first for a live-vintage replay).
Every run stamps RESULTS.md with the commit, the vintage mode, the replay window
and the exact package versions that produced the numbers, because "out-of-sample
AUC 0.80" means nothing without knowing whether it came from real ALFRED vintages
or from the release-lag proxy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import subprocess
import sys
import warnings
from importlib import metadata
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from macro_nowcaster.backtest.pseudo_realtime import evaluate, replay
from macro_nowcaster.config import get_settings
from macro_nowcaster.data.fred_client import get_client
from macro_nowcaster.features.transforms import standardized_panel
from macro_nowcaster.models.dfm import fit_activity_factor, fit_pca_factor
from macro_nowcaster.models.recession import fit_nowcast

ROOT = Path(__file__).resolve().parents[1]


def _commit() -> str:
    """Short SHA plus whether the tree was dirty when the numbers were produced."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}{' (dirty tree)' if dirty else ''}"
    except Exception:  # noqa: BLE001
        return "unknown (not a git checkout)"


def _packages() -> str:
    """Installed versions of the dependencies declared in requirements.txt."""
    names = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            names.append(line.split(">=")[0].split("==")[0].strip())
    out = []
    for n in names:
        try:
            out.append(f"{n} {metadata.version(n)}")
        except metadata.PackageNotFoundError:
            out.append(f"{n} not installed")
    return ", ".join(out)


def _provenance(mode: str, start: str, end: str, seed: str, factor: str,
                dfm_share: str) -> list[str]:
    novintage = os.environ.get("MN_NO_VINTAGE") == "1"
    vintage = ("release-lag proxy only (MN_NO_VINTAGE=1)" if novintage
               else "ALFRED vintages where available, release-lag proxy otherwise")
    return [
        f"commit:           {_commit()}",
        f"data source:      {mode}",
        f"vintage mode:     {vintage if mode != 'SYNTHETIC' else 'synthetic client, lag proxy'}",
        f"replay window:    {start} to {end}",
        f"replay factor:    {factor.upper()} {dfm_share}",
        "recognition lag:  4 months",
        f"seed:             {seed}",
        f"python:           {sys.version.split()[0]} on {platform.system().lower()}",
        f"packages:         {_packages()}",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=str(ROOT / "RESULTS.md"),
                    help="where to write the report (default: RESULTS.md)")
    ap.add_argument("--factor", choices=["pca", "dfm"], default="pca",
                    help="estimator for the replay: pca is fast, dfm is what the site ships")
    args = ap.parse_args()

    settings = get_settings()
    client = get_client(settings)
    mode = "LIVE FRED" if settings.fred_api_key else "SYNTHETIC"
    end = args.end or dt.datetime.now(dt.timezone.utc).date().isoformat()

    print(f"data source: {mode}")
    print(f"replay window: {args.start} -> {end}")
    print(f"replay factor: {args.factor}")
    print("running replay (this is the slow part)...")

    raw = {c: client.get_series(c) for c in settings.codes}
    raw = {k: v for k, v in raw.items() if v is not None and not v.empty}
    final_z = standardized_panel(raw, settings, mode="full")
    # the in-sample benchmark uses the same estimator, so the two AUCs compare
    final_af = (fit_activity_factor(final_z, prefer="dfm") if args.factor == "dfm"
                else fit_pca_factor(final_z))
    final_factor = final_af.factor
    usrec = (client.get_series(settings.recession_flag).resample("ME").mean() > 0.5).astype(int)

    slope = final_z.get("T10Y3M")
    in_sample = fit_nowcast(final_factor, slope, usrec)

    rt = replay(client, settings, start=args.start, end=end, factor=args.factor)
    metrics = evaluate(rt, final_factor, usrec.astype(float))

    # how much of the replay really ran the requested estimator
    methods = rt["factor_method"] if "factor_method" in rt else pd.Series(dtype=str)
    dfm_share = ""
    if args.factor == "dfm" and len(methods):
        n_dfm = int((methods == "dfm").sum())
        dfm_share = (f"(all {n_dfm} months converged)" if n_dfm == len(methods)
                     else f"({n_dfm}/{len(methods)} months converged; the rest fell back to PCA)")
    elif args.factor == "pca":
        dfm_share = "(fit_pca_factor) - the live site ships the DFM"

    prov = _provenance(mode, args.start, end,
                       "7 (SyntheticClient)" if mode == "SYNTHETIC" else "n/a (replay is deterministic)",
                       args.factor, dfm_share)
    lines = [
        "=" * 60,
        "HONEST EVALUATION RESULTS",
        "=" * 60,
        f"data source:                 {mode}",
        f"replay window:               {args.start} to {end}",
        f"replay months evaluated:     {metrics.get('n_periods', 'n/a')}",
        f"replay factor:               {args.factor.upper()} {dfm_share}",
        "",
        f"in-sample recession AUC:     {in_sample.auc:.3f}",
        f"out-of-sample recession AUC: {metrics.get('recession_oos_auc', float('nan')):.3f}",
        f"OOS Brier score:             {metrics.get('recession_oos_brier', float('nan')):.3f}",
        "",
        f"real-time vs final corr:     {metrics.get('composite_realtime_vs_final_corr', float('nan')):.3f}",
        f"composite revision MAE:      {metrics.get('composite_revision_mae', float('nan')):.3f}",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print("\n" + report)

    with open(args.out, "w") as fh:
        fh.write("# Backtest results\n\n")
        fh.write(f"Generated {dt.datetime.now(dt.timezone.utc).date().isoformat()} on {mode} data.\n\n")
        fh.write("## Provenance\n\n```\n" + "\n".join(prov) + "\n```\n\n")
        fh.write("## Results\n\n```\n" + report + "\n```\n")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
