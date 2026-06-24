"""Scheduled refresh flow.

Run directly (python flows/update_flow.py) or register with Prefect for a managed
schedule. Rebuilds the artifact from the latest data and writes a fresh memo.
"""
from __future__ import annotations

import logging

from macro_nowcaster.llm.memo_agent import MemoContext, generate_memo
from macro_nowcaster.pipeline import build_artifact

logging.basicConfig(level=logging.INFO)


def refresh() -> dict:
    art = build_artifact()
    s = art.summary()
    memo, used = generate_memo(MemoContext(
        as_of=s["as_of"], composite=s["composite"], regime=s["regime"],
        nowcast_recprob=s["nowcast_recprob"], lead_recprob=s["lead_recprob"],
        gdp_nowcast=s["gdp_nowcast"], top_tailwinds=s["top_tailwinds"],
        top_drags=s["top_drags"]))
    (art.z_panel.index[-1])
    with open("data/latest_memo.txt", "w") as fh:
        fh.write(memo)
    logging.info("refresh complete as_of=%s used_llm=%s", s["as_of"], used)
    return s


# Optional Prefect wrapper (only if prefect is installed).
try:
    from prefect import flow

    @flow(name="macro-nowcast-refresh")
    def scheduled_refresh():
        return refresh()
except Exception:  # noqa: BLE001
    scheduled_refresh = None


if __name__ == "__main__":
    refresh()
