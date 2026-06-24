"""Automated research memo generation.

When recession probability crosses a threshold (or on demand), assemble the full
macro state, the news decomposition, and any Fed divergence analysis into a prompt
and have Claude draft a desk-style memo. Without ANTHROPIC_API_KEY it falls back to
a clean templated memo so the pipeline never blocks.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

ANTHROPIC_MODEL = "claude-sonnet-4-6"


@dataclass
class MemoContext:
    as_of: str
    composite: float
    regime: str
    nowcast_recprob: float
    lead_recprob: float
    gdp_nowcast: float
    top_tailwinds: list[str]
    top_drags: list[str]
    fed_divergence: str | None = None


def _template_memo(ctx: MemoContext) -> str:
    return (
        f"MACRO NOWCAST MEMO  |  {ctx.as_of}\n"
        f"{'=' * 52}\n"
        f"Activity factor: {ctx.composite:+.2f} sd ({ctx.regime}).\n"
        f"Recession probability: {ctx.nowcast_recprob:.0%} now, "
        f"{ctx.lead_recprob:.0%} over 12 months.\n"
        f"GDP nowcast: {ctx.gdp_nowcast:+.1f}% annualized.\n\n"
        f"Tailwinds: {', '.join(ctx.top_tailwinds)}.\n"
        f"Drags: {', '.join(ctx.top_drags)}.\n"
        + (f"\nFed divergence: {ctx.fed_divergence}\n" if ctx.fed_divergence else "")
        + "\n[Templated memo. Set ANTHROPIC_API_KEY for a written analyst draft.]"
    )


def generate_memo(ctx: MemoContext) -> tuple[str, bool]:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return _template_memo(ctx), False

    prompt = (
        "Write a concise, professional macro research memo (about 200 words) for "
        "a multi-asset portfolio team, based strictly on the data below. Cover the "
        "current state, the balance of risks, and one positioning implication. Do "
        "not invent numbers.\n\n"
        f"As of: {ctx.as_of}\n"
        f"Activity factor: {ctx.composite:+.2f} standard deviations ({ctx.regime})\n"
        f"Recession probability now: {ctx.nowcast_recprob:.0%}\n"
        f"Recession probability 12m ahead: {ctx.lead_recprob:.0%}\n"
        f"GDP nowcast: {ctx.gdp_nowcast:+.1f}% annualized\n"
        f"Tailwinds: {', '.join(ctx.top_tailwinds)}\n"
        f"Drags: {', '.join(ctx.top_drags)}\n"
        + (f"Fed divergence note: {ctx.fed_divergence}\n" if ctx.fed_divergence else "")
    )
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text, True
    except Exception as exc:  # noqa: BLE001
        log.warning("memo LLM call failed: %s", str(exc)[:80])
        return _template_memo(ctx), False
