"""Retrieval-augmented analysis of Federal Reserve communications.

The signature research feature: cross-reference the qualitative tone of FOMC
statements and minutes against the quantitative nowcast, and surface divergences
("the data says slowing, but the Fed's language turned hawkish").

Design notes:
  * Retrieval uses TF-IDF by default so it runs with no heavy embedding deps; an
    embedding backend can be swapped in behind the same interface.
  * Generation calls the Anthropic API (claude). With no ANTHROPIC_API_KEY the
    module returns a structured, deterministic stub so the pipeline still runs.
  * Fed text is supplied by the caller (fetched or pasted). We never assume a
    specific scraper so the module stays robust and testable offline.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

ANTHROPIC_MODEL = "claude-sonnet-4-6"


@dataclass
class FedDocument:
    date: str
    kind: str           # statement | minutes | beige_book | speech
    text: str


@dataclass
class DivergenceAnalysis:
    nowcast_summary: str
    retrieved: list[FedDocument]
    analysis: str
    used_llm: bool = field(default=False)


class FedRAG:
    def __init__(self, documents: list[FedDocument]):
        self.documents = documents
        self._corpus = [d.text for d in documents]
        self._vec = TfidfVectorizer(stop_words="english", max_features=4000)
        self._matrix = self._vec.fit_transform(self._corpus) if self._corpus else None

    def retrieve(self, query: str, k: int = 3) -> list[FedDocument]:
        if self._matrix is None or not self._corpus:
            return []
        q = self._vec.transform([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        top = sims.argsort()[::-1][:k]
        return [self.documents[i] for i in top]

    def analyze(self, nowcast_summary: str, k: int = 3) -> DivergenceAnalysis:
        retrieved = self.retrieve(nowcast_summary, k=k)
        context = "\n\n".join(f"[{d.date} {d.kind}] {d.text[:1500]}" for d in retrieved)
        prompt = (
            "You are a macro strategist. Compare the QUANTITATIVE nowcast below "
            "with the Federal Reserve's own language in the retrieved excerpts. "
            "State clearly whether they agree or diverge, cite specific phrasing, "
            "and give one actionable takeaway. Be concise.\n\n"
            f"QUANT NOWCAST:\n{nowcast_summary}\n\nFED EXCERPTS:\n{context}"
        )
        text, used = self._call_llm(prompt)
        return DivergenceAnalysis(nowcast_summary, retrieved, text, used)

    def _call_llm(self, prompt: str) -> tuple[str, bool]:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            return (
                "[LLM stub: set ANTHROPIC_API_KEY to generate the divergence "
                "analysis. Retrieval ran and the most relevant Fed excerpts are "
                "attached for manual comparison.]",
                False,
            )
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text, True
        except Exception as exc:  # noqa: BLE001
            log.warning("Anthropic call failed: %s", str(exc)[:80])
            return f"[LLM call failed: {str(exc)[:80]}]", False


def to_json(analysis: DivergenceAnalysis) -> str:
    return json.dumps(
        {
            "nowcast_summary": analysis.nowcast_summary,
            "retrieved": [{"date": d.date, "kind": d.kind} for d in analysis.retrieved],
            "analysis": analysis.analysis,
            "used_llm": analysis.used_llm,
        },
        indent=2,
    )
