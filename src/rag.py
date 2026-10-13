"""Source-grounded QA: retrieve -> compose with citations -> refuse when unsupported."""
from __future__ import annotations
from typing import Dict, List
from . import config, kb
from .llm import generate


def cite(u: Dict) -> str:
    dtype = u.get("doc_type", "")
    if dtype == "pdf":
        loc = f"p.{u['loc'].replace('p.', '')}" if "p." not in u["loc"] else u["loc"]
        return f"{u['doc']} ({loc})"
    if dtype == "pptx":
        return f"{u['doc']} ({u['loc']})"
    if dtype == "video":
        return f"{u['doc']} (@{u['loc']})"
    return f"{u['doc']} ({u['loc']})"


def answer(query: str, top_k: int = None, topic_filter: str = "") -> Dict:
    hits = kb.retrieve(query, top_k=top_k, topic_filter=topic_filter)
    if not hits or hits[0].get("score", 0) < config.REFUSAL_THRESHOLD:
        return {
            "answer": ("I can't find this in your uploaded course material, so I won't guess. "
                       "Try asking about: " + ", ".join(list(kb.course_map())[:5]) +
                       ". Anything outside the material is flagged as outside knowledge."),
            "citations": [], "hits": hits[:3], "grounded": False,
        }
    excerpts = "\n".join(f"- [{cite(h)} | {h.get('topic','')}] {h['text'][:500]}" for h in hits)
    prompt = (f"Answer ONLY from these excerpts. Every factual sentence must end with a citation like [1]. "
              f"If the excerpts don't cover it, say so. Question: {query}\n\nEXCERPTS:\n{excerpts}")
    raw = generate(prompt, system="You are a source-grounded tutor. Never invent page/slide numbers.")
    # append human-readable citation line
    cit_line = "\n".join(f"[{i+1}] {cite(h)} — {h.get('topic','')}: “{h['text'][:140]}…”" for i, h in enumerate(hits))
    full = f"{raw}\n\nSources:\n{cit_line}"
    return {"answer": full, "citations": [cite(h) for h in hits], "hits": hits, "grounded": True}
