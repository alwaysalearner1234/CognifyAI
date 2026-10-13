"""Learner model: per-topic Bayesian Knowledge Tracing + diagnostic for cold start."""
from __future__ import annotations
import json
from typing import Dict, List
from . import config


def _blank(topic: str) -> Dict:
    return {"p": config.BKT_P_INIT, "attempts": 0, "correct": 0}


def load() -> Dict:
    f = config.LEARNER_FILE
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save(state: Dict) -> None:
    config.LEARNER_FILE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def update(topic: str, correct: bool, state: Dict = None) -> Dict:
    state = state if state is not None else load()
    s = state.get(topic, _blank(topic))
    p = s["p"]
    G, S, T = config.BKT_P_GUESS, config.BKT_P_SLIP, config.BKT_P_TRANSIT
    if correct:
        p_given = (p * (1 - S)) / max(1e-9, p * (1 - S) + (1 - p) * G)
    else:
        p_given = (p * S) / max(1e-9, p * S + (1 - p) * (1 - G))
    s["p"] = round(min(0.99, p_given + (1 - p_given) * T), 4)
    s["attempts"] += 1
    s["correct"] += 1 if correct else 0
    state[topic] = s
    save(state)
    return state


def mastery(state: Dict = None) -> Dict[str, float]:
    state = state if state is not None else load()
    return {t: v["p"] for t, v in state.items()}


def weakest(state: Dict = None, k: int = 3) -> List[str]:
    m = mastery(state)
    return sorted(m, key=lambda t: m[t])[:k]


def diagnostic_questions() -> List[Dict]:
    """Cold-start diagnostic: 1 probe per major topic (used by UI before any history)."""
    from . import kb
    probes = []
    for topic in list(kb.DEFAULT_TOPICS)[:4]:
        hits = kb.retrieve(topic, top_k=1, topic_filter=topic) or kb.retrieve(topic, top_k=1)
        if hits:
            from .rag import cite
            h = hits[0]
            probes.append({"topic": topic, "source": cite(h),
                           "question": f"Diagnostic ({topic}): what does this mean in your own words? “{h['text'][:200]}…”",
                           "answer": h["text"][:200], "type": "short"})
    return probes
