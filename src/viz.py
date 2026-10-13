"""Optional enhancements: flashcards, study schedule (forgetting curve), course map data."""
from __future__ import annotations
from typing import List, Dict
import math


def flashcards(weak_topics: List[str], hits_by_topic: Dict[str, List[Dict]]) -> List[Dict]:
    cards = []
    for t in weak_topics:
        for h in hits_by_topic.get(t, [])[:3]:
            front = f"[{t}] Explain: {h['text'][:120]}…"
            back = f"{h['text'][:300]}\n— {h['doc']} ({h['loc']})"
            cards.append({"front": front, "back": back, "topic": t})
    return cards


def study_schedule(weak_topics: List[str], mastery: Dict[str, float], days_left: int = 7) -> List[Dict]:
    """Simple SM-2/forgetting-curve-inspired allocation: weaker + sooner = more sessions."""
    if not weak_topics:
        return []
    weights = {t: (1.0 - mastery.get(t, 0.3)) + 0.2 for t in weak_topics}
    tot = sum(weights.values())
    plan = []
    for i, t in enumerate(weak_topics):
        sessions = max(1, round(days_left * weights[t] / tot))
        interval = max(1, round(2.5 * (1 - mastery.get(t, 0.3)) + 1))
        plan.append({"topic": t, "sessions": sessions,
                     "interval_days": interval,
                     "plan": f"Day {(i % days_left) + 1}: {t} ({sessions} sessions, revise every {interval}d)"})
    return plan
