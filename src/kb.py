"""Knowledge base: topic tagging + TF-IDF vector store persisted as JSON/pickle.
Every unit keeps origin link (doc + loc). Topics modelled as {topic, subtopic, concepts, prerequisites}.
"""
from __future__ import annotations
import json, pickle, re
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .llm import generate, active_provider

# Default course map (Physics Mechanics). Auto-extended when new topics are tagged.
DEFAULT_TOPICS = {
    "Newton's Laws": {"subtopics": ["First law & inertia", "F=ma", "Action-reaction"],
                       "prerequisites": ["Vectors & Kinematics"]},
    "Friction": {"subtopics": ["Static vs kinetic", "Inclined planes"], "prerequisites": ["Newton's Laws"]},
    "Work, Energy & Power": {"subtopics": ["Work-energy theorem", "Conservation of energy"],
                             "prerequisites": ["Newton's Laws"]},
    "Vectors & Kinematics": {"subtopics": ["Displacement/velocity", "Projectile motion"],
                             "prerequisites": []},
}

KEYWORD_TOPICS = {
    "newton": "Newton's Laws", "inertia": "Newton's Laws", "f=ma": "Newton's Laws",
    "action-reaction": "Newton's Laws", "friction": "Friction", "static friction": "Friction",
    "kinetic": "Friction", "work": "Work, Energy & Power", "energy": "Work, Energy & Power",
    "power": "Work, Energy & Power", "conservation": "Work, Energy & Power",
    "velocity": "Vectors & Kinematics", "projectile": "Vectors & Kinematics",
    "vector": "Vectors & Kinematics", "kinematics": "Vectors & Kinematics",
}


def keyword_tag(text: str) -> Tuple[str, str, List[str]]:
    low = text.lower()
def keyword_tag(text: str) -> Tuple[str, str, List[str]]:
    low = text.lower()
    scores: Dict[str, int] = {}
    for kw, topic in KEYWORD_TOPICS.items():
        c = low.count(kw)
        if c:
            # title/first-120-chars matches weigh 3x (fixes "without friction" mis-tag)
            if kw in low[:120]:
                c += 2
            scores[topic] = scores.get(topic, 0) + c
    if scores:
        topic = max(scores, key=lambda t: scores[t])
        subs = DEFAULT_TOPICS.get(topic, {}).get("subtopics", [""])
        sub = next((s for s in subs if any(w in low for w in s.lower().split() if len(w) > 3)), subs[0] if subs else "")
        concepts = [kw for kw in KEYWORD_TOPICS if kw in low][:4]
        return topic, sub, concepts
    return "General", "", []


def llm_tag(units: List[Dict]) -> None:
    """Tag units with LLM when available; else keyword fallback. Mutates units in place."""
    if active_provider() == "local":
        for u in units:
            t, s, c = keyword_tag(u["text"])
            u["topic"], u["subtopic"], u["concepts"] = t, s, c
        return
    # batch in groups of 6 to save calls
    for i in range(0, len(units), 6):
        batch = units[i:i + 6]
        joined = "\n".join(f"[{j}] {u['text'][:400]}" for j, u in enumerate(batch))
        prompt = ("Tag each excerpt with: Topic | Subtopic | key concepts (comma-separated). "
                  "Use consistent topic names; reuse existing topics when possible.\n"
                  f"Existing topics: {list(DEFAULT_TOPICS)}\n\nEXCERPTS:\n{joined}\n\n"
                  "Return one line per excerpt: idx | Topic | Subtopic | concept1, concept2")
        try:
            out = generate(prompt, system="You are a curriculum tagger. Be consistent and concise.")
            for line in out.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4 and parts[0].isdigit():
                    u = batch[int(parts[0])]
                    u["topic"] = parts[1][:60] or u.get("topic", "General")
                    u["subtopic"] = parts[2][:80]
                    u["concepts"] = [c.strip()[:40] for c in parts[3].split(",") if c.strip()][:5]
        except Exception:
            pass
        for u in batch:
            if not u.get("topic"):
                t, s, c = keyword_tag(u["text"])
                u["topic"], u["subtopic"], u["concepts"] = t, s, c


# ---------- persistence + retrieval ----------
def load_units() -> List[Dict]:
    f = config.KB_UNITS_FILE
    if not f.exists():
        return []
    return json.loads(f.read_text(encoding="utf-8"))


def save_units(units: List[Dict]) -> None:
    config.KB_DIR.mkdir(parents=True, exist_ok=True)
    config.KB_UNITS_FILE.write_text(json.dumps(units, indent=1, ensure_ascii=False), encoding="utf-8")
    _rebuild_index(units)


def add_units(new: List[Dict]) -> List[Dict]:
    units = load_units()
    seen = {(u["doc"], u["loc"], u["text"][:80]) for u in units}
    fresh = [u for u in new if (u["doc"], u["loc"], u["text"][:80]) not in seen]
    if fresh:
        llm_tag(fresh)
        units.extend(fresh)
        save_units(units)
    return fresh


def _rebuild_index(units: List[Dict]) -> None:
    if not units:
        if config.KB_INDEX_FILE.exists():
            config.KB_INDEX_FILE.unlink()
        return
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
    mat = vec.fit_transform([u["text"] for u in units])
    with open(config.KB_INDEX_FILE, "wb") as fh:
        pickle.dump({"vectorizer": vec, "matrix": mat}, fh)


def _load_index():
    if not config.KB_INDEX_FILE.exists():
        return None, None
    d = pickle.loads(config.KB_INDEX_FILE.read_bytes())
    return d["vectorizer"], d["matrix"]


def retrieve(query: str, top_k: int = None, topic_filter: str = "") -> List[Dict]:
    units = load_units()
    if not units:
        return []
    vec, mat = _load_index()
    if vec is None:
        _rebuild_index(units)
        vec, mat = _load_index()
    sims = cosine_similarity(vec.transform([query]), mat)[0]
    scored = sorted(zip(sims, units), key=lambda x: -x[0])
    out = []
    for s, u in scored:
        if topic_filter and u.get("topic") != topic_filter:
            continue
        uu = dict(u)
        uu["score"] = float(s)
        out.append(uu)
        if len(out) >= (top_k or config.TOP_K):
            break
    return out


def course_map() -> Dict:
    """Returns {topic: {count, subtopics, prerequisites}} for flow-map viz."""
    units = load_units()
    m: Dict = {}
    for t, meta in DEFAULT_TOPICS.items():
        m[t] = {"count": 0, "subtopics": meta["subtopics"], "prerequisites": meta["prerequisites"]}
    for u in units:
        t = u.get("topic") or "General"
        m.setdefault(t, {"count": 0, "subtopics": [], "prerequisites": []})
        m[t]["count"] += 1
        if u.get("subtopic") and u["subtopic"] not in m[t]["subtopics"]:
            m[t]["subtopics"].append(u["subtopic"])
    return m
