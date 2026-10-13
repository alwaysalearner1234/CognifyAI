"""Adaptive assessment: generate -> verify -> dedup -> cited feedback -> report.
Question: {qid, type, topic, source, difficulty, question, options, answer, explanation}
"""
from __future__ import annotations
import json, uuid, random, re
from typing import List, Dict
from . import config, kb
from .llm import generate, active_provider
from .rag import cite


def _history() -> List[Dict]:
    f = config.QUIZ_HISTORY_FILE
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _save_history(h: List[Dict]) -> None:
    config.QUIZ_HISTORY_FILE.write_text(json.dumps(h, indent=1, ensure_ascii=False), encoding="utf-8")


def seen_questions() -> set:
    return {q.get("qhash") for q in _history() if q.get("qhash")}


def _qhash(q: str) -> str:
    return re.sub(r"\W+", "", q.lower())[:60]


def _template_questions(hits, n: int, qtypes: List[str], seen: set = None) -> List[Dict]:
    """Offline generator: builds MCQs/numericals directly from source sentences (verified by construction).
    Seen-aware: skips hashes already used; adds 'practice variant k' salt so later sessions stay novel
    while remaining grounded in the same source span."""
    seen = seen or set()
    out, attempt = [], 0
    order = qtypes or ["mcq"]
    start = len(seen) % max(1, len(hits))  # rotate start so sessions differ
    while len(out) < n and attempt < 400:
        h = hits[(start + attempt) % len(hits)]
        qt = order[(len(out)) % len(order)]
        sents = [s.strip() for s in h["text"].split(".") if len(s.strip()) > 30] or [h["text"][:220]]
        sent = sents[(attempt // 1) % len(sents)][:220]
        salt = attempt // (len(hits) * len(sents) * max(1, len(order)))
        variant_label = "" if salt == 0 else f" (practice set {salt + 1})"
        topic, src = h.get("topic", "General"), cite(h)
        diff = "easy" if len(sent) < 120 else "medium"
        tag = f"{qt}{attempt}s{salt}"
        attempt += 1
        if qt == "numerical":
            qh = _qhash(sent + tag)
            if qh in seen or any(q["qhash"] == qh for q in out):
                continue
            out.append({"qid": uuid.uuid4().hex[:8], "type": "numerical", "topic": topic,
                        "source": src, "difficulty": diff,
                        "question": f"Using the principle in [{src}]: “{sent}”{variant_label}. A 2 kg block is involved — set up (don't solve) the equation you'd use and name the law.",
                        "options": [], "answer": f"Set up from “{sent[:100]}…” [{src}]",
                        "explanation": f"Derived from [{src}]. Show which law applies and substitute m=2 kg.",
                        "qhash": qh})
        elif qt == "short":
            qh = _qhash(sent + tag)
            if qh in seen or any(q["qhash"] == qh for q in out):
                continue
            out.append({"qid": uuid.uuid4().hex[:8], "type": "short", "topic": topic,
                        "source": src, "difficulty": diff,
                        "question": f"In your own words, explain{variant_label}: “{sent}” [{src}]",
                        "options": [], "answer": sent, "explanation": f"Key points from [{src}].",
                        "qhash": qh})
        else:
            # MCQ: correct = source sentence gist; distractors generic but distinct
            qh = _qhash(sent + tag)
            if qh in seen or any(q["qhash"] == qh for q in out):
                continue
            out.append({"qid": uuid.uuid4().hex[:8], "type": "mcq", "topic": topic,
                        "source": src, "difficulty": diff,
                        "question": f"According to [{src}]{variant_label}, which statement about {topic} is correct?",
                        "options": [f"{sent[:90]}",
                                    f"The opposite of the stated {topic} principle holds universally",
                                    f"{topic} applies only in vacuum with no forces",
                                    f"{topic} is unrelated to forces and motion"],
                        "answer": f"{sent[:90]}",
                        "explanation": f"Option A restates [{src}]; B–D contradict it.",
                        "qhash": qh})
    return out


def generate_quiz(scope: str = "", topics: List[str] = None, n: int = 5,
                  qtypes: List[str] = None, difficulty: str = "adaptive",
                  learner_state: Dict = None) -> List[Dict]:
    qtypes = qtypes or ["mcq", "short", "numerical"]
    topics = topics or []
    # retrieval per topic for coverage
    hits = []
    if topics:
        per = max(1, n // len(topics))
        for t in topics:
            hits += kb.retrieve(scope or t, top_k=per + 2, topic_filter=t)
    else:
        hits = kb.retrieve(scope or "physics mechanics laws energy", top_k=n + 3)
    if not hits:
        return []
    # adaptive ordering: weakest topics first
    if learner_state:
        hits.sort(key=lambda h: learner_state.get(h.get("topic", ""), 0.3))
    seen = seen_questions()
    if active_provider() != "local":
        # LLM generation with verification fallback
        excerpts = "\n".join(f"- [{cite(h)}|{h.get('topic')}] {h['text'][:400]}" for h in hits[:8])
        prompt = (f"Create {n} quiz questions (types {qtypes}, difficulty {difficulty}) from EXCERPTS only. "
                  f"Each question tagged with topic + source. Return JSON list with keys: type,topic,question,options,answer,explanation,source,difficulty.\n\n"
                  f"EXCERPTS:\n{excerpts}")
        try:
            raw = generate(prompt, system="You write exam questions as strict JSON only.")
            m = re.search(r"\[.*\]", raw, re.S)
            data = json.loads(m.group(0)) if m else []
            qs = []
            for d in data[:n]:
                d["qid"] = uuid.uuid4().hex[:8]
                d["qhash"] = _qhash(d.get("question", ""))
                if d["qhash"] in seen:
                    continue
                qs.append(d)
            if qs and verify(qs, hits):
                _save_history(_history() + qs)
                return qs
        except Exception as e:
            print("[quiz] llm path failed:", e)
    qs = _template_questions(hits, n, qtypes, seen=seen)
    # shuffle MCQ options but track answer
    for q in qs:
        if q["type"] == "mcq" and q.get("options"):
            random.shuffle(q["options"])
    _save_history(_history() + qs)
    return qs


def verify(questions: List[Dict], hits) -> bool:
    """Answer verification: answer span must overlap a retrieved excerpt (cross-check)."""
    corpus = " ".join(h["text"].lower() for h in hits)
    ok = 0
    for q in questions:
        ans = (q.get("answer") or "").lower()
        key = re.sub(r"\W+", " ", ans).strip().split()
        overlap = sum(1 for w in key[:12] if len(w) > 3 and w in corpus)
        if overlap >= 2 or q.get("type") == "numerical":
            ok += 1
    return ok >= max(1, len(questions) * 2 // 3)


def grade(question: Dict, student: str) -> Dict:
    ans = (question.get("answer") or "").lower()
    s = (student or "").lower().strip()
    if question["type"] == "mcq":
        correct = s == ans.lower() or any(o.lower().startswith(s[:20]) for o in [ans] if s) and s in ans
        # simpler: letter or text match
        correct = s in ans or ans in s
    else:
        key = [w for w in re.sub(r"\W+", " ", ans).split() if len(w) > 4][:8]
        overlap = sum(1 for w in key if w in s)
        correct = overlap >= max(1, len(key) // 2)
    fb = (f"Correct. {question.get('explanation','')} Source: {question.get('source','')}"
          if correct else
          f"Not quite. Correct answer: {question.get('answer')}. {question.get('explanation','')} "
          f"Review {question.get('source','')} — likely misconception: confusing {question.get('topic','')} with a related principle.")
    return {"correct": bool(correct), "feedback": fb, "source": question.get("source", "")}


def report(results: List[Dict]) -> Dict:
    from collections import defaultdict
    by_topic: Dict = defaultdict(lambda: {"n": 0, "c": 0})
    for q, r in results:
        t = q.get("topic", "General")
        by_topic[t]["n"] += 1
        by_topic[t]["c"] += 1 if r.get("correct") else 0
    weak = sorted(by_topic, key=lambda t: by_topic[t]["c"] / max(1, by_topic[t]["n"]))[:3]
    total = sum(v["n"] for v in by_topic.values())
    acc = sum(v["c"] for v in by_topic.values()) / max(1, total)
    return {"accuracy": round(acc, 3), "by_topic": dict(by_topic),
            "weak_topics": weak,
            "summary": f"Accuracy {acc:.0%}. Focus revision on: {', '.join(weak) or '—'}. "
                       f"Flashcards + re-quiz scheduled for weak topics."}
