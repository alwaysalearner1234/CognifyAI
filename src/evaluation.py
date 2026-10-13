"""System evaluation (RAGAS-style, dependency-free) + simulated students.
Metrics: faithfulness, answer relevancy, context precision/recall.
"""
from __future__ import annotations
import json, random
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import kb, rag, assessment, learner


def _cos(a: str, b: str) -> float:
    try:
        v = TfidfVectorizer(stop_words="english").fit_transform([a, b])
        return float(cosine_similarity(v[0], v[1])[0][0])
    except Exception:
        return 0.0


def faithfulness(answer: str, contexts: List[str]) -> float:
    """Fraction of answer sentences overlapping retrieved contexts (claim support)."""
    sents = [s.strip() for s in answer.replace("?", ".").split(".") if len(s.strip()) > 15]
    if not sents:
        return 0.0
    corpus = " ".join(contexts).lower()
    import re
    sup = 0
    for s in sents:
        if s.lower().startswith(("based on", "sources:", "note:")):
            sup += 1
            continue
        keys = [w for w in re.sub(r"\W+", " ", s.lower()).split() if len(w) > 4][:6]
        if not keys or sum(1 for w in keys if w in corpus) >= max(1, len(keys) // 2):
            sup += 1
    return round(sup / len(sents), 3)


def evaluate_qa(testset: List[Dict]) -> Dict:
    rows, precs, recs, faiths, rels = [], [], [], [], []
    for t in testset:
        res = rag.answer(t["question"])
        ctxs = [h["text"] for h in res["hits"]]
        if t.get("expect_refusal"):
            ok = not res["grounded"]
            rows.append({"q": t["question"], "refused": ok, "pass": ok})
            precs.append(1.0 if ok else 0.0)
            continue
        gold = t.get("gold_loc", "")
        retr_locs = [f"{h['doc']} {h['loc']}" for h in res["hits"]]
        hit = any(gold.lower() in r.lower() or r.lower() in gold.lower() for r in retr_locs) if gold else len(ctxs) > 0
        precs.append(1.0 if hit else 0.0)
        recs.append(1.0 if hit else 0.0)
        f = faithfulness(res["answer"], ctxs) if res["grounded"] else 1.0
        r = _cos(t["question"], res["answer"])
        faiths.append(f)
        rels.append(round(r, 3))
        rows.append({"q": t["question"][:70], "grounded": res["grounded"],
                     "faithfulness": f, "relevancy": round(r, 3), "ctx_hit": hit})
    def avg(x): return round(sum(x) / len(x), 3) if x else 0.0
    return {"n": len(testset), "context_precision": avg(precs), "context_recall": avg(recs),
            "faithfulness": avg(faiths), "answer_relevancy": avg(rels), "rows": rows}


def simulate_students(sessions: int = 5) -> Dict:
    """3 profiles x N sessions; reports mastery gains + question repetition rate.
    Isolated: backs up quiz history so global dedup store doesn't starve the sim."""
    import random as _r
    _r.seed(42)
    from . import config as _cfg
    backup = _cfg.QUIZ_HISTORY_FILE.read_text(encoding="utf-8") if _cfg.QUIZ_HISTORY_FILE.exists() else None
    if _cfg.QUIZ_HISTORY_FILE.exists():
        _cfg.QUIZ_HISTORY_FILE.unlink()
    try:
        profiles = {"fast": 0.85, "average": 0.6, "struggling": 0.35}
        out = {}
        all_q = []
        for name, p in profiles.items():
            state = {}
            for s in range(sessions):
                qs = assessment.generate_quiz(scope="mechanics", n=4,
                                              learner_state={t: v.get("p", .3) for t, v in state.items()} or None)
                for q in qs:
                    all_q.append(q["qhash"])
                    correct = _r.random() < p
                    cur = state.get(q.get("topic", "General"), {"p": 0.3})
                    G, S, T = 0.2, 0.1, 0.15
                    pp = cur["p"]
                    pg = (pp * (1 - S)) / max(1e-9, pp * (1 - S) + (1 - pp) * G) if correct else (pp * S) / max(1e-9, pp * S + (1 - pp) * (1 - G))
                    cur["p"] = round(min(.99, pg + (1 - pg) * T), 4)
                    state[q.get("topic", "General")] = cur
            end = sum(v["p"] for v in state.values()) / max(1, len(state))
            out[name] = {"mastery_gain": round(end - 0.3, 3), "final_mastery": round(end, 3),
                         "topics": {k: v["p"] for k, v in state.items()}}
        total = len(all_q)
        uniq = len(set(all_q))
        out["question_repetition_rate"] = round(1 - uniq / max(1, total), 3)
        out["sessions"] = sessions
        return out
    finally:
        if backup is not None:
            _cfg.QUIZ_HISTORY_FILE.write_text(backup, encoding="utf-8")
        elif _cfg.QUIZ_HISTORY_FILE.exists():
            _cfg.QUIZ_HISTORY_FILE.unlink()
