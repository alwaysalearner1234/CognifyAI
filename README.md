# CognifyAI — Source-Grounded Adaptive Tutor (Track D)

AI study companion that unifies lecture videos, textbooks, and slides into a **source-cited knowledge base**,
then runs **adaptive assessments + personalized tutoring** on top.

## Quickstart
```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env   # add GOOGLE_API_KEY or OPENAI_API_KEY (optional — offline mode works)
python scripts/seed_sample.py
streamlit run app.py
```

## Workflow (demo script, 3–10 min video)
1. **Ingest tab** — upload PDF/PPTX/MP4 + click Ingest (or "Load bundled Physics sample"). Show units with `doc (p./slide/@timestamp)` origins + topic tags + 🖼 figure flags.
2. **Chat tab** — ask in-scope Q ("State work-energy theorem") → cited answer with Sources list; ask off-scope ("quantum entanglement") → declined + flagged.
3. **Quiz tab** — Generate quiz → answer → cited feedback per answer → Finish report (weak topics + misconceptions). Re-generate shows **no repeated questions**.
4. **Dashboard** — BKT mastery bars update after every quiz.
5. **Evaluation tab** — Run RAG metrics + simulated students; read faithfulness/relevancy/precision/recall + mastery gains + repetition rate.
6. **Revision tab** — flashcards + spaced study schedule for weak topics (+ Hindi explanation toggle in Chat).

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md).

## Grounding method
TF-IDF retrieval (top-k=5) over units carrying `{doc, loc}` origin; refusal if max cosine < 0.10;
LLM prompted with EXCERPTS-only + per-sentence citation requirement; local offline fallback returns
extractive cited answer so demo never hallucinates without a key.

## Learner model
Per-topic **Bayesian Knowledge Tracing**: P(L) init 0.30, P(T) 0.15, P(G) 0.20, P(S) 0.10; updates on every
graded quiz answer; cold start via diagnostic probes; weakest-first quiz ordering; mastery ≥ 0.75.

## Evaluation
`src/evaluation.py` (RAGAS-style, dependency-free): faithfulness (claim→context overlap),
answer relevancy (Q–A cosine), context precision/recall vs `tests/test_set.json` gold locations
(includes off-material refusal cases); `simulate_students()` runs fast/average/struggling profiles
across N sessions → mastery gains + question repetition rate.

## Repo layout
```
app.py  src/{config,llm,ingestion,kb,rag,assessment,learner,evaluation,viz}.py
data/sample_physics/  tests/test_set.json  scripts/seed_sample.py
```
