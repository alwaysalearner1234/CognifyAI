# Architecture — CognifyAI

```
[PDF/PPTX/MP4/SRT/TXT] → ingestion.py → units {doc,loc,text,topic,has_figure}
        → kb.py (LLM/keyword tag + TF-IDF index, units.json)
        → rag.py (retrieve top-k → EXCERPTS prompt → cited answer / refusal)
        → assessment.py (generate → verify → dedup → grade → report)
        → learner.py (BKT per-topic) → dashboard / adaptive ordering
        → evaluation.py (faithfulness, relevancy, precision/recall, simulation)
        → app.py (Streamlit: Ingest/Chat/Quiz/Dashboard/Evaluation/Revision)
```

## Design decisions
- **No vector-DB server**: sklearn TF-IDF + pickle keeps install <1 min and judging reproducible; swap to Chroma/OpenAI embeddings later without changing unit schema.
- **LLM-agnostic** (`src/llm.py`): Gemini 1.5 Flash → GPT-4o-mini → offline extractive fallback. Offline mode guarantees *no hallucination without a key* (evaluation honesty).
- **Multimodal**: PyMuPDF text+image extraction (page → `p.N`, thumbnails saved), python-pptx text+picture (slide → `slide N`), faster-whisper transcripts (90 s windows → `mm:ss`) with `.srt` sidecar fallback; figures flagged `has_figure` and surfaced in UI.
- **Grounding**: origin preserved end-to-end (ingest → retrieve → cite → quiz source tag → feedback). Refusal threshold 0.10 tuned on sample KB.
- **Assessment validity**: template questions are *correct by construction* (answer = source span); LLM questions pass overlap verification; `qhash` dedup across sessions.
- **Personalization**: BKT is simple, explainable, and measurable; simulation harness proves gains scale with sessions.

## Scaling beyond prototype
Replace TF-IDF with `sentence-transformers + Chroma`, add CLIP/pytesseract captions for figures,
Whisper-large transcription service, SM-2 scheduler with push reminders, audio TTS tutoring.
