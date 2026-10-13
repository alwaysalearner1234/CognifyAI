"""CognifyAI — source-grounded adaptive tutor (Track D). Run: streamlit run app.py"""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from src import config, ingestion, kb, rag, assessment, learner, evaluation
from src import viz as vizmod
from src.llm import active_provider, hindi_explain

st.set_page_config(page_title="CognifyAI — Adaptive Tutor", layout="wide")
st.title("CognifyAI · Source-Grounded Adaptive Tutor")
st.caption(f"LLM backend: **{active_provider()}** (set GOOGLE_API_KEY/OPENAI_API_KEY in .env for fluent mode) · "
           f"KB units: **{len(kb.load_units())}**")

tabs = st.tabs(["📥 Ingest & KB", "💬 Tutor Chat", "📝 Adaptive Quiz", "📊 Dashboard",
                "🧪 Evaluation", "🗂 Revision"])

# ---------- 1. Ingest ----------
with tabs[0]:
    st.header("Multimodal Knowledge Base")
    st.write("Upload lecture videos, textbooks (PDF), slides (PPTX), transcripts. No manual preprocessing — "
             "every unit keeps its origin (page / slide / timestamp) and is tagged to topic → subtopic → concepts.")
    files = st.file_uploader("Upload course files", type=["pdf", "pptx", "mp4", "mp3", "wav", "srt", "vtt", "txt", "md"],
                             accept_multiple_files=True)
    if st.button("Ingest uploaded files", disabled=not files):
        total = 0
        for f in files:
            p = config.UPLOAD_DIR / f.name
            p.write_bytes(f.getbuffer())
            try:
                new = ingestion.ingest_file(p)
                fresh = kb.add_units(new)
                total += len(fresh)
            except Exception as e:
                st.error(f"{f.name}: {e}")
        st.success(f"Ingested {total} new units. KB now {len(kb.load_units())}.")
    if st.button("Load bundled Physics sample"):
        from pathlib import Path as P
        units = []
        for fp in (P("data") / "sample_physics").glob("*.*"):
            units += ingestion.ingest_file(fp)
        st.success(f"Added {len(kb.add_units(units))} units.")
    st.subheader("Course map (topics → prerequisites)")
    cmap = kb.course_map()
    if cmap:
        rows = [{"topic": t, "units": v["count"], "prerequisites": ", ".join(v["prerequisites"]),
                 "subtopics": ", ".join(v["subtopics"][:4])} for t, v in cmap.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with st.expander("Browse units (with origin links)"):
        for u in kb.load_units()[-30:]:
            fig = " 🖼" if u.get("has_figure") else ""
            st.markdown(f"**{u.get('topic','')}** · {u['doc']} ({u['loc']}){fig} — {u['text'][:220]}…")

# ---------- 2. Chat ----------
with tabs[1]:
    st.header("Source-grounded tutor chat")
    st.write("Every answer cites excerpts that open the exact page/slide/timestamp. "
             "Off-material queries are declined, outside knowledge is flagged.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    q = st.text_input("Ask from your uploaded material", key="chat_q")
    col1, col2 = st.columns([1, 1])
    hindi = col1.checkbox("Also explain in Hindi 🇮🇳")
    tfilter = col2.selectbox("Scope (topic filter)", [""] + sorted(kb.course_map()),
                             format_func=lambda x: "All topics" if not x else x)
    if st.button("Ask", key="ask_btn") and q:
        res = rag.answer(q, topic_filter=tfilter)
        st.session_state.chat.append((q, res))
        # learner model updates lightly from conversation depth (asking = exposure, not mastery)
    for qq, res in reversed(st.session_state.chat[-10:]):
        st.markdown(f"**You:** {qq}")
        st.markdown(res["answer"])
        if hindi and res["grounded"]:
            st.info(hindi_explain(res["answer"][:600]))
        if not res["grounded"]:
            st.warning("⚠️ Flagged: outside uploaded material — no citation claimed.")
        st.divider()

# ---------- 3. Quiz ----------
with tabs[2]:
    st.header("Adaptive assessment")
    state = learner.load()
    if not state:
        st.info("👋 New student (cold start): take the diagnostic or just start — BKT mastery starts at 0.30/topic.")
        if st.button("Run diagnostic quiz"):
            dq = learner.diagnostic_questions()
            st.session_state.diagnostic = dq
            for d in dq:
                st.markdown(f"**[{d['topic']}]** {d['question']}")
    scope = st.text_input("Scope (keywords, e.g. 'friction incline')", value="mechanics")
    topics = st.multiselect("Topics", sorted(kb.course_map()), default=learner.weakest(state)[:2] if state else [])
    n = st.slider("Questions", 3, 10, 5)
    qtypes = st.multiselect("Types", ["mcq", "short", "numerical"], default=["mcq", "short", "numerical"])
    if st.button("Generate quiz (verified, deduped)"):
        qs = assessment.generate_quiz(scope=scope, topics=topics, n=n, qtypes=qtypes,
                                      learner_state={t: v["p"] for t, v in state.items()})
        st.session_state.quiz = qs
        if not qs:
            st.error("KB empty or no coverage — ingest files first.")
    qs = st.session_state.get("quiz", [])
    results = []
    if qs:
        for i, qu in enumerate(qs):
            st.markdown(f"**Q{i+1} [{qu.get('type')}] ({qu.get('topic')} · {qu.get('difficulty')})** "
                        f"📍 {qu.get('source')}")
            st.write(qu["question"])
            if qu.get("options"):
                for j, o in enumerate(qu["options"]):
                    st.write(f"{chr(65+j)}. {o}")
            ans = st.text_input(f"Your answer Q{i+1}", key=f"ans_{qu['qid']}")
            if ans:
                r = assessment.grade(qu, ans)
                st.markdown(r["feedback"])
                results.append((qu, {"correct": r["correct"]}))
                learner.update(qu.get("topic", "General"), r["correct"])
        if results and st.button("Finish & show report"):
            rep = assessment.report(results)
            st.json(rep)
            st.session_state.last_report = rep

# ---------- 4. Dashboard ----------
with tabs[3]:
    st.header("Learner dashboard")
    state = learner.load()
    if not state:
        st.info("No data yet — complete a quiz or diagnostic.")
    else:
        df = pd.DataFrame([{"topic": t, "mastery": v["p"], "attempts": v["attempts"],
                            "status": "Mastered ✅" if v["p"] >= config.MASTERY_THRESHOLD else "Learning 🌱"}
                           for t, v in state.items()])
        st.plotly_chart(px.bar(df, x="topic", y="mastery", color="status", range_y=[0, 1],
                               title="Per-topic mastery (BKT)"), use_container_width=True)
        st.dataframe(df, use_container_width=True)
        st.write(f"Weak topics: **{', '.join(learner.weakest(state))}**")

# ---------- 5. Evaluation ----------
with tabs[4]:
    st.header("System evaluation")
    st.write("RAGAS-style metrics (faithfulness, answer relevancy, context precision/recall) on a team-built test set, "
             "plus simulated-student personalization runs.")
    if st.button("Run RAG evaluation on tests/test_set.json"):
        ts = json.loads(Path("tests/test_set.json").read_text(encoding="utf-8"))
        with st.spinner("Evaluating…"):
            out = evaluation.evaluate_qa(ts)
        st.json({k: v for k, v in out.items() if k != "rows"})
        st.dataframe(pd.DataFrame(out["rows"]), use_container_width=True)
    sess = st.slider("Simulation sessions", 2, 10, 5)
    if st.button("Run simulated students (fast/average/struggling)"):
        with st.spinner("Simulating…"):
            sim = evaluation.simulate_students(sessions=sess)
        st.json(sim)

# ---------- 6. Revision ----------
with tabs[5]:
    st.header("Revision material & study schedule")
    state = learner.load()
    weak = learner.weakest(state) if state else list(kb.course_map())[:2]
    days = st.slider("Days before exam", 1, 30, 7)
    if st.button("Generate flashcards + schedule"):
        by_topic = {}
        for t in weak:
            by_topic[t] = kb.retrieve(t, top_k=3, topic_filter=t)
        cards = vizmod.flashcards(weak, by_topic)
        for c in cards:
            with st.expander(f"🔖 {c['front'][:90]}"):
                st.write(c["back"])
        st.subheader("Study schedule (forgetting-curve spaced)")
        sched = vizmod.study_schedule(weak, {t: learner.load().get(t, {}).get("p", 0.3) for t in weak}, days)
        st.dataframe(pd.DataFrame(sched), use_container_width=True)
