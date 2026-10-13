"""Unified LLM client: Gemini -> OpenAI -> local fallback. Keeps prototype runnable with zero API key."""
from __future__ import annotations
import os
from . import config


def active_provider() -> str:
    gk = (config.GOOGLE_API_KEY or "").strip()
    ok = (config.OPENAI_API_KEY or "").strip()
    # keys that are empty or missing libs don't count
    try:
        import google.generativeai  # noqa
        has_gem = True
    except Exception:
        has_gem = False
    try:
        import openai  # noqa
        has_oai = True
    except Exception:
        has_oai = False
    pref = (config.LLM_PROVIDER or "auto").lower()
    if pref in ("gemini", "openai", "local"):
        if pref == "gemini" and (not gk or not has_gem):
            return "local"
        if pref == "openai" and (not ok or not has_oai):
            return "local"
        return pref
    if gk and has_gem:
        return "gemini"
    if ok and has_oai:
        return "openai"
    return "local"


def generate(prompt: str, system: str = "", max_tokens: int = 800) -> str:
    provider = active_provider()
    if provider == "gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GOOGLE_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            full = (system + "\n\n" + prompt) if system else prompt
            resp = model.generate_content(full)
            return (resp.text or "").strip()
        except Exception as e:
            print(f"[llm] gemini failed, falling back to local: {e}")
    if provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model="gpt-4o-mini", messages=msgs, max_tokens=max_tokens, temperature=0.3
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[llm] openai failed, falling back to local: {e}")
    return _local_generate(prompt, system)


def _local_generate(prompt: str, system: str = "") -> str:
    """Extractive/template fallback so demo works offline. Caller passes source-grounded prompts;
    we echo a structured grounded answer rather than hallucinating."""
    # The RAG layer already builds excerpts; local mode just formats them.
    # If prompt contains 'EXCERPTS:', return first excerpts as the answer body.
    if "EXCERPTS:" in prompt:
        body = prompt.split("EXCERPTS:", 1)[1][:1800]
        return (
            "Based on your uploaded course material (offline mode — no LLM key configured):\n\n"
            f"{body.strip()}\n\n"
            "Note: connect a Gemini/OpenAI key in .env for fluent explanations. "
            "All claims above are copied from cited excerpts."
        )
    if "QUIZ" in prompt.upper() or "QUESTION" in prompt.upper():
        return "LOCAL_MODE_QUESTION"
    return prompt[:1200]


def hindi_explain(text: str) -> str:
    """Optional Hindi explanation helper (req 6d). Uses LLM if available else prefix."""
    p = active_provider()
    if p == "local":
        return "(Hindi explanation needs an LLM key. English source-backed answer above.)\n" + text
    return generate(f"Explain the following in simple Hindi (Hinglish ok), keeping technical terms in English:\n\n{text}",
                    system="You are a bilingual tutor.")
