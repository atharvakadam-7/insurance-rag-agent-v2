from __future__ import annotations

from langchain_groq import ChatGroq

from .config import GROQ_API_KEY, GROQ_FALLBACK_MODEL, GROQ_MODEL, LLM_TIMEOUT_SECONDS, require_groq_key

_llm = None


def get_llm():
    global _llm
    require_groq_key()
    if _llm is not None:
        return _llm

    primary = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
        timeout=LLM_TIMEOUT_SECONDS,
    )

    if GROQ_FALLBACK_MODEL:
        fallback = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_FALLBACK_MODEL,
            temperature=0,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        _llm = primary.with_fallbacks([fallback])
    else:
        _llm = primary

    return _llm