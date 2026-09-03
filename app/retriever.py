from __future__ import annotations

import logging

from langchain_core.documents import Document

from .config import CANDIDATE_K, TOP_K
from .hybrid import bm25_rank, expand_parents, flashrank_rerank, reciprocal_rank_fusion
from .store import (
    bm25_path,
    get_vectorstore,
    load_json,
    parents_path,
    similarity_search,
)

logger = logging.getLogger("insurance_agent")

_bm25_cache: dict | None = None
_parents_cache: dict | None = None


def invalidate_caches() -> None:
    global _bm25_cache, _parents_cache
    _bm25_cache = None
    _parents_cache = None


def _load_bm25():
    global _bm25_cache
    if _bm25_cache is None:
        _bm25_cache = load_json(bm25_path(), {"ids": [], "texts": [], "metadatas": []})
    return _bm25_cache


def _load_parents():
    global _parents_cache
    if _parents_cache is None:
        _parents_cache = load_json(parents_path(), {})
    return _parents_cache


def _bm25_docs() -> list[Document]:
    data = _load_bm25()
    docs = []
    for text, meta in zip(data.get("texts") or [], data.get("metadatas") or []):
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def _matches_policy(meta: dict, policy_filter: str | None) -> bool:
    if not policy_filter:
        return True
    needle = policy_filter.lower().strip()
    blob = " ".join(
        str(meta.get(key) or "")
        for key in ("policy_name", "insurer", "product", "source")
    ).lower().replace("_", " ").replace("-", " ")
    return needle in blob


def hybrid_retrieve(
    query: str,
    k: int = TOP_K,
    policy_filter: str | None = None,
    candidate_k: int = CANDIDATE_K,
) -> list[Document]:
    vectorstore = get_vectorstore()
    chroma_filter = None
    dense = similarity_search(vectorstore, query, k=candidate_k, metadata_filter=chroma_filter)
    sparse = bm25_rank(query, _load_bm25().get("texts") or [], _bm25_docs(), k=candidate_k)
    fused = [doc for doc, _ in reciprocal_rank_fusion([dense, sparse])]
    if policy_filter:
        fused = [d for d in fused if _matches_policy(d.metadata or {}, policy_filter)]
        dense_f = [d for d in dense if _matches_policy(d.metadata or {}, policy_filter)]
        if not fused:
            fused = dense_f
    reranked = flashrank_rerank(query, fused[: max(candidate_k, k)], top_n=max(k * 2, k))
    expanded = expand_parents(reranked, _load_parents())
    return expanded[:k]


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("policy_name") or d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        section = d.metadata.get("section", "")
        header = f"[Doc {i} | {source} | page {page}"
        if section:
            header += f" | {section}"
        header += "]"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


def docs_to_citations(docs: list[Document]) -> list[dict]:
    citations = []
    seen = set()
    for d in docs:
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("section"),
        )
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "policy_name": d.metadata.get("policy_name") or "",
                "insurer": d.metadata.get("insurer") or "",
                "source": d.metadata.get("source") or "",
                "page": d.metadata.get("page"),
                "section": d.metadata.get("section") or "",
                "snippet": (d.page_content or "")[:280],
            }
        )
    return citations
