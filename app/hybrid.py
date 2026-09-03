from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.metadata.get("id") or f"{doc.metadata.get('source')}:{doc.page_content[:80]}"
            scores[key] += 1.0 / (k + rank)
            docs[key] = doc
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(docs[key], score) for key, score in ordered]


def bm25_rank(query: str, corpus_texts: list[str], corpus_docs: list[Document], k: int) -> list[Document]:
    if not corpus_texts or not query.strip():
        return []
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    tokenized_query = query.lower().split()
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [corpus_docs[i] for i in ranked_idx if scores[i] > 0]


def flashrank_rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    if not docs:
        return []
    try:
        from flashrank import Ranker, RerankRequest
    except Exception:
        return docs[:top_n]
    try:
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        ordered = []
        for item in results:
            idx = item["id"] if isinstance(item, dict) else getattr(item, "id", None)
            if idx is None:
                continue
            ordered.append(docs[int(idx)])
        return (ordered or docs)[:top_n]
    except Exception:
        return docs[:top_n]


def expand_parents(docs: list[Document], parents: dict) -> list[Document]:
    expanded: list[Document] = []
    seen = set()
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id in parents and parent_id not in seen:
            seen.add(parent_id)
            payload = parents[parent_id]
            expanded.append(
                Document(page_content=payload["text"], metadata=payload.get("metadata") or doc.metadata)
            )
        elif not parent_id:
            key = doc.metadata.get("id") or doc.page_content[:40]
            if key not in seen:
                seen.add(key)
                expanded.append(doc)
    return expanded or docs
