from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.documents import Document

from .config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    DATABASE_URL,
    INDEX_DIR,
    VECTOR_BACKEND,
)
from .embeddings import get_embeddings

logger = logging.getLogger("insurance_agent")

PARENTS_FILE = "parents.json"
BM25_FILE = "bm25.json"
MANIFEST_FILE = "manifest.json"


def index_dir() -> Path:
    path = Path(INDEX_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parents_path() -> Path:
    return index_dir() / PARENTS_FILE


def bm25_path() -> Path:
    return index_dir() / BM25_FILE


def manifest_path() -> Path:
    return index_dir() / MANIFEST_FILE


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def get_vectorstore():
    embeddings = get_embeddings()
    if VECTOR_BACKEND == "pgvector":
        if not DATABASE_URL:
            raise RuntimeError("VECTOR_BACKEND=pgvector requires DATABASE_URL")
        from langchain_postgres import PGVector

        return PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
        )
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def delete_by_source(vectorstore, source: str) -> None:
    try:
        if hasattr(vectorstore, "delete") and VECTOR_BACKEND == "pgvector":
            vectorstore.delete(filter={"source": source})
            return
        collection = getattr(vectorstore, "_collection", None)
        if collection is None:
            return
        existing = collection.get(where={"source": source})
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)
    except Exception as exc:
        logger.warning("Could not delete existing chunks for %s: %s", source, exc)


def add_documents(vectorstore, docs: list[Document]) -> None:
    ids = [d.metadata.get("id") for d in docs]
    if all(ids):
        # Belt-and-suspenders: even with full-content hashing, don't let one
        # genuine duplicate ID (e.g. truly identical boilerplate chunk
        # appearing twice) crash the entire batch upsert. Keep first occurrence.
        seen: set[str] = set()
        deduped_docs, deduped_ids = [], []
        for doc, doc_id in zip(docs, ids):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            deduped_docs.append(doc)
            deduped_ids.append(doc_id)
        vectorstore.add_documents(deduped_docs, ids=deduped_ids)
    else:
        vectorstore.add_documents(docs)


def similarity_search(vectorstore, query: str, k: int, metadata_filter: dict | None = None):
    kwargs = {"k": k}
    if metadata_filter:
        if VECTOR_BACKEND == "pgvector":
            kwargs["filter"] = metadata_filter
        else:
            kwargs["filter"] = metadata_filter
    try:
        return vectorstore.similarity_search(query, **kwargs)
    except Exception:
        return vectorstore.similarity_search(query, k=k)


def count_indexed() -> int:
    try:
        store = get_vectorstore()
        if VECTOR_BACKEND == "pgvector":
            return len(load_json(bm25_path(), {}).get("ids") or [])
        collection = getattr(store, "_collection", None)
        if collection is None:
            return 0
        return collection.count()
    except Exception:
        return 0


def list_policies() -> list[dict]:
    data = load_json(bm25_path(), {})
    metas = data.get("metadatas") or []
    seen: dict[str, dict] = {}
    for meta in metas:
        name = (meta or {}).get("policy_name") or (meta or {}).get("source")
        if not name or name in seen:
            continue
        seen[name] = {
            "policy_name": (meta or {}).get("policy_name") or name,
            "insurer": (meta or {}).get("insurer", "unknown"),
            "source": (meta or {}).get("source", ""),
        }
    return sorted(seen.values(), key=lambda x: x["policy_name"].lower())
