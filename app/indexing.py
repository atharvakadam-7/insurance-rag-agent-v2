from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path

from langchain_core.documents import Document

from .chunking import parent_child_chunks
from .config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    INDEX_DIR,
    MIN_CHUNK_CHARS,
    OCR_MIN_CHARS,
    PARENT_CHUNK_SIZE,
    VECTOR_BACKEND,
)
from .pdf_extract import extract_pdf_pages, infer_policy_meta
from .retriever import invalidate_caches
from .store import (
    add_documents,
    bm25_path,
    delete_by_source,
    get_vectorstore,
    load_json,
    manifest_path,
    parents_path,
    save_json,
)

logger = logging.getLogger("insurance_agent")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_paths(data_dir: str | None = None) -> list[str]:
    folder = Path(data_dir or DATA_DIR)
    if not folder.is_dir():
        return []
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() == ".pdf")


def _merge_parents(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    merged.update(incoming)
    return merged


def _rewrite_bm25_for_sources(existing: dict, source: str, new_docs: list[Document]) -> dict:
    ids, texts, metas = [], [], []
    for i, text, meta in zip(
        existing.get("ids") or [],
        existing.get("texts") or [],
        existing.get("metadatas") or [],
    ):
        if (meta or {}).get("source") == source:
            continue
        ids.append(i)
        texts.append(text)
        metas.append(meta)
    for doc in new_docs:
        ids.append(doc.metadata.get("id"))
        texts.append(doc.page_content)
        metas.append(doc.metadata)
    return {"ids": ids, "texts": texts, "metadatas": metas}


def index_pdf(path: str, vectorstore=None) -> dict:
    vectorstore = vectorstore or get_vectorstore()
    meta = infer_policy_meta(path)
    pages = extract_pdf_pages(path, ocr_min_chars=OCR_MIN_CHARS)
    children, parents = parent_child_chunks(
        pages,
        meta,
        parent_size=PARENT_CHUNK_SIZE,
        child_size=CHILD_CHUNK_SIZE,
        child_overlap=CHILD_CHUNK_OVERLAP,
        min_chars=MIN_CHUNK_CHARS,
    )
    if not children:
        logger.warning("No usable chunks from %s (empty or scanned without OCR)", path)
        return {"source": path, "chunks": 0, "pages": len(pages)}

    delete_by_source(vectorstore, path)
    add_documents(vectorstore, children)

    all_parents = load_json(parents_path(), {})
    all_parents = {k: v for k, v in all_parents.items() if v.get("metadata", {}).get("source") != path}
    save_json(parents_path(), _merge_parents(all_parents, parents))

    bm25 = load_json(bm25_path(), {"ids": [], "texts": [], "metadatas": []})
    save_json(bm25_path(), _rewrite_bm25_for_sources(bm25, path, children))
    invalidate_caches()
    return {"source": path, "chunks": len(children), "pages": len(pages), "policy_name": meta["policy_name"]}


def build_index(data_dir: str | None = None, force: bool = False) -> dict:
    pdfs = _pdf_paths(data_dir)
    if not pdfs:
        logger.warning("No PDFs found in %s", data_dir or DATA_DIR)
        return {"indexed": 0, "skipped": 0, "files": []}

    manifest = load_json(manifest_path(), {"embedding_model": None, "files": {}})
    if force or manifest.get("embedding_model") != EMBEDDING_MODEL:
        if VECTOR_BACKEND != "pgvector" and os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
        os.makedirs(INDEX_DIR, exist_ok=True)
        save_json(parents_path(), {})
        save_json(bm25_path(), {"ids": [], "texts": [], "metadatas": []})
        manifest = {"embedding_model": EMBEDDING_MODEL, "files": {}}

    vectorstore = get_vectorstore()
    current_hashes = {path: file_sha256(path) for path in pdfs}
    indexed, skipped = [], []

    removed = [src for src in (manifest.get("files") or {}) if src not in current_hashes]
    for src in removed:
        delete_by_source(vectorstore, src)
        parents = load_json(parents_path(), {})
        save_json(
            parents_path(),
            {k: v for k, v in parents.items() if v.get("metadata", {}).get("source") != src},
        )
        bm25 = load_json(bm25_path(), {"ids": [], "texts": [], "metadatas": []})
        save_json(bm25_path(), _rewrite_bm25_for_sources(bm25, src, []))

    for path, digest in current_hashes.items():
        prev = (manifest.get("files") or {}).get(path) or {}
        if not force and prev.get("hash") == digest:
            skipped.append(path)
            continue
        result = index_pdf(path, vectorstore=vectorstore)
        manifest.setdefault("files", {})[path] = {
            "hash": digest,
            "chunks": result["chunks"],
        }
        indexed.append(result)

    manifest["embedding_model"] = EMBEDDING_MODEL
    save_json(manifest_path(), manifest)
    invalidate_caches()
    return {"indexed": len(indexed), "skipped": len(skipped), "files": indexed}
