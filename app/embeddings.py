from __future__ import annotations

from langchain_community.embeddings import FastEmbedEmbeddings

from .config import EMBEDDING_MODEL

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings
