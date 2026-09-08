import os
from dotenv import load_dotenv

load_dotenv(override=True)

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Check Groq's console for current model names — they deprecate models
# without much warning. This was correct at time of writing.
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
print(f"[startup] Using GROQ_MODEL={GROQ_MODEL}")

# --- Embeddings / vector store ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "policies")
# "chroma" (default, local/free-tier friendly) or "pgvector" (needs DATABASE_URL)
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma")
DATABASE_URL = os.getenv("DATABASE_URL")  # only used if VECTOR_BACKEND=pgvector

# --- Ingestion / chunking ---
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_DIR = os.getenv("INDEX_DIR", "index_store")  # holds bm25.json, parents.json, manifest.json
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "500"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "100"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "40"))
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "40"))  # page text shorter than this triggers OCR fallback

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "4"))
# How many candidates dense + BM25 each pull before fusion/rerank narrows to TOP_K
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))


def require_groq_key():
    """Called lazily (not at import time) so ingest.py and tests don't need
    a Groq key just to build the vectorstore."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")