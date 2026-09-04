# Insurance Policy RAG Agent

An agent that answers questions about insurance policy PDFs and calculates claim reimbursements. Live at https://insurance-rag-agent-v2-production.up.railway.app

Ask it what a co-payment clause says, or what you'd get back on a claim, and it retrieves the relevant clause, cites the page it came from, and runs the math in code rather than guessing the arithmetic.

## How it works

**Ingestion.** PDFs are converted to markdown with `pymupdf4llm`, which keeps section structure intact. Pages that come back mostly blank (scanned pages) go through OCR as a fallback. Text gets cleaned of encoding artifacts before it's chunked. Chunking splits each document into small child chunks for search and larger parent chunks for context — a search hits a 500-character fragment, but the agent reads the full paragraph around it.

**Retrieval.** A query runs through two searches at once: a dense vector search (Chroma) and a keyword search (BM25). Insurance text has terms like "sub-limit" or "co-payment" that keyword search catches more reliably than embeddings alone, so the two results are merged with reciprocal rank fusion, then reranked with a cross-encoder (`flashrank`). If the question names a policy, results are filtered to that policy before ranking, so a question about HDFC doesn't pull in a Star Health clause.

**Calculation.** The agent doesn't do arithmetic. It retrieves the coverage percentage, co-pay, deductible, sub-limit, and room-rent cap from the policy text, then hands those numbers to a plain Python function that applies them in the right order: waiting-period check, room-rent proportionate deduction, deductible, sub-limit, coverage percentage, co-pay.

**Auditing.** Every query logs the exact tool calls made, the arguments passed to the calculator, and the answer returned, as JSON in `audit.log`. If a number looks wrong later, you can trace exactly what the model extracted and what it fed into the math.

## Stack

FastAPI, LangGraph, Groq (`qwen/qwen3.6-27b`, falls back to `openai/gpt-oss-20b`), Chroma, `rank_bm25`, `flashrank`, `fastembed` for embeddings, `pymupdf4llm` and `rapidocr` for extraction. Deployed on Railway with the index built into the Docker image at build time.

## Running it locally

```bash
git clone https://github.com/atharvakadam-7/insurance-rag-agent-v2
cd insurance-rag-agent-v2
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add a `.env` file:
```
GROQ_API_KEY=your_key
GROQ_MODEL=qwen/qwen3.6-27b
GROQ_FALLBACK_MODEL=openai/gpt-oss-20b
```

Build the index and start the server:
```bash
python ingest.py
uvicorn app.main:app --reload
```

Add your own PDFs to `data/` and re-run `ingest.py`. Files that haven't changed are skipped, so you're not re-embedding the whole set every time.

## Evaluation

`evals/run_eval.py` checks whether retrieval surfaces the right source document for a set of gold questions covering waiting periods, co-payment, room-rent limits, exclusions, day-care coverage, maternity, sum-insured tiers, and the claims process.

```bash
python evals/run_eval.py
```

Currently passes 8 of 8.

## What it gets right and what it doesn't

Retrieval and calculation are split apart on purpose, so the model can't miscalculate a reimbursement, but it can still misread a clause and hand the calculator a wrong number with total confidence. The audit log exists so a wrong answer can be traced back to what the model actually extracted, not just what it printed.

The ingestion pipeline skips unchanged files but has no document versioning. If a policy PDF is updated mid-year, you'd re-run ingestion and get a fresh index, with no record of exactly what changed. Fine at a handful of PDFs, not built for hundreds.

Groq's free tier has tight rate limits for a multi-step agent, since search, calculation, and the final answer can be three separate calls. The client retries automatically, so queries still complete, just slower under load than on a paid tier.