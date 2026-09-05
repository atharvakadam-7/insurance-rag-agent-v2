# Insurance Policy RAG Agent

An agent that answers questions about insurance policy PDFs and calculates claim reimbursements. Live at https://insurance-rag-agent-v2-production.up.railway.app

Ask it what a co-payment clause says, or what you'd get back on a claim. It retrieves the relevant clause, cites the page it came from, and runs the math in code instead of guessing the arithmetic.

## How it works

**Ingestion.** `pymupdf4llm` converts PDFs to markdown and keeps the section structure intact. Scanned pages that come back mostly blank fall back to OCR. Text gets cleaned of encoding artifacts before chunking. Chunking splits each document into small child chunks for search and larger parent chunks for context: a search hits a 500-character fragment, but the agent reads the full paragraph around it.

**Retrieval.** A query runs through two searches at once: a dense vector search (Chroma) and a keyword search (BM25). Insurance text uses terms like "sub-limit" or "co-payment" that keyword search catches more reliably than embeddings alone, so the two results merge through reciprocal rank fusion, then a cross-encoder (`flashrank`) reranks them. If the question names a policy, the agent filters results to that policy before ranking, so a question about HDFC doesn't pull in a Star Health clause. Retrieved chunk content is capped at 600 characters per chunk before it reaches the LLM, to stay under Groq's free-tier per-minute token limits during multi-step tool calls.

**Calculation.** The agent doesn't do arithmetic. It pulls the coverage percentage, co-pay, deductible, sub-limit, and room-rent cap from the policy text, then hands those numbers to a plain Python function that applies them in order: waiting-period check, room-rent proportionate deduction, deductible, sub-limit, coverage percentage, co-pay.

**Search discipline.** The agent is capped at 2 retrieval searches per question, enforced in code rather than prompted. A `pre_model_hook` counts tool calls and forces the model to stop searching and answer once it hits the limit. Past that point it either calculates with what it has, answers with what it has, or states plainly that the documents don't cover the question.

**Auditing.** Every query logs the exact tool calls made, the arguments passed to the calculator, and the answer returned, as JSON in `audit.log`. Trace a wrong number back to what the model actually extracted, not just what it printed.

## Stack

FastAPI, LangGraph, Groq (`qwen/qwen3.6-27b`), Chroma, `rank_bm25`, `flashrank`, `fastembed` for embeddings, `pymupdf4llm` and `rapidocr` for extraction. Deployed on Railway with the index built into the Docker image at build time.

## Running it locally

```bash
git clone https://github.com/atharvakadam-7/insurance-rag-agent-v2
cd insurance-rag-agent-v2
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add a `.env` file:
GROQ_API_KEY=your_key
GROQ_MODEL=qwen/qwen3.6-27b


Build the index and start the server:
```bash
python ingest.py
uvicorn app.main:app --reload
```

Add your own PDFs to `data/` and re-run `ingest.py`. Unchanged files get skipped, so you're not re-embedding the whole set every time.

## Testing

21 unit tests cover claim math, chunking, and hybrid rerank logic. No network or LLM calls, so they run in under a second:

```bash
python -m pytest tests/ -v
```

GitHub Actions runs them automatically on every push (`.github/workflows/tests.yml`).

## Evaluation

Retrieval quality: `evals/run_eval.py` checks whether hybrid retrieval surfaces the right source document across 15 gold questions covering general policy terms (waiting periods, co-payment, room-rent limits, exclusions, day-care, maternity, sum-insured, claims process) and insurer-specific clauses (Star Health, HDFC).

```bash
python evals/run_eval.py
```
**15/15 passed.**

Answer accuracy: `evals/run_answer_eval.py` runs the full agent (retrieval, claim calculation, and LLM synthesis) and checks whether the correct reimbursement figure shows up in the final answer. This catches cases where retrieval finds the right clause but the agent still misreads or miscalculates it.

```bash
python evals/run_answer_eval.py
```
**4/4 passed** on the current sample. This eval calls the live Groq API and needs `GROQ_API_KEY`. Four claim scenarios is a small sample: solid signal, not exhaustive coverage. It's not wired into CI, since that would need a live API key stored as a secret, and free-tier rate limits would make CI runs flaky through no fault of the code.

## What it gets right and what it doesn't

Retrieval and calculation stay split apart on purpose, so the model can't miscalculate a reimbursement. It can still misread a clause and hand the calculator a wrong number with total confidence. The audit log exists so you can trace a wrong answer back to what the model actually extracted.

The ingestion pipeline skips unchanged files but keeps no document versioning. Update a policy PDF mid-year and you get a fresh index with no record of what changed. Fine at a handful of PDFs, not built for hundreds.

Groq's free tier enforces tight rate limits for a multi-step agent: per-minute caps on both input and output tokens, plus a separate daily token cap. A single question can take 3 or more calls (search, calculation, final answer). The client retries automatically on per-minute limits, so queries still complete, just slower under load. The daily cap is a hard stop until it resets. The 600-character cap on retrieved chunk content is a rate-limit accommodation, not a design ideal, and it's worth relaxing on a paid tier for richer context per answer.

## Changelog

- Fixed: removed the `gpt-oss-20b` fallback model. It doesn't support tool calling, and it silently corrupted agent responses (empty answers, recursion-limit loops) whenever the primary model hit a rate limit and LangChain's `with_fallbacks` swapped models mid-conversation.
- Fixed: added `reasoning_effort="none"` to the primary LLM call. qwen's hidden chain-of-thought tokens were consuming the entire `max_tokens` budget, leaving nothing for the actual answer.
- Fixed: the agent was ignoring its own "search at most twice" system-prompt rule. A `pre_model_hook` now counts `search_policy_docs` calls in code and forces the model to stop searching and answer once it hits the limit.
- Changed: capped retrieved chunk content at 600 characters in `format_docs()` to stay under Groq free-tier input-token-per-minute limits during multi-step tool calls.
- Added: a 21-test pytest suite (`tests/test_claim.py`, `tests/test_chunking.py`, plus the existing `tests/test_hybrid.py`) and a GitHub Actions workflow that runs them on every push.
