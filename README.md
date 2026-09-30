# Insurance Policy RAG Agent


https://github.com/user-attachments/assets/9cc82a5c-a3e1-48c3-8c41-aebb171ef565





An agent that answers questions about insurance policy PDFs and calculates claim reimbursements.

Ask it what a co-payment clause says, or what you'd get back on a claim. It retrieves the relevant clause, cites the page it came from, and runs the math in code instead of guessing the arithmetic.

## How it works

**Ingestion.** `pymupdf4llm` converts PDFs to markdown and keeps the section structure intact. Scanned pages that come back mostly blank fall back to OCR. Text gets cleaned of encoding artifacts before chunking. Chunking splits each document into small child chunks for search and larger parent chunks for context: a search hits a 500-character fragment, but the agent reads the full paragraph around it.

**Retrieval.** A query runs through two searches at once: a dense vector search (Chroma) and a keyword search (BM25). Insurance text uses terms like "sub-limit" or "co-payment" that keyword search catches more reliably than embeddings alone, so the two results merge through reciprocal rank fusion, then a cross-encoder (`flashrank`) reranks them. If the question names a policy, the agent filters results to that policy before ranking, so a question about HDFC doesn't pull in a Star Health clause. Retrieved chunk content is capped at 600 characters per chunk before it reaches the LLM, to stay under Groq's free-tier per-minute token limits during multi-step tool calls.

**Calculation.** The agent doesn't do arithmetic. It pulls the coverage percentage, co-pay, deductible, sub-limit, and room-rent cap from the policy text, then hands those numbers to a plain Python function that applies them in order: waiting-period check, room-rent proportionate deduction, deductible, sub-limit, coverage percentage, co-pay.

**Search discipline.** The agent is hard-capped at 2 retrieval searches per question, enforced inside the search tool itself, not just prompted. Past the limit, `search_policy_docs` returns a fixed refusal string instead of running another search. An earlier version of this enforcement lived in a `pre_model_hook` that injected a system-message warning before the limit-reaching call. That approach was advisory: the model sometimes ignored it under pressure on harder questions. The current version is a hard gate at the tool level, using LangGraph's `InjectedState` to check the message history from inside the tool.

**Monitoring.** Every query is traced end-to-end in LangSmith: each LLM call, each tool call with its arguments and return value, latency per step, and token usage. A flat JSON audit log (`audit.log`) is kept as a lightweight secondary record, but LangSmith is the primary way to inspect what happened on a specific query.

**Auditing.** Every query logs the exact tool calls made, the arguments passed to the calculator, and the answer returned, as JSON in `audit.log`. Trace a wrong number back to what the model actually extracted, not just what it printed.

## Stack

FastAPI, LangGraph, Groq (`qwen/qwen3.8-27b`), Chroma, `rank_bm25`, `flashrank`, `fastembed` for embeddings, `pymupdf4llm` and `rapidocr` for extraction, LangSmith for tracing and monitoring. Deployed on Railway with the index built into the Docker image at build time.

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
GROQ_MODEL=qwen/qwen3.8-27b

# Optional — enables LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=insurance-rag-agent-v2
# Only needed if your LangSmith workspace is on a non-US region
LANGCHAIN_ENDPOINT=https://apac.api.smith.langchain.com
```

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
**4/4 passed**, confirmed on repeated runs including the previously flaky senior-co-payment case (see Changelog). This eval calls the live Groq API and needs `GROQ_API_KEY`. Four claim scenarios is a small sample: solid signal, not exhaustive coverage. It's not wired into CI, since that would need a live API key stored as a secret, and free-tier rate limits would make CI runs flaky through no fault of the code.

## What it gets right and what it doesn't

Retrieval and calculation stay split apart on purpose, so the model can't miscalculate a reimbursement. It can still misread a clause and hand the calculator a wrong number with total confidence. LangSmith traces and the audit log exist so you can trace a wrong answer back to what the model actually extracted.

The search agent generates a fresh query string on every run rather than using a fixed one, so two runs of the same question can search slightly differently. On most questions this doesn't matter: the relevant clause ranks near the top regardless of phrasing. On narrower questions where the needed clause competes with several similar sections (the age-based co-payment clause is the clearest example), that variance can occasionally mean a 2-search budget doesn't surface it. Asking with the policy's own terminology ("co-payment percentage for a senior citizen aged 65") retrieves more consistently than a casual phrasing of the same question.

The ingestion pipeline skips unchanged files but keeps no document versioning. Update a policy PDF mid-year and you get a fresh index with no record of what changed. Fine at a handful of PDFs, not built for hundreds.

Groq's free tier enforces tight rate limits for a multi-step agent: per-minute caps on both input and output tokens, plus a separate daily token cap that appears to age out on a rolling window rather than resetting at a fixed time. A single question can take 3 or more calls (search, calculation, final answer). The client retries automatically on per-minute limits, so queries still complete, just slower under load. The daily cap is a hard stop until it recovers. The 600-character cap on retrieved chunk content works around Groq's rate limits, not an ideal design choice. A paid tier would let it relax for richer context per answer.

## Changelog

- Fixed: removed the `gpt-oss-20b` fallback model. It doesn't support tool calling, and it silently corrupted agent responses (empty answers, recursion-limit loops) whenever the primary model hit a rate limit and LangChain's `with_fallbacks` swapped models mid-conversation.
- Fixed: added `reasoning_effort="none"` to the primary LLM call. qwen's hidden chain-of-thought tokens were consuming the entire `max_tokens` budget, leaving nothing for the actual answer.
- Fixed: the agent was ignoring its own "search at most twice" system-prompt rule. A first attempt at fixing it (a `pre_model_hook` injecting a stop instruction) turned out to be advisory rather than enforced. The model sometimes searched again anyway on harder questions, still hitting the recursion limit. Root-caused via a LangSmith trace showing repeated search loops on the failing case. Real fix: `search_policy_docs` now enforces the limit itself via `InjectedState`, returning a refusal string instead of a real result past 2 calls. Confirmed fixed across repeated runs.
- Fixed: the model would sometimes skip calling `calculate_claim_reimbursement` entirely and compute a reimbursement figure in prose instead, right after exhausting its 2-search budget. Measured a 50% failure rate on a targeted set of test questions before the fix. Root cause: the search-limit system notice told the model to stop searching but didn't explicitly mandate calling the calculator as its next move. Rewrote the notice to require the tool call as its first branch. Retested against the same failing cases: failure rate dropped to effectively zero.
- Changed: capped retrieved chunk content at 600 chars in `format_docs()` to stay under Groq free-tier input-token-per-minute limits during multi-step tool calls.
- Added: a 21-test pytest suite (`tests/test_claim.py`, `tests/test_chunking.py`, plus the existing `tests/test_hybrid.py`) and a GitHub Actions workflow that runs them on every push.
- Added: LangSmith tracing to step through agent runs in detail, replacing the flat audit log as the primary way to debug a specific query. Setup required a non-default `LANGCHAIN_ENDPOINT` for accounts on a non-US LangSmith region; see the `.env` example above.
