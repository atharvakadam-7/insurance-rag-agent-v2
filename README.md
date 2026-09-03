# Insurance Policy RAG Agent

An agentic RAG (Retrieval-Augmented Generation) system that answers questions about insurance policy documents — coverage, exclusions, waiting periods, co-payments — and calculates claim reimbursements with real policy math, not LLM guesswork.

**Live demo:** https://insurance-rag-agent-v2-production.up.railway.app

---

## Why this exists

Insurance policy documents are long, dense, and full of clauses that change based on exact conditions (age, entry date, treatment type, room category). A plain vector-search chatbot tends to either hallucinate numbers or miss the specific clause that actually applies. This project treats retrieval and calculation as two separate, verifiable steps: the agent must *find* the relevant clause before it's allowed to *calculate* anything, and the math itself is done in code, not by the LLM.

---

## Architecture

```
                        ┌─────────────────────┐
                        │   FastAPI /query     │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │  LangGraph ReAct Agent │
                        │  (Groq qwen3.6-27b)    │
                        └──────────┬───────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
  ┌───────────▼──────────┐ ┌───────▼────────┐ ┌──────────▼─────────┐
  │  search_policy_docs   │ │ calculate_claim │ │ compare_policy_    │
  │  (hybrid retrieval)   │ │ _reimbursement   │ │ clauses            │
  └───────────┬──────────┘ └───────┬────────┘ └────────────────────┘
              │                    │
   ┌──────────▼──────────┐  ┌──────▼───────┐
   │ Dense (Chroma) +     │  │ claim.py:     │
   │ Sparse (BM25) fusion │  │ deductible →  │
   │ → flashrank rerank   │  │ room-rent cap │
   │ → parent expansion   │  │ → sub-limit → │
   └──────────────────────┘  │ coverage % →  │
                              │ co-pay        │
                              └──────────────┘
```

### Ingestion pipeline (`ingest.py` → `app/indexing.py`)
1. **Extraction** (`pdf_extract.py`) — `pymupdf4llm` converts each PDF page to markdown, preserving structure. Pages with too little extracted text (scanned/image pages) automatically fall back to OCR via `rapidocr`.
2. **Cleaning** — a `clean_text()` pass strips PDF encoding artifacts (mangled non-breaking spaces, mojibake dashes) that otherwise leak into the model's final answer.
3. **Chunking** (`chunking.py`) — parent-child splitting: large (~2000 char) parent chunks preserve full context, small (~500 char) child chunks are what actually gets embedded and searched. Section headers (ALL-CAPS lines) are detected and attached as metadata so citations can reference a named section, not just a page number.
4. **Indexing** — child chunks go into a Chroma vectorstore (dense) and a BM25 index (sparse); parent chunks go into a lookup store used for context expansion after retrieval. Hash-based caching means unchanged PDFs are skipped on re-ingest.

### Retrieval (`retriever.py`, `hybrid.py`)
1. Dense search (Chroma) and sparse search (BM25) each return their own candidate list.
2. **Reciprocal Rank Fusion (RRF)** merges both ranked lists into one, so a chunk that scores well on *either* semantic similarity or exact keyword match surfaces near the top — this matters a lot for insurance text, where an exact term like "sub-limit" or "co-payment" can be more reliable than embedding similarity alone.
3. **flashrank** re-ranks the fused candidates with a cross-encoder for a final relevance pass.
4. **Parent expansion** — the top child chunks are swapped for their parent chunks before being handed to the LLM, so the model sees full surrounding context instead of an isolated 500-character fragment.
5. **Policy filtering** — if the user names a specific insurer/policy, results are filtered to that policy's metadata before ranking, preventing clauses from different policies (different co-pay rules, different waiting periods) from bleeding into one answer.

### Claim calculation (`claim.py`)
A pure-Python function, not an LLM guess. Applies, in order:
1. Waiting-period check (blocks the claim outright if active)
2. Room-rent proportionate deduction (if claimed room rent exceeds the policy's per-day cap, the *entire* claim — not just the room charge — is scaled down proportionately, matching how Indian health insurers actually calculate this)
3. Deductible
4. Sub-limit cap
5. Coverage percentage
6. Co-payment

The agent's job is only to retrieve the correct percentages/caps from the policy text and pass them as arguments — it never does the arithmetic itself.

### Agent orchestration (`agent.py`)
A LangGraph `create_react_agent` with a system prompt enforcing:
- Always search before answering (never answer from general knowledge)
- Natural-language search queries, not keyword strings (matches semantic search behavior much better)
- Max two search attempts before giving up cleanly
- If no restricting clause (co-pay, deductible, cap) is found for a claim, treat that as "no restriction applies" and calculate full reimbursement — rather than incorrectly returning "not found"
- Cite the source document and page/section for every claim
- Never leak raw tool-call syntax into the final answer

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | async, typed, fast to iterate |
| Orchestration | LangGraph (`create_react_agent`) | tool-calling agent loop with built-in recursion limits |
| LLM | Groq (`qwen/qwen3.6-27b`, fallback `openai/gpt-oss-20b`) | fast inference, free tier available, tool-calling support |
| Vector store | Chroma | lightweight, no external DB needed |
| Sparse retrieval | `rank_bm25` | keyword-exact matching to complement embeddings |
| Reranking | `flashrank` | cross-encoder rerank without a heavy dependency |
| Embeddings | `fastembed` (BAAI/bge-small-en-v1.5) | ONNX-based, no PyTorch — critical for staying under free-tier RAM limits |
| PDF parsing | `pymupdf4llm` + `rapidocr` fallback | structure-aware markdown extraction with OCR for scanned pages |
| Deployment | Docker on Railway | free tier includes it fully baked at build time (no cold-start indexing) |

---

## Evaluation

`evals/run_eval.py` runs a gold-question set (`evals/gold_questions.json`) against the retrieval pipeline and checks whether the expected source document is present in the top results.

**Current result: 8/8 (100%) retrieval accuracy** across questions covering waiting periods, co-payment rules, room-rent limits, exclusions, day-care procedures, maternity coverage, sum-insured tiers, and claims process.

```bash
python evals/run_eval.py
```

---

## Setup

```bash
git clone https://github.com/atharvakadam-7/insurance-rag-agent-v2
cd insurance-rag-agent-v2
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
GROQ_MODEL=qwen/qwen3.6-27b
GROQ_FALLBACK_MODEL=openai/gpt-oss-20b
```

Build the index from the PDFs in `data/`:
```bash
python ingest.py
```

Run the server:
```bash
uvicorn app.main:app --reload
```

Query it:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the co-payment for insured persons above 60?"}'
```

---

## Example queries

- *"What is the waiting period for pre-existing diseases?"* — compares waiting periods across both indexed policies with buy-back/reduction options
- *"Reimbursement on a ₹50,000 claim at age 65 under Star Comprehensive?"* — retrieves the applicable co-pay clause, calculates the exact reimbursement, and cites the page it came from
- *"What is the room rent limit under HDFC Optima Secure?"* — surfaces plan-variant-specific caps (Lite / Select / Select Optional)

---

## What's in `data/`

Two sample policy PDFs are included for demo purposes:
- `star_comprehensive.pdf`
- `hdfc_optima_secure.pdf`

Add your own PDFs to `data/` and re-run `python ingest.py` to index them — unchanged files are automatically skipped on subsequent runs via content-hash caching.

---

## Known limitations

- Free-tier Groq rate limits (8K TPM / 30 RPM on `qwen/qwen3.6-27b`) mean multi-step agentic queries can hit 429s and retry — the client handles this automatically but it adds latency under load.
- Retrieval eval currently checks source-document presence, not exact answer correctness — a stricter answer-level eval (e.g. checking the calculated reimbursement figure matches expected) is a natural next step.
- No conversation memory between queries yet — each `/query` call is stateless.
