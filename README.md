# Insurance policy agent

Agentic RAG system: FastAPI + LangGraph + Groq (Llama 3.3) + Chroma.
The agent retrieves from insurance policy PDFs and can run reimbursement
calculations, instead of just answering from retrieved text.

## Architecture

```
User -> FastAPI /query -> LangGraph agent (ReAct loop)
                              |-- search_policy_docs (Chroma retriever)
                              |-- calculate_claim_reimbursement
                              |-- compare_policy_clauses
                              -> Groq LLM (Llama 3.3)
```

## 1. Local setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your real Groq API key in .env
```

Get a free Groq API key at https://console.groq.com — takes two minutes,
no card required as of this writing. Check their model list too; model
names in `.env.example` get deprecated without much notice.

## 2. Add your policy documents

Drop PDF files into `data/`. Public policy wordings from IRDAI-regulated
insurers (LIC, HDFC Ergo, ICICI Lombard, Star Health, etc.) are the
intended source — they're public documents, no licensing issue.

## 3. Build the index

```bash
python ingest.py
```

Re-run this any time you add or change PDFs. It wipes and rebuilds the
whole index rather than appending — appending to a stale index is how you
get duplicate or contradictory chunks.

## 4. Run locally

```bash
uvicorn app.main:app --reload
```

Test it:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Does this policy cover COVID hospitalization?"}'
```

## 5. Run with Docker

```bash
docker build -t insurance-rag-agent .
docker run -p 8000:8000 --env-file .env insurance-rag-agent
```

The image bakes in the embedding model AND the vectorstore built from
whatever is in `data/` at build time. If you change the PDFs, rebuild the
image — there's no way to update a running container's index in place on
this setup, and Render's free tier wipes disk on every deploy anyway.

## 6. Deploy to Render

1. Push this repo to GitHub.
2. New Web Service on Render -> connect the repo -> Docker runtime
   (Render auto-detects the Dockerfile).
3. Add `GROQ_API_KEY` as an environment variable in Render's dashboard —
   never commit it, `.env` is already gitignored.
4. Deploy. Cold starts on the free tier are slow (spins down after
   inactivity) — mention this if you demo it live, don't let a recruiter
   think it's broken while it wakes up.

## Known limitations

- No conversation memory — every `/query` call is stateless. Multi-turn
  follow-ups ("what about my second claim?") won't have context. Fixing
  this means adding a session/thread ID and LangGraph's checkpointer.
- No reranking or hybrid search yet — pure vector similarity. That's the
  Phase 5 upgrade from the roadmap, not done here.
- No eval suite yet — you don't actually know the retrieval quality is
  good, you're assuming it. Ragas comes in Phase 4.
- `create_react_agent`'s exact keyword arguments can shift between
  LangGraph versions. If `agent.py` throws a TypeError on `prompt=`, check
  the installed version's signature — don't assume this code is
  permanently correct.
- `ingest.py` uses `PyPDFLoader` from `langchain-community`, which
  Anthropic's own research confirms is being sunset by the LangChain team
  (see langchain-ai/langchain-community#674) — it still works today, but
  it's not where new development is going. Fine for getting this running
  now; don't leave it there if you're still using this repo in a few
  months. Standalone replacements like `langchain-pymupdf4llm` exist.
