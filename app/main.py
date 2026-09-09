from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel
import json
import logging
from datetime import datetime
import os
import uuid
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


from .agent import build_agent
from .pdf_extract import clean_text

app = FastAPI(title="Insurance Policy Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

_agent = None
_checkpointer_conn = None

NOT_FOUND_ANSWER = "The provided policy documents do not contain information about this."

# Audit logging — every tool call, every decision
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
if not audit_logger.handlers:
    handler = logging.FileHandler("audit.log")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)

def get_agent():
    global _agent, _checkpointer_conn
    if _agent is None:
        _checkpointer_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
        checkpointer = SqliteSaver(_checkpointer_conn)
        _agent = build_agent(checkpointer=checkpointer)
    return _agent

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    session_id: str

def log_audit(event: dict):
    """Log structured audit event as JSON."""
    event["timestamp"] = datetime.utcnow().isoformat()
    audit_logger.info(json.dumps(event))


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/audit")
def audit(n: int = 20):
    """Return the last n query events from audit.log as JSON."""
    log_path = "audit.log"
    if not os.path.exists(log_path):
        return {"entries": []}
    
    with open(log_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    
    return {"count": len(entries), "entries": entries[-n:]}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
	
    session_id = req.session_id or str(uuid.uuid4())
    agent = get_agent()
    tool_calls = []

    try:
        # Log the incoming query
        log_audit({"event": "query_start", "question": req.question,"session_id": session_id})

        result = agent.invoke(
            {"messages": [HumanMessage(content=req.question)]},
            config={"recursion_limit": 25,  "configurable": {"thread_id": session_id}},
        )

        final_message = result["messages"][-1]

        # Extract tool calls from the agent's message history
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "tool": tc.get("name", "unknown"),
                        "args": tc.get("args", {}),
                    })

        if isinstance(final_message.content, list):
            answer = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in final_message.content
            )
        else:
            answer = str(final_message.content)

        answer = clean_text(answer)

        if not answer.strip():
            answer = NOT_FOUND_ANSWER

        # Log the successful query with full context
        log_audit({
            "event": "query_complete",
            "question": req.question,
            "tool_calls": tool_calls,
            "answer_preview": answer[:200],
            "status": "success",
        })

        return QueryResponse(answer=answer, session_id=session_id)

    except GraphRecursionError:
        log_audit({
            "event": "query_complete",
            "question": req.question,
            "tool_calls": tool_calls,
            "status": "recursion_limit_exceeded",
        })
        return QueryResponse(answer=NOT_FOUND_ANSWER,  session_id=session_id)

    except Exception as e:
        log_audit({
            "event": "query_error",
            "question": req.question,
            "tool_calls": tool_calls,
            "error": str(e),
            "status": "error",
        })
        raise HTTPException(status_code=500, detail=str(e))

