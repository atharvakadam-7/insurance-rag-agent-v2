from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from .agent import build_agent
from .pdf_extract import clean_text

app = FastAPI(title="Insurance Policy Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

_agent = None

NOT_FOUND_ANSWER = "The provided policy documents do not contain information about this."


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    agent = get_agent()

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=req.question)]},
            config={"recursion_limit": 25},
        )
        final_message = result["messages"][-1]

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

        return QueryResponse(answer=answer)

    except GraphRecursionError:
        return QueryResponse(answer=NOT_FOUND_ANSWER)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))