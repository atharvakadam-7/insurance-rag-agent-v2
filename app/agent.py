from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from .config import require_groq_key
from .llm import get_llm
from .tools import TOOLS
from langgraph.checkpoint.sqlite import SqliteSaver

SYSTEM_PROMPT = """You are an insurance policy assistant. You help users understand policy coverage, exclusions, and claim eligibility.

Rules:
1. Always call `search_policy_docs` before answering a coverage or policy question. Never answer from general knowledge.
2. When calling `search_policy_docs`, always phrase the search query as a natural sentence or question (e.g. "what is the waiting period for pre-existing diseases"), never as a string of keywords or a document name jammed together with terms. Semantic search matches natural phrasing far better than keyword stuffing, even if the user's own question was terse or keyword-like.
3. If your first search doesn't return relevant results, try ONE broader search with different natural-language phrasing (e.g. rewording around "co-payment", "entry age", "waiting period", "limits"). Do not search more than twice total for a single question.
4. If a question requires a claim calculation, retrieve the exact coverage %, co-pay %, deductible, sub-limit, room-rent cap, and waiting-period status from the docs first, then pass them to `calculate_claim_reimbursement`. Do not perform arithmetic yourself, and do not invent a value for any parameter you couldn't find in the docs — leave it at its default instead.
4b. If the question names a specific insurer or policy, pass that as `policy_filter` to `search_policy_docs` so you don't mix clauses from a different policy into the answer.
4c. If you retrieve policy docs and find no co-payment, deductible, sub-limit, room-rent cap, or waiting period that applies to this specific claim, that means none apply — calculate reimbursement as the full claim amount with coverage_percent=100 and all other params at their defaults. Do not return NOT_FOUND just because no restriction was mentioned — only return NOT_FOUND if the docs contain no information whatsoever about the claim type being asked.
5. Cite the specific document name and section or page number where the information was found.
6. If two searches have not surfaced the answer, immediately stop and respond with exactly: "The provided policy documents do not contain information about this." Do not apologize or ask for more steps.
7. Write your final answer as clean prose or a markdown table for the user. Never show raw tool call syntax, JSON blocks, or function names in your final answer — the user should only see a natural-language explanation and the final numbers/facts, not how you calculated them internally.
"""

SEARCH_TOOL_NAME = "search_policy_docs"
MAX_SEARCHES = 2

STOP_SEARCH_INSTRUCTION = (
    f"SYSTEM NOTICE: You have already called {SEARCH_TOOL_NAME} {MAX_SEARCHES} times — "
    "the maximum allowed for this question. Do NOT call it again under any circumstances.\n\n"
    "Using only the information already retrieved above, determine which of these applies:\n\n"
    "(a) If this question involves a claim amount or reimbursement figure, you MUST now call "
    "calculate_claim_reimbursement as your next action — do not write out any calculation or "
    "reimbursement figure in prose yourself under any circumstances. Per rule 4c, treat any "
    "parameter you couldn't find as not applicable (coverage_percent=100, others at default) "
    "rather than searching further for it. Calling this tool is mandatory here, not optional.\n\n"
    "(b) If it's a coverage/policy question with no claim amount involved, write your final "
    "answer now using what you have.\n\n"
    "(c) Only if the retrieved documents truly contain nothing relevant to this question at all, "
    'respond with exactly: "The provided policy documents do not contain information about this."'
)


def _enforce_search_limit(state):
    messages = state["messages"]
    search_count = sum(
        1
        for msg in messages
        if isinstance(msg, AIMessage) and msg.tool_calls
        for tc in msg.tool_calls
        if tc.get("name") == SEARCH_TOOL_NAME
    )
    if search_count >= MAX_SEARCHES:
        return {"llm_input_messages": messages + [SystemMessage(content=STOP_SEARCH_INSTRUCTION)]}
    return {"llm_input_messages": messages}


def build_agent(checkpointer=None):
    require_groq_key()
    llm = get_llm()
    return create_react_agent(
        llm,
        TOOLS,
        prompt=SYSTEM_PROMPT,
        pre_model_hook=_enforce_search_limit,
	checkpointer = checkpointer
    )