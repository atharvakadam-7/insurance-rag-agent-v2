from langchain_core.tools import tool
from pydantic import ValidationError

from .claim import calculate_claim
from .retriever import format_docs, hybrid_retrieve


from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from .retriever import format_docs, hybrid_retrieve

SEARCH_LIMIT = 2

@tool
def search_policy_docs(
    query: str,
    state: Annotated[dict, InjectedState],
    policy_filter: str = "",
) -> str:
    """Search the insurance policy documents... (same docstring as before)"""
    prior_searches = sum(
        1
        for msg in state["messages"]
        if hasattr(msg, "tool_calls") and msg.tool_calls
        for tc in msg.tool_calls
        if tc.get("name") == "search_policy_docs"
    )
    if prior_searches >= SEARCH_LIMIT:
        return (
            "SEARCH LIMIT REACHED. No further searches will be performed. "
            "You must now answer using only what you've already retrieved: "
            "calculate a reimbursement with default values for anything not "
            "found, answer the coverage question with what you have, or "
            "state that the documents don't cover this."
        )
    try:
        docs = hybrid_retrieve(query, policy_filter=policy_filter or None)
        if not docs:
            return "No relevant policy sections found for that query."
        return format_docs(docs)
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

@tool
def calculate_claim_reimbursement(
    claim_amount: float,
    coverage_percent: float = 100.0,
    deductible: float = 0.0,
    copay_percent: float = 0.0,
    sublimit: float = 0.0,
    room_rent_cap: float = 0.0,
    room_rent_claimed: float = 0.0,
    waiting_period_active: bool | str = False,
) -> str:
    """Calculate the estimated reimbursement for a claim, applying room-rent
    proportionate deduction, deductible, sub-limit, coverage %, and co-pay
    in the correct order. Retrieve every percentage/limit used here from the
    policy docs first via search_policy_docs — never guess a number.

    claim_amount: total claim value in rupees.
    coverage_percent: coverage percentage from the policy, e.g. 80 for 80%.
    deductible: excess that applies before coverage kicks in.
    copay_percent: co-payment percentage the insured bears, e.g. 10 for 10%.
    sublimit: rupee cap on this claim category, if any (0 = no sub-limit).
    room_rent_cap / room_rent_claimed: if the policy caps room rent per day
      and the claimed room rent exceeds it, the whole claim is scaled down
      proportionately. Pass both as 0 if not applicable or not a hospitalization claim.
    waiting_period_active: True if a waiting period still blocks this claim
      entirely (check the docs for the condition's specific waiting period).
    """
    try:
        # Coerce string bools to actual bools (some LLMs pass "False" as string)
        if isinstance(waiting_period_active, str):
            waiting_period_active = waiting_period_active.strip().lower() not in ("false", "0", "no", "")

        # Validate numeric types — if the agent passed bad types, fail gracefully
        try:
            claim_amount = float(claim_amount)
            coverage_percent = float(coverage_percent)
            deductible = float(deductible)
            copay_percent = float(copay_percent)
            sublimit = float(sublimit) if sublimit else None
            room_rent_cap = float(room_rent_cap) if room_rent_cap else None
            room_rent_claimed = float(room_rent_claimed) if room_rent_claimed else None
        except (ValueError, TypeError) as e:
            return f"Error: invalid numeric argument. {str(e)} Ensure all amounts are numbers."

        result = calculate_claim(
            claim_amount=claim_amount,
            coverage_percent=coverage_percent,
            deductible=deductible,
            copay_percent=copay_percent,
            sublimit=sublimit,
            room_rent_cap=room_rent_cap,
            room_rent_claimed=room_rent_claimed,
            waiting_period_active=waiting_period_active,
        )
        return result.as_text()
    except Exception as e:
        return f"Error calculating claim: {str(e)}"


TOOLS = [search_policy_docs, calculate_claim_reimbursement]