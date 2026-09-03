from langchain_core.tools import tool

from .claim import calculate_claim
from .retriever import format_docs, hybrid_retrieve


@tool
def search_policy_docs(query: str, policy_filter: str = "") -> str:
    """Search the insurance policy documents for information relevant to the
    query. Use this whenever the user asks about coverage, exclusions,
    waiting periods, or any specific clause in a policy. Always call this
    before answering a coverage question — never answer from memory.
    policy_filter: optional insurer or policy name to restrict results to
    (e.g. "Star Health"), leave empty to search all policies."""
    docs = hybrid_retrieve(query, policy_filter=policy_filter or None)
    if not docs:
        return "No relevant policy sections found for that query."
    return format_docs(docs)


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
    if isinstance(waiting_period_active, str):
        waiting_period_active = waiting_period_active.strip().lower() not in ("false", "0", "no", "")
    result = calculate_claim(
        claim_amount=claim_amount,
        coverage_percent=coverage_percent,
        deductible=deductible,
        copay_percent=copay_percent,
        sublimit=sublimit or None,
        room_rent_cap=room_rent_cap or None,
        room_rent_claimed=room_rent_claimed or None,
        waiting_period_active=waiting_period_active,
    )
    return result.as_text()


@tool
def compare_policy_clauses(clause_a: str, clause_b: str) -> str:
    """Compare two policy clause texts (already retrieved via
    search_policy_docs) on coverage, exclusions, and conditions. Pass the
    actual clause text, not a policy name — this tool doesn't retrieve
    anything itself."""
    return (
        "Compare these two policy clauses on coverage, exclusions, "
        f"and conditions:\n\nClause A:\n{clause_a}\n\nClause B:\n{clause_b}"
    )


TOOLS = [search_policy_docs, calculate_claim_reimbursement, compare_policy_clauses]
