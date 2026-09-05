"""
Answer-level eval: runs the full agent (retrieval + calculation + LLM
synthesis) and checks whether the expected reimbursement figure actually
appears in the final answer. This is the eval that catches a hallucinated
co-pay or misread clause producing a confidently wrong number — the
retrieval eval only confirms the right document was found, not that the
agent extracted the right numbers from it.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import re
from pathlib import Path

from app.agent import build_agent
from langchain_core.messages import HumanMessage

GOLD_PATH = Path(__file__).parent / "gold_claims.json"


def normalize_number(n: int) -> list[str]:
    plain = str(n)
    comma = f"{n:,}"
    # Indian lakh-style grouping, e.g. 100000 -> "1,00,000"
    s = plain
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        # group the rest in pairs of 2 from the right
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        indian = ",".join(groups) + "," + last3
    else:
        indian = s
    return [plain, comma, indian, f"₹{plain}", f"₹{comma}", f"₹{indian}",
            f"Rs. {comma}", f"Rs {comma}", f"Rs. {indian}", f"Rs {indian}"]

from langgraph.errors import GraphRecursionError

import time

def run():
    gold = json.loads(GOLD_PATH.read_text())
    agent = build_agent()

    passed = 0
    for i, case in enumerate(gold):
        if i > 0:
            time.sleep(20)  # let the rate limit window reset between cases
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=case["question"])]},
                config={"recursion_limit": 25},
            )
            final = result["messages"][-1]
            answer = final.content if isinstance(final.content, str) else str(final.content)
        except GraphRecursionError:
            answer = "[recursion limit hit]"
        except Exception as e:
            answer = f"[error: {e}]"

        candidates = normalize_number(case["expected_reimbursement"])
        found = any(c in answer for c in candidates)

        status = "PASS" if found else "FAIL"
        if found:
            passed += 1
        print(f"{status} {case['id']}: {case['question']}")
        if not found:
            print(f"  expected one of {candidates}")
            print(f"  got: {answer[:300]}")

    print(f"\n{passed}/{len(gold)} answer-accuracy checks passed")
if __name__ == "__main__":
    run()