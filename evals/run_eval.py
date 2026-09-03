"""Retrieval eval against evals/gold_questions.json. Does not call Groq."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retriever import hybrid_retrieve
from app.store import count_indexed


def keywords_hit(text: str, keywords: list[str]) -> bool:
    blob = text.lower()
    return any(k.lower() in blob for k in keywords)


def main() -> int:
    gold_path = Path(__file__).with_name("gold_questions.json")
    questions = json.loads(gold_path.read_text(encoding="utf-8"))
    if count_indexed() == 0:
        print("SKIP: no index (run python ingest.py with PDFs in data/)")
        return 0

    passed = 0
    for item in questions:
        docs = hybrid_retrieve(item["question"], policy_filter=item.get("policy"))
        combined = "\n".join(d.page_content for d in docs)
        ok = keywords_hit(combined, item.get("expected_keywords") or [])
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"{status} {item['id']}: {item['question']}")
        if not ok:
            print(f"  keywords={item.get('expected_keywords')} n_docs={len(docs)}")

    total = len(questions)
    print(f"\n{passed}/{total} retrieval checks passed")
    return 0 if passed >= max(1, total // 2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
