from __future__ import annotations

import hashlib
import re
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

HEADING_RE = re.compile(
    r"^(#{1,4}\s+.+|[0-9]{1,2}\.[0-9.]{0,8}\s+\S.+|[A-Z][A-Z0-9 /&(),.-]{8,})$",
    re.MULTILINE,
)


def _split_sections(text: str) -> list[tuple[str, str]]:
    if not text.strip():
        return []
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("body", text)]
    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("preamble", preamble))
    for i, match in enumerate(matches):
        title = match.group(0).lstrip("#").strip()[:180]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        if body:
            sections.append((title, body))
    return sections or [("body", text)]


def parent_child_chunks(
    pages: list[dict],
    file_meta: dict,
    parent_size: int,
    child_size: int,
    child_overlap: int,
    min_chars: int,
) -> tuple[list[Document], dict[str, dict]]:
    """Build child Documents plus a parent_id -> {text, metadata} map."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=max(80, child_overlap),
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
    )
    children: list[Document] = []
    parents: dict[str, dict] = {}

    for page in pages:
        page_no = page.get("page", "?")
        sections = _split_sections(page.get("text") or "")
        for section_title, section_text in sections:
            parent_docs = parent_splitter.create_documents([section_text])
            if not parent_docs:
                parent_docs = [Document(page_content=section_text)]
            for p_idx, parent_doc in enumerate(parent_docs):
                parent_text = parent_doc.page_content.strip()
                if len(parent_text) < min_chars:
                    continue
                parent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_meta.get('source')}:{page_no}:{section_title}:{p_idx}"))
                parent_meta = {
                    **file_meta,
                    "page": page_no,
                    "section": section_title,
                    "parent_id": parent_id,
                }
                parents[parent_id] = {"text": parent_text, "metadata": parent_meta}
                child_docs = child_splitter.create_documents([parent_text])
                for c_idx, child in enumerate(child_docs):
                    content = child.page_content.strip()
                    if len(content) < min_chars:
                        continue
                    # Hash the FULL content, not a slice — insurance PDFs repeat
                    # near-identical boilerplate (standard exclusion clauses,
                    # page headers) across pages, so a truncated prefix collides
                    # far more often than you'd expect and Chroma rejects the
                    # whole upsert batch on any duplicate ID.
                    child_id = hashlib.sha1(
                        f"{parent_id}:{c_idx}:{content}".encode()
                    ).hexdigest()
                    children.append(
                        Document(
                            page_content=content,
                            metadata={
                                **parent_meta,
                                "id": child_id,
                                "parent_id": parent_id,
                            },
                        )
                    )
    return children, parents
