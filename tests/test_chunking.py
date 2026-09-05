import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import parent_child_chunks, _split_sections


def test_split_sections_no_headings_returns_body():
    sections = _split_sections("Just plain text with no headings at all.")
    assert sections == [("body", "Just plain text with no headings at all.")]


def test_split_sections_empty_text_returns_empty():
    assert _split_sections("") == []
    assert _split_sections("   ") == []


def test_split_sections_with_numbered_heading():
    text = "1.1 Waiting Period\nSome content about waiting periods here."
    sections = _split_sections(text)
    assert len(sections) >= 1
    assert any("Waiting Period" in title for title, _ in sections)


def test_parent_child_chunks_basic():
    pages = [{"page": 1, "text": "A" * 500 + "\n\n" + "B" * 500}]
    file_meta = {"source": "test.pdf", "policy_name": "test", "insurer": "unknown"}
    children, parents = parent_child_chunks(
        pages, file_meta,
        parent_size=2000, child_size=500, child_overlap=100, min_chars=40,
    )
    assert len(children) > 0
    assert len(parents) > 0
    for child in children:
        assert "parent_id" in child.metadata
        assert child.metadata["parent_id"] in parents


def test_parent_child_chunks_filters_short_content():
    pages = [{"page": 1, "text": "short"}]  # under min_chars
    file_meta = {"source": "test.pdf", "policy_name": "test", "insurer": "unknown"}
    children, parents = parent_child_chunks(
        pages, file_meta,
        parent_size=2000, child_size=500, child_overlap=100, min_chars=40,
    )
    assert children == []
    assert parents == {}


def test_child_id_uses_full_content_hash_not_just_prefix():
    # Two children sharing an identical 80-char prefix but differing later
    # must get different ids — this is the DuplicateIDError fix from v2 notes.
    prefix = "X" * 80
    pages = [{
        "page": 1,
        "text": prefix + "AAAA" + "\n\n" + prefix + "BBBB",
    }]
    file_meta = {"source": "test.pdf", "policy_name": "test", "insurer": "unknown"}
    children, _ = parent_child_chunks(
        pages, file_meta,
        parent_size=2000, child_size=2000, child_overlap=0, min_chars=10,
    )
    ids = [c.metadata["id"] for c in children]
    assert len(ids) == len(set(ids)), "duplicate chunk IDs despite differing content"