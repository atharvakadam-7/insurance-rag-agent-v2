from langchain_core.documents import Document

from app.hybrid import expand_parents, reciprocal_rank_fusion


def test_rrf_prefers_overlap():
    a = Document(page_content="a", metadata={"id": "1"})
    b = Document(page_content="b", metadata={"id": "2"})
    c = Document(page_content="c", metadata={"id": "3"})
    fused = reciprocal_rank_fusion([[a, b, c], [b, a, c]])
    assert fused[0][0].metadata["id"] in {"1", "2"}
    assert fused[0][1] > fused[-1][1]


def test_expand_parents_dedupes():
    child1 = Document(page_content="c1", metadata={"id": "c1", "parent_id": "p"})
    child2 = Document(page_content="c2", metadata={"id": "c2", "parent_id": "p"})
    parents = {"p": {"text": "full parent", "metadata": {"parent_id": "p", "section": "X"}}}
    out = expand_parents([child1, child2], parents)
    assert len(out) == 1
    assert out[0].page_content == "full parent"
