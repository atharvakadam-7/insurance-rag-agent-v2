from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger("insurance_agent")

# All the "fancy" dash/hyphen look-alikes we want collapsed to a plain "-".
_DASH_CHARS = {
    "\u2010",  # hyphen
    "\u2011",  # non-breaking hyphen
    "\u2012",  # figure dash
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\u2015",  # horizontal bar
    "\u2212",  # minus sign
}
_OCR = None
_OCR_FAILED = False


def _get_ocr():
    global _OCR, _OCR_FAILED
    if _OCR_FAILED:
        return None
    if _OCR is not None:
        return _OCR
    try:
        from rapidocr_onnxruntime import RapidOCR

        _OCR = RapidOCR()
    except Exception as exc:
        logger.info("OCR unavailable (%s); scanned pages will be skipped", exc)
        _OCR_FAILED = True
        return None
    return _OCR


def infer_policy_meta(path: str) -> dict:
    name = Path(path).stem
    lower = name.lower().replace("_", " ").replace("-", " ")
    insurer = "unknown"
    for token, label in (
        ("hdfc", "HDFC Ergo"),
        ("star", "Star Health"),
        ("icici", "ICICI Lombard"),
        ("lic", "LIC"),
        ("niva", "Niva Bupa"),
        ("care", "Care Health"),
        ("new india", "New India Assurance"),
        ("oriental", "Oriental"),
        ("united", "United India"),
    ):
        if token in lower:
            insurer = label
            break
    return {
        "source": path,
        "policy_name": name,
        "insurer": insurer,
        "product": name,
    }


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("â¯â¥â¯", " >= ")
    text = text.replace("â¥", " >= ")
    text = text.replace("â¤", " <= ")
    text = text.replace("â¯", " ")
    text = text.replace("â", "-")
    text = text.replace("\xa0", " ")
    text = text.replace("\u202f", " ")
    text = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in text)
    text = "".join("-" if ch in _DASH_CHARS else ch for ch in text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def _ocr_page(page) -> str:
    ocr = _get_ocr()
    if ocr is None:
        return ""
    try:
        pix = page.get_pixmap(dpi=150)
        result, _ = ocr(pix.tobytes("png"))
        if not result:
            return ""
        return "\n".join(row[1] for row in result if len(row) > 1)
    except Exception as exc:
        logger.warning("OCR failed on a page: %s", exc)
        return ""


def extract_pdf_pages(path: str, ocr_min_chars: int = 40) -> list[dict]:
    """Return [{page, text, tables_markdown}] using pymupdf / pymupdf4llm."""
    pages: list[dict] = []
    try:
        import pymupdf4llm

        chunks = pymupdf4llm.to_markdown(path, page_chunks=True)
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("markdown") or ""
                meta = chunk.get("metadata") or {}
                page_no = meta.get("page", i)
            else:
                text = str(chunk)
                page_no = i
            pages.append(
                {
                    "page": int(page_no) + 1 if isinstance(page_no, int) and page_no < 10000 else i + 1,
                    "text": clean_text(text),
                }
            )
    except Exception as exc:
        logger.warning("pymupdf4llm failed for %s (%s); falling back to PyMuPDF", path, exc)
        pages = []

    if pages and any(p["text"] for p in pages):
        return _ocr_fill(path, pages, ocr_min_chars)

    import fitz

    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = clean_text(page.get_text("text") or "")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return _ocr_fill(path, pages, ocr_min_chars)


def _ocr_fill(path: str, pages: list[dict], ocr_min_chars: int) -> list[dict]:
    need_ocr = [p for p in pages if len(p["text"]) < ocr_min_chars]
    if not need_ocr:
        return pages
    import fitz

    doc = fitz.open(path)
    for p in pages:
        if len(p["text"]) >= ocr_min_chars:
            continue
        idx = p["page"] - 1
        if 0 <= idx < len(doc):
            ocr_text = clean_text(_ocr_page(doc[idx]))
            if len(ocr_text) > len(p["text"]):
                p["text"] = ocr_text
                p["ocr"] = True
    doc.close()
    return pages