"""
Stage 1 — PDF Text Extraction for Legal/Compliance RAG Pipeline
================================================================
Input  : data/raw/          (category subfolders with PDFs)
Output : data/processed/cleaned_documents/  (same folder structure, .md files)
"""

import re
import logging
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_DIR  = Path("data/raw")
OUTPUT_DIR = Path("data/processed/cleaned_documents")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "extraction.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def remove_standalone_page_numbers(text: str) -> str:
    """
    Remove lines that contain only a page number (integer),
    which pdfplumber often picks up from footers/headers.
    """
    lines = text.splitlines()
    cleaned = [
        line for line in lines
        if not re.fullmatch(r"\s*\d{1,4}\s*", line)
    ]
    return "\n".join(cleaned)


def normalize_whitespace(text: str) -> str:
    """
    • Collapse multiple consecutive blank lines into one.
    • Strip trailing whitespace from every line.
    • Ensure the result does not start or end with blank lines.
    """
    lines = [line.rstrip() for line in text.splitlines()]

    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue          # skip consecutive blank lines
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).strip()


def clean_page_text(raw_text: str) -> str:
    """
    Apply all cleaning steps to a single page's raw text.
    Order matters:
      1. Remove standalone page numbers first (before collapsing blanks).
      2. Normalize whitespace last.
    """
    text = remove_standalone_page_numbers(raw_text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: Path) -> str:
    """
    Extract and clean text from every page of *pdf_path*.

    Returns a single string with <!-- PAGE: X --> markers
    separating the pages.  Empty pages are skipped silently.
    """
    sections: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, 1):
            raw = page.extract_text()

            # Skip pages with no extractable text (scanned images, etc.)
            if not raw or not raw.strip():
                log.debug("    Page %d/%d — no text, skipped.", page_num, total)
                continue

            cleaned = clean_page_text(raw)
            if not cleaned:
                continue

            sections.append(f"<!-- PAGE: {page_num} -->\n\n{cleaned}")
            log.debug("    Page %d/%d — extracted %d chars.", page_num, total, len(cleaned))

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def mirror_output_path(pdf_path: Path, input_dir: Path, output_dir: Path) -> Path:
    """
    Compute the output .md path that mirrors the input category structure.

    Example
    -------
    input_dir  = data/raw
    pdf_path   = data/raw/banking/kredits/document.pdf
    output_dir = data/processed/cleaned_documents
    → returns   data/processed/cleaned_documents/banking/kredits/document.md
    """
    relative = pdf_path.relative_to(input_dir)
    return output_dir / relative.with_suffix(".md")


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path) -> bool:
    """
    Extract, clean, and save one PDF.
    Returns True on success, False on failure.
    """
    out_path = mirror_output_path(pdf_path, INPUT_DIR, OUTPUT_DIR)

    log.info("→ %s", pdf_path.relative_to(INPUT_DIR))

    try:
        content = extract_pdf(pdf_path)
    except Exception as exc:
        log.error("  ✗ Extraction failed: %s", exc)
        return False

    if not content.strip():
        log.warning("  ⚠ No extractable text found — file skipped.")
        return False

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        log.error("  ✗ Could not write output file: %s", exc)
        return False

    log.info("  ✓ Saved → %s  (%d chars)", out_path, len(content))
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_DIR.exists():
        log.error("Input directory not found: %s", INPUT_DIR)
        return

    pdf_files = sorted(INPUT_DIR.rglob("*.pdf"))
    if not pdf_files:
        log.warning("No PDF files found under: %s", INPUT_DIR)
        return

    log.info("Input  : %s", INPUT_DIR)
    log.info("Output : %s", OUTPUT_DIR)
    log.info("PDFs   : %d file(s) found", len(pdf_files))
    log.info("=" * 60)

    success = failed = 0
    for pdf_path in pdf_files:
        if process_pdf(pdf_path):
            success += 1
        else:
            failed += 1

    log.info("=" * 60)
    log.info("Finished.  Success: %d | Failed/Skipped: %d", success, failed)
    log.info("Output folder: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()