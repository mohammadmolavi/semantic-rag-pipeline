from pathlib import Path


SUPPORTED_SUFFIXES = {".docx", ".txt", ".pdf"}


def load_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_docx(path: str | Path) -> str:
    try:
        from docx import Document
    except ImportError as error:
        raise RuntimeError("Install python-docx to read .docx files.") from error

    document = Document(path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    return "\n".join(paragraph for paragraph in paragraphs if paragraph)

def load_pdf(path: str | Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as error:
        raise RuntimeError("Install pymupdf to read .pdf files.") from error

    document = fitz.open(path)
    pages = [page.get_text("text").strip() for page in document]
    return "\n".join(page for page in pages if page)


def load_document(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".docx":
        return load_docx(path)
    if suffix == ".txt":
        return load_text_file(path)
    if suffix == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
