from pathlib import Path


SUPPORTED_SUFFIXES = {
    ".docx",
    ".txt",
    ".pdf",
}


class DocumentLoadError(ValueError):
    """Raised when a supported document cannot be decoded safely."""


def load_text_file(
    path: str | Path,
) -> str:
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise DocumentLoadError(f"Could not read TXT document: {error}") from error


def _format_docx_paragraph(
    paragraph,
) -> str:
    text = paragraph.text.strip()

    if not text:
        return ""

    style_name = getattr(
        paragraph.style,
        "name",
        "",
    ).strip().casefold()

    if style_name == "title":
        return f"# {text}"

    if style_name.startswith(
        "heading "
    ):
        level_text = (
            style_name
            .removeprefix(
                "heading "
            )
            .strip()
        )

        if level_text.isdigit():
            level = max(
                1,
                min(
                    int(
                        level_text
                    ),
                    6,
                ),
            )

            return (
                f"{'#' * level} {text}"
            )

    return text


def _format_docx_cell(
    cell,
) -> str:
    from docx.table import Table

    from docx.text.paragraph import Paragraph

    parts = []

    for block in cell.iter_inner_content():
        if isinstance(
            block,
            Paragraph,
        ):
            text = " ".join(
                block.text.split()
            )

            if text:
                parts.append(
                    text
                )

        elif isinstance(
            block,
            Table,
        ):
            nested_table = _format_docx_table(
                block
            )

            if nested_table:
                parts.append(
                    " ; ".join(
                        nested_table.splitlines()
                    )
                )

    return " ; ".join(
        parts
    )


def _format_docx_table(
    table,
) -> str:
    rows: list[list[str]] = []

    for row in table.rows:
        values = []

        seen_cells: set[int] = set()

        for cell in row.cells:
            cell_id = id(
                cell._tc
            )

            if cell_id in seen_cells:
                continue

            seen_cells.add(
                cell_id
            )

            values.append(
                _format_docx_cell(
                    cell
                )
            )

        if any(
            values
        ):
            rows.append(
                values
            )

    if not rows:
        return ""

    if len(rows) == 1:
        return " | ".join(
            value
            for value in rows[0]
            if value
        )

    headers = rows[0]

    lines = [
        " | ".join(
            value
            for value in headers
            if value
        )
    ]

    for row in rows[1:]:
        values = []

        for index, value in enumerate(
            row
        ):
            if not value:
                continue

            header = (
                headers[index]
                if index < len(headers)
                else ""
            )

            formatted_value = (
                f"{header}: {value}"
                if header
                else value
            )

            values.append(
                formatted_value
            )

        if values:
            lines.append(
                " | ".join(
                    values
                )
            )

    return "\n".join(
        line
        for line in lines
        if line
    )


def load_docx(
    path: str | Path,
) -> str:
    try:
        from docx import Document

        from docx.table import Table

        from docx.text.paragraph import Paragraph

    except ImportError as error:
        raise RuntimeError(
            "Install python-docx to read .docx files."
        ) from error

    try:
        document = Document(path)
        blocks = []

        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                text = _format_docx_paragraph(block)
            elif isinstance(block, Table):
                text = _format_docx_table(block)
            else:
                continue

            if text:
                blocks.append(text)
    except Exception as error:
        raise DocumentLoadError(f"Could not read DOCX document: {error}") from error

    return "\n\n".join(
        blocks
    )


def load_pdf(
    path: str | Path,
) -> str:
    try:
        import pymupdf

    except ImportError as error:
        raise RuntimeError(
            "Install pymupdf to read .pdf files."
        ) from error

    try:
        with pymupdf.open(path) as document:
            pages = [
                page.get_text("text", sort=True).strip()
                for page in document
            ]
    except Exception as error:
        raise DocumentLoadError(f"Could not read PDF document: {error}") from error

    return "\n\n".join(
        page
        for page in pages
        if page
    )


def load_document(
    path: str | Path,
) -> str:
    suffix = Path(
        path
    ).suffix.lower()

    if suffix == ".docx":
        return load_docx(
            path
        )

    if suffix == ".txt":
        return load_text_file(
            path
        )

    if suffix == ".pdf":
        return load_pdf(
            path
        )

    raise ValueError(
        f"Unsupported file type: {suffix or 'unknown'}"
    )
