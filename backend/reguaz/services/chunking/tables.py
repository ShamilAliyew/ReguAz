from __future__ import annotations


def markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def markdown_header(headers: list[str]) -> str:
    if not headers:
        return ""
    return f"{markdown_row(headers)}\n{markdown_row(['---'] * len(headers))}"


def nonempty_rows(rows: list[list[str]]) -> list[list[str]]:
    """Drop empty form rows without dropping partially completed legal rows."""
    return [row for row in rows if any(cell.strip(" _.-") for cell in row)]


def row_groups(
    headers: list[str], rows: list[list[str]], *, max_tokens: int, count_tokens
) -> list[tuple[int, int, list[list[str]]]]:
    header = markdown_header(headers)
    groups: list[tuple[int, int, list[list[str]]]] = []
    current: list[list[str]] = []
    start = 1
    for row_number, row in enumerate(nonempty_rows(rows), start=1):
        candidate = "\n".join(filter(None, [header, *(markdown_row(r) for r in [*current, row])]))
        if current and count_tokens(candidate) > max_tokens:
            groups.append((start, row_number - 1, current))
            current = [row]
            start = row_number
        else:
            current.append(row)
    if current:
        groups.append((start, start + len(current) - 1, current))
    return groups
