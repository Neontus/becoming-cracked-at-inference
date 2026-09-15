#!/usr/bin/env python3
"""Validate the small frontmatter contract used by the portfolio."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("title", "summary", "date", "updated", "stage", "order", "draft", "tags")


def parse_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text()
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        raise ValueError("missing YAML frontmatter")

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if line.startswith("  - ") or not line.strip():
            continue
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip('"')
    return metadata, match.group(2)


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        metadata, body = parse_frontmatter(path)
    except ValueError as exc:
        return [str(exc)]

    for key in REQUIRED:
        if key not in metadata:
            errors.append(f"missing `{key}`")
    for key in ("date", "updated"):
        if key in metadata:
            try:
                date.fromisoformat(metadata[key])
            except ValueError:
                errors.append(f"`{key}` must use YYYY-MM-DD")
    if metadata.get("draft") not in {"true", "false"}:
        errors.append("`draft` must be true or false")
    try:
        int(metadata.get("order", ""))
    except ValueError:
        errors.append("`order` must be an integer")
    if metadata.get("draft") == "false":
        if len(metadata.get("summary", "")) < 30:
            errors.append("published `summary` must be descriptive")
        if "Replace this" in body or "—" in body and "| —" in body:
            errors.append("published note still contains template placeholders")
    return errors


def main() -> int:
    failed = False
    for path in sorted((ROOT / "public-notes").glob("*.md")):
        errors = validate(path)
        if errors:
            failed = True
            for error in errors:
                print(f"{path.relative_to(ROOT)}: {error}")
        else:
            print(f"ok  {path.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

