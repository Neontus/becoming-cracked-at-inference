#!/usr/bin/env python3
"""Create a public-note draft from the repository template."""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 3:
        print(
            'Usage: python3 scripts/new_public_note.py slug "Post title"',
            file=sys.stderr,
        )
        return 2

    slug, title = sys.argv[1].strip(), sys.argv[2].strip()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        print("Slug must contain lowercase letters, numbers, and single hyphens.", file=sys.stderr)
        return 2
    if not title:
        print("Title cannot be empty.", file=sys.stderr)
        return 2

    destination = ROOT / "posts" / f"{slug}.md"
    if destination.exists():
        print(f"Refusing to overwrite {destination.relative_to(ROOT)}", file=sys.stderr)
        return 1

    today = date.today().isoformat()
    template = (ROOT / "scripts" / "templates" / "public-note.md").read_text()
    destination.write_text(template.replace("{{date}}", today).replace("{{title}}", title))
    print(destination.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

