#!/usr/bin/env python3
"""Create a dated session note from the repository template."""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print('Usage: python3 scripts/new_session.py "topic"', file=sys.stderr)
        return 2

    today = date.today().isoformat()
    topic = sys.argv[1].strip()
    destination = ROOT / "notes" / "sessions" / today[:7] / f"{today}-{slugify(topic)}.md"
    if destination.exists():
        print(f"Refusing to overwrite {destination.relative_to(ROOT)}", file=sys.stderr)
        return 1

    template = (ROOT / "scripts" / "templates" / "session-note.md").read_text()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("{{date}}", today).replace("{{topic}}", topic))
    print(destination.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

