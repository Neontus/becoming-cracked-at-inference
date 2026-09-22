#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 posts/<slug>.md /path/to/Neontus.github.io" >&2
  exit 2
fi

source_note="$1"
site_repo="$2"

if [[ ! -f "$source_note" ]]; then
  echo "Note not found: $source_note" >&2
  exit 1
fi

if [[ "$(basename "$(dirname "$source_note")")" != "posts" ]]; then
  echo "Source must be a Markdown file inside posts/." >&2
  exit 1
fi

if [[ ! -f "$site_repo/package.json" ]] || ! grep -q '"name": "neontus"' "$site_repo/package.json"; then
  echo "Destination does not look like the Neontus portfolio repository: $site_repo" >&2
  exit 1
fi

if grep -Eq '^draft:[[:space:]]*true[[:space:]]*$' "$source_note"; then
  echo "Refusing to publish a draft. Set draft: false after review." >&2
  exit 1
fi

python3 "$(dirname "$0")/check_public_notes.py"
mkdir -p "$site_repo/content/writing"
cp "$source_note" "$site_repo/content/writing/$(basename "$source_note")"

source_assets="$(dirname "$source_note")/assets"
if [[ -d "$source_assets" ]]; then
  mkdir -p "$site_repo/public/writing/assets"
  cp -R "$source_assets/." "$site_repo/public/writing/assets/"
fi

echo "Published source to $site_repo/content/writing/$(basename "$source_note")"
echo "Next: run npm run build in the site repository, then review and commit both repositories."
