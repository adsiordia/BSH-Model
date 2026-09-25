#!/usr/bin/env bash
# Publish site/index.html to a Hugging Face static Space.
#
#   hf auth login                                      # once, interactive
#   ./deploy/sync_huggingface.sh <hf-username>/<space-name>
#
# Creates the Space if it does not exist, then uploads the current build.
# The page is a single self-contained file, so there is nothing else to ship.
set -euo pipefail

REPO="${1:?usage: sync_huggingface.sh <hf-username>/<space-name>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="$ROOT/site/index.html"
CARD="$ROOT/deploy/huggingface_README.md"

[ -f "$SITE" ] || { echo "No build at $SITE — run: python site/build.py" >&2; exit 1; }
command -v hf >/dev/null || { echo "hf CLI not found — pip install -U huggingface_hub" >&2; exit 1; }
hf auth whoami >/dev/null 2>&1 || { echo "Not logged in — run: hf auth login" >&2; exit 1; }

SIZE_MB=$(awk "BEGIN{printf \"%.2f\", $(stat -c%s "$SITE")/1048576}")
echo "==> build is ${SIZE_MB} MB"

echo "==> creating the Space if it does not exist"
hf repos create "$REPO" --repo-type space --space-sdk static --exist-ok

echo "==> uploading index.html"
hf upload "$REPO" "$SITE" index.html --repo-type space \
  --commit-message "Update explorer build ($(date -u +%Y-%m-%d))"

echo "==> uploading the Space card"
hf upload "$REPO" "$CARD" README.md --repo-type space \
  --commit-message "Update Space card"

echo
echo "==> live at https://huggingface.co/spaces/$REPO"
