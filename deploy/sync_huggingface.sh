#!/usr/bin/env bash
# Publish site/index.html to a Hugging Face static Space.
#
#   ./deploy/sync_huggingface.sh <hf-username>/<space-name>
#
# Credentials, in the order they are tried:
#   1. $HF_TOKEN
#   2. a token file -- $HF_TOKEN_FILE, else ~/.hf_token  (chmod 600)
#   3. whatever `hf auth login` stored
#
# On a shared machine prefer 1 or 2: `hf auth login` overwrites the stored
# token for every other user of the box. Create a WRITE token at
# https://huggingface.co/settings/tokens then:
#
#   umask 077 && printf '%s' 'hf_xxx' > ~/.hf_token
#
# The script refuses to publish into a namespace you do not own.
set -euo pipefail

REPO="${1:?usage: sync_huggingface.sh <hf-username>/<space-name>}"
NS="${REPO%%/*}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="$ROOT/site/index.html"
CARD="$ROOT/deploy/huggingface_README.md"

[ -f "$SITE" ] || { echo "No build at $SITE — run: python site/build.py" >&2; exit 1; }
command -v hf >/dev/null || { echo "hf CLI not found — pip install -U huggingface_hub" >&2; exit 1; }

TOKFILE="${HF_TOKEN_FILE:-$HOME/.hf_token}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$TOKFILE" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$TOKFILE")"
  export HF_TOKEN
  echo "==> using the token in $TOKFILE"
fi

# hf colours its output, so strip ANSI escapes before parsing
WHOAMI="$(hf auth whoami 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')" || true
WHO="$(printf '%s\n' "$WHOAMI" | sed -n 's/^user: *//p' | tr -d '[:space:]')"
[ -n "$WHO" ] || { echo "Not authenticated. Set HF_TOKEN, write $TOKFILE, or run: hf auth login" >&2; exit 1; }
ORGS="$(printf '%s\n' "$WHOAMI" | sed -n 's/^orgs: *//p' | tr -d '[:space:]')"

echo "==> authenticated as: $WHO${ORGS:+  (orgs: $ORGS)}"
if [ "$NS" != "$WHO" ] && [[ ",$ORGS," != *",$NS,"* ]]; then
  echo >&2
  echo "REFUSING TO PUBLISH: you asked for the namespace '$NS' but this token" >&2
  echo "belongs to '$WHO'${ORGS:+ (orgs: $ORGS)}. Publishing would go to the wrong account." >&2
  echo "Use '$WHO/<space-name>', or supply a token for '$NS'." >&2
  exit 1
fi

echo "==> build is $(awk "BEGIN{printf \"%.2f\", $(stat -c%s "$SITE")/1048576}") MB"
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
