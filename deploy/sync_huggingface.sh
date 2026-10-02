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
# Private by default: the page inlines the full assay data and unpublished
# predictions. Flip it on the Hub (Settings -> Change visibility), or:
#   VISIBILITY=public ./deploy/sync_huggingface.sh <user>/<space>
VIS="${VISIBILITY:-private}"
echo "==> creating the Space if it does not exist  (visibility: $VIS)"
if [ "$VIS" = "private" ]; then
  hf repos create "$REPO" --repo-type space --space-sdk static --private --exist-ok
else
  hf repos create "$REPO" --repo-type space --space-sdk static --exist-ok
fi

# Uploaded through the Python API rather than `hf upload`: the CLI re-runs
# create_repo() without a space_sdk, which defaults to Gradio and fails with
# 402 (Gradio Spaces need PRO). Static Spaces are free; upload_file() does not
# touch repo creation at all.
# The "By enzyme" tab fetches a pocket structure per run, so site/pockets/ has
# to travel with the page. Everything else is still inlined in index.html.
POCKETS="$ROOT/site/pockets"
if [ -d "$POCKETS" ]; then
  echo "==> $(ls "$POCKETS"/*.pdb 2>/dev/null | wc -l) pocket structures to upload"
fi

echo "==> uploading index.html, the Space card and the structures"
REPO="$REPO" SITE="$SITE" CARD="$CARD" POCKETS="$POCKETS" python - <<'PYEOF'
import os
from huggingface_hub import HfApi
api, repo = HfApi(), os.environ["REPO"]
msg = "Update explorer build"
for local, remote in ((os.environ["SITE"], "index.html"),
                      (os.environ["CARD"], "README.md")):
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=repo, repo_type="space", commit_message=msg)
    print(f"    uploaded {remote}")
pk = os.environ.get("POCKETS", "")
if pk and os.path.isdir(pk):
    # one commit for the whole folder rather than 351 separate uploads
    api.upload_folder(folder_path=pk, path_in_repo="pockets", repo_id=repo,
                      repo_type="space", commit_message="Update pocket structures")
    print(f"    uploaded pockets/ ({len(os.listdir(pk))} files)")
PYEOF

echo
echo "==> live at https://huggingface.co/spaces/$REPO"
