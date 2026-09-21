#!/usr/bin/env bash
# Refresh vendor/switchyard from a Switchyard commit or branch, source-only.
#
#   scripts/update_vendor.sh main                    # latest main
#   scripts/update_vendor.sh <commit-sha>             # pin exactly
#
# Review the diff before committing: `git diff --stat vendor/switchyard` and re-run
# scripts/build.sh + scripts/e2e.sh to confirm jevjudge's compiled profiles and route
# configs still match what Switchyard's judge-backed algorithms actually send.
set -euo pipefail
cd "$(dirname "$0")/.."
REF="${1:?usage: scripts/update_vendor.sh <commit-or-branch>}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== fetching NVIDIA-NeMo/Switchyard @ $REF"
git clone --quiet https://github.com/NVIDIA-NeMo/Switchyard.git "$TMP/switchyard"
git -C "$TMP/switchyard" checkout --quiet "$REF"
SHA="$(git -C "$TMP/switchyard" rev-parse HEAD)"

rm -rf vendor/switchyard
mkdir -p vendor/switchyard
tar --exclude='.git' --exclude='target' --exclude='**/__pycache__' --exclude='.pytest_cache' \
  -C "$TMP/switchyard" -cf - . | tar -C vendor/switchyard -xf -
rm -f vendor/switchyard/switchyard_rust/_switchyard_rust*.so
echo "$SHA" > vendor/switchyard/VENDORED_COMMIT

echo "== vendored $SHA"
sed -i.bak "s#\[\`[0-9a-f]\{7,40\}\`\](https://github.com/NVIDIA-NeMo/Switchyard/commit/[0-9a-f]\{40\})#[\`${SHA:0:7}\`](https://github.com/NVIDIA-NeMo/Switchyard/commit/${SHA})#" vendor/NOTICE.md
rm -f vendor/NOTICE.md.bak
echo "Update vendor/NOTICE.md's date and re-run scripts/build.sh + scripts/e2e.sh."
