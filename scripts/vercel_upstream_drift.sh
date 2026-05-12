#!/usr/bin/env bash
# Compute whether vercel/vercel changed packages/detect-agent since the last run.
# Intended for GitHub Actions; optionally writes key=value lines to GITHUB_OUTPUT.
#
# Required env:
#   UPSTREAM_STATE_DIR     - directory holding last_packages_detect_agent_commit.txt
#   UPSTREAM_VERCEL_MIRROR - path to (or parent of) bare clone of github.com/vercel/vercel.git
#   UPSTREAM_DIFF_FILE     - where to write unified diff when drift is detected
#
# Optional:
#   GITHUB_OUTPUT          - when set, writes should_sync, new_sha, merge_base, last_recorded,
#                            diff_path, compare_url (booleans as true/false strings)

set -euo pipefail

UPSTREAM_STATE_DIR="${UPSTREAM_STATE_DIR:?UPSTREAM_STATE_DIR is required}"
UPSTREAM_VERCEL_MIRROR="${UPSTREAM_VERCEL_MIRROR:?UPSTREAM_VERCEL_MIRROR is required}"
UPSTREAM_DIFF_FILE="${UPSTREAM_DIFF_FILE:?UPSTREAM_DIFF_FILE is required}"

path='packages/detect-agent'
LAST_SHA_FILE="${UPSTREAM_STATE_DIR}/last_packages_detect_agent_commit.txt"

append_kv() {
  if [[ -z "${GITHUB_OUTPUT:-}" ]]; then
    return 0
  fi
  printf '%s=%s\n' "$1" "$2" >>"$GITHUB_OUTPUT"
}

mkdir -p "$UPSTREAM_STATE_DIR"

if [[ ! -d "$UPSTREAM_VERCEL_MIRROR" ]]; then
  git clone --mirror https://github.com/vercel/vercel.git "$UPSTREAM_VERCEL_MIRROR"
fi

git -C "$UPSTREAM_VERCEL_MIRROR" fetch --prune origin

# Mirror clones expose the default branch as refs/heads/main, not origin/main.
main_ref="main"
if ! git -C "$UPSTREAM_VERCEL_MIRROR" rev-parse -q --verify "${main_ref}^{commit}" >/dev/null 2>&1; then
  main_ref="origin/main"
fi
if ! git -C "$UPSTREAM_VERCEL_MIRROR" rev-parse -q --verify "${main_ref}^{commit}" >/dev/null 2>&1; then
  echo "failed to resolve main branch tip (tried main, origin/main) in ${UPSTREAM_VERCEL_MIRROR}" >&2
  exit 1
fi

NEW_SHA="$(git -C "$UPSTREAM_VERCEL_MIRROR" log -1 --format=%H "$main_ref" -- "$path")"
if [[ -z "$NEW_SHA" ]]; then
  echo "failed to resolve latest commit for $path on ${main_ref}" >&2
  exit 1
fi

if [[ ! -f "$LAST_SHA_FILE" ]]; then
  printf '%s\n' "$NEW_SHA" >"$LAST_SHA_FILE"
  echo "bootstrapped upstream state (no sync): $NEW_SHA" >&2
  append_kv should_sync false
  append_kv new_sha "$NEW_SHA"
  exit 0
fi

LAST_SHA="$(tr -d '[:space:]' <"$LAST_SHA_FILE" || true)"
if [[ -z "$LAST_SHA" ]]; then
  printf '%s\n' "$NEW_SHA" >"$LAST_SHA_FILE"
  echo "repaired empty state file -> $NEW_SHA" >&2
  append_kv should_sync false
  append_kv new_sha "$NEW_SHA"
  exit 0
fi

if [[ "$LAST_SHA" == "$NEW_SHA" ]]; then
  echo "no upstream drift: $NEW_SHA" >&2
  append_kv should_sync false
  append_kv new_sha "$NEW_SHA"
  exit 0
fi

echo "upstream drift: $LAST_SHA -> $NEW_SHA" >&2

if ! git -C "$UPSTREAM_VERCEL_MIRROR" cat-file -e "${LAST_SHA}^{commit}" 2>/dev/null; then
  echo "widening fetch for stored sha" >&2
  git -C "$UPSTREAM_VERCEL_MIRROR" fetch --prune origin "${LAST_SHA}" || true
fi

if ! git -C "$UPSTREAM_VERCEL_MIRROR" cat-file -e "${LAST_SHA}^{commit}" 2>/dev/null; then
  echo "stored sha $LAST_SHA not found; delete ${LAST_SHA_FILE} to re-bootstrap" >&2
  exit 1
fi

MB="$(git -C "$UPSTREAM_VERCEL_MIRROR" merge-base "$LAST_SHA" "$NEW_SHA" 2>/dev/null || true)"
if [[ -z "$MB" ]]; then
  MB="$LAST_SHA"
fi

git -C "$UPSTREAM_VERCEL_MIRROR" diff "${MB}..${NEW_SHA}" -- "$path" >"$UPSTREAM_DIFF_FILE" || true

COMPARE_URL="https://github.com/vercel/vercel/compare/${MB}...${NEW_SHA}"

append_kv should_sync true
append_kv new_sha "$NEW_SHA"
append_kv branch_slug "${NEW_SHA:0:7}"
append_kv merge_base "$MB"
append_kv last_recorded "$LAST_SHA"
append_kv diff_path "$UPSTREAM_DIFF_FILE"
append_kv compare_url "$COMPARE_URL"

exit 0
