# Shared helpers for Krishiv Claude Code hooks. Sourced, not executed.
# Hooks must never fail the session: every helper degrades to a warning.

# Engine repo root: the Claude project dir, else derived from this file.
krishiv_engine_dir() {
  if [ -n "${CLAUDE_PROJECT_DIR:-}" ] && [ -d "$CLAUDE_PROJECT_DIR" ]; then
    printf '%s\n' "$CLAUDE_PROJECT_DIR"
  else
    (cd "$(dirname "$0")/../.." 2>/dev/null && pwd)
  fi
}

# Internal docs repo: $KRISHIV_INTERNAL, else a sibling of the main checkout
# (resolved via the git common dir, so it works from worktrees too).
krishiv_internal_dir() {
  if [ -n "${KRISHIV_INTERNAL:-}" ]; then
    if [ -d "$KRISHIV_INTERNAL" ]; then
      (cd "$KRISHIV_INTERNAL" && pwd)
      return 0
    fi
    return 1
  fi
  common=$(git -C "$1" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || return 1
  cand="$common/../../krishiv-internal"
  [ -d "$cand" ] || return 1
  (cd "$cand" && pwd)
}

# mtime in epoch seconds (GNU stat, then BSD stat); 0 if unknown.
krishiv_mtime() {
  stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0
}
