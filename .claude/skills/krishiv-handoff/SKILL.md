---
name: krishiv-handoff
description: Use at the START of every Krishiv session (to pick up state from the repos) and at the END of every session or when stopping mid-task (to leave state so any other session/model can continue). Covers STATUS.md, the active plan, handoff notes, git/PR state.
---

# Krishiv session handoff (ADR 0031)

Paths: engine repo = this repo (or a worktree of it); `INTERNAL` = `../krishiv-internal`, or from a worktree
`$(git rev-parse --path-format=absolute --git-common-dir)/../../krishiv-internal`, or `$KRISHIV_INTERNAL`.
Nothing lives in chat memory: if it is not in the repos, the next session does not know it.

## Start checklist
1. Read `CLAUDE.md` (engine root). The SessionStart hook already printed `INTERNAL/STATUS.md` and the newest
   handoff head; if it did not, run `.claude/hooks/session-start.sh`.
2. Read in full: `INTERNAL/STATUS.md` → the active plan it links (`INTERNAL/plans/P<n>-*.md`) → the newest
   `INTERNAL/handoffs/YYYY-MM-DD-*.md`.
3. Repo state, both repos:
   - `git -C <engine> status --short && git -C <engine> log --oneline -10 && git -C <engine> worktree list`
   - `git -C INTERNAL status --short && git -C INTERNAL log --oneline -5`
   - `gh pr list --state open` and `gh run list --limit 5` (engine).
4. Reconcile: in-progress task in STATUS.md ↔ worktree/branch ↔ open PR. If they disagree, trust git, and fix
   STATUS.md first.
5. Pick work: resume the in-progress task, else the next unchecked task in the active plan whose `Depends on` is
   done. Check the task's `Model:` line (Opus for open design questions / hard subsystems, Sonnet for specified
   work). Do not start a new phase without the user's gate approval (`INTERNAL/plans/README.md`, Gate reviews).
6. Blockers listed under "waiting on user" stay blocked until the user answers — ask, don't work around.

## End checklist (also when stopping half-way)
1. Engine: commit work in progress on the task branch (never on `main`); run fmt/clippy/tests and note results.
   Push the branch (`git push -u origin <branch>`) and open/update a draft PR (`gh pr create --draft`) — only when
   the session is allowed to push; never push `main`, never `--force`.
2. Plan: tick finished task checkboxes in `INTERNAL/plans/P<n>-*.md` and link the PR.
3. `INTERNAL/STATUS.md` — keep these sections current: Current phase; Active tasks (task id, branch, worktree path,
   model, state); Blockers / waiting on user; Next 3 tasks; Metrics (golden pass rate, PySpark pass rate, latest
   benchmark); Links (active plan, latest handoff). Update the `_Last updated_` line.
4. Write `INTERNAL/handoffs/YYYY-MM-DD-<topic>.md` (one per session; add `-2` if the name exists):
   ```
   # Handoff YYYY-MM-DD: <topic>
   Session: <model>, <task ids>
   ## Done            (with commits / PR links)
   ## Half-done       (what, branch, worktree path, exact next command)
   ## Tried and failed (and why — so nobody repeats it)
   ## Decisions/questions for the user
   ## Next            (ordered; task ids)
   ```
5. Commit in INTERNAL directly (`git -C INTERNAL add STATUS.md plans handoffs && git -C INTERNAL commit`); docs repo
   changes do not need a PR. Use the commit trailer given in the session.
6. The Stop hook warns if engine code changed but STATUS.md did not — treat that as a missed step 3.

## Rules
- Hard stops (ask the user): publishing; pushing/merging `main`; deleting data/containers; touching other projects.
- Never put secrets, tokens or machine credentials in STATUS.md or handoffs.
