---
name: krishiv-adr
description: Use when a Krishiv decision must be recorded or changed — a new architectural choice, reversing/refining an Accepted ADR, adding a heavy dependency, changing a public surface or on-disk format, or closing an experiment. ADRs live in krishiv-internal/adr, never in the engine repo.
---

# Write or supersede an ADR (ADR 0001, 0031)

ADRs live in `../krishiv-internal/adr/` (index: `adr/README.md`, template: `adr/template.md`). Decisions change
only through an ADR; the spec (`../krishiv-internal/spec/`) must stay consistent with ADRs. ADR drafting is an
**Opus** task.

## When an ADR is required
- Any change to an Accepted decision (algorithm, crate layout/layering, public API/protocol, on-disk format,
  config/error-class/metric naming policy, dependency policy).
- Heavy new dependency (C/C++ build, large transitive tree, or a core-path crate) — ADR 0028.
- Each experiment's outcome (ADR 0032): create or update the ADR it informs.

## Checklist — new ADR
1. Next number: `ls ../krishiv-internal/adr | sort | tail -3` → NNNN = highest + 1 (4 digits).
2. Copy `adr/template.md` → `adr/NNNN-<kebab-slug>.md`. Fill every section: Status (`Proposed`), Date, Deciders
   (`gopal tathe, with Claude`), Supersedes, Context (with evidence links: `research/`, `experiments/`,
   benchmark results), Decision (concrete names, numbers, config keys, crates), Alternatives (table), Consequences,
   Validation (experiment/test + decision rule), Related.
3. Add the row to `adr/README.md`.
4. Update affected spec chapters and plan tasks in the same commit; mention the ADR id in them.
5. Status stays **Proposed** until the user accepts it. Ask the user; record acceptance (`Accepted`, date).

## Checklist — supersede (never edit an Accepted decision in place)
1. Write the new ADR as above with `Supersedes: <old>`.
2. In the old ADR change only the Status line to `Superseded by NNNN` (plus a one-line pointer). Leave its body.
3. Update `adr/README.md`, the spec, plans, `CLAUDE.md` hard rules and affected `crates/*/CLAUDE.md` in the engine
   repo (via a normal engine PR) if they cite the old decision.
4. Partial refinements that do not reverse the decision (e.g. "pending E05 validation" → validated) may update
   Status/Validation in place with a dated note.

## Commit
Commit in krishiv-internal directly (`git -C ../krishiv-internal add adr spec plans && git -C ../krishiv-internal commit`)
with the session trailer. Then note the ADR in `STATUS.md` (blockers section if awaiting user acceptance).
