---
name: krishiv-experiment
description: Use when running or closing a Krishiv experiment (Wave 0 E01–E10, Wave 1 E11–E18, Wave 2 E19–E26) or when a decision needs data — prototype, measure against kill criteria within a time-box, write decision.md and update the ADR.
---

# Run an experiment (ADR 0032)

Read by path: `../krishiv-internal/adr/0032-experiment-driven-decisions.md`; `../krishiv-internal/experiments/README.md`;
the experiment folder `../krishiv-internal/experiments/<wave>/<Exx-slug>/README.md`; the ADR(s) it informs;
`../krishiv-internal/spec/12-experiments.md`. Experiment design and decisions are **Opus**; prototype coding may be Sonnet.

## Before running (write it down first — binding)
1. README.md has: question, options, hypothesis, method (datasets, scale, machine), metrics, **kill criteria /
   decision rule**, **time-box**, ADR to update. If anything is missing, write it and get user sign-off before
   measuring. Kill criteria never change after data is seen without user sign-off.
2. Confirm the harness pieces needed exist (`krishiv-benchmark` skill); laptop results are relative A/B only.

## Running
3. Prototypes are **throwaway**: a worktree/branch `exp/<exx-slug>`, never merged into `main`. Only harness and
   benchmark code is kept (via a normal PR).
4. Implement options behind the same trait boundary so the loser can be revisited later.
5. Measure p50 **and** p99, CPU, peak memory, spill, S3 requests/cost where relevant — via perf-analyst.
   Raw JSON → `experiments/<wave>/<Exx>/results/` (machine, commit, dataset, config included).
6. Watch the time-box. When it expires and no option clears the rule: **pick the simpler option** and move on.

## Closing
7. `experiments/<wave>/<Exx>/decision.md`: data summary → rule applied → decision → residual risks.
8. Update the ADR via the `krishiv-adr` skill (e.g. "Accepted — pending E05" → Accepted, or a superseding ADR) and
   the spec chapter it touches.
9. Update the plan checkbox and `STATUS.md`; Wave 2 decisions also add a CI regression gate.
10. Commit internal docs directly in krishiv-internal; harness changes go through an engine PR.

Not experiments (CI tests from day one): Murmur3 compatibility, batch == stream invariant.
