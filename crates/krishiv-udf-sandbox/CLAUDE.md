# krishiv-udf-sandbox

**Layer:** L1 (depends on `krishiv-common` only among Krishiv crates; `cargo xtask layering`). Used by the UDF worker
pool (`krishiv-udf-pool`, P3-T014) and by the binary's hidden trampoline subcommand; never by the PyO3 module.

## Purpose
The OS sandbox for Python UDF workers in server mode (P3-T014b; design `design/P3-T014-connect-udfs-sandbox.md` §4 in
krishiv-internal). No root and no user namespace: a trampoline process applies a parent-death signal,
`PR_SET_NO_NEW_PRIVS`, rlimits, Landlock and seccomp, then `exec`s the interpreter, which starts confined.

## Key files
Paths are relative to this crate.
- `src/lib.rs` the public surface: `enter_and_exec`, `trampoline_main`, `EXIT_SANDBOX_FAILED`, re-exports
- `src/spec.rs` `SandboxSpec` (interpreter, args, environment, read paths, private directory, `Limits`, `Network`,
  `allow_fork`, `parent_pid`) and its encoding as trampoline arguments (`to_args`/`from_args`; same binary on both
  sides, so the encoding is unversioned and never stored)
- `src/probe.rs` `probe()` and `Capabilities::require`: what the kernel can enforce, and the refusal
  `KRISHIV_UDF_SANDBOX_UNAVAILABLE` (0A000)
- `src/enter.rs` the entry order (pdeathsig and parent check, no-new-privs, rlimits, Landlock, seccomp, `exec`)
- `src/fs_rules.rs` the Landlock ruleset and the ELF `PT_INTERP` reader (the loader needs the execute right)
- `src/seccomp.rs` the two seccomp filters (`EPERM` denylist; `ENOSYS` for `clone3`)
- `src/error.rs` `SandboxError`
- `tests/confinement.rs` re-runs the test binary as the trampoline and checks each layer with a real `python3`

## Extension points
- A system call to deny: add it to `ALWAYS_DENIED` in `src/seccomp.rs` (or a conditional rule in `deny_rules`), and a
  check to `tests/confinement.rs`. Review list: every addition says which escape it closes.
- A path every worker needs: `SandboxSpec::system_read_paths`. Paths specific to one interpreter come from the pool's
  startup record of that interpreter (note §3.5), not from here.
- A new Landlock right: raise `TARGET` in `src/fs_rules.rs` only with a test that the older ABIs still enforce the rest
  (the ruleset is best effort except for a TCP allowlist, which is a hard requirement).

## Invariants
- Seccomp is installed last, after Landlock: it forbids calls the earlier steps make.
- No layer is dropped silently: `Capabilities::require` refuses a spec the kernel cannot enforce (no Landlock, no
  seccomp, a TCP allowlist below Landlock ABI 4), and `restrict_self` reporting `NotEnforced` is an error.
- The worker never creates a socket unless ports are allowed; then only TCP streams, connect only, to those ports.
- Inherited descriptors above 2 are closed by the worker module before it reads client code, not here (closing a
  descriptor by number needs `unsafe`); the trampoline is started by `Command`, whose own descriptors are close-on-exec.
- `enter_and_exec` is called only in a process that exists to become the worker; it never runs in the server.
- `#![forbid(unsafe_code)]`: the `landlock` and `seccompiler` crates hold the system-call `unsafe`.
- No `unwrap`/`expect`/`panic` in library code; errors carry an error class and SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0022 (UDF isolation), 0029 (`unsafe` placement), 0046 (N1: client code is admin-only in server mode until this
sandbox is in use), 0028 (the `landlock` and `seccompiler` dependencies)

## Tests
`cargo nextest run -p krishiv-udf-sandbox --no-tests=pass` · `cargo clippy -p krishiv-udf-sandbox --all-targets -- -D warnings`.
The confinement tests need Linux with Landlock and seccomp and a `python3` (`KRISHIV_TEST_PYTHON` overrides); they fail
rather than skip without them, and `probe_matches_the_kernel` asserts Landlock ABI 6 or later when `CI` is set.
