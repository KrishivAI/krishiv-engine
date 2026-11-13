# krishiv-udf-sandbox

**Layer:** L1 (depends on `krishiv-common` only among Krishiv crates; `cargo xtask layering`). Used by the UDF worker
pool (`krishiv-udf-pool`, P3-T014) and by the binary's hidden trampoline subcommand; never by the PyO3 module.
**The pool must start workers from one long-lived thread:** the parent-death signal fires when the thread that
started the trampoline exits, not when the server does.

## Purpose
The OS sandbox for Python UDF workers in server mode (P3-T014b; design `design/P3-T014-connect-udfs-sandbox.md` §4 in
krishiv-internal). No root and no user namespace: a trampoline process applies a parent-death signal,
`PR_SET_NO_NEW_PRIVS`, rlimits, Landlock and seccomp, then `exec`s the interpreter, which starts confined.

## Key files
Paths are relative to this crate.
- `src/lib.rs` the public surface: `enter_and_exec`, `confine_current_thread` (no `exec`; tests), `trampoline_main`, `EXIT_SANDBOX_FAILED`, re-exports
- `src/spec.rs` `SandboxSpec` (interpreter, args, environment, read paths, private directory, `Limits`, `Network`,
  `allow_fork`, `parent_pid`) and its encoding as trampoline arguments (`to_args`/`from_args`; same binary on both
  sides, so the encoding is unversioned and never stored)
- `src/probe.rs` `probe()` and `Capabilities::require`: what the kernel can enforce, and the refusal
  `KRISHIV_UDF_SANDBOX_UNAVAILABLE` (0A000)
- `src/enter.rs` the entry order (setsid, pdeathsig and parent check, no-new-privs, rlimits, Landlock, seccomp, `exec`)
- `src/fs_rules.rs` the Landlock ruleset and the ELF `PT_INTERP` reader (the loader needs the execute right)
- `src/seccomp.rs` the seccomp filters (x86_64: `ENOSYS` for x32 numbers, installed first; the `EPERM` denylist;
  `ENOSYS` for `clone3`)
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
- The worker never creates a socket unless ports are allowed; then only plain TCP streams (protocol 0 or 6) over IPv4
  or IPv6, connect only, to those ports. MPTCP and SCTP are refused because Landlock's TCP rules do not cover them.
- The worker cannot reconfigure another process: signals, `prlimit64`, `setpriority`/`ioprio_set` (process, group or
  user), `sched_set*`, `migrate_pages` name only pid 0 or its own pid. Same-uid processes (the server) are otherwise
  open to all of these, and Landlock has no hook for them.
- Minimum Landlock ABI per guarantee: files (read, write, execute) ABI 1; path truncation ABI 3, and below that the
  seccomp `truncate` rule (always installed); TCP port allowlist ABI 4 (refused below); signal and abstract-socket
  scoping ABI 6, and below that the seccomp signal rules and the `socket()` rule. `probe_matches_the_kernel` asserts
  ABI 6 or later on CI.
- **Inherited descriptors (design note §9):** the trampoline does not close descriptors above 2 (closing one by
  number needs `unsafe`). The worker module (PR-26b) must, as **the first statement of its entry point**, before any
  `sys.path` change and before importing anything from an artifact, pyfile or archive directory, call
  `os.closerange(3, ...)`; the interpreter must run with `-I`; and PR-26b carries the test that a descriptor the
  server opened without close-on-exec does not reach a UDF (`udf_clean_env_and_fds`). The trampoline itself is
  started by `Command`, whose descriptors are close-on-exec.
- The worker runs in its own session (`setsid`): no controlling terminal, no terminal signals from the server's
  process group; `TIOCSTI`/`TIOCLINUX` on an inherited terminal are refused as well.
- `enter_and_exec` and `confine_current_thread` run only in a process that exists to become the worker, never in the
  server.
- `#![forbid(unsafe_code)]`: the `landlock` and `seccompiler` crates hold the system-call `unsafe`; the x32 filter is
  plain BPF data.
- No `unwrap`/`expect`/`panic` in library code; errors carry an error class and SQLSTATE.

## What the sandbox does not guarantee (note §4.6)
- Kernel bugs, and system calls the denylist misses (review list: each addition names the escape it closes).
- Metadata of paths outside the rules: Landlock does not restrict `stat`, `readlink`, `getxattr`, `access` or inotify,
  so a worker can learn names, sizes and times of server files, but not read or change them.
- Code the worker wrote: the private directory has no execute right (no `execve`), but `mmap(PROT_EXEC)` of a file
  it wrote works; such code runs under the same layers.
- Side channels, and CPU starvation without cgroups (note §4.5).
- Data a UDF is given: it sees its input rows by design.

## Tests owned by later PRs (note §5)
PR-26b (the pool and the worker module): the positive numpy/pandas/pyarrow run with threads, `udf_clean_env_and_fds`
(the descriptor half), `sandbox_off_warns` and `sandbox_unavailable_refuses` through the pool's config, and the CI job
that runs this crate's suite inside a default Docker container (Docker's seccomp profile decides whether Landlock is
reachable, the refusal path). "Another session's artifacts" and "the spill root" are the same rule as the `outside`
case here.

## ADRs (in `../krishiv-internal/adr/`)
0022 (UDF isolation), 0029 (`unsafe` placement), 0046 (N1: client code is admin-only in server mode until this
sandbox is in use), 0028 (the `landlock` and `seccompiler` dependencies)

## Tests
`cargo nextest run -p krishiv-udf-sandbox --no-tests=pass` · `cargo clippy -p krishiv-udf-sandbox --all-targets -- -D warnings`.
The confinement tests need Linux with Landlock and seccomp and a `python3` (`KRISHIV_TEST_PYTHON` overrides); they fail
rather than skip without them, and `probe_matches_the_kernel` asserts Landlock ABI 6 or later when `CI` is set.
