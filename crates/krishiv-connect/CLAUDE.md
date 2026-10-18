# krishiv-connect

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P2a (protos, decoder, Connect-to-IR translation: P2-T091a); the gRPC server is P3.

## Purpose
Spark Connect (protocols 3.5, 4.0, 4.1, 4.2): the vendored protocol types and the translation of a Connect plan into
the logical-plan IR of `krishiv-plan`. The server (P3-T011, P3-T012) extends this decoder; it does not replace it.

## Invariants
- Protos are vendored per tag, verbatim, under `proto/spark-connect/<ver>/` at the repository root (ADR 0003; only
  4.2 so far; on the PR-size allowlist). Behaviour is gated on the client's declared version (P3-T010).
- `src/proto/v<ver>/*.pb.rs` and `tests/fixtures/spark-connect-<ver>.fds` are **generated, never edited**:
  `cargo run --locked --manifest-path tools/connect-protogen/Cargo.toml -- --write` (CI runs `--check`). The tool is a
  workspace of its own (protox, a pure-Rust protobuf compiler, and prost-build), so `protoc` is not needed anywhere and
  the product dependency graph does not carry the compiler. Generated code is excluded from the PR-size cap, from
  diff-coverage (`xtask` allowlist) and from the typo check.
- **Decode only through `wire::decode_*`.** prost is built with `no-recursion-limit` (its fixed limit of 100 rejects
  ordinary plans, and Spark's own limit is 1024), so `wire` bounds size, message count, decoded size and nesting with
  an iterative walk before prost recurses, and rejects fields the version does not define. Cargo unifies the feature,
  so it applies to every crate linked with this one: `clippy.toml` disallows prost's decoding methods everywhere else
  (a gRPC codec must call `wire`, not `prost::Message::decode`).
- `translate` turns the decoded proto into the IR: `plan_id` becomes the relation's `Origin` (and `ColumnRef::plan_id`
  for attributes and stars); canonical operator names go through `krishiv_plan::Builtin`; read options and paths go
  through `OptionValue::from_user` (ADR 0038) and the proto's plaintext map is never logged or put in an error. The
  proto types derive `Debug`, which prints option values: never format a decoded proto into an error or log.
- Anything outside the supported subset fails with `KRISHIV_UNSUPPORTED_FEATURE` naming the proto message and field,
  or with Spark's own `CONNECT_INVALID_PLAN.*` class for an invalid plan; never a silent approximation.
- Krishiv extensions only in `google.protobuf.Any` slots, package `krishiv.v1`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0005, 0038

## Test
`cargo nextest run -p krishiv-connect --no-tests=pass` · `cargo clippy -p krishiv-connect --all-targets -- -D warnings`
· goldens: `tests/fixtures/pyspark-4.2/*.json` come from real PySpark 4.2 client calls
(`tests/scripts/gen_connect_goldens.py`) · fuzz: `cd fuzz && cargo +nightly fuzz run connect_decode`
