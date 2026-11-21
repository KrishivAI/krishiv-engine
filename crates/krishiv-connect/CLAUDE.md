# krishiv-connect

**Layer:** L6 (may depend only on crates in L6 or lower; `cargo xtask layering`).  
**Real code arrives:** P2a (protos, decoder, Connect-to-IR translation: P2-T091a); the gRPC server is P3.

## Purpose
Spark Connect (protocols 3.5, 4.0, 4.1, 4.2, 4.3; one 4.3 superset decoder): the vendored protocol types and the translation of a Connect plan into
the logical-plan IR of `krishiv-plan`. The server (P3-T011, P3-T012) extends this decoder; it does not replace it.

## Key files
- `src/wire.rs` the only decode path (`wire::decode_*`, `decode_*_with(bytes, &DecodeOptions)`; `decode_request` for any RPC request, the `Request` trait the
  generated layout implements): size, count, depth (`krishiv.connect.maxPlanDepth`) and field-version checks
- `src/proto/mod.rs` versioned generated protos; `src/proto/v4_3/spark.connect.pb.rs` generated messages (4.2 is kept only for the golden descriptor) (do not edit)
- `src/translate/mod.rs` `translate_*` entry points, `command_not_supported` (the server runs `sql_command` and `register_function`; every other command is 56K00 until T012); `src/translate/relation.rs` and `expression.rs` (IR building)
- `src/translate/relation_udf.rs` (P3-T014 PR-26d part 2) `MapPartitions` (28), `GroupMap` (31), `CoGroupMap` (32) with a Python UDF: the eval types Spark's planner accepts, the stateful and Scala fields refused, the UDF's reference from the same `UdfHost` inline calls use (`relation_python_udf`, `translate_relation_with`; refused without one)
- `src/translate/literal.rs` literals (scalars, intervals, nested array/map/struct, specialized arrays); `src/translate/local.rs` the host protocol for local relations; `src/translate/cached.rs` the reserved table a `cached_remote_relation` reads and the ids the translator records for the host; `src/translate/schema.rs` schema texts; `src/translate/names.rs` names and plan-id origins
- `src/translate/udf.rs` Python UDFs (P3-T014): `python_udf` reads a `CommonInlineUserDefinedFunction` into a `PythonUdf`
  (scalar eval types 100/101/200/250; Scala/Java 56K00), `UdfHost` is the server's hook naming an inline call
  (`translate_relation_with`; without a host inline UDFs are refused), `translate_command` reads `register_function`
- `src/translate/catalog.rs` (P3-T082, PR-46) `catalog_call`: a `Relation.catalog` message (36 operations) as a `CatalogCall` (Spark's overload by `db_name`; `createTable` / `createExternalTable` as `CreateTable`: the `path` becomes the last option, a struct schema becomes a DDL column list, a non-struct is `ASSERTION_FAILURE`); the translator records it as `LocalRequest::Catalog` and the host answers it (`krishiv-connect-server` `execute/catalog_call.rs`); `translate/catalog/tests.rs`
- `src/translate/errors.rs` `KRISHIV_UNSUPPORTED_FEATURE` and `CONNECT_INVALID_PLAN.*` errors
- `src/artifact.rs` `parse_artifact_name`: `AddArtifacts` names (prefixes, segments, fragments), fuzzed by
  `connect_artifact_name` in `fuzz/`
- `src/extension.rs` the `krishiv.v1` extensions (P3-T015): `resolve(any, slot)` reads an `Any` once (registered message, or `KRISHIV_UNSUPPORTED_FEATURE` naming the capability), the registry, `capabilities()`; messages mirror `proto/krishiv/v1/krishiv.proto` (additive only, compared by a test)
- `src/arrow.rs` result columns Spark sends in another Arrow shape than the engine holds (geometry/geography)
- `tests/fuzz_regressions.rs` fuzz findings (inputs that once ran out of memory) must end in a typed error
- `src/translate/write.rs` (PR-24) `write_spec` / `write_v2_spec`: a `WriteOperation` / `WriteOperationV2` as the writer state (`WriteSpec`, `WriteV2Spec`; Spark 4.3 `transformWriteOperation{,V2}` order), refusing V1 `clustering_columns`, `with_schema_evolution` and V2 partition transforms; INVALID_ENUM for unknown save methods and modes; `Debug` hides paths, option values and table names; `translate/write/tests.rs`
- `src/coverage.rs` (P3-T012a) the coverage table: an exhaustive `match` per `oneof` (Relation, Expression, Command, Catalog, Literal, DataType) with a row per arm (status, refused fields, IR target, task, group), so a regenerated proto with a new arm does not compile until it has a row; `src/bin/connect-coverage-dump.rs` prints it as JSON for `cargo xtask connect-coverage [--check]`, which writes `docs/reference/connect/coverage.md` and `tests/connect/coverage.json` (CI checks them in the docs-site job). Update the row in the same commit that translates an arm; `tests/coverage.rs` fails when a row disagrees with the translator
- `tests/goldens.rs` golden runner over `tests/snapshots/` and `tests/fixtures/pyspark-4.2/`

## Extension points
- New relation or expression: a translation arm in `src/translate/relation.rs` or `expression.rs` building an existing IR node.
- New protocol version: the generator in `tools/connect-protogen` (repository root), then a new `src/proto/v<ver>/`
  (it also writes the server stub `crates/krishiv-connect-server/src/service/v<ver>.pb.rs`).

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
- Deep messages (more than a few levels) decode, drop and translate on a thread with a big stack (`stack`): prost's
  recursion costs about 3 KiB of stack per level in a debug build, more than the 2 MiB of a test or async worker. The
  decoded `Decoded<M>` drops there too. Never hold or drop a deeply nested proto outside it.
- `translate` turns the decoded proto into the IR: `plan_id` becomes the relation's `Origin` (and `ColumnRef::plan_id`
  for attributes and stars); canonical operator names go through `krishiv_plan::Builtin`; read options and paths go
  through `OptionValue::from_user` (ADR 0038); nothing formats option values into an error. The generated `Debug` of
  every message (`debug.pb.rs`) prints the plan's structure, but no string, byte string, map or `Any` payload (SQL
  text, literals, names, options, paths), only their sizes. `Decoded<M>` wipes option values, paths and `Any` payloads
  with zeros before it drops the message (`Scrub`, `scrub.pb.rs`); a proto built in Rust and passed to `translate_*`
  stays the caller's to wipe.
- SQL expression strings (`selectExpr`, `where("...")`, `expr()`) are parsed eagerly while translating, by
  `krishiv_sql::parse_and_lower_named_expression` with the session's `TranslateConf::parser` (filled by
  `krishiv-connect-server` from `Session::parser_options`); `ToDF`, pivots and `TIME` literals wait for IR nodes
  or later groups (`src/coverage.rs` lists every arm).
- **Local relations** (`createDataFrame`): the translator is synchronous, so the host builds them. It translates the
  plan once with `LocalRelations::collecting()` (records a `LocalRequest` per `local_relation` /
  `cached_local_relation`, answers with placeholders), builds the relations (`Session::local_relation`), then
  translates again with `LocalRelations::resolved(..)`: the n-th request gets the n-th relation. A plan without
  local relations is translated once. `TranslateConf::default()` has no host: a local relation is
  `KRISHIV_UNSUPPORTED_FEATURE`. `chunked_cached_local_relation` stays 56K00.
- Schema texts (`translate/schema.rs`): DDL, else JSON (`parseTypeWithFallback`); a local relation's non-struct type is
  wrapped as `struct<value: t>`, a data source's must be a struct (`CONNECT_INVALID_PLAN.INVALID_SCHEMA_NON_STRUCT_TYPE`).
  `SubqueryAlias.qualifier ++ [alias]` is one `ALIAS_PART_SEP`-joined alias.
- A `oneof` arm that is not translated (`Relation`, `Expression`, `Literal`, `DataType`, later `Command` and `Catalog`)
  fails with Spark's `CONNECT_INVALID_PLAN.INVALID_ONE_OF_FIELD_NOT_SUPPORTED` (SQLSTATE 56K00; parameters fullName,
  the upper-cased field name and the field number, as `InvalidInputErrors.invalidOneOfField`; owner decision 6,
  `design/P3-connect-mapping.md` s6). A supported arm with an unsupported **field** fails with
  `KRISHIV_UNSUPPORTED_FEATURE` naming the message and field, and an invalid plan with Spark's other
  `CONNECT_INVALID_PLAN.*` classes; never a silent approximation.
- Krishiv extensions only in `google.protobuf.Any` slots, package `krishiv.v1`: add one by a message in `proto/krishiv/v1/krishiv.proto`, a prost type and a registry row in `extension.rs`, and a translator arm; a stock client never receives one unless it sent one.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0005, 0038

## Tests
`cargo nextest run -p krishiv-connect --no-tests=pass` · `cargo clippy -p krishiv-connect --all-targets -- -D warnings`
· goldens: `tests/fixtures/pyspark-4.2/*.json` come from real PySpark 4.2 client calls
(`tests/scripts/gen_connect_goldens.py`) · fuzz: `cd fuzz && cargo +nightly fuzz run connect_decode`
