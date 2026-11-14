# krishiv-catalogs

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).
**Real code arrives:** P3 (PR-16, T020b).

## Purpose
External catalogs behind `krishiv_plan`'s `Catalog` trait: the Hive metastore and Unity Catalog (read-only). A catalog is
input, not a trusted party (design `P3-lakehouse.md` §5.6): everything it returns is checked before anything reads it.

## Layout
- `src/location.rs` the locations a catalog returns: `LocationPolicy`, `RemoteLocationPolicy` (object-store schemes only; no
  `file://`, no credentials in the URL, no `..`), `check_location` (raises `KRISHIV_CATALOG_LOCATION_REFUSED`, which names the
  catalog and the table and never the location, and logs a catalog fault on `krishiv::catalog`; it is not an authorization
  denial). The allowlists of ADR 0046 §3a and the egress policy join through the trait when they exist.
- `src/hive/` (feature `hive`; the plain `thrift` runtime, default features off) the Hive metastore: `types.rs` the structs of
  `hive_metastore.thrift` by field id (read-only; unknown or mistyped fields are skipped, a list or map of the wrong element
  type is refused), `limits.rs` the budget of one call (a deadline for the whole call, bytes read, decoded elements, longest
  string; `thrift` bounds none of these as a total), `client.rs` `HmsClient` (blocking, buffered binary; a guarded socket
  spends the deadline and the byte budget on every read, so a server that trickles bytes cannot hold a call; the type
  readers spend the element budget before they allocate; a missing table is `None`, `MetaException` an error; no SASL, no
  framed transport), `catalog.rs` `HiveCatalog` (a read-only `Catalog`: a table Spark wrote takes its schema from
  `spark.sql.sources.schema[.part.N]` and its partition columns from `numPartCols`/`partCol.N`, any other from the
  metastore's columns, partition columns last; a damaged Spark schema property is a catalog fault, never the placeholder
  columns; `sd.location` and the serde `path` go through `check_location`; a failure is `KRISHIV_CATALOG_FAILURE`, a limit
  `KRISHIV_CATALOG_LIMIT_EXCEEDED`, both naming the catalog and table with the detail in the log, cut short; the connection is
  dropped on a failure), `test_server.rs` a fake metastore over real Thrift binary (tests only).
- Not here yet: the write half of the Hive catalog, the Unity Catalog client, the Docker acceptance against a real metastore
  (the default port, transport and SASL are settings and unverified until then).

## Invariants
- A refusal of a catalog-supplied location is a catalog fault, never `KRISHIV_INSUFFICIENT_PERMISSIONS`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028, 0046

## Tests
`cargo nextest run -p krishiv-catalogs --no-tests=pass` · `cargo clippy -p krishiv-catalogs --all-targets -- -D warnings`
