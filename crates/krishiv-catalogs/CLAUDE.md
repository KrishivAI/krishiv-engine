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
  `hive_metastore.thrift` by field id (read-only, unknown or mistyped fields skipped), `client.rs` `HmsClient` (blocking;
  connect, read and write timeouts from `krishiv.catalog.requestTimeout`, message size and nesting limits through
  `TConfiguration`; a missing table is `None`, `MetaException` an error; no SASL), `catalog.rs` `HiveCatalog`
  (a read-only `Catalog`: a table Spark wrote takes its schema from `spark.sql.sources.schema[.part.N]`, any other from the
  metastore's columns, partition columns last; the location goes through `check_location`; a failure is
  `KRISHIV_CATALOG_FAILURE` naming the catalog and table, detail in the log; the connection is dropped on a failure),
  `test_server.rs` a fake metastore over real Thrift binary (tests only).
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
