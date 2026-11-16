# krishiv-catalogs

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).
**Real code arrives:** P3 (PR-16, T020b, PB-07).

## Purpose
External catalogs behind `krishiv_plan`'s `Catalog` trait: the embedded SQLite catalog (read-write), the Hive metastore, Iceberg REST and Unity Catalog (read-only). A catalog is
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
- `src/iceberg_rest/` (feature `iceberg-rest`; PR-17, decision 8: Krishiv's own client, not `iceberg-catalog-rest`, which follows a
  server's `uri` override with the caller's token) the Iceberg REST read client, wire forms from the spec and the Java client at
  `apache-iceberg-1.12.0`: `config.rs` the declaration (`RestConfig`: URI, warehouse, `RestAuth` none/static token/OAuth2 client
  credentials, `RestLimits`, the OneLake form, `http://` only when allowed, S3 Tables refused), `transport.rs` async `reqwest` on a
  thread of its own with a current-thread runtime (the caller blocks on a `std` channel with the exact deadline, so dropping or
  calling inside a runtime never panics; no redirects, no ambient proxy, `Accept-Encoding: identity` and any other encoding
  refused, body capped while read), `auth.rs` the token source per request (client-credentials token reused until a tenth of its
  lifetime is left, the lifetime capped at a day; a 401 drops only the token it refused), `encode.rs` RFC 3986 path segments, namespace separator, Java `URLEncoder` form bodies,
  `prefix`/separator safety, JSON depth pre-scan, `types.rs` the response bodies from `serde_json::Value` (catalog, table and credential
  `config` values become `SecretString`s; `read-restrictions` fail closed), `error.rs` `RestError` -> `KRISHIV_CATALOG_FAILURE`/
  `KRISHIV_CATALOG_LIMIT_EXCEEDED` (server messages and `error_description` stay out of the error), `client.rs` `RestClient`
  (`/v1/config` once, fetched outside the lock, only `prefix` and `namespace-separator` taken, `uri` ignored; one deadline per
  call, a paged listing included; page/entry caps, a repeated or over-4-KiB token stops it; empty/`.`/`..` names and any path
  the URL parser rewrites refused; a 404 is "absent" only with a `NoSuch*` error model; credential prefixes as full URL,
  bucket or bare scheme; `load_table` checks every returned location and refuses read restrictions and server-side planning;
  `load_credentials` only when advertised), `test_server.rs` a fake HTTP/1.1 server (tests only).
- `src/sqlite/` (feature `sqlite`; PB-07, ADR 0014 item 6, ADR 0037 `per_query`; `rusqlite` with the bundled C amalgamation, default features off) the embedded catalog:
  `mod.rs` `SqliteCatalog` (`Catalog` + `CatalogWrite`, `open` / `open_in_memory`, `table_details`, `list_namespaces`, `list_tables`; one connection behind a
  mutex, WAL, `synchronous=FULL`, a 10 s busy timeout; one `BEGIN IMMEDIATE` transaction per change, which bumps the epoch only when it changes
  something, so a killed process leaves the catalog as it was and two processes take turns; `epoch()` reads the file, so another process's change moves
  it; one-part namespaces, `default` exists and is not droppable, names `[A-Za-z0-9_]+` stored folded; a table's version is the epoch of the change
  that made it what it is, no history, so `time_travel` is off and a load at another version finds nothing; no staging, R1 is open), `store.rs` the
  schema (`PRAGMA user_version` = format version 1, a newer file is refused), rows and their JSON columns (types as Spark catalog strings read back by
  `parse_data_type`), `write.rs` the write half (the shared `apply_changes` for `alter_table`, same error classes as `StoreCatalog`), `tests.rs` the shared
  contract suite (`krishiv_plan::analyzer::contract`, in memory and on a file) plus a restart, two connections, threads, two processes and a `kill -9`
  mid-DDL (the test binary re-run as the child). One folding rule, stored in the file (`settings` table; opening it with the other `case_sensitive` is refused; the query's `NameForm` is ignored); `table_details(id, policy)` checks the stored location; `single_part_namespace()` is true; `epoch()` never goes backwards or repeats, also after a read fault. ADR 0028 numbers of `--features sqlite`: 4 new runtime crates (rusqlite 0.43, libsqlite3-sys 0.40 with the bundled SQLite C amalgamation, fallible-iterator, fallible-streaming-iterator; vcpkg is a build-script dependency of libsqlite3-sys), none in the default build; clean build of rusqlite + libsqlite3-sys alone (dev profile, 4 jobs) 29.5 s wall, 24.6 s user, which is the added clean-build time (measured in a scratch crate, not the whole workspace; the release-profile number is not measured). It never touches a table's files: `drop_table` removes the row, `purge` has nothing to purge.
- Not here yet: the `Catalog` over the REST client (metadata mapped through `krishiv-connectors-iceberg`; it also decides which
  table `config` keys reach a store: `s3.endpoint`, signer and refresh endpoints, `remote-signing-config`, delegated-token keys,
  with the endpoint-override case of `catalog_location_confined`; a table's vended `token` for later requests is PR-58b), a fuzz
  target for the REST response parsers, the write half of the
  Hive catalog, the Unity Catalog client, the Docker acceptance against a real metastore
  (the default port, transport and SASL are settings and unverified until then).

## Invariants
- A refusal of a catalog-supplied location is a catalog fault, never `KRISHIV_INSUFFICIENT_PERMISSIONS`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028, 0046

## Tests
`cargo nextest run -p krishiv-catalogs --no-tests=pass` · `cargo clippy -p krishiv-catalogs --all-targets -- -D warnings`
