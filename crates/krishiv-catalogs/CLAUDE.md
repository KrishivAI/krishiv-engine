# krishiv-catalogs

**Layer:** L5 (may depend only on crates in L5 or lower; `cargo xtask layering`).
**Real code:** P3 (PR-16, T020b, PB-07).

## Purpose
External catalogs behind `krishiv_plan`'s `Catalog` trait: the embedded SQLite catalog (read-write), the Hive metastore, Iceberg REST, Unity Catalog and AWS Glue (read-only). A catalog is
input, not a trusted party (design `P3-lakehouse.md` §5.6): everything it returns is checked before anything reads it.

## Layout
- `src/location.rs` the locations a catalog returns: `LocationPolicy`, `RemoteLocationPolicy` (object-store schemes only; no
  `file://`, no credentials in the URL, no `..`), `check_location` (raises `KRISHIV_CATALOG_LOCATION_REFUSED`, which names the
  catalog and the table and never the location, and logs a catalog fault on `krishiv::catalog`; it is not an authorization
  denial). The allowlists of ADR 0046 §3a and the egress policy join through the trait when they exist.
- HMS egress (P3-T067 §5.2 S10): `HmsConfig::egress` (an `EgressScope` of `krishiv-store`) makes `HmsClient` dial only the
  resolved addresses the rule allows for the declared host (exempt by name); `None` is no rule (embedded).
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
  lifetime is left, the lifetime capped at a day; a 401 drops only the token it refused), `delegated.rs` (P3-T067 §3.3,
  PR-58b) `auth = delegated`: `RestCaller` (identity key, principal, `delegated_by`, `SubjectToken`) taken by the
  `*_as` client calls; `forward` sends the caller's token only when its validated `aud` names the catalog's audience
  and never a signed assertion; `exchange` is RFC 8693 (subject, actor = the catalog's client-credentials token or
  none, audience, scope), cached per (identity key, claims fingerprint) inside the catalog's own source (a token
  without `expires_in` lives no longer than its subject token, at most a day, and serves only a caller whose subject
  token is valid), a refused subject token remembered 30 s by its whole-token fingerprint, `maxExchangesPerMinute`
  per (principal, claims) and `maxDistinctExchangesPerMinute` per principal, the principal keyed without groups
  (`IdentityKey::limit_key`) and charged only for a request sent to the IdP (`KRISHIV_CATALOG_LIMIT_EXCEEDED`), one
  `krishiv::audit` record per attempt (a failed actor fetch included); no user token, an expired one or an IdP
  400/401/403 for the subject is `KRISHIV_CATALOG_DELEGATION_UNAVAILABLE` (the OAuth `error` code only), never a
  fallback to the server identity; `invalid_client`/`unauthorized_client` or an actor that cannot be fetched is the
  catalog's failure, never remembered against the user, and a failed actor fetch is repeated from memory for 30 s;
  the actor token goes to the token endpoint only, never to the catalog; `/v1/config` is read with the catalog's own
  token in every other mode and kept per catalog, and on a delegated catalog with the caller's own delegated token,
  kept per (principal without groups, claims), least recently used out past 4,096; `delegated_tests.rs` with a scripted token endpoint, `encode.rs` RFC 3986 path segments, namespace separator, Java `URLEncoder` form bodies,
  `prefix`/separator safety, JSON depth pre-scan, `types.rs` the response bodies from `serde_json::Value` (catalog, table and credential
  `config` values become `SecretString`s; `read-restrictions` fail closed), `error.rs` `RestError` -> `KRISHIV_CATALOG_FAILURE`/
  `KRISHIV_CATALOG_LIMIT_EXCEEDED` (server messages and `error_description` stay out of the error), `client.rs` `RestClient`
  (`/v1/config` once, fetched outside the lock, only `prefix` and `namespace-separator` taken, `uri` ignored; one deadline per
  call, a paged listing included; page/entry caps, a repeated or over-4-KiB token stops it; empty/`.`/`..` names and any path
  the URL parser rewrites refused; a 404 is "absent" only with a `NoSuch*` error model; credential prefixes as full URL,
  bucket or bare scheme; `load_table` checks every returned location and refuses read restrictions and server-side planning;
  `load_credentials` only when advertised; `load_table_snapshots_as` sends `snapshots=all|refs`; `commit_table_as` (PR-36c part 2) one JSON `POST .../tables/{t}` with the same token, egress client and `X-Iceberg-Access-Delegation` as `load_table`, resent once only after a 401 with a renewable token, never after a 5xx or a timeout, body capped at `max_response_bytes` before sending, a 200's locations checked (a refused one is a failure after sending), `CommitReply` (status, `Retry-After`, body) or `CommitFailure { sent }` from `transport.rs`'s `call_tracked` (unsent only for an egress refusal, a request never built or connected, no token); `commit_tests.rs`), `http_test_server.rs` (crate level, aliased as `test_server` here) a fake HTTP/1.1 server (tests only).
- `src/sqlite/` (feature `sqlite`; PB-07, ADR 0014 item 6, ADR 0037 `per_query`; `rusqlite` with the bundled C amalgamation, default features off) the embedded catalog:
  `mod.rs` `SqliteCatalog` (`Catalog` + `CatalogWrite`, `open` / `open_in_memory`, `table_details`, `list_namespaces`, `list_tables`; one connection behind a
  mutex, WAL, `synchronous=FULL`, a 10 s busy timeout; one `BEGIN IMMEDIATE` transaction per change, which bumps the epoch only when it changes
  something, so a killed process leaves the catalog as it was and two processes take turns; `epoch()` reads the file, so another process's change moves
  it; one-part namespaces, `default` exists and is not droppable, names `[A-Za-z0-9_]+` stored folded; a table's version is the epoch of the change
  that made it what it is, no history, so `time_travel` is off and a load at another version finds nothing; no staging, R1 is open), `store.rs` the
  schema (`PRAGMA user_version` = format version 2, a version 1 file gains `python_functions` when opened, a newer file is refused), rows and their JSON columns (types as Spark catalog strings read back by
  `parse_data_type`), `functions.rs` Python UDFs per tenant (P3-T072c: `put_python_function` / `drop_python_function` / `python_function` / `python_functions`, payload capped at 8 MiB, 256 functions and 64 MiB of payload per tenant (checked in the write transaction), one row per (tenant, folded name), `StoredPythonFunction`, no payload in `Debug`), `write.rs` the write half (the shared `apply_changes` for `alter_table`, same error classes as `StoreCatalog`), `tests.rs` the shared
  contract suite (`krishiv_plan::analyzer::contract`, in memory and on a file) plus a restart, two connections, threads, two processes and a `kill -9`
  mid-DDL (the test binary re-run as the child). One folding rule, stored in the file (`settings` table; opening it with the other `case_sensitive` is refused; the query's `NameForm` is ignored); `table_details(id, policy)` checks the stored location; `single_part_namespace()` is true; `epoch()` never goes backwards or repeats, also after a read fault. ADR 0028 numbers of `--features sqlite`: 4 new runtime crates (rusqlite 0.43, libsqlite3-sys 0.40 with the bundled SQLite C amalgamation, fallible-iterator, fallible-streaming-iterator; vcpkg is a build-script dependency of libsqlite3-sys), none in the default build; clean build of rusqlite + libsqlite3-sys alone (dev profile, 4 jobs) 29.5 s wall, 24.6 s user, which is the added clean-build time (measured in a scratch crate, not the whole workspace; the release-profile number is not measured). It never touches a table's files: `drop_table` removes the row, `purge` has nothing to purge.
- `src/unity/` (feature `unity`; PR-16, design `P3-lakehouse.md` §3; no new external dependency: `reqwest`/`serde_json`/`url` and `krishiv-store` as `iceberg_rest`, and no code shared with it, so either feature builds alone) the Unity Catalog read-only client for the OSS UC REST API (v0.7.0, `/api/2.1/unity-catalog`): `config.rs` `UnityConfig` (server URI, one UC catalog, static bearer token as `SecretString`, `http://` only when allowed, `egress` the policy connector, `None` = no rule) and `UnityLimits` (timeout, bytes, depth, listing entries/pages, columns), `transport.rs` the same shape as the Iceberg REST one (own thread, GET only; the client is `PolicyConnector::catalog_client` and the host is checked before every request; no redirects, no ambient proxy, identity encoding, body capped while read), `error.rs` `UnityError` -> `KRISHIV_CATALOG_FAILURE`/`KRISHIV_CATALOG_LIMIT_EXCEEDED` (server messages never in the error text), `client.rs` `UnityClient` (`catalogs`, `schemas`, paged `tables`, `tables/{full_name}`; one deadline per call; repeated or over-4-KiB page token, page and entry caps; names that would reshape a URL are refused or absent; a 404 is "absent" only for the exact v0.7.0 codes `TABLE_NOT_FOUND`/`SCHEMA_NOT_FOUND`; `CATALOG_NOT_FOUND`, bare `NOT_FOUND`, `ENDPOINT_NOT_FOUND` and the rest are failures), `catalog.rs` `UnityCatalog` (a read-only `Catalog`: no write half, so `capabilities().writes` is false and the capability check raises `UNSUPPORTED_FEATURE.CATALOG_OPERATION`; only `MANAGED`/`EXTERNAL` tables are read (a view or other type is `UNSUPPORTED_FEATURE.TABLE_OPERATION`, before the location check); `DELTA`/`PARQUET`/`CSV` give `UnityFormat` + a location through `check_location`, every other format, Iceberg included, is `UNSUPPORTED_FEATURE.TABLE_OPERATION`; columns from `type_text` by `position`, partitions by `partition_index`, width counted with nested fields), `tests.rs` recorded-shape responses from a fake server (`src/http_test_server.rs`, shared with `iceberg_rest` tests) incl. the read half of the shared contract suite. Unity's Iceberg REST endpoint (`/api/2.1/unity-catalog/iceberg`) is not built, and whether the OSS image serves it, and the 404 codes against a live server, are unverified until the Docker check.
- `src/glue/` (feature `glue`; PR-17 leftover, design `P3-lakehouse.md` §2.3; no AWS SDK: `reqwest`/`serde_json`/`url`/`sha2` and `krishiv-store`, no code shared with the other clients) the AWS Glue Data Catalog read-only client: `config.rs` `GlueConfig` (region, endpoint override (path-less), static keys as `SecretString`, optional session token and catalog id, `http://` only when allowed, `egress`) and `GlueLimits`, `sigv4.rs` Signature Version 4 for one JSON POST on `sha2` + a small HMAC (checked against RFC 4231 and AWS's `get-vanilla` vector; signed headers `content-type;host;x-amz-date;[x-amz-security-token;]x-amz-target`), `transport.rs` the Unity transport shape but POST with sensitive auth headers, `error.rs` `GlueError` (server text never in an error), `client.rs` `GlueClient` (`GetDatabases`, paged `GetTables`, `GetTable`, JSON 1.1 `X-Amz-Target: AWSGlue.*`; absence is only `EntityNotFoundException` on `GetTable`), `catalog.rs` `GlueCatalog` (read-only `Catalog`, names folded to lower case; format from `Parameters`: `table_type=ICEBERG` (location = `metadata_location`), `spark.sql.sources.provider` delta/parquet/csv, else Hive input format/serde parquet or OpenCSV; views and anything else `UNSUPPORTED_FEATURE.TABLE_OPERATION`; partition keys last), `tests.rs` a fake server replaying Glue-shaped answers (the labelled recorded-fixture path: LocalStack serves Glue only in its paid tier, so there is no Docker acceptance).
- `src/declared.rs` (feature `session`, P3-T067 §2.3, PR-58b) `IcebergRestFactory`, the session registry's `iceberg-rest` factory: a
  `RestClient` per declared catalog (auth none/token/client-credentials with `audience`, or delegated; the built
  `CallerCatalog` maps each call's `CatalogCaller` to a `RestCaller`; `option.prefix` refused at startup), its
  requests on a `PolicyConnector` scoped to the catalog's declared hosts with no redirects. `transport.rs` builds no
  `reqwest::Client` of its own: `PolicyConnector::catalog_client` (krishiv-store) and a host check before every request.
  `declared/table.rs` `RestCatalog` (`load_table` / `load_declared`: the `metadata` of the `loadTable` answer parsed by
  `krishiv-connectors-iceberg` (`IcebergMetadata`), its schema the columns (Iceberg `doc` the comment), the snapshot read the
  version; the answer is cached 30 s per table and identity key (the caller's on a `delegated` catalog, the catalog's
  otherwise; `refresh` drops every identity's entry and moves `epoch`); `list_namespaces`/`list_tables` with the caller's
  token; `RestTable::open` (`DeclaredTable`) opens the scan through `IcebergTable::from_metadata` and
  `IcebergTableProvider::open` on a blocking thread, with the query's I/O and paths (the session confines them in server
  mode, ADR 0046 §3a); a response asking for remote signing is refused at open; table `config`
  signer and refresh-endpoint keys are dropped, never followed or sent anywhere; its `token` is kept only with a vended
  answer); `declared/vended.rs` (§4.1, §4.3,
  §4.4, unit 3) `storage = vended`: `VendedCache` (per process, per catalog instance, identity key and table, `krishiv.catalog.vended.maxEntries`
  LRU, only credential keys kept; an entry is dropped at its answer's expiry (earliest credential expiry, else
  fetched + `vendedDefaultTtl`) in `get` and `put`; a `RestCatalog` drops its instance's entries when dropped and the
  instance is remembered as dead, so nothing is kept for it later and two runtimes declaring one
  name never share an answer; a `loadTable` answer replaces the held one only when fetched later, under one lock; the
  table `token` of a `loadTable` answer is kept with it and is the bearer of that table's `loadCredentials` and reload
  only, a 401 or 419 falling back to the caller's token), `VendedSource` (the cached credential while it outlives the refresh
  skew, else `loadCredentials` with the same caller or a table reload (endpoint not advertised, or a 404); a second
  fetch while the cache still holds the answer it served means the store refused it, so it refreshes; a refreshed answer that narrows the prefix below the scope's root serves the scope with the one credential of the prefixes inside it, refused when they differ), one store scope per vended prefix through `ScanInputs::stores`,
  `VendedPaths` (a location outside every scope, or no credential, is `KRISHIV_CATALOG_LOCATION_REFUSED` before any
  request: no ambient fallback), the §4.4 block-cache scope; `iceberg_rest/vended.rs` prefix forms (full URL on
  segments, bucket, scheme family), the Iceberg 1.12.0 credential keys and their expiries; `EmbeddedLocations` (embedded use
  also accepts absolute `file:` locations). Writes (PR-36c part 2): `option.write = true` (default `false`, read-only; anything but true/false refused at startup) gives a table loaded at its current version a `RestCommitter` (the connector's `RestCommitTransport` over `commit_table_as` and an uncached `snapshots=all|refs` load, for the caller that loaded it, on the blocking pool), so its provider writes; the session does not route `INSERT`/DML to declared tables yet. Tests: `declared/tests.rs` (names, auth), `declared/read_tests.rs` (a table the
  connector's `testing::write_table` wrote, served by the fake REST server), `declared/write_tests.rs` (appends over loopback HTTP into a fake catalog applying requests: committed and read back by SQL, a lost 200 probed, a 409 rebased, read-only by default); `declared/vended_read_tests.rs`: a table
  `testing::write_table_with` wrote to `s3://bucket/wh` with a writer key on `src/s3_test_server.rs` (a loopback S3 stand-in, tests only:
  PUT/GET range/HEAD/DELETE in memory, records each request's SigV4 key id, `deny` (403 `AccessDenied`) and `expire`
  (403 `ExpiredToken`; a bare 403 on HEAD, as real S3) per key id), read by a session with no S3 key:
  every request signed with the vended key, a credential inside the refresh skew replaced through `loadCredentials`
  (with the table token) or a table reload, a key storage refuses as expired mid-read refreshed and never served
  again, a path read never signed with the vended key (`vended_prefix_does_not_open_path_reads`), remote signing
  refused, `vended_secrets_never_shown`; time moves by the cache's `VendedClock` (`VendedCache::with_clock`, the test factory `ClockedFactory`), never by sleeping). Feature `testing` exports the stand-in and `write_s3_table` for other crates' tests
  (krishiv-connect-server `tests/delegated_catalog.rs`: `vended_blocks_not_shared_across_delegated_users`, `catalog_mode_matrix`).
- Not here yet: Unity vended credentials, distributed reads of declared tables (refused by the session), a fuzz
  target for the Unity Catalog response parsers (§5.6; the REST one is `fuzz/fuzz_targets/iceberg_rest_response.rs` over `iceberg_rest/fuzz_entry.rs`, seeds `fuzz/seeds/iceberg_rest_response/`), the T021 decision on `MANAGED` Delta tables of a UC server (they may be catalog-managed: UC-ratified commits not yet in `_delta_log`, so a by-path read can be stale; `UnityTable` says so), the write half of the
  Hive catalog, the Docker acceptance against a real metastore and a real Unity Catalog server
  (the default port, transport and SASL are settings and unverified until then).

## Invariants
- A refusal of a catalog-supplied location is a catalog fault, never `KRISHIV_INSUFFICIENT_PERMISSIONS`.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0014, 0028, 0046

## Tests
`cargo nextest run -p krishiv-catalogs --no-tests=pass` · `cargo nextest run -p krishiv-catalogs --features glue` (the Glue fixtures; the default set does not compile them) · `cargo clippy -p krishiv-catalogs --all-targets -- -D warnings`
