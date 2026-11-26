---
description: "Upgrade from Krishiv 0.2 to 0.3: behaviour changes, new defaults and the settings and APIs to check before you move."
---

# Upgrading from 0.2

Most 0.2 programs run unchanged on 0.3. This page lists what can change the result or the failure of a program that
worked on 0.2, in the order you are likely to meet it. Krishiv follows semantic versioning only from 1.0, so a 0.x
release may change its public API.

## Before you upgrade

1. Pin the new version in a test environment: `pip install "krishiv==0.3.0"`. Python 3.11 or newer is required, as in 0.2.
2. Run your tests and your queries once. Every item below fails with a named error class or changes a visible value, so
   a failure points at one of them.
3. Rust users: the crates are versioned together. Move every `krishiv-*` dependency to `0.3.0` at once.

<!-- BREAKING-LIST: merge the Rust public-API breaks from the cargo-semver-checks run against 0.2.0 (release checklist, sonnet-8) into "Rust API changes" below, each with its migration line. -->

## Spark version and behaviour

- **Spark 4.3 is the default profile.** `spark.version`, `version()` and the Spark Connect `spark_version` answer `4.3.0`
  where 0.2 answered 4.2. Behaviour that differs between Spark 4.2 and 4.3 now follows 4.3. A program that compared the
  version string needs a new value.
- **Unsupported `spark.sql.legacy.*` settings fail.** Setting a `spark.sql.legacy.*` key that Krishiv does not honour to
  a non-default value now fails with `KRISHIV_UNSUPPORTED_FEATURE`; 0.2 ignored it. Remove the setting, or set it back
  to its default.
- **A NULL written into a NOT NULL column fails** with `NOT_NULL_ASSERT_VIOLATION`, as in Spark 4.2 and later. It does not
  fail under `spark.sql.storeAssignmentPolicy=LEGACY`.
- **JSON reads.** `spark.read.json` reads nested JSON as structs, arrays and maps. Inferred fields are ordered by name,
  a top-level array of objects is one row per object, a malformed line adds `_corrupt_record`, and repeated field names
  raise `COLUMN_ALREADY_EXISTS`. A schema inferred on 0.2 may have a different column order.
- **Partitioned reads check the directory layout.** Two base paths raise `CONFLICTING_DIRECTORY_STRUCTURES` and
  differing partition columns raise `CONFLICTING_PARTITION_COLUMN_NAMES`, as in Spark 4.2.0. Set
  `spark.sql.files.ignoreInvalidPartitionPaths` or the `ignoreInvalidPartitionPaths` read option to skip invalid paths.
- **Partition-indexed expressions in `UNION ALL`.** `rand`, `uuid(seed)`, `monotonically_increasing_id()` and
  `spark_partition_id()` inside a branch now read the branch's own partition index, and a union of one-partition inputs
  is one partition. Set `krishiv.sql.unionOutputPartitioning` (or `spark.sql.unionOutputPartitioning`) to `false` for
  the old layout.
- **`LIKE` and `ILIKE` over `UTF8_LCASE`** now match instead of being refused, and `randn` matches Spark's values.

## Python

- **`pyarrow>=16` is installed with the package.** `pip install "krishiv[pyarrow]"` still works. `toPandas` needs the
  `pandas` extra: `pip install "krishiv[pandas]"`.
- **JVM-only APIs raise a named error.** `spark.sparkContext`, `df.rdd`, `krishiv.SparkConf`, `krishiv.pandas`,
  `krishiv.ml`, `SparkSession.builder.enableHiveSupport()` and the JDBC reader and writer raise
  `KRISHIV_UNSUPPORTED_FEATURE` and name the DataFrame way. Cluster settings (`master("yarn")`, `spark.executor.*`,
  `spark.driver.*`, `spark.submit.*`) log a warning and have no effect.
- **`https://` reads that infer a schema are refused**, as in Spark; give an explicit schema. See
  [Data sources](data-sources.md).

## Running the server (new in 0.3)

If you start `krishiv server` for the first time, these rules are new for you rather than changed:

- A server that listens on a non-loopback address refuses to start without authentication, and a bearer-token or OIDC
  setup also needs TLS. See [Securing a single-node server](single-node-server-security.md).
- With authentication on, the admin listener (`/metrics` and the web console) asks for credentials and checks the role
  (viewer, operator or admin).
- Catalogs are declared at startup only. `SET` and `RESET` of `krishiv.catalog.<name>.*` and `spark.sql.catalog.<name>`
  fail with `CANNOT_MODIFY_STATIC_CONFIG` for every role.
- `SHOW CATALOGS` and `spark.catalog.listCatalogs` list every declared catalog the caller may see, including catalogs
  no query has used yet.
- In a Rust library session with grants, `DROP VIEW` checks the Drop privilege on the view, as `DROP TABLE` does.
- Spark Connect refuses a plan nested deeper than `krishiv.connect.maxPlanDepth` (default 1024, at most 4096).

## Rust API changes

<!-- BREAKING-LIST: one entry per break reported by cargo-semver-checks against 0.2.0, with the old and the new call. -->

See the [Rust reference](../reference/rust/index.md) for the 0.3 signatures.

## New settings you may want

The [configuration reference](../reference/config/index.md) lists every key. The ones most upgrades look at:
`krishiv.memory.limit` (the default is 80% of a container's memory limit, at least 512 MiB below it, never under half), `krishiv.spill.dir`,
`krishiv.history.path`, `krishiv.workload.resourceGroups.file`, `krishiv.flightSql.enabled` and
`krishiv.write.idempotencyKey`. No 0.2 setting was renamed in the changes recorded so far.

<!-- RENAMES: confirm against the semver and config diff before release; if a key was renamed, list old and new here. -->
