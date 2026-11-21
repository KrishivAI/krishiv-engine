// SPDX-License-Identifier: Apache-2.0
//! The config entries of the lakehouse connectors: the Delta log pool and snapshot cache, the
//! catalog parse limits (P3-T021 design note section 3.2, P3 lakehouse design section 5.6) and
//! the Iceberg manifest cache (P3-T026 design note section 2.3).

use std::time::Duration;

use super::MIB;
use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// Threads of the process-wide pool that runs Delta kernel calls.
pub static DELTA_LOG_THREADS: ConfigEntry = ConfigEntry {
    key: "krishiv.delta.logThreads",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(4)),
    doc: "Experimental. Threads of the one process-wide pool on which Delta log work runs (log \
          replay, snapshots, commits, checkpoints): at most this many Delta kernel calls run at \
          once across all queries, and a query waits for a free thread (a cancellable wait, \
          measured by the `delta.log_pool.wait_ms` metric). Read when the pool first starts.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// Most bytes the runtime's cache of Delta snapshots holds.
pub static DELTA_SNAPSHOT_CACHE_MAX_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.delta.snapshotCache.maxBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(64 * MIB)),
    doc: "Experimental. Most bytes, estimated, the runtime's cache of Delta snapshots holds: \
          each table's latest snapshot and its live-file listing, kept across \
          queries so the next one reads only the newer log. The least recently used table \
          leaves first; a table whose listing alone is larger keeps only its snapshot, and its \
          files are listed again by each query. Read when the runtime starts.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// Most retries of a Delta commit that lost a race (delta-spark's `maxCommitAttempts`).
pub static DELTA_MAX_COMMIT_ATTEMPTS: ConfigEntry = ConfigEntry {
    key: "krishiv.delta.maxCommitAttempts",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(10_000_000)),
    doc: "Most times a Delta write retries its commit after another writer committed first and \
          no conflict was found: up to this many retries, so one more attempt in all, as \
          delta-spark counts them. Past it the write fails with \
          `DELTA_MAX_COMMIT_RETRIES_EXCEEDED`.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: Some("spark.databricks.delta.maxCommitAttempts"),
};

/// Longest time a Delta commit retries.
pub static WRITE_COMMIT_RETRY_TIMEOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.write.commitRetryTimeout",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(Duration::from_secs(30 * 60))),
    doc: "Experimental. Longest time all the attempts of one table-format commit may take \
          together (a Delta commit retried after losing races). Past it the write fails with \
          `DELTA_MAX_COMMIT_RETRIES_EXCEEDED`; delta-spark has no such bound.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: None,
};

/// The caller's key that makes a retried write commit once (E34).
pub static WRITE_IDEMPOTENCY_KEY: ConfigEntry = ConfigEntry {
    key: "krishiv.write.idempotencyKey",
    value_type: ValueType::String,
    default: None,
    doc: "A key the caller chooses for one logical write. A write whose key is already recorded \
          in the target table's commits (a Delta `txn` action, an Iceberg snapshot summary \
          property, a commit-log entry) writes nothing and succeeds, so a job that retries a \
          statement appends once. The key is scoped to the table. The per-write option of the \
          same name wins over this setting. An empty key is invalid. When unset, writes are not \
          deduplicated across statements. Set it per logical write: while it is set on a session, \
          every later write to the same table with the same key writes nothing. A key lasts as long \
          as the commit that holds it (Delta `txn` retention, Iceberg snapshot expiry, the \
          commit log). Iceberg's writer honours the key, but no SQL or DataFrame write reaches it \
          yet. A directory without a commit log refuses a key. UPDATE, DELETE and MERGE refuse a key set \
          on the session (row-level writes do not take one yet). A writer from before this \
          setting drops the keys of a commit-log checkpoint it writes.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// Largest catalog metadata a lakehouse connector parses.
pub static CATALOG_MAX_METADATA_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.catalog.maxMetadataBytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(256 * MIB)),
    doc: "Largest piece of table metadata a lakehouse connector parses, counted on decompressed \
          bytes: a catalog response, an Iceberg metadata file, a Delta commit file \
          (`_delta_log/N.json`), `_last_checkpoint` or a `.crc` file, or a checkpoint's Parquet \
          footer. A larger one fails the load with `KRISHIV_CATALOG_LIMIT_EXCEEDED` instead of \
          being read. A legitimate Delta commit of about a million added files can pass the \
          default; raise this key for such a table.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// Widest table schema a lakehouse connector accepts.
pub static CATALOG_MAX_COLUMNS: ConfigEntry = ConfigEntry {
    key: "krishiv.catalog.maxColumns",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(10_000)),
    doc: "Most columns, nested fields included, of a table schema a lakehouse connector accepts \
          from table metadata (a Delta `metaData` schema string, an Iceberg schema). A wider one \
          fails the load with `KRISHIV_CATALOG_LIMIT_EXCEEDED`.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// Bytes of decoded Iceberg manifests kept for planning.
pub static ICEBERG_MANIFEST_CACHE_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.iceberg.manifestCache.bytes",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(64 * MIB)),
    doc: "Estimated bytes of decoded Iceberg manifest entries kept in memory, least recently used \
          first out, so that planning a snapshot again (a self-join, a CTE, the next query on the \
          same table) does not fetch and parse its manifests again. Manifests are immutable, so \
          entries are keyed by path and length and never go stale. One cache per Iceberg storage \
          registration, shared by every table opened through it; 0 disables it.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// Most token exchanges per minute for one principal and one set of exchange claims (P3-T067 §3.3).
pub static CATALOG_DELEGATION_MAX_EXCHANGES_PER_MINUTE: ConfigEntry = ConfigEntry {
    key: "krishiv.catalog.delegation.maxExchangesPerMinute",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(60)),
    doc: "Most RFC 8693 token exchanges a catalog with `auth = delegated` makes in one minute for \
          one principal and one set of exchange claims (scope, audience, groups). An exchanged \
          token is reused until it expires, so a client that sends a fresh token with the same \
          claims on every call costs one exchange per exchanged-token lifetime. Past the limit a \
          catalog call fails with `KRISHIV_CATALOG_LIMIT_EXCEEDED` until the minute is over.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// Most distinct claim sets exchanged per minute for one principal (P3-T067 §3.3).
pub static CATALOG_DELEGATION_MAX_DISTINCT_EXCHANGES_PER_MINUTE: ConfigEntry = ConfigEntry {
    key: "krishiv.catalog.delegation.maxDistinctExchangesPerMinute",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(20)),
    doc: "Most distinct sets of exchange claims (scope, audience, groups) a catalog with \
          `auth = delegated` exchanges tokens for in one minute for one principal: the ceiling a \
          client meets that varies its claims to get past \
          `krishiv.catalog.delegation.maxExchangesPerMinute`. Past it a catalog call fails with \
          `KRISHIV_CATALOG_LIMIT_EXCEEDED` until the minute is over.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// How long one catalog call may take before it fails.
pub static CATALOG_REQUEST_TIMEOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.catalog.requestTimeout",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(Duration::from_secs(30))),
    doc: "Longest one request to an external catalog (Hive metastore, Unity Catalog, Iceberg REST) \
          may take, connect and response included. A stalled catalog fails the call with \
          `KRISHIV_CATALOG_FAILURE` naming the catalog. Redirects from a catalog are never followed.",
    scope: Scope::Cluster,
    settable_by: SettableBy::Startup,
    version_added: "0.2.0",
    spark_alias: None,
};

/// Most data files one Iceberg write task keeps open at once.
pub static ICEBERG_WRITE_MAX_OPEN_FILES: ConfigEntry = ConfigEntry {
    key: "krishiv.iceberg.write.maxOpenFiles",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(100)),
    doc: "Most data files one Iceberg write task (one driver of an append) keeps open at once, \
          one per partition it is writing. Past the cap the oldest-opened file is closed, and a \
          partition seen again after its file closed gets a new file, so a write over more \
          partitions per task than this writes more, smaller files. Read when the write is \
          planned.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: None,
};

/// How many snapshots an Iceberg commit validates between its start and the
/// branch's head.
pub static ICEBERG_COMMIT_MAX_INTERVENING: ConfigEntry = ConfigEntry {
    key: "krishiv.iceberg.commit.maxIntervening",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(1000)),
    doc: "Most snapshots committed to a branch between the snapshot an Iceberg write (an \
          INSERT OVERWRITE, delete, update or merge) started from and the branch's current head \
          that its commit validates for conflicts. An INSERT OVERWRITE validates only under the \
          write option isolation-level, and without validate-from-snapshot-id it validates the \
          whole history, which counts here. A commit that finds more fails with \
          KRISHIV_ICEBERG_COMMIT_CONFLICT rather than validating part of the history; appends \
          validate nothing and ignore it. Read when the write is planned. Krishiv has no catalog \
          that commits Iceberg overwrites, deletes, updates or merges yet, so until it runs them \
          this has no effect.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

/// Whether an Iceberg append writes Theta sketches of its columns into a Puffin statistics file.
pub static ICEBERG_WRITE_STATISTICS_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.iceberg.write.statistics.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Whether an append into an Iceberg table (`INSERT`, CTAS) also writes a distinct-count \
          sketch (`apache-datasketches-theta-v1`) of each of its first 100 top-level primitive columns \
          (not `uuid`) into a Puffin statistics file, committed with the snapshot in the same request, \
          so the optimizer and other engines read the table's distinct counts without a scan. A \
          snapshot gets statistics only when every row it holds is sketched: the table was empty, or \
          the previous snapshot has statistics of the column; otherwise the column, or the whole \
          file, is left out. Statistics never fail a write: on any failure, or a catalog refusing \
          the update, the write commits without them. A table opts out with the table property \
          `krishiv.write.statistics.enabled=false`. Off by default until the write cost of the sketches \
          is measured. Read when the write is planned.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: None,
};

#[cfg(test)]
mod tests {
    use super::ICEBERG_WRITE_STATISTICS_ENABLED;
    use crate::config::{Config, ConfigValue, Scope};

    /// PR-54b: statistics on Iceberg writes, off by default until U3 measures the write cost.
    #[test]
    fn iceberg_write_statistics_enabled_parses_and_defaults() {
        let mut c = Config::new();
        let get = |c: &Config| {
            c.get(&ICEBERG_WRITE_STATISTICS_ENABLED)
                .and_then(ConfigValue::as_bool)
        };
        assert_eq!(get(&c), Some(false));
        c.set("krishiv.iceberg.write.statistics.enabled", "true")
            .unwrap();
        assert_eq!(get(&c), Some(true));
        assert!(
            c.set("krishiv.iceberg.write.statistics.enabled", "sometimes")
                .is_err()
        );
        assert_eq!(ICEBERG_WRITE_STATISTICS_ENABLED.scope, Scope::Session);
    }
}
