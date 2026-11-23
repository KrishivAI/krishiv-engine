// SPDX-License-Identifier: Apache-2.0
//! The config entries of Python UDFs and their Arrow batches (P2-T059).

use std::borrow::Cow;
use std::time::Duration;

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// `krishiv.sql.execution.arrow.maxRecordsPerBatch` (Spark's
/// `spark.sql.execution.arrow.maxRecordsPerBatch`).
pub static ARROW_MAX_RECORDS_PER_BATCH: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.execution.arrow.maxRecordsPerBatch",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(10_000)),
    doc: "Most rows in one Arrow batch handed to a Python UDF; zero or a negative value means \
          no limit (the input batches are passed as they are).",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.execution.arrow.maxRecordsPerBatch"),
    settable_by: SettableBy::User,
};

/// `krishiv.sql.execution.arrow.pyspark.validateSchema.enabled` (Spark's
/// `spark.sql.execution.arrow.pyspark.validateSchema.enabled`, Spark 4.1+; `SQLConf.scala` L4765-4772 at
/// v4.2.0).
pub static ARROW_PYSPARK_VALIDATE_SCHEMA: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.execution.arrow.pyspark.validateSchema.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "When true, the result batches of `mapInPandas` and `mapInArrow` are checked against the \
          declared schema as Spark 4.1+ checks them (types equal, a non-nullable declared column \
          only from a non-nullable one), failing with `ARROW_TYPE_MISMATCH`. When false, Spark \
          checks nothing and Krishiv still refuses a result unlike the declared schema, with \
          `PYTHON_EXCEPTION` (a documented divergence).",
    scope: Scope::Session,
    version_added: "0.2.0",
    spark_alias: Some("spark.sql.execution.arrow.pyspark.validateSchema.enabled"),
    settable_by: SettableBy::User,
};

/// `krishiv.sql.execution.pythonUDF.arrow.enabled` (Spark's
/// `spark.sql.execution.pythonUDF.arrow.enabled`).
pub static PYTHON_UDF_ARROW_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.execution.pythonUDF.arrow.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Whether `@udf` Python UDFs run Arrow-batched. Only the Arrow path is implemented: \
          `false` is accepted with a warning and still runs Arrow-batched, whose return-type \
          coercion differs from Spark's pickled-row path in some cases (a documented \
          divergence).",
    scope: Scope::Session,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.execution.pythonUDF.arrow.enabled"),
    settable_by: SettableBy::User,
};

/// `krishiv.sql.execution.pythonUDTF.arrow.enabled` (Spark's
/// `spark.sql.execution.pythonUDTF.arrow.enabled`, `SQLConf.scala` at
/// v4.2.0: `true`; `false` at v3.5 to v4.1).
pub static PYTHON_UDTF_ARROW_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.execution.pythonUDTF.arrow.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Whether a Python `@udtf` without `useArrow` is Arrow-optimized (eval type 301) rather \
          than a plain UDTF (300). The client reads it when it defines the UDTF (PySpark's \
          `_create_py_udtf`) and sends the eval type it chose; the engine runs that eval type \
          and never reads this setting. The default is Spark 4.2.0's (`true`); Spark 3.5 to \
          4.1 default to `false`.",
    scope: Scope::Session,
    version_added: "0.2.1",
    spark_alias: Some("spark.sql.execution.pythonUDTF.arrow.enabled"),
    settable_by: SettableBy::User,
};

/// `krishiv.python.udf.threads`.
pub static PYTHON_UDF_THREADS: ConfigEntry = ConfigEntry {
    key: "krishiv.python.udf.threads",
    value_type: ValueType::Int,
    default: Some(ConfigValue::Int(0)),
    doc: "Threads of the embedded Python UDF pool, fixed when the pool starts (the first query \
          calling a Python UDF). 0 picks one: the executor's thread count on a free-threaded \
          Python, else 1 (with the GIL more threads only contend).",
    scope: Scope::Cluster,
    version_added: "0.1.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};

/// `krishiv.udf.sandbox` (P3-T014b; design `P3-T014-connect-udfs-sandbox.md` §4.4).
pub static UDF_SANDBOX: ConfigEntry = ConfigEntry {
    key: "krishiv.udf.sandbox",
    value_type: ValueType::Enum(&["on", "off"]),
    default: Some(ConfigValue::String(Cow::Borrowed("on"))),
    doc: "Whether the server's Python UDF workers run in the OS sandbox (Landlock, seccomp and \
          resource limits; no network, no other programs, no files outside the allowed paths). \
          `on` refuses server-side Python UDFs with `KRISHIV_UDF_SANDBOX_UNAVAILABLE` when the \
          kernel cannot enforce it. `off` runs them with resource limits, a clean environment \
          and no inherited descriptors only, and logs a warning at startup.",
    scope: Scope::Cluster,
    version_added: "0.2.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};

/// `krishiv.python.worker.executables` (P3-T014, design §3.5).
pub static PYTHON_WORKER_EXECUTABLES: ConfigEntry = ConfigEntry {
    key: "krishiv.python.worker.executables",
    value_type: ValueType::String,
    default: Some(ConfigValue::String(Cow::Borrowed("python3"))),
    doc: "Comma-separated Python interpreters the server's UDF workers may run (absolute paths, \
          or names looked up on the server's `PATH` at startup). Each needs Python 3.11 or later \
          with `pyspark-client` (or `pyspark`) of the clients' version, `pyarrow` and `pandas`. A \
          worker runs the listed interpreter whose `major.minor` is the client's, else the first \
          usable one.",
    scope: Scope::Cluster,
    version_added: "0.2.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};

/// `krishiv.python.worker.idleTimeout` (P3-T014, design §3.4).
pub static PYTHON_WORKER_IDLE_TIMEOUT: ConfigEntry = ConfigEntry {
    key: "krishiv.python.worker.idleTimeout",
    value_type: ValueType::Duration,
    default: Some(ConfigValue::Duration(Duration::from_secs(60))),
    doc: "How long an idle Python UDF worker is kept for the next task of its session before it \
          is stopped.",
    scope: Scope::Cluster,
    version_added: "0.2.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};

/// `krishiv.python.worker.maxPerSession` (P3-T014, design §3.4).
pub static PYTHON_WORKER_MAX_PER_SESSION: ConfigEntry = ConfigEntry {
    key: "krishiv.python.worker.maxPerSession",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(8)),
    doc: "Most Python UDF worker processes one session runs at once; further UDF calls wait \
          for one of them.",
    scope: Scope::Cluster,
    version_added: "0.2.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};

/// `krishiv.python.worker.maxTotal` (P3-T014, design §3.4).
pub static PYTHON_WORKER_MAX_TOTAL: ConfigEntry = ConfigEntry {
    key: "krishiv.python.worker.maxTotal",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(64)),
    doc: "Most Python UDF worker processes the server runs at once, over all sessions; at the \
          limit an idle worker of another session is stopped, else the call waits.",
    scope: Scope::Cluster,
    version_added: "0.2.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};

/// `krishiv.python.worker.maxPerTenant` (P3-T072).
pub static PYTHON_WORKER_MAX_PER_TENANT: ConfigEntry = ConfigEntry {
    key: "krishiv.python.worker.maxPerTenant",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(32)),
    doc: "Most Python UDF worker processes one tenant runs at once, over all its sessions, and \
          most calls it runs at once; further calls wait, so one tenant cannot take every worker \
          or call thread. Sessions without a tenant count per principal; with authentication off there is no such cap. A tenant's workers are \
          never shared with another tenant.",
    scope: Scope::Cluster,
    version_added: "0.3.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};

/// `krishiv.python.worker.tenantMemoryLimit` (P3-T072).
pub static PYTHON_WORKER_TENANT_MEMORY_LIMIT: ConfigEntry = ConfigEntry {
    key: "krishiv.python.worker.tenantMemoryLimit",
    value_type: ValueType::Size,
    default: None,
    doc: "Address-space limit (`RLIMIT_AS`) of each Python UDF worker of a tenant, when lower than \
          `krishiv.python.worker.memoryLimit`. Unset: the general limit applies.",
    scope: Scope::Cluster,
    version_added: "0.3.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};

/// `krishiv.python.worker.memoryLimit` (P3-T014, design §3.4).
pub static PYTHON_WORKER_MEMORY_LIMIT: ConfigEntry = ConfigEntry {
    key: "krishiv.python.worker.memoryLimit",
    value_type: ValueType::Size,
    default: Some(ConfigValue::Size(1 << 30)),
    doc: "Address-space limit of each Python UDF worker (`RLIMIT_AS`). A UDF that allocates \
          past it gets a `MemoryError` or ends its worker; the server is not affected.",
    scope: Scope::Cluster,
    version_added: "0.2.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};

/// `krishiv.python.worker.sandbox.allowFork` (P3-T014b, design §7 item 2).
pub static PYTHON_WORKER_ALLOW_FORK: ConfigEntry = ConfigEntry {
    key: "krishiv.python.worker.sandbox.allowFork",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Lets sandboxed Python UDFs fork (`multiprocessing`, `os.fork`). The other layers stay: \
          a forked process still cannot run other programs, open sockets or read files outside \
          the allowed paths. Spark's workers may fork; Krishiv's may not by default.",
    scope: Scope::Cluster,
    version_added: "0.2.0",
    spark_alias: None,
    settable_by: SettableBy::Startup,
};

#[cfg(test)]
mod tests {
    use super::PYTHON_UDTF_ARROW_ENABLED;
    use crate::config::{Config, ConfigValue};

    /// Spark 4.2.0's default: a bare `@udtf` is Arrow-optimized (the goldens' `defaults`), under Spark's key
    /// too.
    #[test]
    fn the_python_udtf_arrow_entry_defaults_to_spark_4_2() {
        let mut c = Config::new();
        let get = |c: &Config| {
            c.get(&PYTHON_UDTF_ARROW_ENABLED)
                .and_then(ConfigValue::as_bool)
        };
        assert_eq!(get(&c), Some(true));
        c.set("spark.sql.execution.pythonUDTF.arrow.enabled", "false")
            .expect("a boolean");
        assert_eq!(get(&c), Some(false));
        assert!(
            c.set("spark.sql.execution.pythonUDTF.arrow.enabled", "maybe")
                .is_err()
        );
    }
}
