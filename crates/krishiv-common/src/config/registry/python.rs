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
