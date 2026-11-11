// SPDX-License-Identifier: Apache-2.0
//! The config entries of Python UDFs and their Arrow batches (P2-T059).

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
