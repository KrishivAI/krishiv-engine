// SPDX-License-Identifier: Apache-2.0
//! The data source a `USING` provider names, resolved as Spark's `DataSource.lookupDataSource` does; one
//! resolver for `CREATE TABLE`, `CREATE TEMPORARY VIEW` and `INSERT OVERWRITE DIRECTORY`.

/// Spark 4.2.0's `DataSource.backwardCompatibilityMap` plus the canonical built-in `FileFormat` class names:
/// the fully qualified names that load a built-in source (matched exactly, as Spark does). A name outside
/// this table is looked up whole, so `org.apache.spark.sql.csv` is `DATA_SOURCE_NOT_FOUND`.
const PROVIDER_CLASSES: &[(&str, &str)] = &[
    ("org.apache.spark.sql.jdbc", "jdbc"),
    ("org.apache.spark.sql.jdbc.DefaultSource", "jdbc"),
    (
        "org.apache.spark.sql.execution.datasources.jdbc.DefaultSource",
        "jdbc",
    ),
    ("org.apache.spark.sql.execution.datasources.jdbc", "jdbc"),
    ("org.apache.spark.sql.json", "json"),
    ("org.apache.spark.sql.json.DefaultSource", "json"),
    ("org.apache.spark.sql.execution.datasources.json", "json"),
    (
        "org.apache.spark.sql.execution.datasources.json.DefaultSource",
        "json",
    ),
    (
        "org.apache.spark.sql.execution.datasources.json.JsonFileFormat",
        "json",
    ),
    ("org.apache.spark.sql.parquet", "parquet"),
    ("org.apache.spark.sql.parquet.DefaultSource", "parquet"),
    (
        "org.apache.spark.sql.execution.datasources.parquet",
        "parquet",
    ),
    (
        "org.apache.spark.sql.execution.datasources.parquet.DefaultSource",
        "parquet",
    ),
    (
        "org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat",
        "parquet",
    ),
    ("org.apache.spark.sql.execution.datasources.orc", "orc"),
    (
        "org.apache.spark.sql.execution.datasources.orc.DefaultSource",
        "orc",
    ),
    (
        "org.apache.spark.sql.execution.datasources.orc.OrcFileFormat",
        "orc",
    ),
    ("org.apache.spark.sql.hive.orc", "orc"),
    ("org.apache.spark.sql.hive.orc.DefaultSource", "orc"),
    (
        "org.apache.spark.sql.execution.datasources.v2.orc.OrcDataSourceV2",
        "orc",
    ),
    ("com.databricks.spark.csv", "csv"),
    (
        "org.apache.spark.sql.execution.datasources.csv.CSVFileFormat",
        "csv",
    ),
    (
        "org.apache.spark.sql.execution.datasources.text.TextFileFormat",
        "text",
    ),
    ("com.databricks.spark.xml", "xml"),
    ("org.apache.spark.ml.source.libsvm", "libsvm"),
    ("org.apache.spark.ml.source.libsvm.DefaultSource", "libsvm"),
];

/// The short names of the sources Spark (or an engine feature) registers; a written name is matched to
/// them ignoring case, and to nothing else.
const SHORT_NAMES: &[&str] = &[
    "parquet",
    "csv",
    "json",
    "text",
    "orc",
    "avro",
    "jdbc",
    "hive",
    "noop",
    "console",
    "rate",
    "delta",
    "iceberg",
    "kafka",
    "memory",
    "socket",
    "xml",
    "libsvm",
    "binaryfile",
    "image",
];

/// The data source a `USING` provider names: a built-in class name or legacy alias maps to its short name
/// (matched exactly, as Spark's map is), a registered short name matches ignoring case and is returned
/// lowercase, and any other name is returned as written, since Spark looks it up (and reports it) as
/// written.
pub(crate) fn short_provider(written: &str) -> String {
    if let Some((_, short)) = PROVIDER_CLASSES.iter().find(|(class, _)| *class == written) {
        return (*short).to_owned();
    }
    let lower = written.to_lowercase();
    if SHORT_NAMES.contains(&lower.as_str()) {
        lower
    } else {
        written.to_owned()
    }
}
