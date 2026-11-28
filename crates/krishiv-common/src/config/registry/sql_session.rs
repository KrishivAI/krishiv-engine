// SPDX-License-Identifier: Apache-2.0
//! The SQL-session entries (`krishiv.sql.*` with a Spark alias): ANSI mode, time zone, path, collations and the
//! store-assignment, CTE and map-key policies.

use std::borrow::Cow;

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// ANSI SQL mode (Spark 4 default: on).
pub static ANSI_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.ansi.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "When true, follow ANSI SQL semantics: invalid casts and arithmetic overflow raise \
          errors instead of returning NULL.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.ansi.enabled"),
};

/// Session time zone.
pub static SESSION_TIME_ZONE: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.session.timeZone",
    value_type: ValueType::TimeZone,
    default: None,
    doc: "Session time zone: a region ID such as `America/Los_Angeles` or a zone offset. \
          When unset, the local system time zone is used, as in Spark.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.session.timeZone"),
};

/// How many fraction digits a `TIME` read from a CSV or JSON file keeps.
pub static IO_TEXT_TIME_NANOS: ConfigEntry = ConfigEntry {
    key: "krishiv.io.text.timeNanos",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Whether a `TIME` value read from a CSV or JSON file keeps nine fraction digits (Spark \
          4.3.0) or is cut to six (Spark 4.2 with the default `TIME` grammar; a `timeFormat` pattern keeps every digit it reads). The session \
          sets it from its Spark profile: `true` under the 4.3 profile, `false` under older ones.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.1",
    spark_alias: None,
};

/// Whether CSV and JSON file-read schema inference may give `TIME`.
pub static IO_TEXT_INFER_TIME: ConfigEntry = ConfigEntry {
    key: "krishiv.io.text.inferTime",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Whether CSV and JSON file-read schema inference may \
          give `TIME` while `spark.sql.timeType.enabled` is on (Spark 4.3.0); Spark 4.2 infers a \
          time-only string as `TIMESTAMP` either way. The session sets it from its Spark profile: \
          `true` under the 4.3 profile, `false` under older ones. `schema_of_csv` follows the \
          session's Spark profile directly.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.1",
    spark_alias: None,
};

/// Which suffixes name the compression codec of a CSV, JSON or text file.
pub static IO_TEXT_COMPRESSION_SUFFIXES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.text.compressionSuffixes",
    value_type: ValueType::Enum(&["exact", "anyCase", "extended"]),
    default: Some(ConfigValue::String(Cow::Borrowed("extended"))),
    doc: "Which file suffixes make a CSV, JSON or text read decompress a file. `exact`: `.gz`, \
          `.deflate`, `.bz2`, `.lz4` and `.snappy` in lower case (Spark 3.5). `anyCase`: the same \
          in any case (Spark 4.0). `extended`: also `.gzip`, `.zst` and `.zstd`, in any case \
          (Spark 4.1 and later). A file with another suffix is read as text; a `.zst` or `.zstd` \
          file under the first two is `KRISHIV_UNSUPPORTED_FEATURE`. The session sets it from its \
          Spark profile.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.1",
    spark_alias: None,
};

/// Whether Parquet schema inference opens zero-byte files.
pub static IO_PARQUET_INFER_EMPTY_FILES: ConfigEntry = ConfigEntry {
    key: "krishiv.io.parquet.inferFromEmptyFiles",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Whether Parquet schema inference opens a zero-byte file and fails on it with \
          `FAILED_READ_FILE.CANNOT_READ_FILE_FOOTER` (Spark up to 4.2: the first file, or every \
          file under `mergeSchema`) or leaves it out (Spark 4.3.0). No read scans a zero-byte \
          file. The session sets it from its Spark profile: `true` under profiles before 4.3.",
    scope: Scope::Session,
    settable_by: SettableBy::Admin,
    version_added: "0.3.1",
    spark_alias: None,
};

/// Whether `SET PATH` is allowed.
pub static SQL_PATH_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.path.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Whether `SET PATH` (the SQL standard routine and relation search path) is enabled.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.path.enabled"),
};

/// Whether schema-level default collations (`CREATE|ALTER SCHEMA .. DEFAULT COLLATION`) are allowed.
pub static SCHEMA_LEVEL_COLLATIONS_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.collation.schemaLevel.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Whether a schema may have a default collation (`CREATE SCHEMA .. DEFAULT COLLATION`, \
          `ALTER SCHEMA .. DEFAULT COLLATION`). Off, as in Spark 4.2, the clause is \
          `UNSUPPORTED_FEATURE.SCHEMA_LEVEL_COLLATIONS`. On, the schema stores it and `DESCRIBE \
          SCHEMA` shows it; tables, views and functions are not created in such a schema yet \
          (Spark gives their strings the schema's collation; Krishiv does not apply it).",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: Some("spark.sql.collation.schemaLevel.enabled"),
};

/// Whether a table name may carry a time travel suffix (`name@v123`, `name@20240101000000000`).
pub static TIME_TRAVEL_AT_SYNTAX_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.timeTravel.atSyntax.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Whether a table name in a query can carry a time travel suffix: `name@v123` (a version) or \
          `name@20240101000000000` (yyyyMMddHHmmssSSS). This setting only enables parsing the \
          suffix: reading an older version or timestamp is not supported yet, so a read with a \
          suffix is `UNSUPPORTED_FEATURE.TIME_TRAVEL`. Off, `@` in a table name is \
          `UNSUPPORTED_FEATURE.TIME_TRAVEL_AT_SYNTAX` at parse time (Spark 4.3).",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.2.0",
    spark_alias: Some("spark.sql.timeTravel.atSyntax.enabled"),
};

/// What `DEFAULT_PATH` expands to.
pub static SQL_DEFAULT_PATH: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.defaultPath",
    value_type: ValueType::String,
    default: Some(ConfigValue::String(Cow::Borrowed(""))),
    doc: "What `DEFAULT_PATH` expands to: a comma-separated `SET PATH` element list without `PATH`. \
          Empty means `system.builtin, system.session, current_schema`.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.defaultPath"),
};

/// Table-insertion type coercion policy.
pub static STORE_ASSIGNMENT_POLICY: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.storeAssignmentPolicy",
    value_type: ValueType::Enum(&["ANSI", "LEGACY", "STRICT"]),
    default: Some(ConfigValue::String(Cow::Borrowed("ANSI"))),
    doc: "Type coercion rules when inserting a value into a column of a different type.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.storeAssignmentPolicy"),
};

/// How nested `WITH` definitions with the same name resolve.
pub static CTE_PRECEDENCE_POLICY: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.legacy.ctePrecedencePolicy",
    value_type: ValueType::Enum(&["EXCEPTION", "LEGACY", "CORRECTED"]),
    default: Some(ConfigValue::String(Cow::Borrowed("CORRECTED"))),
    doc: "How nested WITH definitions with the same name resolve: EXCEPTION fails on a \
          nested definition shadowing an outer one, LEGACY inlines outer definitions first, \
          CORRECTED lets inner definitions shadow outer ones.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.legacy.ctePrecedencePolicy"),
};

/// Map key deduplication policy.
pub static MAP_KEY_DEDUP_POLICY: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.mapKeyDedupPolicy",
    value_type: ValueType::Enum(&["EXCEPTION", "LAST_WIN"]),
    default: Some(ConfigValue::String(Cow::Borrowed("EXCEPTION"))),
    doc: "How duplicate map keys are handled by the map functions (`map`, `map_concat`, \
          `map_from_arrays`, `map_from_entries`, `str_to_map`): `EXCEPTION` fails the query, \
          `LAST_WIN` keeps the value inserted last.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.1.0",
    spark_alias: Some("spark.sql.mapKeyDedupPolicy"),
};
