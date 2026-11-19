// SPDX-License-Identifier: Apache-2.0
//! The Spark SQL settings a Spark Connect client reads from the server when it starts a session or runs
//! `createDataFrame`, with Spark 4.3.0's defaults (`SQLConf.scala`): a client treats a missing key as an error.

use std::borrow::Cow;

use crate::config::{ConfigEntry, ConfigValue, Scope, SettableBy, ValueType};

/// `spark.sql.timestampType`.
pub static SQL_TIMESTAMP_TYPE: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.timestampType",
    value_type: ValueType::Enum(&["TIMESTAMP_LTZ", "TIMESTAMP_NTZ"]),
    default: Some(ConfigValue::String(Cow::Borrowed("TIMESTAMP_LTZ"))),
    doc: "The type a bare `TIMESTAMP` literal, `to_timestamp` and `make_timestamp` produce (a bare `TIMESTAMP` cast target or column type does not follow it yet): `TIMESTAMP_LTZ` (with the session time zone) or `TIMESTAMP_NTZ` (without a zone).",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.timestampType"),
};

/// `spark.sql.session.localRelationCacheThreshold`.
pub static LOCAL_RELATION_CACHE_THRESHOLD: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.session.localRelationCacheThreshold",
    value_type: ValueType::Int32 { min: 0 },
    default: Some(ConfigValue::Int(1024 * 1024)),
    doc: "The size in bytes from which a client uploads a local relation (`createDataFrame`) once and refers to it by hash. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.session.localRelationCacheThreshold"),
};

/// `spark.sql.session.localRelationSizeLimit`.
pub static LOCAL_RELATION_SIZE_LIMIT: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.session.localRelationSizeLimit",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(3 * 1024 * 1024 * 1024)),
    doc: "The limit in bytes on how large a chunked cached local relation can be. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.session.localRelationSizeLimit"),
};

/// `spark.sql.session.localRelationChunkSizeRows`.
pub static LOCAL_RELATION_CHUNK_SIZE_ROWS: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.session.localRelationChunkSizeRows",
    value_type: ValueType::Int32 { min: 1 },
    default: Some(ConfigValue::Int(10_000)),
    doc: "The rows per chunk when a client splits a local relation. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.session.localRelationChunkSizeRows"),
};

/// `spark.sql.session.localRelationChunkSizeBytes`.
pub static LOCAL_RELATION_CHUNK_SIZE_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.session.localRelationChunkSizeBytes",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(16 * 1024 * 1024)),
    doc: "The bytes per chunk when a client splits a local relation; the smaller of this and the batch size applies. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.session.localRelationChunkSizeBytes"),
};

/// `spark.sql.session.localRelationBatchOfChunksSizeBytes`.
pub static LOCAL_RELATION_BATCH_OF_CHUNKS_SIZE_BYTES: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.session.localRelationBatchOfChunksSizeBytes",
    value_type: ValueType::PositiveInt,
    default: Some(ConfigValue::Int(1024 * 1024 * 1024)),
    doc: "The memory a client uses to collect chunks of a local relation before it uploads them. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.session.localRelationBatchOfChunksSizeBytes"),
};

/// `spark.sql.execution.pandas.convertToArrowArraySafely`.
pub static PANDAS_CONVERT_TO_ARROW_ARRAY_SAFELY: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.execution.pandas.convertToArrowArraySafely",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(true)),
    doc: "Whether a pandas to Arrow conversion checks for overflow and truncation. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.execution.pandas.convertToArrowArraySafely"),
};

/// `spark.sql.execution.pandas.inferPandasDictAsMap`.
pub static PANDAS_INFER_DICT_AS_MAP: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.execution.pandas.inferPandasDictAsMap",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Whether a Python dict in a pandas column is inferred as a map (else a struct). Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.execution.pandas.inferPandasDictAsMap"),
};

/// `spark.sql.pyspark.inferNestedDictAsStruct.enabled`.
pub static PYSPARK_INFER_NESTED_DICT_AS_STRUCT: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.pyspark.inferNestedDictAsStruct.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Whether a nested Python dict is inferred as a struct (else a map). Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.pyspark.inferNestedDictAsStruct.enabled"),
};

/// `spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled`.
pub static PYSPARK_LEGACY_INFER_ARRAY_FROM_FIRST: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Whether the element type of a Python list is inferred from its first element only. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled"),
};

/// `spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled`.
pub static PYSPARK_LEGACY_INFER_MAP_FROM_FIRST: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Whether the key and value types of a Python dict are inferred from its first pair only. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled"),
};

/// `spark.sql.execution.arrow.useLargeVarTypes`.
pub static ARROW_USE_LARGE_VAR_TYPES: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.execution.arrow.useLargeVarTypes",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "Whether Arrow data for strings and binaries uses the large (64-bit offset) variable-width types. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.execution.arrow.useLargeVarTypes"),
};

/// `spark.sql.execution.pandas.structHandlingMode`.
pub static PANDAS_STRUCT_HANDLING_MODE: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.execution.pandas.structHandlingMode",
    value_type: ValueType::ExactEnum(&["legacy", "row", "dict"]),
    default: Some(ConfigValue::String(Cow::Borrowed("legacy"))),
    doc: "How a client converts a struct column when it builds a pandas DataFrame: `legacy`, `row` (a Row object) or `dict` (a dict with suffixed names for repeated fields). Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.execution.pandas.structHandlingMode"),
};

/// `spark.sql.repl.eagerEval.enabled`.
pub static REPL_EAGER_EVAL_ENABLED: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.repl.eagerEval.enabled",
    value_type: ValueType::Bool,
    default: Some(ConfigValue::Bool(false)),
    doc: "When true, a notebook or REPL shows the top rows of a DataFrame without an explicit `show()`. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.repl.eagerEval.enabled"),
};

/// `spark.sql.repl.eagerEval.maxNumRows`.
pub static REPL_EAGER_EVAL_MAX_NUM_ROWS: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.repl.eagerEval.maxNumRows",
    value_type: ValueType::Int32 { min: i32::MIN },
    default: Some(ConfigValue::Int(20)),
    doc: "The most rows eager evaluation shows. Only used when `krishiv.sql.repl.eagerEval.enabled` is true. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.repl.eagerEval.maxNumRows"),
};

/// `spark.sql.repl.eagerEval.truncate`.
pub static REPL_EAGER_EVAL_TRUNCATE: ConfigEntry = ConfigEntry {
    key: "krishiv.sql.repl.eagerEval.truncate",
    value_type: ValueType::Int32 { min: i32::MIN },
    default: Some(ConfigValue::Int(20)),
    doc: "The most characters eager evaluation shows for a cell. Only used when `krishiv.sql.repl.eagerEval.enabled` is true. Read by Spark Connect clients.",
    scope: Scope::Session,
    settable_by: SettableBy::User,
    version_added: "0.3.0",
    spark_alias: Some("spark.sql.repl.eagerEval.truncate"),
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, lookup};

    /// Spark 4.3.0 `SQLConf.scala` defaults, by Spark key.
    #[test]
    fn the_client_read_keys_exist_with_spark_defaults() {
        let c = Config::new();
        for (key, want) in [
            (
                "spark.sql.timestampType",
                ConfigValue::String("TIMESTAMP_LTZ".into()),
            ),
            (
                "spark.sql.session.localRelationCacheThreshold",
                ConfigValue::Int(1_048_576),
            ),
            (
                "spark.sql.session.localRelationSizeLimit",
                ConfigValue::Int(3_221_225_472),
            ),
            (
                "spark.sql.session.localRelationChunkSizeRows",
                ConfigValue::Int(10_000),
            ),
            (
                "spark.sql.session.localRelationChunkSizeBytes",
                ConfigValue::Int(16_777_216),
            ),
            (
                "spark.sql.session.localRelationBatchOfChunksSizeBytes",
                ConfigValue::Int(1_073_741_824),
            ),
            (
                "spark.sql.execution.pandas.convertToArrowArraySafely",
                ConfigValue::Bool(true),
            ),
            (
                "spark.sql.execution.pandas.inferPandasDictAsMap",
                ConfigValue::Bool(false),
            ),
            (
                "spark.sql.pyspark.inferNestedDictAsStruct.enabled",
                ConfigValue::Bool(false),
            ),
            (
                "spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled",
                ConfigValue::Bool(false),
            ),
            (
                "spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled",
                ConfigValue::Bool(false),
            ),
            (
                "spark.sql.execution.arrow.useLargeVarTypes",
                ConfigValue::Bool(false),
            ),
            (
                "spark.sql.execution.pandas.structHandlingMode",
                ConfigValue::String("legacy".into()),
            ),
            ("spark.sql.repl.eagerEval.enabled", ConfigValue::Bool(false)),
            ("spark.sql.repl.eagerEval.maxNumRows", ConfigValue::Int(20)),
            ("spark.sql.repl.eagerEval.truncate", ConfigValue::Int(20)),
        ] {
            let entry = lookup(key).unwrap_or_else(|| panic!("{key} is not registered"));
            assert_eq!(c.get(entry), Some(&want), "{key}");
        }
    }

    #[test]
    fn timestamp_type_takes_either_type_in_any_case_and_refuses_others() {
        let mut c = Config::new();
        c.set("spark.sql.timestampType", "timestamp_ntz").unwrap();
        assert_eq!(
            c.get(&SQL_TIMESTAMP_TYPE),
            Some(&ConfigValue::String("TIMESTAMP_NTZ".into()))
        );
        assert!(c.set("krishiv.sql.timestampType", "DATETIME").is_err());
    }

    /// Each setting answers to its `krishiv.sql.*` key and to the Spark key, and both reach one entry.
    #[test]
    fn each_client_read_key_is_reachable_by_its_krishiv_name_and_spark_alias() {
        for key in [
            "krishiv.sql.timestampType",
            "krishiv.sql.session.localRelationCacheThreshold",
            "krishiv.sql.session.localRelationSizeLimit",
            "krishiv.sql.session.localRelationChunkSizeRows",
            "krishiv.sql.session.localRelationChunkSizeBytes",
            "krishiv.sql.session.localRelationBatchOfChunksSizeBytes",
            "krishiv.sql.execution.pandas.convertToArrowArraySafely",
            "krishiv.sql.execution.pandas.inferPandasDictAsMap",
            "krishiv.sql.pyspark.inferNestedDictAsStruct.enabled",
            "krishiv.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled",
            "krishiv.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled",
            "krishiv.sql.execution.arrow.useLargeVarTypes",
            "krishiv.sql.execution.pandas.structHandlingMode",
            "krishiv.sql.repl.eagerEval.enabled",
            "krishiv.sql.repl.eagerEval.maxNumRows",
            "krishiv.sql.repl.eagerEval.truncate",
        ] {
            let entry = lookup(key).unwrap_or_else(|| panic!("{key} is not registered"));
            assert_eq!(entry.key, key);
            let alias = entry
                .spark_alias
                .unwrap_or_else(|| panic!("{key} has no Spark alias"));
            assert!(std::ptr::eq(lookup(alias).unwrap(), entry), "{alias}");
        }
    }

    #[test]
    fn client_read_settings_refuse_what_spark_refuses() {
        let mut c = Config::new();
        assert!(
            c.set("krishiv.sql.session.localRelationCacheThreshold", "-1")
                .is_err()
        );
        assert!(
            c.set("krishiv.sql.session.localRelationCacheThreshold", "0")
                .is_ok()
        );
        let key = "krishiv.sql.execution.pandas.structHandlingMode";
        assert!(c.set(key, "ROW").is_err());
        assert!(c.set(key, "row").is_ok());
        assert_eq!(
            ValueType::Int32 { min: 0 }.to_string(),
            "32-bit int (min 0)"
        );
        assert_eq!(ValueType::Int32 { min: i32::MIN }.to_string(), "32-bit int");
        for key in [
            "krishiv.sql.session.localRelationCacheThreshold",
            "krishiv.sql.session.localRelationChunkSizeRows",
            "krishiv.sql.repl.eagerEval.maxNumRows",
            "krishiv.sql.repl.eagerEval.truncate",
        ] {
            assert!(c.set(key, "2147483647").is_ok(), "{key} at i32::MAX");
            assert!(c.set(key, "2147483648").is_err(), "{key} above i32::MAX");
        }
        // The three `longConf` entries stay 64-bit.
        assert!(
            c.set(
                "krishiv.sql.session.localRelationChunkSizeBytes",
                "2147483648"
            )
            .is_ok()
        );
    }
}
