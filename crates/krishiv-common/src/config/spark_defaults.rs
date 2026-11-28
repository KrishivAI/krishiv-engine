// SPDX-License-Identifier: Apache-2.0
//! What Spark answers for a SQL config Krishiv does not register itself, when it is unset (acceptance
//! SV-3, ADR 0008).
//!
//! Spark's `spark.conf.get(key)` of an unset key answers the default of the entry `SQLConf` registers for it
//! (`SQLConf.getConfString`, v4.3.0 `SQLConf.scala:9801-9808`): a value, no value for an optional entry, or
//! the value of the entry a fallback entry reads; only a key no entry registers is `SQL_CONF_NOT_FOUND`.
//! A PySpark client reads such keys on its own (`arrow/test_parity_arrow.py` reads
//! `spark.sql.execution.arrow.pyspark.enabled` in its setup). Krishiv registers the settings it honours
//! (`registry.rs`), and those are asked first; for every other entry the default profile's release registers,
//! [`spark_default`] gives what that release answers. The table (`spark_defaults_table.rs`) is generated from
//! the release by `tests/scripts/gen_spark_sql_defaults.py`.
//!
//! Such a key changes nothing in Krishiv: a value set for it is kept and ignored, as before. Answering
//! Spark's default for it unset is what a stock client expects of the release it talks to.

mod table {
    use super::SparkDefault;
    include!("spark_defaults_table.rs");
}
use table::SPARK_SQL_DEFAULTS;

/// What `spark.conf.get` answers for an unset entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparkDefault {
    /// The entry's default, as Spark prints it.
    Value(&'static str),
    /// An optional entry: no value (`<undefined>` in `SET`), and no error.
    NoValue,
    /// The entry reads this other key when unset.
    Fallback(&'static str),
}

/// A SQL config entry of the default profile's Spark release that Krishiv does not register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkEntry {
    /// The answer for the entry unset.
    pub default: SparkDefault,
    /// A static (`StaticSQLConf`) key: Spark's `isModifiable` is false for it.
    pub is_static: bool,
}

/// The Spark entry registered under `key`, when the default profile's release registers one (Krishiv's own
/// entries are not looked up here: ask [`super::lookup`] first).
#[must_use]
pub fn spark_default(key: &str) -> Option<SparkEntry> {
    let i = SPARK_SQL_DEFAULTS
        .binary_search_by(|(k, ..)| (*k).cmp(key))
        .ok()?;
    let (_, default, is_static) = SPARK_SQL_DEFAULTS[i];
    Some(SparkEntry { default, is_static })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_and_unique() {
        assert!(SPARK_SQL_DEFAULTS.len() > 500);
        assert!(SPARK_SQL_DEFAULTS.windows(2).all(|w| w[0].0 < w[1].0));
    }

    /// Every fallback names an entry Spark registers (or one Krishiv registers), so a chain ends.
    #[test]
    fn fallbacks_name_registered_entries() {
        for (key, default, _) in SPARK_SQL_DEFAULTS {
            if let SparkDefault::Fallback(to) = default {
                assert!(
                    spark_default(to).is_some() || super::super::lookup(to).is_some(),
                    "{key} falls back to the unregistered {to}"
                );
            }
        }
    }

    #[test]
    fn the_keys_of_the_acceptance_rows() {
        // SV-3: Spark 4.3.0 `spark.conf.get` of the unset key; the arrow key falls back to the deprecated one,
        // which 4.3.0 defaults to true (3.5-4.2: false).
        assert_eq!(
            spark_default("spark.sql.execution.arrow.pyspark.enabled").map(|e| e.default),
            Some(SparkDefault::Fallback("spark.sql.execution.arrow.enabled"))
        );
        assert_eq!(
            spark_default("spark.sql.execution.arrow.enabled").map(|e| e.default),
            Some(SparkDefault::Value("true"))
        );
        assert_eq!(
            spark_default("spark.sql.execution.arrow.maxRecordsPerBatch").map(|e| e.default),
            Some(SparkDefault::Value("10000"))
        );
        assert_eq!(
            spark_default("spark.sql.adaptive.customCostEvaluatorClass").map(|e| e.default),
            Some(SparkDefault::NoValue)
        );
        assert!(spark_default("spark.sql.catalogImplementation").is_some_and(|e| e.is_static));
        assert_eq!(spark_default("spark.sql.no.such"), None);
    }
}
