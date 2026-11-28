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
    /// A row of an older release: its line, the key, and its `(default, static)` there (`None`: not registered).
    type OlderRow = (&'static str, &'static str, Option<(SparkDefault, bool)>);
    include!("spark_defaults_table.rs");
}
use table::{SPARK_SQL_DEFAULTS, SPARK_SQL_DEFAULTS_BEFORE};

mod core_table {
    include!("spark_core_keys_table.rs");
}
use core_table::SPARK_CORE_KEYS;

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

/// The Spark entry registered under `key` in the Spark release line `line` (`"3.5"` to `"4.3"`, a session
/// profile's `line()`; any other line reads 4.3's), when that release registers one. Krishiv's own entries are
/// not looked up here: ask [`super::lookup`] first.
///
/// ```
/// use krishiv_common::config::{SparkDefault, spark_default};
/// // Spark 4.2 turned the Arrow path on by default; 4.1 and older answer false.
/// let arrow = |line| spark_default(line, "spark.sql.execution.arrow.enabled").map(|e| e.default);
/// assert_eq!(arrow("4.3"), Some(SparkDefault::Value("true")));
/// assert_eq!(arrow("4.1"), Some(SparkDefault::Value("false")));
/// ```
#[must_use]
pub fn spark_default(line: &str, key: &str) -> Option<SparkEntry> {
    if let Ok(i) =
        SPARK_SQL_DEFAULTS_BEFORE.binary_search_by(|(l, k, _)| (*l, *k).cmp(&(line, key)))
    {
        let (.., older) = SPARK_SQL_DEFAULTS_BEFORE[i];
        return older.map(|(default, is_static)| SparkEntry { default, is_static });
    }
    let i = SPARK_SQL_DEFAULTS
        .binary_search_by(|(k, ..)| (*k).cmp(key))
        .ok()?;
    let (_, default, is_static) = SPARK_SQL_DEFAULTS[i];
    Some(SparkEntry { default, is_static })
}

/// Whether the default profile's release registers `key` as a Spark config entry that is not a SQL one
/// (`spark.executor.memory`, `spark.ui.port`): Spark refuses to change such a key at runtime
/// (`RuntimeConfig.requireNonStaticConf`, v4.3.0: `CANNOT_MODIFY_CONFIG` while
/// `spark.sql.legacy.setCommandRejectsSparkCoreConfs` is true, its default).
#[must_use]
pub fn is_spark_core_key(key: &str) -> bool {
    SPARK_CORE_KEYS.binary_search(&key).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_and_unique() {
        assert!(SPARK_SQL_DEFAULTS.len() > 500);
        assert!(SPARK_SQL_DEFAULTS.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(SPARK_CORE_KEYS.len() > 300);
        assert!(SPARK_CORE_KEYS.windows(2).all(|w| w[0] < w[1]));
        assert!(
            SPARK_SQL_DEFAULTS_BEFORE
                .windows(2)
                .all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1))
        );
        assert!(
            SPARK_SQL_DEFAULTS_BEFORE
                .iter()
                .all(|(l, ..)| ["3.5", "4.0", "4.1", "4.2"].contains(l))
        );
    }

    /// Spark 4.3.0 Connect server, 2026-11-28: `spark.conf.set` of these is `CANNOT_MODIFY_CONFIG`; of
    /// `spark.yarn.queue`, `spark.kubernetes.namespace`, `spark.master` and `spark.app.name` it succeeds (their
    /// entries are not loaded, or they are not entries); no SQL key is a core key.
    #[test]
    fn the_core_keys_of_the_oracle_run() {
        for key in [
            "spark.executor.memory",
            "spark.driver.host",
            "spark.python.worker.reuse",
            "spark.history.fs.logDirectory",
            "spark.ui.port",
            "spark.kryo.registrator",
            "spark.redaction.regex",
        ] {
            assert!(is_spark_core_key(key), "{key}");
        }
        for key in [
            "spark.yarn.queue",
            "spark.kubernetes.namespace",
            "spark.master",
            "spark.app.name",
            "spark.sql.ansi.enabled",
            "spark.sql.warehouse.dir",
        ] {
            assert!(!is_spark_core_key(key), "{key}");
        }
        assert!(
            SPARK_CORE_KEYS
                .iter()
                .all(|k| spark_default("4.3", k).is_none())
        );
    }

    /// Every fallback names an entry Spark registers (or one Krishiv registers), so a chain ends.
    #[test]
    fn fallbacks_name_registered_entries() {
        for line in ["3.5", "4.0", "4.1", "4.2", "4.3"] {
            let keys = SPARK_SQL_DEFAULTS
                .iter()
                .map(|(k, ..)| *k)
                .chain(SPARK_SQL_DEFAULTS_BEFORE.iter().map(|(_, k, _)| *k));
            for key in keys {
                if let Some(SparkEntry {
                    default: SparkDefault::Fallback(to),
                    ..
                }) = spark_default(line, key)
                {
                    assert!(
                        spark_default(line, to).is_some() || super::super::lookup(to).is_some(),
                        "{line}: {key} falls back to the unregistered {to}"
                    );
                }
            }
        }
    }

    /// Spark answers a bytes or time entry's converted value (`SQLConf.getConfString`: `stringConverter(readFrom)`),
    /// not its default's text (review of a95ba3b26, finding 1; Spark 4.3.0 `spark.conf.get`).
    #[test]
    fn a_bytes_or_time_entry_answers_its_converted_value() {
        for (key, want) in [
            ("spark.sql.broadcastTimeout", "300000ms"),
            ("spark.sql.execution.arrow.maxBytesPerBatch", "67108864b"),
            (
                "spark.sql.adaptive.coalescePartitions.minPartitionSize",
                "1048576b",
            ),
        ] {
            assert_eq!(
                spark_default("4.3", key).map(|e| e.default),
                Some(SparkDefault::Value(want)),
                "{key}"
            );
        }
    }

    /// A session of an older profile reads its release's rows (finding 3): the arrow key is false before 4.2
    /// (Spark 3.5.9, 4.0.5 and 4.1.4; 4.2.1 and 4.3.0 answer true).
    #[test]
    fn an_older_release_answers_its_own_default() {
        assert_eq!(
            spark_default("4.2", "spark.sql.execution.arrow.enabled").map(|e| e.default),
            Some(SparkDefault::Value("true"))
        );
        for line in ["3.5", "4.0", "4.1"] {
            assert_eq!(
                spark_default(line, "spark.sql.execution.arrow.enabled").map(|e| e.default),
                Some(SparkDefault::Value("false")),
                "{line}"
            );
        }
        assert_eq!(
            spark_default("4.3", "spark.sql.execution.arrow.enabled").map(|e| e.default),
            Some(SparkDefault::Value("true"))
        );
        // Any other line reads 4.3's.
        assert_eq!(
            spark_default("9.9", "spark.sql.execution.arrow.enabled"),
            spark_default("4.3", "spark.sql.execution.arrow.enabled")
        );
    }

    #[test]
    fn the_keys_of_the_acceptance_rows() {
        // SV-3: Spark 4.3.0 `spark.conf.get` of the unset key; the arrow key falls back to the deprecated one,
        // which 4.2.1 and 4.3.0 default to true (3.5.9-4.1.4: false).
        assert_eq!(
            spark_default("4.3", "spark.sql.execution.arrow.pyspark.enabled").map(|e| e.default),
            Some(SparkDefault::Fallback("spark.sql.execution.arrow.enabled"))
        );
        assert_eq!(
            spark_default("4.3", "spark.sql.execution.arrow.enabled").map(|e| e.default),
            Some(SparkDefault::Value("true"))
        );
        assert_eq!(
            spark_default("4.3", "spark.sql.execution.arrow.maxRecordsPerBatch").map(|e| e.default),
            Some(SparkDefault::Value("10000"))
        );
        assert_eq!(
            spark_default("4.3", "spark.sql.adaptive.customCostEvaluatorClass").map(|e| e.default),
            Some(SparkDefault::NoValue)
        );
        assert!(
            spark_default("4.3", "spark.sql.catalogImplementation").is_some_and(|e| e.is_static)
        );
        assert_eq!(spark_default("4.3", "spark.sql.no.such"), None);
    }
}
