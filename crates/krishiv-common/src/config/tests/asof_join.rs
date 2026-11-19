// SPDX-License-Identifier: Apache-2.0
//! The Spark 4.3 ASOF JOIN entries (kept out of `tests.rs`, which is at its size cap).

use super::*;

/// Both entries are off by default, parse booleans, refuse other values, and answer to their
/// Spark names (`spark.sql.join.asofJoin.enabled`, `spark.sql.join.sortMergeAsOfJoin.enabled`).
#[test]
fn asof_join_entries_default_parse_and_alias() {
    let mut c = Config::new();
    for entry in [&SQL_ASOF_JOIN_ENABLED, &SQL_SORT_MERGE_ASOF_JOIN_ENABLED] {
        assert_eq!(
            c.get(entry),
            Some(&ConfigValue::Bool(false)),
            "{}",
            entry.key
        );
        let alias = entry.spark_alias.unwrap();
        assert_eq!(lookup(alias).map(|e| e.key), Some(entry.key));
        assert!(REGISTRY.iter().any(|e| std::ptr::eq(*e, entry)));
        c.set(alias, "true").unwrap();
        assert_eq!(c.get(entry), Some(&ConfigValue::Bool(true)), "{alias}");
        c.set(entry.key, "FALSE").unwrap();
        assert_eq!(
            c.get(entry),
            Some(&ConfigValue::Bool(false)),
            "{}",
            entry.key
        );
        assert!(c.set(alias, "maybe").is_err(), "{alias}");
    }
    assert_eq!(
        SQL_ASOF_JOIN_ENABLED.key,
        "krishiv.sql.join.asofJoin.enabled"
    );
    assert_eq!(
        SQL_SORT_MERGE_ASOF_JOIN_ENABLED.key,
        "krishiv.sql.join.sortMergeAsOfJoin.enabled"
    );
}
