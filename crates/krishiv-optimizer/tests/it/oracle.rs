// SPDX-License-Identifier: Apache-2.0
//! Inferred join filters against Spark 4.2.0 (design P2-T072b acceptance
//! §6.5): per TPC-H query and table, the `isnotnull` filters Spark's
//! `InferFiltersFromConstraints` adds (fixture
//! `tests/fixtures/spark-4.2.0/inferred-join-filters.json`, generator
//! `tests/scripts/gen_inferred_join_filters.py`) against the ones on
//! Krishiv's optimised scans. Spark's filter-own `isnotnull` (beside a
//! written predicate on the same column of that scan) are not inferred here
//! and are set aside. What is left must be exactly [`MISSING`] and
//! [`EXTRA`], each with its reason (`docs/known-differences.md`).

use std::collections::{BTreeMap, BTreeSet};

use krishiv_optimizer::optimize;
use serde_json::Value;

use crate::support;

/// `(query, table, column)` Spark infers and Krishiv does not.
const MISSING: &[(&str, &str, &str)] = &[
    // The join key of a subtree the subquery reads too: kept off the shared
    // `revenue0` body so it is computed once (the join drops a null key anyway).
    ("q15", "lineitem", "l_suppkey"),
    // From a non-equality join condition (`l_quantity < 0.2 * avg(..)`).
    ("q17", "lineitem", "l_quantity"),
    // The key equality is common to every branch of an `OR`.
    ("q19", "lineitem", "l_partkey"),
    ("q19", "part", "p_partkey"),
    // From the `StartsWith` Spark substitutes onto the second `part` scan.
    ("q20", "part", "p_name"),
    // From a non-equality join condition (`ps_availqty > 0.5 * sum(..)`).
    ("q20", "partsupp", "ps_availqty"),
];

/// `(query, table, column)` Krishiv infers and Spark does not: keys of the
/// semi and anti joins that `EXISTS`/`IN`/`NOT EXISTS` become (Spark
/// rewrites those subqueries after its inference).
const EXTRA: &[(&str, &str, &str)] = &[
    ("q04", "lineitem", "l_orderkey"),
    ("q04", "orders", "o_orderkey"),
    ("q20", "part", "p_partkey"),
    ("q20", "supplier", "s_suppkey"),
    ("q22", "orders", "o_custkey"),
];

type Key = (String, String, String);

#[test]
fn tpch_isnotnull_matches_spark_but_for_the_listed_differences() {
    let path = support::root().join("tests/fixtures/spark-4.2.0/inferred-join-filters.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let ctx = support::ctx();
    let (mut missing, mut extra) = (BTreeSet::<Key>::new(), BTreeSet::<Key>::new());
    for q in 1..=22 {
        let name = format!("q{q:02}");
        let spark = spark_isnotnull(&doc["tpch"][&name]);
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let krishiv = scan_isnotnull(&optimize(&plan).unwrap().root().to_string());
        let tables: BTreeSet<&String> = spark.keys().chain(krishiv.keys()).collect();
        for table in tables {
            let (s, k) = (spark.get(table), krishiv.get(table));
            let empty = BTreeSet::new();
            let (s, k) = (s.unwrap_or(&empty), k.unwrap_or(&empty));
            for c in s.difference(k) {
                missing.insert((name.clone(), table.clone(), c.clone()));
            }
            for c in k.difference(s) {
                extra.insert((name.clone(), table.clone(), c.clone()));
            }
        }
    }
    let listed = |l: &[(&str, &str, &str)]| -> BTreeSet<Key> {
        l.iter()
            .map(|(q, t, c)| ((*q).to_owned(), (*t).to_owned(), (*c).to_owned()))
            .collect()
    };
    assert_eq!(
        missing,
        listed(MISSING),
        "Spark infers these, Krishiv does not"
    );
    assert_eq!(extra, listed(EXTRA), "Krishiv infers these, Spark does not");
}

/// Spark's inferred `isnotnull` columns per table of one query, without the
/// filter-own ones.
fn spark_isnotnull(entry: &Value) -> BTreeMap<String, BTreeSet<String>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (rel, cats) in entry["scans"].as_object().unwrap() {
        let table = rel
            .split("default.")
            .nth(1)
            .unwrap()
            .split(['#', '['])
            .next()
            .unwrap()
            .to_owned();
        let written: Vec<&str> = cats["written"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap())
            .collect();
        let set = out.entry(table).or_default();
        for p in cats["isnotnull"].as_array().unwrap() {
            let col = column_of(p.as_str().unwrap());
            if !written.iter().any(|w| reads(w, &col)) {
                set.insert(col);
            }
        }
    }
    out
}

/// The `isnotnull` columns of the filters directly over each table's scans
/// in Krishiv's plan text (shared scans printed once, read by id).
fn scan_isnotnull(text: &str) -> BTreeMap<String, BTreeSet<String>> {
    let lines: Vec<&str> = text.lines().collect();
    let mut shared: BTreeMap<String, String> = BTreeMap::new();
    for l in &lines {
        if let (Some(t), Some(id)) = (scan_table(l), l.split("(shared #").nth(1)) {
            shared.insert(id.trim_end_matches(')').to_owned(), t);
        }
    }
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (i, l) in lines.iter().enumerate() {
        let table = scan_table(l).or_else(|| {
            l.split("Shared #")
                .nth(1)
                .and_then(|r| r.split(',').next())
                .and_then(|id| shared.get(id).cloned())
        });
        let Some(table) = table else { continue };
        let set = out.entry(table).or_default();
        for above in lines[..i].iter().rev() {
            let Some(pred) = above.split("Filter ").nth(1) else {
                break;
            };
            for part in pred.split("isnotnull(").skip(1) {
                set.insert(strip_id(part.split(')').next().unwrap_or("")));
            }
        }
    }
    out
}

fn scan_table(line: &str) -> Option<String> {
    let rest = line.split("Scan spark_catalog.default.").nth(1)?;
    Some(rest.split(' ').next()?.to_owned())
}

fn column_of(isnotnull: &str) -> String {
    strip_id(
        isnotnull
            .trim_start_matches("isnotnull(")
            .trim_end_matches(')'),
    )
}

fn strip_id(c: &str) -> String {
    c.split('#').next().unwrap_or(c).to_owned()
}

/// Whether Spark's written predicate `w` reads column `col` (a whole word).
fn reads(w: &str, col: &str) -> bool {
    w.match_indices(col).any(|(i, _)| {
        let before = w[..i].chars().next_back();
        let after = w[i + col.len()..].chars().next();
        let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        !word(before) && !word(after)
    })
}
