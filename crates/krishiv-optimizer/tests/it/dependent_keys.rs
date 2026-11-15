// SPDX-License-Identifier: Apache-2.0
//! Group keys computed from other column keys leave the keys (see
//! `dependent_keys.rs`): the shapes it applies to and each condition that
//! keeps a key. The values, overflow included, are checked against the
//! unrewritten plan on the executor, in `krishiv-exec`'s planner tests.

use std::sync::Arc;

use krishiv_optimizer::optimize;
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};
use krishiv_plan::{Grouping, LogicalPlan, PlanNode};

use crate::support;

/// `h(ip int, l bigint, f double, s string, c string collate utf8_lcase,
/// d date, x int)`.
fn catalog() -> MemoryCatalog {
    let col = |name: &str, t: &str| TableColumn {
        name: name.to_owned(),
        data_type: krishiv_types::parse_data_type(t).unwrap(),
        nullable: true,
        char_varchar: None,
    };
    MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "h",
        vec![
            col("ip", "int"),
            col("l", "bigint"),
            col("f", "double"),
            col("s", "string"),
            col("c", "string collate utf8_lcase"),
            col("d", "date"),
            col("x", "int"),
        ],
        None,
    )
}

/// The key counts of the aggregates of `sql`, before and after optimizing
/// (outermost first); the output and the fixed point are checked too.
fn keys(sql: &str) -> (Vec<usize>, Vec<usize>, String) {
    let ctx = support::ctx_over(catalog());
    let resolved = support::resolve(sql, &ctx);
    let optimized = optimize(&resolved).unwrap();
    assert_eq!(
        optimized.root().resolved_output(),
        resolved.root().resolved_output(),
        "{sql}: the output keeps its attributes (id, name, type, nullability)"
    );
    let again = optimize(&optimized).unwrap();
    assert!(
        again.structurally_eq(&optimized),
        "{sql}: a fixed point\n{}",
        again.root()
    );
    (
        key_counts(resolved.root()),
        key_counts(optimized.root()),
        optimized.root().to_string(),
    )
}

fn key_counts(root: &Arc<LogicalPlan>) -> Vec<usize> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        if let PlanNode::Aggregate {
            grouping: Grouping::GroupBy(k),
            ..
        } = &p.node
        {
            out.push(k.len());
        }
        p.for_each_child(&mut |c| stack.push(c));
    }
    out
}

#[test]
fn clickbench_q36_groups_by_one_key() {
    let sql = "SELECT ip, ip - 1, ip - 2, ip - 3, COUNT(*) AS c FROM h \
               GROUP BY ip, ip - 1, ip - 2, ip - 3 ORDER BY c DESC LIMIT 10";
    let (before, after, text) = keys(sql);
    assert_eq!(before, vec![4], "{text}");
    assert_eq!(after, vec![1], "{text}");
    // The shifted keys are still computed, by the aggregate.
    assert!(text.contains("(ip#0 - 3)"), "{text}");
}

#[test]
fn shapes_that_lose_their_dependent_keys() {
    for (sql, want) in [
        // A dependent key that is not selected.
        ("SELECT COUNT(*) FROM h GROUP BY ip, ip + 1", 1),
        // Over two column keys, and a literal key.
        (
            "SELECT ip, x, ip * x, COUNT(*) FROM h GROUP BY ip, x, ip * x",
            2,
        ),
        ("SELECT ip, 'k', COUNT(*) FROM h GROUP BY ip, 'k'", 1),
        // Functions of a string, a date and a bigint.
        (
            "SELECT s, upper(s), COUNT(*) FROM h GROUP BY s, upper(s)",
            1,
        ),
        ("SELECT d, year(d), COUNT(*) FROM h GROUP BY d, year(d)", 1),
        ("SELECT l, l % 7, SUM(x) FROM h GROUP BY l, l % 7", 1),
        // A filter on the dependent key above the aggregate.
        (
            "SELECT ip, ip - 1, COUNT(*) FROM h GROUP BY ip, ip - 1 HAVING ip - 1 > 5",
            1,
        ),
    ] {
        let (_, after, text) = keys(sql);
        assert_eq!(after, vec![want], "{sql}\n{text}");
    }
}

#[test]
fn conditions_that_keep_a_key() {
    for (sql, why) in [
        (
            "SELECT f, 1 / f, COUNT(*) FROM h GROUP BY f, 1 / f",
            "0.0 and -0.0 form one group of f but two of 1 / f",
        ),
        (
            "SELECT c, length(c), COUNT(*) FROM h GROUP BY c, length(c)",
            "a collated string groups values that differ",
        ),
        (
            "SELECT ip, x + 1, COUNT(*) FROM h GROUP BY ip, x + 1",
            "x + 1 reads a column that is not a key",
        ),
        (
            "SELECT ip, ip + rand(), COUNT(*) FROM h GROUP BY ip, ip + rand()",
            "nondeterministic",
        ),
        (
            "SELECT ip + 1, ip + 2, COUNT(*) FROM h GROUP BY ip + 1, ip + 2",
            "no bare column key",
        ),
        (
            "SELECT 'k', COUNT(*) FROM h GROUP BY 'k'",
            "the aggregate would become global",
        ),
    ] {
        let (before, after, text) = keys(sql);
        assert_eq!(before, after, "{sql}: {why}\n{text}");
    }
}

#[test]
fn rollup_keeps_its_keys() {
    let sql = "SELECT ip, ip - 1, COUNT(*) FROM h GROUP BY ROLLUP(ip, ip - 1)";
    let ctx = support::ctx_over(catalog());
    let resolved = support::resolve(sql, &ctx);
    let optimized = optimize(&resolved).unwrap();
    let text = optimized.root().to_string();
    // A rollup's sets are lowered to an expand and a plain aggregate; a key
    // of a set (and `spark_grouping_id`) is never a function of another.
    assert!(text.contains("(ip#0 - 1)"), "{text}");
}
