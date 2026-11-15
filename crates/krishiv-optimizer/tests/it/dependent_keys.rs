// SPDX-License-Identifier: Apache-2.0
//! Group keys determined by other column keys (`dependent_group_keys`):
//! the shapes it finds and each condition that keeps a key. The physical
//! planner groups by the rest; the values and errors are checked against
//! Spark's per-row grouping in `krishiv-exec`'s planner tests.

use krishiv_optimizer::dependent_group_keys;
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};
use krishiv_plan::{Grouping, PlanNode};

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

/// The key counts of the first aggregate of `sql` (as analyzed): all of
/// them, and those that are not dependent.
fn keys(sql: &str) -> (Vec<usize>, Vec<usize>, String) {
    let ctx = support::ctx_over(catalog());
    let resolved = support::resolve(sql, &ctx);
    let mut all = Vec::new();
    let mut kept = Vec::new();
    let mut stack = vec![resolved.root()];
    while let Some(p) = stack.pop() {
        if let PlanNode::Aggregate {
            grouping: Grouping::GroupBy(k),
            ..
        } = &p.node
        {
            all.push(k.len());
            kept.push(dependent_group_keys(k).iter().filter(|d| !**d).count());
        }
        p.for_each_child(&mut |c| stack.push(c));
    }
    (all, kept, resolved.root().to_string())
}

#[test]
fn clickbench_q36_groups_by_one_key() {
    let sql = "SELECT ip, ip - 1, ip - 2, ip - 3, COUNT(*) AS c FROM h \
               GROUP BY ip, ip - 1, ip - 2, ip - 3 ORDER BY c DESC LIMIT 10";
    let (before, after, text) = keys(sql);
    assert_eq!(before, vec![4], "{text}");
    assert_eq!(after, vec![1], "{text}");
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
