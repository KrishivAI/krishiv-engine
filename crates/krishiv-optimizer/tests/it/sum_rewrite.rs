// SPDX-License-Identifier: Apache-2.0
//! `SUM(x + c)` as `SUM(x) + c * COUNT(x)`: the shapes it applies to and
//! each condition that keeps it away (see `sum_rewrite.rs`). The values are
//! checked against the unrewritten plan on the executor, in
//! `krishiv-exec`'s planner tests.

use krishiv_optimizer::optimize;
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};

use crate::support;

/// `s(ti tinyint, si smallint, i int, l bigint, f double, d decimal(10,2), g int)`.
fn catalog() -> MemoryCatalog {
    let col = |name: &str, t: &str| TableColumn {
        name: name.to_owned(),
        data_type: krishiv_types::parse_data_type(t).unwrap(),
        nullable: true,
    };
    MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "s",
        vec![
            col("ti", "tinyint"),
            col("si", "smallint"),
            col("i", "int"),
            col("l", "bigint"),
            col("f", "double"),
            col("d", "decimal(10,2)"),
            col("g", "int"),
        ],
        None,
    )
}

/// The optimised plan of `sql`, as text.
fn plan(sql: &str) -> String {
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
    optimized.root().to_string()
}

/// The line of the plan that prints the aggregate's computations.
fn aggregate_line(text: &str) -> &str {
    text.lines()
        .find(|l| {
            l.trim_start_matches(['+', '-', ' '])
                .starts_with("Aggregate")
        })
        .unwrap_or_else(|| panic!("no aggregate in\n{text}"))
}

fn rewritten(sql: &str) -> bool {
    let text = plan(sql);
    let has_count = text.contains("count(");
    // A rewritten aggregate computes the shifted sums from one sum and one
    // count; the originals are gone.
    assert!(
        has_count || !text.contains("1L *"),
        "{sql}: no count without a product\n{text}"
    );
    has_count
}

#[test]
fn narrow_columns_with_a_small_constant_are_rewritten() {
    // Spark types `smallint + 1` as an `int` addition over a cast column;
    // CSE then computes the shared cast once below the aggregate.
    let text = plan("SELECT SUM(si + 1) FROM s");
    assert!(text.contains("CAST(si#1 AS INT) AS cse_"), "{text}");
    assert!(text.contains("sum(cse_"), "{text}");
    assert!(text.contains("count(cse_"), "{text}");
    assert!(!text.contains("(CAST(si#1 AS INT) + 1)"), "{text}");
    for sql in [
        "SELECT SUM(si + 1) FROM s",
        "SELECT SUM(1 + si) FROM s",
        "SELECT SUM(si - 2) FROM s",
        "SELECT SUM(2 - si) FROM s",
        "SELECT SUM(ti + 100) FROM s",
        "SELECT SUM(si + -5) FROM s",
        "SELECT SUM(si - -5) FROM s",
        "SELECT SUM(5 - si) FROM s",
        "SELECT SUM(si + 65536) FROM s",
        "SELECT SUM(si - 65536) FROM s",
        "SELECT SUM(si + 0) FROM s",
        "SELECT SUM(CAST(si AS BIGINT) + 1) FROM s",
        "SELECT SUM(CAST(ti AS INT) + 7) FROM s",
        "SELECT g, SUM(si + 1) FROM s GROUP BY g",
        "SELECT g, SUM(si + 1) FROM s GROUP BY g HAVING SUM(si + 1) > 3 ORDER BY g",
    ] {
        assert!(rewritten(sql), "{sql}\n{}", plan(sql));
    }
}

#[test]
fn many_shifted_sums_share_one_sum_and_one_count() {
    let items: Vec<String> = (0..90).map(|k| format!("SUM(si + {k})")).collect();
    let sql = format!("SELECT SUM(si), {} FROM s", items.join(", "));
    let text = plan(&sql);
    // The aggregate computes `sum(si)`, and for the 90 shifted sums one
    // `sum` and one `count` of the shared cast, computed once below it.
    let aggregate = aggregate_line(&text);
    assert_eq!(aggregate.matches("count(cse_").count(), 1, "{text}");
    assert_eq!(aggregate.matches("sum(cse_").count(), 1, "{text}");
    assert_eq!(aggregate.matches("sum(si#1)").count(), 1, "{text}");
    assert!(text.contains("CAST(si#1 AS INT) AS cse_"), "{text}");
}

#[test]
fn a_sum_of_the_column_already_there_is_reused() {
    let text = plan("SELECT SUM(CAST(si AS INT)), SUM(si + 1) FROM s");
    // The aggregate's own `sum` serves both items; CSE computes the shared
    // cast once below the aggregate.
    assert_eq!(
        aggregate_line(&text).matches("sum(cse_").count(),
        1,
        "{text}"
    );
    assert!(text.contains("count(cse_"), "{text}");
    assert!(text.contains("CAST(si#1 AS INT) AS cse_"), "{text}");
}

#[test]
fn conditions_that_keep_the_sum_as_it_is() {
    for (sql, why) in [
        // `INT_MAX + 1` overflows per row: Spark raises or wraps.
        ("SELECT SUM(i + 1) FROM s", "an int column can overflow"),
        ("SELECT SUM(i - 1) FROM s", "an int column can overflow"),
        ("SELECT SUM(l + 1) FROM s", "a bigint column can overflow"),
        // A `TINYINT` stays a `TINYINT`: `127 + 1Y` overflows per row, as
        // `32767 + 1S` does for a `SMALLINT`.
        (
            "SELECT SUM(ti + 1Y) FROM s",
            "a tinyint addition can overflow per row",
        ),
        (
            "SELECT SUM(ti - -1Y) FROM s",
            "a tinyint addition can overflow per row",
        ),
        (
            "SELECT SUM(1Y - ti) FROM s",
            "a tinyint subtraction can overflow per row",
        ),
        (
            "SELECT SUM(si + 1S) FROM s",
            "a smallint addition can overflow per row",
        ),
        (
            "SELECT SUM(si + 65537) FROM s",
            "the constant is past the bound",
        ),
        (
            "SELECT SUM(si + 100000) FROM s",
            "the constant is past the bound",
        ),
        (
            "SELECT SUM(CAST(i AS BIGINT) + 1) FROM s",
            "an int column's accumulation bound is too wide",
        ),
        (
            "SELECT SUM(f + 1.5) FROM s",
            "floating-point sums are not associative",
        ),
        ("SELECT SUM(si + 0.5) FROM s", "a decimal addition"),
        ("SELECT SUM(d + 1) FROM s", "a decimal addition"),
        (
            "SELECT SUM(DISTINCT si + 1) FROM s",
            "DISTINCT counts each value once",
        ),
        (
            "SELECT SUM(si + 1) FILTER (WHERE g > 1) FROM s",
            "FILTER restricts both sides",
        ),
        ("SELECT SUM(si + ti) FROM s", "no constant"),
        ("SELECT SUM(si * 2) FROM s", "not an addition"),
        ("SELECT SUM(si + si) FROM s", "no constant"),
        ("SELECT SUM(si + g) FROM s", "no constant"),
        ("SELECT SUM(si) FROM s", "nothing added"),
        (
            "SELECT SUM(g + 1) FROM (SELECT g FROM s)",
            "an int column can overflow",
        ),
        (
            "SELECT AVG(si + 1) FROM s",
            "only SUM is rewritten (AVG would follow SUM and COUNT)",
        ),
    ] {
        assert!(!rewritten(sql), "{sql} ({why})\n{}", plan(sql));
    }
}

#[test]
fn an_expression_over_the_sum_is_left_as_it_is() {
    // The item is `sum(..) * 2`, not a plain sum.
    let text = plan("SELECT SUM(si + 1) * 2 FROM s");
    assert!(!text.contains("count("), "{text}");
}

#[test]
fn aggregates_in_subqueries_are_rewritten_too() {
    let text = plan("SELECT g FROM s WHERE g > (SELECT SUM(si + 1) FROM s)");
    assert!(text.contains("count(cse_"), "{text}");
}
