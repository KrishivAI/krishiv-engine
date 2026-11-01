// SPDX-License-Identifier: Apache-2.0
//! Aggregate `FILTER (WHERE ...)`, several `DISTINCT` aggregates in one
//! query and `FROM ... SELECT ... [SELECT ...]` through SQL. Expected
//! values follow Spark 4.2's definitions (`group-by-filter.sql`,
//! `group-by.sql` and `subquery-in-from.sql`): a filter keeps the rows
//! where it is true (a NULL is not true), `count` skips NULL arguments,
//! `sum` of no rows is NULL, and every `SELECT` of a `FROM`-first query
//! reads the shared `FROM`, combined with `UNION ALL`.
#![forbid(unsafe_code)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Seven rows `(a, b, p)`: `a` groups them (1: rows 1–2, 2: rows 3–4, 3:
/// row 5, NULL: rows 6–7); `b` and `p` hold NULLs.
fn session() -> R<Session> {
    let s = Session::new()?;
    let columns: Vec<(&str, DataType, ArrayRef)> = vec![
        (
            "a",
            DataType::Int32,
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(3),
                None,
                None,
            ])),
        ),
        (
            "b",
            DataType::Int32,
            Arc::new(Int32Array::from(vec![
                Some(10),
                Some(20),
                Some(10),
                None,
                Some(30),
                Some(10),
                None,
            ])),
        ),
        (
            "p",
            DataType::Boolean,
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                Some(true),
                Some(true),
                None,
                Some(true),
                Some(false),
            ])),
        ),
    ];
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|(n, t, _)| Field::new(*n, t.clone(), true))
            .collect::<Vec<_>>(),
    ));
    let arrays = columns.into_iter().map(|c| c.2).collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    s.register_batches("t", schema, vec![batch])?;
    Ok(s)
}

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        DataType::Float64 => format!("{:?}", c.as_primitive::<Float64Type>().value(row)),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        other => format!("<{other}>"),
    }
}

/// The rows (`a|b|c`) of `sql`, in order.
fn rows(s: &Session, sql: &str) -> R<Vec<String>> {
    block_on(async {
        let df = s.sql(sql).await?;
        let batches = df.collect().await?;
        Ok(batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows()).map(|r| {
                    b.columns()
                        .iter()
                        .map(|c| cell(c.as_ref(), r))
                        .collect::<Vec<_>>()
                        .join("|")
                })
            })
            .collect())
    })
}

/// The rows of `sql`, sorted (for order-free `UNION ALL` comparisons).
fn sorted_rows(s: &Session, sql: &str) -> R<Vec<String>> {
    let mut out = rows(s, sql)?;
    out.sort();
    Ok(out)
}

/// The error class of `sql`, which must fail.
fn class(s: &Session, sql: &str) -> R<String> {
    match rows(s, sql) {
        Ok(_) => Err(format!("{sql} did not fail").into()),
        Err(e) => match e.downcast::<KrishivError>() {
            Ok(e) => Ok(e.error_class().name().to_owned()),
            Err(_) => Err("not a KrishivError".into()),
        },
    }
}

/// `FILTER` keeps the rows where the condition is true: `count(*)` the
/// passing rows, `count(b)` those with non-NULL `b`, `sum` and friends
/// the passing values.
#[test]
fn filter_reads_only_passing_rows() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT COUNT(*) FILTER (WHERE p), COUNT(b) FILTER (WHERE p), \
             SUM(b) FILTER (WHERE p), AVG(b) FILTER (WHERE p), \
             MIN(b) FILTER (WHERE p), MAX(b) FILTER (WHERE p) FROM t",
        )?,
        ["4|3|30|10.0|10|10"]
    );
    // Beside unfiltered calls, which still read all rows.
    assert_eq!(
        rows(
            &s,
            "SELECT COUNT(*), COUNT(b), SUM(b), SUM(b) FILTER (WHERE p) FROM t",
        )?,
        ["7|5|80|30"]
    );
    // A condition true of no row: counts are 0, other values NULL.
    assert_eq!(
        rows(
            &s,
            "SELECT COUNT(*) FILTER (WHERE false), SUM(b) FILTER (WHERE false), \
             COUNT(b) FILTER (WHERE p AND false) FROM t",
        )?,
        ["0|NULL|0"]
    );
    assert_eq!(
        rows(&s, "SELECT COUNT(*) FILTER (WHERE p) FROM t WHERE 1 = 0")?,
        ["0"]
    );
    assert_eq!(
        rows(&s, "SELECT SUM(b) FILTER (WHERE p) FROM t WHERE 1 = 0")?,
        ["NULL"]
    );
    // A NULL condition row counts where the condition is NULL.
    assert_eq!(
        rows(
            &s,
            "SELECT COUNT(*) FILTER (WHERE p IS NULL), SUM(b) FILTER (WHERE p IS NULL) FROM t",
        )?,
        ["1|30"]
    );
    // Moment aggregates read only the passing rows too: `b` is 10 and 20
    // where `a = 1` (population variance 25) and 10 on every `p` row (0).
    assert_eq!(
        rows(
            &s,
            "SELECT VAR_POP(b) FILTER (WHERE a = 1), VAR_POP(b) FILTER (WHERE p), \
             STDDEV_SAMP(b) FILTER (WHERE a = 1) FROM t",
        )?,
        ["25.0|0.0|7.0710678118654755"]
    );
    Ok(())
}

/// `FILTER` under `GROUP BY`, and `FIRST`/`LAST` over passing rows.
#[test]
fn filter_grouped_first_last() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT a, COUNT(*) FILTER (WHERE p), SUM(b) FILTER (WHERE p) \
             FROM t GROUP BY a ORDER BY a",
        )?,
        ["NULL|1|10", "1|1|10", "2|2|10", "3|0|NULL"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT FIRST(b) FILTER (WHERE p), LAST(b) FILTER (WHERE a = 2) FROM t",
        )?,
        ["10|NULL"]
    );
    Ok(())
}

/// One `DISTINCT` aggregate with a filter, and several `DISTINCT`
/// aggregates with different arguments in one query.
#[test]
fn distinct_aggregates_with_filters() -> R {
    let s = session()?;
    assert_eq!(
        rows(&s, "SELECT COUNT(DISTINCT b) FILTER (WHERE p) FROM t")?,
        ["1"]
    );
    assert_eq!(
        rows(&s, "SELECT COUNT(DISTINCT a) FILTER (WHERE p) FROM t")?,
        ["2"]
    );
    assert_eq!(
        rows(&s, "SELECT COUNT(DISTINCT a), COUNT(DISTINCT b) FROM t")?,
        ["3|3"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT COUNT(DISTINCT a) FILTER (WHERE p), COUNT(DISTINCT b) FILTER (WHERE p) FROM t",
        )?,
        ["2|1"]
    );
    // The same argument set twice, with and without a filter.
    assert_eq!(
        rows(
            &s,
            "SELECT COUNT(DISTINCT a), COUNT(DISTINCT a) FILTER (WHERE p) FROM t",
        )?,
        ["3|2"]
    );
    // Several arguments, with and without a filter (`count` needs every
    // argument non-NULL).
    assert_eq!(rows(&s, "SELECT COUNT(DISTINCT a, b) FROM t")?, ["4"]);
    assert_eq!(
        rows(&s, "SELECT COUNT(DISTINCT a, b) FILTER (WHERE p) FROM t")?,
        ["2"]
    );
    // Grouped.
    assert_eq!(
        rows(
            &s,
            "SELECT p, COUNT(DISTINCT a), COUNT(DISTINCT b) FROM t GROUP BY p ORDER BY p",
        )?,
        ["NULL|1|1", "false|1|1", "true|2|1"]
    );
    Ok(())
}

/// `DISTINCT` aggregates beside filtered and unfiltered plain ones.
#[test]
fn distinct_aggregates_beside_plain_ones() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT COUNT(DISTINCT a), SUM(b) FILTER (WHERE p) FROM t",
        )?,
        ["3|30"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT a, COUNT(DISTINCT b) FILTER (WHERE p), SUM(b) \
             FROM t GROUP BY a ORDER BY a",
        )?,
        ["NULL|1|10", "1|1|30", "2|1|10", "3|0|30"]
    );
    // `FILTER` on the plain aggregate of a grouped multi-`DISTINCT` query.
    assert_eq!(
        rows(
            &s,
            "SELECT a, COUNT(DISTINCT b), SUM(b) FILTER (WHERE p) \
             FROM t GROUP BY a ORDER BY a",
        )?,
        ["NULL|1|10", "1|2|10", "2|1|10", "3|1|NULL"]
    );
    Ok(())
}

/// One `DISTINCT` aggregate without a filter plans as one aggregate,
/// and a filter may read a grouping key.
#[test]
fn single_distinct_and_group_key_filter() -> R {
    let s = session()?;
    assert_eq!(rows(&s, "SELECT COUNT(DISTINCT a) FROM t")?, ["3"]);
    assert_eq!(rows(&s, "SELECT AVG(DISTINCT b) FROM t")?, ["20.0"]);
    assert_eq!(
        rows(
            &s,
            "SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a ORDER BY a",
        )?,
        ["NULL|1", "1|2", "2|1", "3|1"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT a, COUNT(*) FILTER (WHERE a > 1) FROM t GROUP BY a ORDER BY a",
        )?,
        ["NULL|0", "1|0", "2|2", "3|1"]
    );
    // A rewrite with one `DISTINCT` set beside an unfiltered twin.
    assert_eq!(
        rows(
            &s,
            "SELECT COUNT(DISTINCT b), COUNT(DISTINCT b) FILTER (WHERE p) FROM t",
        )?,
        ["3|1"]
    );
    Ok(())
}

/// The multi-`DISTINCT` rewrite plans an expansion with a group id.
#[test]
fn multi_distinct_plans_an_expand() -> R {
    let s = session()?;
    for sql in [
        "SELECT COUNT(DISTINCT a), COUNT(DISTINCT b) FROM t",
        "SELECT COUNT(DISTINCT a), COUNT(DISTINCT b) FROM t GROUP BY a",
    ] {
        let plan = rows(&s, &format!("EXPLAIN {sql}"))?.join("\n");
        assert!(plan.contains("Expand"), "{sql}:\n{plan}");
    }
    let plan = rows(&s, "EXPLAIN SELECT COUNT(DISTINCT a) FROM t")?.join("\n");
    assert!(!plan.contains("Expand"), "single set plans as one:\n{plan}");
    Ok(())
}

/// `FILTER` misuse and rewrites the engine does not do.
#[test]
fn filter_rejections() -> R {
    let s = session()?;
    assert_eq!(
        class(&s, "SELECT SUM(b) FILTER (WHERE a) FROM t")?,
        "INVALID_AGGREGATE_FILTER.NOT_BOOLEAN"
    );
    assert_eq!(
        class(
            &s,
            "SELECT SUM(b) FILTER (WHERE RANK() OVER (ORDER BY a) > 1) FROM t",
        )?,
        "INVALID_AGGREGATE_FILTER.CONTAINS_WINDOW_FUNCTION"
    );
    // Several `DISTINCT` sets over grouping sets.
    assert_eq!(
        class(
            &s,
            "SELECT COUNT(DISTINCT a), COUNT(DISTINCT b) FROM t GROUP BY GROUPING SETS ((a), (b))",
        )?,
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    Ok(())
}

/// One `SELECT` over `FROM` reads as the plain query does.
#[test]
fn from_first_single_select() -> R {
    let s = session()?;
    let plain = rows(&s, "SELECT a, b FROM t ORDER BY a, b")?;
    assert_eq!(
        plain,
        [
            "NULL|NULL",
            "NULL|10",
            "1|10",
            "1|20",
            "2|NULL",
            "2|10",
            "3|30"
        ]
    );
    assert_eq!(rows(&s, "FROM t SELECT a, b ORDER BY a, b")?, plain);
    // Qualified references read the shared `FROM`.
    assert_eq!(
        rows(&s, "FROM t SELECT t.a WHERE t.a = 2 ORDER BY t.a")?,
        rows(&s, "SELECT t.a FROM t WHERE t.a = 2 ORDER BY t.a")?,
    );
    // A bare `FROM t` returns the relation.
    assert_eq!(
        sorted_rows(&s, "FROM t")?,
        sorted_rows(&s, "SELECT * FROM t")?
    );
    Ok(())
}

/// Several `SELECT`s over one `FROM` combine with `UNION ALL`.
#[test]
fn from_first_several_selects() -> R {
    let s = session()?;
    assert_eq!(
        sorted_rows(&s, "FROM t SELECT a WHERE a = 1 SELECT b WHERE b = 10")?,
        ["1", "1", "10", "10", "10"]
    );
    // Each body groups and orders on its own.
    assert_eq!(
        sorted_rows(
            &s,
            "FROM t SELECT a GROUP BY a ORDER BY a LIMIT 2 \
             SELECT b WHERE b IS NOT NULL ORDER BY b LIMIT 2",
        )?,
        ["1", "10", "10", "NULL"]
    );
    // Branches of different types coerce, as `UNION ALL` does.
    assert_eq!(
        sorted_rows(&s, "FROM t SELECT a SELECT CAST(a AS DOUBLE)")?,
        [
            "1.0", "1.0", "1.0", "1.0", "2.0", "2.0", "2.0", "2.0", "3.0", "3.0", "NULL", "NULL",
            "NULL", "NULL"
        ]
    );
    // `DISTINCT` bodies.
    assert_eq!(
        sorted_rows(&s, "FROM t SELECT DISTINCT a SELECT DISTINCT b")?,
        ["1", "10", "2", "20", "3", "30", "NULL", "NULL"]
    );
    // Branches of different widths fail, as `UNION ALL` does.
    assert_eq!(
        class(&s, "FROM t SELECT a SELECT a, b")?,
        "NUM_COLUMNS_MISMATCH"
    );
    Ok(())
}

/// Shared composite arguments across calls evaluate once per row.
#[test]
fn shared_composite_arguments() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT SUM(b + a), AVG(b + a), COUNT(DISTINCT a), COUNT(DISTINCT b) FROM t",
        )?,
        ["77|19.25|3|3"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT COUNT(DISTINCT a), COUNT(DISTINCT a + 0), COUNT(DISTINCT b) FROM t",
        )?,
        ["3|3|3"]
    );
    Ok(())
}

/// The multi-`DISTINCT` Spark goldens that need no `VALUES` table:
/// `group-by.sql` #16 and `group-by-filter.sql` #68.
#[test]
fn spark_golden_multi_distinct_statements() -> R {
    let s = session()?;
    let sub = "(SELECT 1 AS a, 2 AS b, 3 AS c)";
    assert_eq!(
        rows(
            &s,
            &format!("SELECT COUNT(DISTINCT b), COUNT(DISTINCT b, c) FROM {sub} GROUP BY a"),
        )?,
        ["1|1"]
    );
    assert_eq!(
        rows(
            &s,
            &format!(
                "SELECT COUNT(DISTINCT b) FILTER (WHERE b > 0), \
                 COUNT(DISTINCT b, c) FILTER (WHERE b > 0 AND c > 2) FROM {sub} GROUP BY a"
            ),
        )?,
        ["1|1"]
    );
    Ok(())
}

/// `HAVING` over a rewritten aggregate: calls from the select list
/// resolve through its output; a new `DISTINCT` set stays unsupported.
#[test]
fn having_over_multi_distinct() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT a, COUNT(DISTINCT a), COUNT(DISTINCT b) FROM t GROUP BY a \
             HAVING COUNT(DISTINCT b) > 1 ORDER BY a",
        )?,
        ["1|1|2"]
    );
    assert_eq!(
        class(
            &s,
            "SELECT a, COUNT(DISTINCT a) FROM t GROUP BY a HAVING COUNT(DISTINCT b) > 1 ORDER BY a",
        )?,
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    Ok(())
}

/// A global aggregate over no rows is one row: the regular counts beside
/// several DISTINCT argument sets are 0, not NULL.
#[test]
fn counts_beside_multi_distinct_over_empty_input_are_zero() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT COUNT(*), COUNT(b), COUNT(DISTINCT a), COUNT(DISTINCT b) FROM t WHERE false",
        )?,
        ["0|0|0|0"]
    );
    Ok(())
}

/// Two distinct argument sets over a grouping expression that is no plain column: the expansion
/// computes the key below it and passes the input's columns through.
#[test]
fn distinct_sets_group_by_an_expression() -> R<()> {
    let s = session()?;
    let df = block_on(s.sql(
        "SELECT CAST(count(DISTINCT h) AS STRING), CAST(sum(DISTINCT s) AS STRING) \
         FROM (SELECT * FROM VALUES ('a', 1.0, 10), ('b', 2.0, 20), ('b', 2.0, 30) AS t(h, s, d)) \
         GROUP BY d > 15",
    ))?;
    let mut rows: Vec<String> = Vec::new();
    for b in block_on(df.collect())? {
        for i in 0..b.num_rows() {
            rows.push(format!(
                "{}|{}",
                b.column(0).as_string_view().value(i),
                b.column(1).as_string_view().value(i)
            ));
        }
    }
    rows.sort();
    assert_eq!(rows, ["1|1.0", "1|2.0"]);
    Ok(())
}
