// SPDX-License-Identifier: Apache-2.0
//! `DISTRIBUTE BY`, `CLUSTER BY` and `SORT BY ... DISTRIBUTE BY` (Spark's `RepartitionByExpression`), in queries
//! and after `|>`. The expressions are checked as Spark checks them and evaluated as its shuffle evaluates them;
//! the rows are the input's. Expected results and errors are from a Spark 4.2.0 run.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(async {
        s.sql("CREATE TEMPORARY VIEW t AS SELECT * FROM VALUES (0, 'abc'), (1, 'def') AS t(x, y)")
            .await?
            .collect()
            .await
    })
    .unwrap();
    s
}

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        other => format!("<{other}>"),
    }
}

/// The rows (`a|b`) of `sql`, in order.
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    batches
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
        .collect()
}

fn sorted(s: &Session, sql: &str) -> Vec<String> {
    let mut out = rows(s, sql);
    out.sort();
    out
}

fn error(s: &Session, sql: &str) -> (String, String) {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql}: no error"),
        Err(e) => (e.error_class().name().to_owned(), e.to_string()),
    }
}

/// The error class of `sql` and its message parameters, `key=value` sorted.
fn params(s: &Session, sql: &str) -> (String, Vec<String>) {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql}: no error"),
        Err(e) => {
            let mut p: Vec<String> = e
                .message_parameters()
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            p.sort();
            (e.error_class().name().to_owned(), p)
        }
    }
}

/// Spark evaluates the partitioning expressions while it shuffles, so
/// `x + 'a'` (`'a'` cast to a number) fails with `CAST_INVALID_INPUT`
/// under ANSI, also under a `count(*)` that reads no column; the other
/// cases are in `distribute_by_goldens.rs`.
#[test]
fn the_expressions_are_evaluated() {
    let s = session();
    for sql in [
        "SELECT x FROM t DISTRIBUTE BY x + 'a'",
        "SELECT count(*) FROM (SELECT x FROM t DISTRIBUTE BY x + 'a')",
    ] {
        assert_eq!(error(&s, sql).0, "CAST_INVALID_INPUT", "{sql}");
    }
    s.conf().set("spark.sql.ansi.enabled", "false").unwrap();
    assert_eq!(
        sorted(&s, "SELECT x FROM t DISTRIBUTE BY x + 'a'"),
        ["0", "1"]
    );
}

/// Any expression a projection allows partitions: a column the select list
/// dropped, a map, an array, a literal, `rand()`.
#[test]
fn the_rows_are_the_input_rows() {
    let s = session();
    for sql in [
        "SELECT x FROM t DISTRIBUTE BY y",
        "SELECT x FROM t DISTRIBUTE BY 1",
        "SELECT x FROM t DISTRIBUTE BY rand()",
        "SELECT x FROM t DISTRIBUTE BY array(x)",
        "SELECT x FROM t CLUSTER BY y",
        "SELECT x FROM t DISTRIBUTE BY x SORT BY y",
        "SELECT x FROM t DISTRIBUTE BY map(x, y)",
        "TABLE t |> DISTRIBUTE BY y |> SELECT x",
        "TABLE t |> CLUSTER BY x |> SELECT x",
    ] {
        assert_eq!(sorted(&s, sql), ["0", "1"], "{sql}");
    }
    assert_eq!(rows(&s, "SELECT x FROM t DISTRIBUTE BY x LIMIT 1").len(), 1);
}

/// `SORT BY` after `DISTRIBUTE BY` sorts (here, one partition).
#[test]
fn sort_by_after_distribute_by_sorts() {
    let s = session();
    assert_eq!(
        rows(&s, "SELECT y, x FROM t DISTRIBUTE BY x SORT BY y DESC"),
        ["def|1", "abc|0"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT y, count(*) AS c FROM t GROUP BY y DISTRIBUTE BY c SORT BY c"
        ),
        ["abc|1", "def|1"]
    );
}

/// A name the input lacks is unresolved, with the input's columns proposed;
/// nothing resolves through an aggregate.
#[test]
fn an_unknown_name_is_unresolved() {
    let s = session();
    for (sql, name, proposal) in [
        ("SELECT x AS z FROM t DISTRIBUTE BY zz", "`zz`", "`z`"),
        (
            "SELECT count(*) AS c FROM t GROUP BY y DISTRIBUTE BY x",
            "`x`",
            "`c`",
        ),
        (
            "TABLE t |> SELECT 1 + 2 AS result |> DISTRIBUTE BY x",
            "`x`",
            "`result`",
        ),
    ] {
        assert_eq!(
            params(&s, sql),
            (
                "UNRESOLVED_COLUMN.WITH_SUGGESTION".to_owned(),
                vec![format!("objectName={name}"), format!("proposal={proposal}")]
            ),
            "{sql}"
        );
    }
}

/// Aggregates, windows and generators are refused; a sort above the
/// repartition does not read the aggregate below it, and a sort over no
/// aggregate refuses an aggregate call.
#[test]
fn aggregates_windows_and_generators_are_refused() {
    let s = session();
    // The parameters of a Spark 4.2.0 run; the generator's plan text has
    // no attribute ids here (Spark: `explode(array(x#11))`).
    for (sql, class, param) in [
        (
            "SELECT x FROM t DISTRIBUTE BY sum(x)",
            "UNSUPPORTED_EXPR_FOR_OPERATOR",
            "invalidExprSqls=\"sum(x)\"",
        ),
        (
            "SELECT x FROM t DISTRIBUTE BY x + sum(x) + max(x)",
            "UNSUPPORTED_EXPR_FOR_OPERATOR",
            "invalidExprSqls=\"sum(x)\", \"max(x)\"",
        ),
        (
            "SELECT x FROM t DISTRIBUTE BY row_number() OVER (ORDER BY x)",
            "UNSUPPORTED_EXPR_FOR_OPERATOR",
            "invalidExprSqls=\"row_number() OVER (ORDER BY x ASC NULLS FIRST ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)\"",
        ),
        (
            "SELECT y, count(*) AS c FROM t GROUP BY y DISTRIBUTE BY y SORT BY count(*)",
            "UNSUPPORTED_EXPR_FOR_OPERATOR",
            "invalidExprSqls=\"count(1)\"",
        ),
        (
            "SELECT x FROM t ORDER BY sum(x)",
            "UNSUPPORTED_EXPR_FOR_OPERATOR",
            "invalidExprSqls=\"sum(x)\"",
        ),
        (
            "SELECT x FROM t SORT BY sum(x)",
            "UNSUPPORTED_EXPR_FOR_OPERATOR",
            "invalidExprSqls=\"sum(x)\"",
        ),
        (
            "SELECT x FROM t DISTRIBUTE BY explode(array(x))",
            "UNSUPPORTED_GENERATOR.OUTSIDE_SELECT",
            "plan=RepartitionByExpression [explode(array(x))]",
        ),
    ] {
        assert_eq!(
            params(&s, sql),
            (class.to_owned(), vec![param.to_owned()]),
            "{sql}"
        );
    }
}

/// `CLUSTER BY` sorts, so a map is refused as a sort key.
#[test]
fn cluster_by_a_map_does_not_order() {
    let s = session();
    let (class, _) = error(&s, "SELECT x FROM t CLUSTER BY map(x, y)");
    assert_eq!(class, "DATATYPE_MISMATCH.INVALID_ORDERING_TYPE");
}

/// A correlated reference in a subquery's `DISTRIBUTE BY` is refused.
#[test]
fn an_outer_reference_is_refused() {
    let s = session();
    assert_eq!(
        params(
            &s,
            "SELECT x FROM t WHERE x > (SELECT max(x) FROM t t2 DISTRIBUTE BY t.y)",
        ),
        (
            "UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY.CORRELATED_REFERENCE".to_owned(),
            vec!["sqlExprs=\"y\"".to_owned()]
        )
    );
}

/// `ORDER BY` with `DISTRIBUTE BY` is still refused at parse time.
#[test]
fn order_by_with_distribute_by_is_refused() {
    let s = session();
    let (class, _) = error(&s, "SELECT x, y FROM t ORDER BY x DISTRIBUTE BY y");
    assert_eq!(
        class,
        "UNSUPPORTED_FEATURE.COMBINATION_QUERY_RESULT_CLAUSES"
    );
}
