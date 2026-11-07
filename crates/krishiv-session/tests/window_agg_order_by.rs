// SPDX-License-Identifier: Apache-2.0
//! `ORDER BY` over an aggregate whose select list holds windows reads the select list only (Spark 4.2.0): an
//! aggregate call in it stays in the sort and is `UNSUPPORTED_EXPR_FOR_OPERATOR`, listed as Spark prints it; a column
//! the aggregate does not output is `UNRESOLVED_COLUMN`, which wins. The expected texts are the vendored
//! `window-with-aggregates` goldens and runs of a local Spark 4.2.0.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

/// `(class, invalidExprSqls)` of the error `sql` raises, or `("OK", "")`.
fn outcome(s: &Session, sql: &str) -> (String, String) {
    let r = block_on(async { s.sql(sql).await?.collect().await });
    match r {
        Ok(_) => ("OK".to_owned(), String::new()),
        Err(e) => {
            let sqls = e
                .message_parameters()
                .iter()
                .find(|(k, _)| *k == "invalidExprSqls")
                .map(|(_, v)| v.clone())
                .unwrap_or_default();
            (e.error_class().name().to_owned(), sqls)
        }
    }
}

#[test]
fn every_unsupported_golden_statement_lists_the_calls_as_spark_does() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/spark-golden/v4.2.0/results/window-with-aggregates.sql.out");
    let text = std::fs::read_to_string(path).unwrap();
    let s = Session::new().unwrap();
    let mut checked = 0;
    for block in text.split("-- !query\n").skip(1) {
        let Some((sql, rest)) = block.split_once("-- !query schema\n") else {
            continue;
        };
        if !rest.contains("\"errorClass\" : \"UNSUPPORTED_EXPR_FOR_OPERATOR\"") {
            continue;
        }
        let want = rest
            .split("\"invalidExprSqls\" : \"")
            .nth(1)
            .and_then(|r| r.split("\"\n").next())
            .unwrap()
            .replace("\\\"", "\"");
        let (class, got) = outcome(&s, sql);
        assert_eq!(class, "UNSUPPORTED_EXPR_FOR_OPERATOR", "{sql}");
        assert_eq!(got, want, "{sql}");
        checked += 1;
    }
    assert_eq!(checked, 9, "the golden holds nine such statements");
}

#[test]
fn calls_with_several_arguments_and_repeats_are_listed_as_written() {
    let s = Session::new().unwrap();
    let from = "FROM VALUES (1, 10, 5), (2, 20, 6), (1, 30, 7) AS t(a, b, c) GROUP BY a, b, c";
    // Spark 4.2.0 (local run): `invalidExprSqls` for each ORDER BY.
    for (order, want) in [
        ("max_by(b, c)", "\"max_by(b, c)\""),
        ("count(DISTINCT b)", "\"count(DISTINCT b)\""),
        (
            "SUM(b) + AVG(c), MAX(c)",
            "\"sum(b)\", \"avg(c)\", \"max(c)\"",
        ),
        ("SUM(b), SUM(b) + 1", "\"sum(b)\", \"sum(b)\""),
        ("abs(SUM(b))", "\"sum(b)\""),
        ("COUNT(*), SUM(c)", "\"count(1)\", \"sum(c)\""),
    ] {
        let sql = format!("SELECT a, SUM(SUM(b)) OVER () AS w {from} ORDER BY {order}");
        assert_eq!(
            outcome(&s, &sql),
            ("UNSUPPORTED_EXPR_FOR_OPERATOR".to_owned(), want.to_owned()),
            "ORDER BY {order}"
        );
    }
}

#[test]
fn an_unresolved_column_wins_and_select_list_keys_still_sort() {
    let s = Session::new().unwrap();
    let rows = "FROM VALUES (1, 10, 5), (2, 20, 6), (1, 30, 7) AS t(a, b, c) GROUP BY a";
    // Spark 4.2.0 (local run): `b` is not an output of the aggregate, so the column error comes first.
    for order in ["SUM(b)", "max_by(b, c)", "count(DISTINCT b)"] {
        let sql = format!("SELECT a, SUM(SUM(b)) OVER () AS w {rows} ORDER BY {order}");
        assert_eq!(
            outcome(&s, &sql).0,
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            "ORDER BY {order}"
        );
    }
    // An alias of an aggregate in the select list, a window output and a grouping column sort fine.
    for order in ["sb", "w, a", "a"] {
        let sql =
            format!("SELECT a, SUM(b) AS sb, SUM(SUM(b)) OVER () AS w {rows} ORDER BY {order}");
        assert_eq!(outcome(&s, &sql).0, "OK", "ORDER BY {order}");
    }
}
