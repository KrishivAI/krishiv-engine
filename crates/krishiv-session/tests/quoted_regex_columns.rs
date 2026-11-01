// SPDX-License-Identifier: Apache-2.0
//! `spark.sql.parser.quotedRegexColumnNames`: a backquoted name in a select item is a Java regular
//! expression over the column names, matched in full and ignoring case unless the session is case
//! sensitive; elsewhere (`WHERE`, `GROUP BY`) it stays a plain name.
#![cfg(test)]

use arrow_array::ArrayRef;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.parser.quotedRegexColumnNames=true")).unwrap();
    block_on(s.sql(
        "CREATE OR REPLACE TEMPORARY VIEW t2 AS SELECT * FROM VALUES \
         (1, 1, 1, 2), (2, 1, 2, 3), (2, 2, 2, 3) AS t2(A, B, c, d)",
    ))
    .unwrap();
    s
}

fn cell(column: &ArrayRef, row: usize) -> String {
    match column.data_type() {
        DataType::Int32 => column.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => column.as_primitive::<Int64Type>().value(row).to_string(),
        other => panic!("unexpected type {other}"),
    }
}

/// The column names and the rows (cells as text) of `sql`.
fn run(s: &Session, sql: &str) -> (Vec<String>, Vec<String>) {
    let df = block_on(s.sql(sql)).unwrap();
    let names = block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let mut rows = Vec::new();
    for b in block_on(df.collect()).unwrap() {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| cell(c, r)).collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    (names, rows)
}

#[test]
fn a_regex_selects_the_columns_it_matches_in_full() {
    let s = session();
    assert_eq!(
        run(&s, "SELECT `(a)?+.+` FROM t2 WHERE a = 1").0,
        ["B", "c", "d"]
    );
    // The case of the pattern does not matter while the session ignores case.
    assert_eq!(
        run(&s, "SELECT `(A|b)` FROM t2 WHERE a = 2").1,
        ["2|1", "2|2"]
    );
    assert_eq!(
        run(&s, "SELECT t.`(a|B)?+.+` FROM t2 t WHERE a = 2").0,
        ["c", "d"]
    );
    // A regex that matches nothing selects nothing.
    assert!(run(&s, "SELECT `(e|f)` FROM t2").0.is_empty());
}

#[test]
fn a_regex_expands_inside_a_call_and_only_in_a_select_item() {
    let s = session();
    assert_eq!(run(&s, "SELECT SUM(`(a|b|c)?+.+`) FROM t2").1, ["8"]);
    // A plain name in `GROUP BY` is not a regex.
    assert!(block_on(s.sql("SELECT SUM(a) FROM t2 GROUP BY `(a)`")).is_err());
    assert_eq!(run(&s, "SELECT SUM(a) FROM t2 GROUP BY `a`").1, ["1", "4"]);
}

#[test]
fn a_case_sensitive_session_matches_case_exactly() {
    let s = session();
    block_on(s.sql("SET spark.sql.caseSensitive=true")).unwrap();
    assert_eq!(run(&s, "SELECT `(a)?+.+` FROM t2").0, ["A", "B", "c", "d"]);
    assert_eq!(run(&s, "SELECT `(A)?+.+` FROM t2").0, ["B", "c", "d"]);
}

#[test]
fn without_the_setting_a_backquoted_name_is_a_column() {
    let s = Session::new().unwrap();
    assert!(block_on(s.sql("SELECT `(a)?+.+` FROM VALUES (1) AS t(a)")).is_err());
}
