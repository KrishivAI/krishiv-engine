// SPDX-License-Identifier: Apache-2.0
//! Recursive CTEs end to end (`WITH RECURSIVE`). Rows and errors are
//! Spark 4.2.0's.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let mut row = Vec::new();
            for c in b.columns() {
                row.push(if c.is_null(i) {
                    "NULL".to_owned()
                } else {
                    match c.data_type() {
                        arrow_schema::DataType::Int32 => c
                            .as_primitive::<arrow_array::types::Int32Type>()
                            .value(i)
                            .to_string(),
                        arrow_schema::DataType::Int64 => c
                            .as_primitive::<arrow_array::types::Int64Type>()
                            .value(i)
                            .to_string(),
                        other => format!("<{other}>"),
                    }
                });
            }
            out.push(row);
        }
    }
    out
}

fn class_of(s: &Session, sql: &str) -> ErrorClass {
    let result = block_on(async { s.sql(sql).await?.collect().await });
    match result {
        Ok(_) => panic!("{sql}: expected an error"),
        Err(e) => e.error_class(),
    }
}

fn column(rows: Vec<Vec<String>>) -> Vec<String> {
    rows.into_iter().map(|mut r| r.remove(0)).collect()
}

#[test]
fn counts_to_ten() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 10) \
         SELECT n FROM r",
    ));
    assert_eq!(got, (1..=10).map(|n| n.to_string()).collect::<Vec<_>>());
}

#[test]
fn a_reference_in_the_outer_query_twice() {
    let s = Session::new().unwrap();
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) \
         SELECT a.n, b.n FROM r a JOIN r b ON a.n = b.n ORDER BY 1",
    );
    assert_eq!(got.len(), 3);
}

#[test]
fn union_without_all_is_refused() {
    let s = Session::new().unwrap();
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION SELECT n + 1 FROM r WHERE n < 3) SELECT * FROM r"
        ),
        ErrorClass::UnionNotSupportedInRecursiveCte
    );
}

#[test]
fn a_reference_in_the_anchor_is_invalid() {
    let s = Session::new().unwrap();
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (SELECT n FROM r UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT * FROM r"
        ),
        ErrorClass::InvalidRecursiveCte
    );
}

#[test]
fn the_level_limit_applies_even_if_the_recursion_would_end() {
    let s = Session::new().unwrap();
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 150) SELECT * FROM r"
        ),
        ErrorClass::RecursionLevelLimitExceeded
    );
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) MAX RECURSION LEVEL 200 AS \
         (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 150) SELECT count(*) FROM r",
    );
    assert_eq!(got, vec![vec!["150".to_owned()]]);
}

#[test]
fn an_outer_limit_ends_an_unbounded_recursion() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 0 UNION ALL SELECT n + 1 FROM r) SELECT * FROM r LIMIT 4",
    ));
    assert_eq!(got, ["0", "1", "2", "3"]);
}

#[test]
fn a_limit_is_not_pushed_when_another_recursive_cte_reads_the_cte() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE a(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM a WHERE n < 5), \
         b(m) AS (SELECT 1 UNION ALL SELECT m + 1 FROM b WHERE m < (SELECT max(n) FROM a)) \
         SELECT * FROM a LIMIT 2",
    ));
    assert_eq!(got.len(), 2);
}

#[test]
fn a_reference_in_a_forbidden_place_is_refused() {
    let s = Session::new().unwrap();
    for body in [
        "SELECT max(n) FROM r",
        "SELECT (SELECT max(n) FROM r)",
        "(SELECT n + 1 FROM r ORDER BY n)",
        "SELECT a.n FROM (SELECT 1 AS n) a LEFT JOIN r ON a.n = r.n",
    ] {
        assert_eq!(
            class_of(
                &s,
                &format!(
                    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL {body}) SELECT * FROM r LIMIT 3"
                )
            ),
            ErrorClass::InvalidRecursiveReferencePlace,
            "{body}"
        );
    }
}

#[test]
fn a_limit_is_not_pushed_when_the_cte_is_read_twice() {
    let s = Session::new().unwrap();
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 10) \
         SELECT n, (SELECT count(*) FROM r) FROM r LIMIT 3",
    );
    assert_eq!(got.len(), 3);
    assert!(got.iter().all(|r| r[1] == "10"), "{got:?}");
}

#[test]
fn a_recursive_term_reads_an_earlier_cte() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE one AS (SELECT 1 AS k), r(n) AS (SELECT 1 UNION ALL SELECT n + k FROM r, one WHERE n < 3) \
         SELECT n FROM r",
    ));
    assert_eq!(got, ["1", "2", "3"]);
}

#[test]
fn the_row_limit_applies() {
    let s = Session::new().unwrap();
    // 100,000 rows per round pass the million in eleven rounds.
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (SELECT id FROM range(100000) UNION ALL SELECT n FROM r) \
             SELECT count(*) FROM r"
        ),
        ErrorClass::RecursionRowLimitExceeded
    );
}
