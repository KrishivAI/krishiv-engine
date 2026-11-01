// SPDX-License-Identifier: Apache-2.0
//! The columns a `USING` (or `NATURAL`) join merges away stay reachable by
//! a qualified name, in the select list, `WHERE`, `GROUP BY`, aggregate
//! arguments and `ORDER BY`, and through `t.*`, as Spark's hidden output.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TEMPORARY VIEW nt1 AS SELECT * FROM VALUES ('one', 1), ('two', 2), ('three', 3) AS nt1(k, v1)",
        "CREATE TEMPORARY VIEW nt2 AS SELECT * FROM VALUES ('one', 1), ('two', 22), ('one', 5), ('four', 4) AS nt2(k, v2)",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    s
}

/// Every cell as text (casts to string in the query), rows sorted.
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    let mut out = Vec::new();
    for b in block_on(df.collect()).unwrap() {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(r) {
                        "NULL".to_owned()
                    } else {
                        c.as_string_view().value(r).to_owned()
                    }
                })
                .collect();
            out.push(cells.join("|"));
        }
    }
    out.sort();
    out
}

#[test]
fn qualified_names_reach_the_merged_columns() {
    let s = session();
    let j = "FROM nt1 LEFT JOIN nt2 USING (k)";
    assert_eq!(
        rows(&s, &format!("SELECT string(nt1.k), string(nt2.k) {j}")),
        ["one|one", "one|one", "three|NULL", "two|two"]
    );
    assert_eq!(
        rows(&s, &format!("SELECT string(v2) {j} WHERE nt2.k = 'two'")),
        ["22"]
    );
    assert_eq!(
        rows(
            &s,
            &format!("SELECT string(nt1.k), string(count(*)) {j} GROUP BY nt1.k")
        ),
        ["one|2", "three|1", "two|1"]
    );
    assert_eq!(rows(&s, &format!("SELECT string(count(nt2.k)) {j}")), ["3"]);
    assert_eq!(
        rows(
            &s,
            &format!("SELECT string(nt1.k) {j} GROUP BY nt1.k ORDER BY max(nt2.v2) DESC")
        ),
        ["one", "three", "two"]
    );
}

#[test]
fn qualified_stars_include_the_merged_columns() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT string(nt1.k), string(v1), string(nt2.k), string(v2) FROM nt1 LEFT JOIN nt2 USING (k) WHERE v1 = 3"
        ),
        ["three|3|NULL|NULL"]
    );
    let df = block_on(s.sql("SELECT nt1.*, nt2.* FROM nt1 LEFT JOIN nt2 USING (k)")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    assert_eq!(names, ["k", "v1", "k", "v2"]);
}

#[test]
fn natural_joins_hide_the_same_way() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT string(nt2.k), string(sum(v2)) FROM nt1 NATURAL JOIN nt2 GROUP BY nt2.k ORDER BY nt2.k"
        ),
        ["one|6", "two|22"]
    );
}

#[test]
fn order_by_matches_a_commuted_grouping_expression() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT string(sum(v1)) FROM nt1 GROUP BY length(k) + 1 ORDER BY 1 + length(k)"
        ),
        ["3", "3"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT string(sum(v2)) FROM nt1 JOIN nt2 USING (k) GROUP BY v1 + 1 ORDER BY 1 + v1, max(nt2.k)"
        ),
        ["22", "6"]
    );
}

#[test]
fn where_keeps_the_merged_columns_for_the_select_list() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT string(nt2.k) FROM nt1 FULL OUTER JOIN nt2 USING (k) WHERE nt1.k NOT LIKE 'x%'"
        ),
        ["NULL", "one", "one", "two"]
    );
}
