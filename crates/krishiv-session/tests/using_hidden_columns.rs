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

#[test]
fn a_qualified_star_lists_the_merged_away_key_first() {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TEMPORARY VIEW sa AS SELECT * FROM VALUES ('x', 'one') AS sa(a, k)",
        "CREATE TEMPORARY VIEW sb AS SELECT * FROM VALUES ('one', 'y') AS sb(k, b)",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
    // Spark: the hidden key, then the output columns.
    for kind in ["RIGHT", "FULL"] {
        let df = block_on(s.sql(&format!("SELECT sa.* FROM sa {kind} JOIN sb USING (k)"))).unwrap();
        let schema = block_on(df.schema()).unwrap();
        let names: Vec<_> = schema.fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(names, ["k", "a"], "{kind}");
    }
}

/// A join above a `NATURAL`/`USING` join reads the merged-away key in its condition and passes it
/// on, so qualified names and `t.*` still reach it.
#[test]
fn a_later_join_condition_reads_the_merged_away_key() {
    let s = session();
    block_on(s.sql(
        "CREATE TEMPORARY VIEW nt3 AS SELECT * FROM VALUES ('one', 4), ('two', 5), ('five', 7) AS nt3(k, v3)",
    ))
    .unwrap();
    assert_eq!(
        rows(
            &s,
            "SELECT nt2.k, nt3.k FROM nt1 NATURAL JOIN nt2 JOIN nt3 ON nt2.k = nt3.k"
        ),
        ["one|one", "one|one", "two|two"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT nt2.k, nt3.k FROM nt1 NATURAL JOIN nt2 LEFT JOIN nt3 ON nt2.k = nt3.k"
        ),
        ["one|one", "one|one", "two|two"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT nt1.k, CAST(nt3.v3 AS STRING), nt2.k FROM nt1 NATURAL JOIN nt2 JOIN nt3 ON nt2.k = nt3.k WHERE nt2.v2 = 22"
        ),
        ["two|5|two"]
    );
}

/// A field read through a dotted column path is named by the last field (Spark wraps the resolved
/// path in an alias); inside another expression's name the whole path is spelled.
#[test]
fn a_dotted_field_path_is_named_by_its_last_field() {
    let s = Session::new().unwrap();
    let names = |sql: &str| -> Vec<String> {
        let df = block_on(s.sql(sql)).unwrap();
        block_on(df.schema())
            .unwrap()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    };
    let from = "FROM VALUES (NAMED_STRUCT('name', 'x', 'id', 1)) AS t(col1)";
    assert_eq!(
        names(&format!("SELECT col1.name, t.col1.id {from}")),
        ["name", "id"]
    );
    assert_eq!(
        names(&format!("SELECT upper(col1.name) {from}")),
        ["upper(col1.name)"]
    );
}
