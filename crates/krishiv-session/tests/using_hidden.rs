// SPDX-License-Identifier: Apache-2.0
//! The columns a `USING` join merges away (`lhs.a`, `rhs.a`): Spark's hidden output, reachable by a
//! qualified name through projections, filters, sorts, windows and pipe boundaries, but not past an
//! alias, an aggregate or a set operation. Rows and errors are from a Spark 4.2.0 run.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

const J: &str = "VALUES (0), (1) lhs(a) |> INNER JOIN VALUES (1), (2) rhs(a) USING (a) ";

fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(i) {
                            "NULL".to_owned()
                        } else if let Some(v) =
                            c.as_primitive_opt::<arrow_array::types::Int64Type>()
                        {
                            v.value(i).to_string()
                        } else {
                            c.as_primitive::<arrow_array::types::Int32Type>()
                                .value(i)
                                .to_string()
                        }
                    })
                    .collect(),
            );
        }
    }
    out
}

fn class(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => panic!("{sql}: no error"),
        Err(e) => e.error_class().name().to_owned(),
    }
}

fn row(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_owned()).collect()
}

/// `pipe-operators` 85: both sides' keys after `extend`, `drop`, `where`, `order by`, `set` and `limit`.
#[test]
fn qualified_keys_survive_a_chain_of_pipe_operators() {
    let s = Session::new().unwrap();
    let sql = format!(
        "{J}|> EXTEND lhs.a + rhs.a AS z1 |> EXTEND lhs.a - rhs.a AS z2 |> DROP z1 |> WHERE z2 = 0 \
         |> ORDER BY lhs.a, rhs.a, z2 |> SET z2 = 4 |> LIMIT 2 |> SELECT lhs.a, rhs.a, z2"
    );
    assert_eq!(rows(&s, &sql), [row(&["1", "1", "4"])]);
}

/// Every projection keeps the merged keys, whatever it outputs, and so does a window.
#[test]
fn projections_and_windows_keep_the_merged_keys() {
    let s = Session::new().unwrap();
    for (tail, want) in [
        ("|> EXTEND 1 AS z |> SELECT rhs.a, z", row(&["1", "1"])),
        ("|> SELECT a |> SELECT rhs.a", row(&["1"])),
        ("|> SELECT a + 1 AS b |> SELECT rhs.a", row(&["1"])),
        ("|> SELECT * |> SELECT rhs.a", row(&["1"])),
        ("|> SELECT rhs.a |> SELECT rhs.a", row(&["1"])),
        (
            "|> EXTEND 1 AS z |> SET z = 2 |> SELECT rhs.a, z",
            row(&["1", "2"]),
        ),
        ("|> EXTEND 1 AS z |> SELECT rhs.*", row(&["1"])),
        ("|> EXTEND 1 AS z |> AGGREGATE sum(rhs.a) AS s", row(&["1"])),
        (
            "|> EXTEND 1 AS z |> EXTEND row_number() OVER (ORDER BY rhs.a) AS r |> SELECT rhs.a, r",
            row(&["1", "1"]),
        ),
    ] {
        let sql = format!("{J}{tail}");
        assert_eq!(rows(&s, &sql), [want], "{sql}");
    }
    // A left join's right key is null where nothing matched.
    let sql = "VALUES (0), (1) lhs(a) |> LEFT JOIN VALUES (1), (2) rhs(a) USING (a) |> EXTEND 1 AS z \
               |> SELECT lhs.a, rhs.a |> ORDER BY 1";
    assert_eq!(rows(&s, sql), [row(&["0", "NULL"]), row(&["1", "1"])]);
}

/// A key read after a window the window itself did not read: the window widens its input.
#[test]
fn a_key_read_after_a_window_widens_through_it() {
    let s = Session::new().unwrap();
    let j = "VALUES (0), (1), (2) lhs(a) |> INNER JOIN VALUES (1), (2), (3) rhs(a) USING (a) ";
    assert_eq!(
        rows(
            &s,
            &format!(
                "{j}|> EXTEND row_number() OVER (ORDER BY a) AS r |> SELECT rhs.a, r |> ORDER BY r"
            )
        ),
        [row(&["1", "1"]), row(&["2", "2"])]
    );
    assert_eq!(
        rows(
            &s,
            &format!("{j}|> EXTEND sum(a) OVER () AS t |> SELECT lhs.a, rhs.a, t |> ORDER BY 1")
        ),
        [row(&["1", "1", "3"]), row(&["2", "2", "3"])]
    );
}

/// An alias, an aggregate and a union end the hidden output, as does a subquery in `FROM`.
#[test]
fn an_alias_aggregate_or_union_ends_the_merged_keys() {
    let s = Session::new().unwrap();
    for sql in [
        format!("{J}|> AS x |> SELECT rhs.a"),
        format!("{J}|> AGGREGATE count(*) GROUP BY a |> SELECT rhs.a"),
        format!("{J}|> UNION ALL (SELECT 1 AS a) |> SELECT rhs.a"),
        "SELECT rhs.a FROM (SELECT * FROM VALUES (0), (1) lhs(a) INNER JOIN VALUES (1), (2) rhs(a) USING (a))"
            .to_owned(),
    ] {
        assert_eq!(class(&s, &sql), "UNRESOLVED_COLUMN.WITH_SUGGESTION", "{sql}");
    }
}

/// Only hidden columns cross a pipe boundary: a column a projection dropped stays dropped.
#[test]
fn a_dropped_column_does_not_come_back_with_the_keys() {
    let s = Session::new().unwrap();
    for sql in [
        format!("{J}|> EXTEND 1 AS z |> SELECT a |> ORDER BY z"),
        format!("{J}|> EXTEND 1 AS z |> SELECT a |> WHERE z = 1"),
    ] {
        assert_eq!(
            class(&s, &sql),
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            "{sql}"
        );
    }
    assert_eq!(
        rows(&s, &format!("{J}|> SELECT a |> ORDER BY rhs.a")),
        [row(&["1"])]
    );
}

/// `pipe-operators` 167: a nested `USING` join on the right, whose side is a second instance of the left
/// side's plan (identical views), reads the inner join's merged key.
#[test]
fn a_nested_using_join_over_a_repeated_plan_reads_its_merged_key() {
    let s = Session::new().unwrap();
    for v in ["t1", "t2", "t3"] {
        let ddl = format!("CREATE TEMPORARY VIEW {v} AS SELECT * FROM VALUES (1) AS grouping(a)");
        block_on(async { s.sql(&ddl).await?.collect().await }).unwrap();
    }
    for sql in [
        "TABLE t1 |> INNER JOIN (t2 jt2 INNER JOIN t3 jt3 USING (a)) USING (a) |> SELECT a, t1.a, jt2.a, jt3.a",
        "SELECT a, t1.a, jt2.a, jt3.a FROM t1 INNER JOIN (t2 jt2 INNER JOIN t3 jt3 USING (a)) USING (a)",
        "SELECT a, x.a, jt2.a, jt3.a FROM (SELECT 1 a) x \
         INNER JOIN ((SELECT 1 a) jt2 INNER JOIN (SELECT 1 a) jt3 USING (a)) USING (a)",
    ] {
        assert_eq!(rows(&s, sql), [row(&["1", "1", "1", "1"])], "{sql}");
    }
}
