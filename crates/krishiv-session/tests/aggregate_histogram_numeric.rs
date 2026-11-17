// SPDX-License-Identifier: Apache-2.0
//! `histogram_numeric(col, nb)` through SQL. Expected values come from a
//! Spark 4.3.0 run (`spark-sql`, local mode) of the same statements: the
//! types of `x`, the argument errors, `NULL` for no value, `FILTER`,
//! `DISTINCT` and `GROUP BY`. Results that depend on how Spark splits a
//! local relation (several partitions merged) are not asserted here; the
//! histogram's unit tests cover the merge against the `group-by.sql` golden.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// The rows of `sql`, cells joined by tabs, with every cell cast to a string.
fn rows(s: &Session, sql: &str) -> R<Vec<String>> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })?;
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(r) {
                        "NULL".to_owned()
                    } else {
                        match c.as_string_opt::<i32>() {
                            Some(a) => a.value(r).to_owned(),
                            None => c.as_string_view().value(r).to_owned(),
                        }
                    }
                })
                .collect();
            out.push(cells.join("\t"));
        }
    }
    Ok(out)
}

fn json_of(s: &Session, from: &str, call: &str) -> R<String> {
    let sql = format!("SELECT to_json(h) FROM (SELECT {call} AS h FROM {from})");
    Ok(rows(s, &sql)?.join("\n"))
}

fn error_class(s: &Session, sql: &str) -> R<String> {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(_) => Err(format!("expected an error: {sql}").into()),
        Err(e) => Ok(e.error_class().name().to_owned()),
    }
}

#[test]
fn the_x_field_has_the_input_type() -> R {
    let s = Session::new()?;
    let cases = [
        (
            "VALUES (1), (2), (3) AS t(c)",
            "array<struct<x:int,y:double>>",
        ),
        (
            "VALUES (1L), (2L), (3L) AS t(c)",
            "array<struct<x:bigint,y:double>>",
        ),
        (
            "VALUES (1F), (2F), (3F) AS t(c)",
            "array<struct<x:float,y:double>>",
        ),
        (
            "VALUES (1D), (2D), (3D) AS t(c)",
            "array<struct<x:double,y:double>>",
        ),
        (
            "VALUES (1S), (2S), (3S) AS t(c)",
            "array<struct<x:smallint,y:double>>",
        ),
        (
            "VALUES (CAST(1 AS TINYINT)), (CAST(2 AS TINYINT)) AS t(c)",
            "array<struct<x:tinyint,y:double>>",
        ),
        (
            "VALUES (CAST(1 AS DECIMAL(4, 2))), (CAST(2 AS DECIMAL(4, 2))) AS t(c)",
            "array<struct<x:decimal(4,2),y:double>>",
        ),
        (
            "VALUES (DATE'2020-01-01'), (DATE'2020-01-05') AS t(c)",
            "array<struct<x:date,y:double>>",
        ),
        (
            "VALUES (TIMESTAMP'2017-03-01 00:00:00') AS t(c)",
            "array<struct<x:timestamp,y:double>>",
        ),
        (
            "VALUES ('1'), ('5.5') AS t(c)",
            "array<struct<x:double,y:double>>",
        ),
        (
            "VALUES (NULL), (NULL) AS t(c)",
            "array<struct<x:double,y:double>>",
        ),
    ];
    for (from, ty) in cases {
        let got = rows(
            &s,
            &format!("SELECT typeof(h) FROM (SELECT histogram_numeric(c, 3) AS h FROM {from})"),
        )?;
        assert_eq!(got, vec![ty.to_owned()], "{from}");
    }
    Ok(())
}

#[test]
fn bins_and_counts_follow_spark() -> R {
    let s = Session::new()?;
    let from = "VALUES (1), (2), (3) AS t(c)";
    assert_eq!(
        json_of(&s, from, "histogram_numeric(c, 3)")?,
        r#"[{"x":1,"y":1.0},{"x":2,"y":1.0},{"x":3,"y":1.0}]"#
    );
    // 1, 4, 5, 6 into three bins: the closest pair (5, 6) merges to 5.5,
    // shown as the int 5.
    assert_eq!(
        json_of(
            &s,
            "VALUES (1), (1), (1), (4), (5), (6) AS t(c)",
            "histogram_numeric(DISTINCT c, 3)"
        )?,
        r#"[{"x":1,"y":1.0},{"x":4,"y":1.0},{"x":5,"y":2.0}]"#
    );
    assert_eq!(
        json_of(
            &s,
            "VALUES (1), (2), (3), (4), (5), (6) AS t(c)",
            "histogram_numeric(c, 3) FILTER (WHERE c > 2)"
        )?,
        r#"[{"x":3,"y":1.0},{"x":4,"y":2.0},{"x":6,"y":1.0}]"#
    );
    assert_eq!(
        json_of(
            &s,
            "VALUES (CAST(1.25 AS DECIMAL(5,1))), (CAST(2.5 AS DECIMAL(5,1))), \
             (CAST(2.75 AS DECIMAL(5,1))), (CAST(9.9 AS DECIMAL(5,1))) AS t(c)",
            "histogram_numeric(c, 3)"
        )?,
        r#"[{"x":1.3,"y":1.0},{"x":2.7,"y":2.0},{"x":9.9,"y":1.0}]"#
    );
    Ok(())
}

#[test]
fn a_group_without_a_value_is_null() -> R {
    let s = Session::new()?;
    let got = rows(
        &s,
        "SELECT k, to_json(h) FROM (SELECT k, histogram_numeric(c, 2) AS h FROM VALUES \
         ('a', 1), ('a', 2), ('a', 10), ('b', 5), ('b', NULL), ('c', NULL) AS t(k, c) GROUP BY k) \
         ORDER BY k",
    )?;
    assert_eq!(
        got,
        vec![
            "a\t[{\"x\":1,\"y\":2.0},{\"x\":10,\"y\":1.0}]",
            "b\t[{\"x\":5,\"y\":1.0}]",
            "c\tNULL",
        ]
    );
    assert_eq!(
        json_of(
            &s,
            "VALUES (1) AS t(c) WHERE c > 5",
            "histogram_numeric(c, 3)"
        )?,
        "NULL"
    );
    Ok(())
}

#[test]
fn nb_is_a_foldable_int_of_at_least_two() -> R {
    let s = Session::new()?;
    let from = "FROM VALUES (1), (5), (9), (10) AS t(c)";
    for nb in ["0", "1", "-3"] {
        assert_eq!(
            error_class(&s, &format!("SELECT histogram_numeric(c, {nb}) {from}"))?,
            "DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE",
            "{nb}"
        );
    }
    assert_eq!(
        error_class(&s, &format!("SELECT histogram_numeric(c, NULL) {from}"))?,
        "DATATYPE_MISMATCH.UNEXPECTED_NULL"
    );
    assert_eq!(
        error_class(&s, &format!("SELECT histogram_numeric(c, c) {from}"))?,
        "DATATYPE_MISMATCH.NON_FOLDABLE_INPUT"
    );
    assert_eq!(
        error_class(
            &s,
            "SELECT histogram_numeric(c, 3) FROM VALUES (true) AS t(c)"
        )?,
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    assert_eq!(
        error_class(
            &s,
            "SELECT histogram_numeric(c, 3) FROM VALUES (array(1)) AS t(c)"
        )?,
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    // A foldable expression, a string, a bigint and a fractional `nb` cast.
    // Spark: `1+2` and `'3'` give 3 bins, `2.5` gives 2.
    for (nb, want) in [
        (
            "1+2",
            r#"[{"x":1,"y":1.0},{"x":5,"y":1.0},{"x":9,"y":2.0}]"#,
        ),
        (
            "'3'",
            r#"[{"x":1,"y":1.0},{"x":5,"y":1.0},{"x":9,"y":2.0}]"#,
        ),
        ("2.5", r#"[{"x":1,"y":1.0},{"x":8,"y":3.0}]"#),
    ] {
        assert_eq!(
            json_of(
                &s,
                "VALUES (1), (5), (9), (10) AS t(c)",
                &format!("histogram_numeric(c, {nb})")
            )?,
            want,
            "{nb}"
        );
    }
    Ok(())
}
