// SPDX-License-Identifier: Apache-2.0
//! `max_by(x, y, k) OVER (..)` and `min_by(x, y, k) OVER (..)`: the `k` best values of `x` in the
//! window frame (Spark 4.2.0 golden `max-min-by-k.sql`, statement "Window function test").
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn rows(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            out.push(
                (0..b.num_columns())
                    .map(|c| {
                        let col = b.column(c);
                        if let Some(v) = col.as_string_opt::<i32>() {
                            v.value(r).to_owned()
                        } else if let Some(v) = col.as_string_view_opt() {
                            v.value(r).to_owned()
                        } else {
                            format!("{:?}", col.slice(r, 1))
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
    }
    out
}

#[test]
fn the_k_best_values_over_a_partition_window() {
    let s = Session::new().unwrap();
    block_on(async {
        s.sql(
            "CREATE TEMPORARY VIEW dept_data AS SELECT * FROM VALUES \
             ('eng', 'ann', 100), ('eng', 'bob', 90), ('eng', 'cy', 80), \
             ('ops', 'dee', 70), ('ops', 'eve', NULL) AS t(dept, emp, salary)",
        )
        .await?
        .collect()
        .await
    })
    .unwrap();
    let got = rows(
        &s,
        "SELECT dept, emp, to_json(max_by(emp, salary, 2) OVER (PARTITION BY dept)), \
         to_json(min_by(emp, salary, 2) OVER (PARTITION BY dept)) FROM dept_data ORDER BY dept, emp",
    );
    assert_eq!(
        got,
        [
            r#"eng|ann|["ann","bob"]|["cy","bob"]"#,
            r#"eng|bob|["ann","bob"]|["cy","bob"]"#,
            r#"eng|cy|["ann","bob"]|["cy","bob"]"#,
            r#"ops|dee|["dee"]|["dee"]"#,
            r#"ops|eve|["dee"]|["dee"]"#,
        ]
    );
}

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(async {
        s.sql(
            "CREATE TEMPORARY VIEW d AS SELECT * FROM VALUES (1,'eng','ann',100), (2,'eng','bob',90), \
             (3,'eng','cy',80), (4,'ops','dee',70), (5,'ops','eve',NULL) AS t(id, dept, emp, salary)",
        )
        .await?
        .collect()
        .await
    })
    .unwrap();
    s
}

/// Spark 4.2.0 (local run, 2026-11-08): a frame with no row that has a `y` gives NULL, not `[]`.
#[test]
fn an_empty_frame_is_null_and_a_running_frame_grows() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(id AS STRING), coalesce(to_json(max_by(emp, salary, 2) OVER (PARTITION BY dept ORDER BY id \
             ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING)), 'NULL') FROM d ORDER BY id"
        ),
        [
            "1|NULL",
            r#"2|["ann"]"#,
            r#"3|["ann","bob"]"#,
            "4|NULL",
            r#"5|["dee"]"#
        ]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(id AS STRING), to_json(max_by(emp, salary, 2) OVER (PARTITION BY dept ORDER BY id)) FROM d \
             ORDER BY id"
        ),
        [
            r#"1|["ann"]"#,
            r#"2|["ann","bob"]"#,
            r#"3|["ann","bob"]"#,
            r#"4|["dee"]"#,
            r#"5|["dee"]"#
        ]
    );
}

/// A map-valued `x` sorts through the comparator, as in the plain form.
#[test]
fn a_map_valued_x_works() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(id AS STRING), to_json(max_by(map('k', emp), salary, 2) OVER (PARTITION BY dept)) FROM d \
             ORDER BY id"
        ),
        [
            r#"1|[{"k":"ann"},{"k":"bob"}]"#,
            r#"2|[{"k":"ann"},{"k":"bob"}]"#,
            r#"3|[{"k":"ann"},{"k":"bob"}]"#,
            r#"4|[{"k":"dee"}]"#,
            r#"5|[{"k":"dee"}]"#
        ]
    );
}

/// The column is named by the call as written over the resolved window, as in Spark.
#[test]
fn the_column_name_is_the_call_as_written() {
    let s = session();
    let names = |sql: &str| -> Vec<String> {
        let b = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
        b[0].schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    };
    assert_eq!(
        names("SELECT max_by(emp, salary, 2) OVER (PARTITION BY dept) FROM d"),
        [
            "max_by(emp, salary, 2) OVER (PARTITION BY dept ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)"
        ]
    );
    assert_eq!(
        names("SELECT id, max_by(emp, salary, 2) OVER (PARTITION BY dept ORDER BY id) FROM d")[1],
        "max_by(emp, salary, 2) OVER (PARTITION BY dept ORDER BY id ASC NULLS FIRST RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
    );
}
