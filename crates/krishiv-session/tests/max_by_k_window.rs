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
