// SPDX-License-Identifier: Apache-2.0
//! Read options on a table reference (`FROM t WITH ('k' = 'v')`) apply to that reference alone
//! (Spark 4.3 fixed two references to one table with different options sharing the first's).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn column(s: &Session, sql: &str) -> Vec<String> {
    let mut out = Vec::new();
    for b in block_on(async { s.sql(sql).await.unwrap().collect().await }).unwrap() {
        let c = b.column(0);
        if let Some(v) = c.as_string_opt::<i32>() {
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        } else {
            let v = c.as_string_view();
            out.extend((0..v.len()).map(|i| v.value(i).to_owned()));
        }
    }
    out
}

fn csv_table() -> (tempfile::TempDir, Session) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.csv"), "a,b\n1,2\n3,4\n").unwrap();
    let s = Session::new().unwrap();
    block_on(s.sql(&format!(
        "CREATE TABLE c (a STRING, b STRING) USING csv LOCATION '{}'",
        dir.path().display()
    )))
    .unwrap();
    (dir, s)
}

#[test]
fn a_reference_reads_with_its_own_options() {
    let (_dir, s) = csv_table();
    // Without the option the header line is a row; with it, it is not.
    assert_eq!(column(&s, "SELECT a FROM c ORDER BY a"), ["1", "3", "a"]);
    assert_eq!(
        column(&s, "SELECT a FROM c WITH ('header' = 'true') ORDER BY a"),
        ["1", "3"]
    );
}

#[test]
fn two_references_with_different_options_each_keep_theirs() {
    let (_dir, s) = csv_table();
    let rows = column(
        &s,
        "SELECT x.a || '-' || y.a FROM c WITH ('header' = 'true') x \
         JOIN c y ON x.a = y.a ORDER BY 1",
    );
    // `x` skips the header line, `y` does not: only the data rows join.
    assert_eq!(rows, ["1-1", "3-3"]);
    let all = column(
        &s,
        "SELECT count(*) || '' FROM c WITH ('header' = 'true') x CROSS JOIN c y",
    );
    assert_eq!(all, ["6"], "2 rows x 3 rows");
}

#[test]
fn an_empty_table_still_reads_with_reference_options() {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    block_on(s.sql(&format!(
        "CREATE TABLE e (a STRING, b STRING) USING csv LOCATION '{}'",
        dir.path().display()
    )))
    .unwrap();
    // The table's columns are declared: no file is needed to know them.
    assert!(column(&s, "SELECT a FROM e WITH ('header' = 'true')").is_empty());
}

#[test]
fn a_reference_option_that_contradicts_the_tables_is_refused() {
    // Spark's `DataSourceUtils.generateDatasourceOptions` (v3.5.0 to v4.3.0): the same key in any
    // case with another value is `_LEGACY_ERROR_TEMP_1151`; the same value keeps the table's.
    // Checked on a Spark 4.2.0 local run (2026-11-07): `header`/`HEADER` = 'true' over a table
    // with `header` = 'false' fails, `HEADER` = 'false' reads `[1, a]`, and the options of a
    // reference to a DataFrame temp view are ignored (`[1, a]` with or without them).
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.csv"), "a,b\n1,2\n").unwrap();
    let s = Session::new().unwrap();
    block_on(s.sql(&format!(
        "CREATE TABLE h (a STRING, b STRING) USING csv OPTIONS ('header' = 'false') LOCATION '{}'",
        dir.path().display()
    )))
    .unwrap();
    assert_eq!(column(&s, "SELECT a FROM h ORDER BY a"), ["1", "a"]);
    for key in ["header", "HEADER"] {
        let e = block_on(async {
            s.sql(&format!("SELECT a FROM h WITH ('{key}' = 'true')"))
                .await?
                .collect()
                .await
        })
        .unwrap_err();
        assert!(
            e.to_string().contains("duplicated key header "),
            "{key}: {e}"
        );
    }
    // The same value, in another case of the key, is fine and reads as the table does.
    assert_eq!(
        column(&s, "SELECT a FROM h WITH ('HEADER' = 'false') ORDER BY a"),
        ["1", "a"]
    );
}

#[test]
fn a_file_views_reference_options_are_ignored() {
    // Spark's `resolveTempView` drops the options of a reference to a view.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.csv"), "a,b\n1,2\n").unwrap();
    let s = Session::new().unwrap();
    block_on(
        s.read()
            .format("csv")
            .option("header", "false")
            .load([dir.path().to_str().unwrap()])
            .create_or_replace_temp_view("v"),
    )
    .unwrap();
    let plain = column(&s, "SELECT _c0 FROM v ORDER BY 1");
    assert_eq!(plain, ["1", "a"]);
    assert_eq!(
        column(&s, "SELECT _c0 FROM v WITH ('header' = 'true') ORDER BY 1"),
        plain
    );
}

#[test]
fn tables_over_the_same_files_keep_their_own_columns() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.csv"), "1,2\n").unwrap();
    let s = Session::new().unwrap();
    for (name, cols) in [("t1", "a STRING, b STRING"), ("t2", "x STRING, y STRING")] {
        block_on(s.sql(&format!(
            "CREATE TABLE {name} ({cols}) USING csv LOCATION '{}'",
            dir.path().display()
        )))
        .unwrap();
    }
    assert_eq!(column(&s, "SELECT a FROM t1 WITH ('sep' = ',')"), ["1"]);
    assert_eq!(column(&s, "SELECT y FROM t2 WITH ('sep' = ',')"), ["2"]);
}
