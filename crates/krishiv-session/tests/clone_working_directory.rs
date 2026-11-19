// SPDX-License-Identifier: Apache-2.0
//! `Session::clone_session` keeps the source's working directory: a copied
//! view over a relative path reads the source's files, not whatever the
//! process's directory holds when the clone is made. A file of its own,
//! because it changes the process's working directory.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_session::Session;

fn rows(s: &Session, sql: &str) -> usize {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.collect())
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum()
}

#[test]
fn a_clone_reads_relative_paths_from_the_source_working_directory() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    std::fs::write(first.path().join("rows.csv"), "a\n1\n").unwrap();
    std::fs::write(second.path().join("rows.csv"), "a\n1\n2\n3\n").unwrap();
    let before = std::env::current_dir().unwrap();

    std::env::set_current_dir(first.path()).unwrap();
    let s = Session::new().unwrap();
    block_on(
        s.sql("CREATE TEMPORARY VIEW v(a int) USING csv OPTIONS (path 'rows.csv', header 'true')"),
    )
    .unwrap();
    std::env::set_current_dir(second.path()).unwrap();
    let c = s.clone_session().unwrap();
    std::env::set_current_dir(before).unwrap();

    assert_eq!(rows(&s, "SELECT * FROM v"), 1);
    assert_eq!(rows(&c, "SELECT * FROM v"), 1);
}
