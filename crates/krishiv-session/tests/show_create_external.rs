// SPDX-License-Identifier: Apache-2.0
//! `SHOW CREATE TABLE` of external tables and views with properties: the location is a line of its
//! own (a `path` option is the location), creating the table makes no directory, and a view lists
//! `TBLPROPERTIES` before `WITH SCHEMA`.
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn show_create(s: &Session, name: &str) -> String {
    let df = block_on(s.sql(&format!("SHOW CREATE TABLE {name}"))).unwrap();
    let batches = block_on(df.collect()).unwrap();
    batches[0].column(0).as_string_view().value(0).to_owned()
}

#[test]
fn a_path_option_and_a_location_are_the_same_external_table() {
    let dir = tempfile::tempdir().unwrap();
    let place = dir.path().join("never/made");
    let s = Session::new().unwrap();
    block_on(s.sql(&format!(
        "CREATE TABLE t1 (a INT) USING parquet OPTIONS ('path' '{}')",
        place.display()
    )))
    .unwrap();
    block_on(s.sql(&format!(
        "CREATE TABLE t2 (a INT) USING parquet LOCATION '{}'",
        place.display()
    )))
    .unwrap();
    assert!(
        !place.exists(),
        "an external table's directory is made by a write"
    );
    let want = format!("LOCATION 'file:{}'\n", place.display());
    for t in ["t1", "t2"] {
        let text = show_create(&s, t);
        assert!(text.ends_with(&want), "{text}");
        assert!(!text.contains("OPTIONS"), "{text}");
    }
}

#[test]
fn a_view_lists_its_properties_before_the_schema_mode() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE tbl (a INT, b INT) USING parquet")).unwrap();
    block_on(s.sql(
        "CREATE VIEW v (aaa, bbb) TBLPROPERTIES ('a' = '1', 'b' = '2') WITH SCHEMA COMPENSATION AS SELECT a, b FROM tbl",
    ))
    .unwrap();
    let text = show_create(&s, "v");
    let props = text.find("TBLPROPERTIES (").unwrap();
    assert!(
        props < text.find("WITH SCHEMA COMPENSATION").unwrap(),
        "{text}"
    );
    assert!(text.contains("'a' = '1',\n  'b' = '2')"), "{text}");
}
