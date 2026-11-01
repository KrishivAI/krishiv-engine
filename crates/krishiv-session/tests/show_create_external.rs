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

#[test]
fn a_view_shows_the_same_statement_as_serde() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE tbl (a INT, b INT) USING parquet")).unwrap();
    block_on(s.sql("CREATE VIEW v (aaa, bbb) AS SELECT a, b FROM tbl")).unwrap();
    assert_eq!(show_create(&s, "v AS SERDE"), show_create(&s, "v"));
    assert!(block_on(s.sql("SHOW CREATE TABLE tbl AS SERDE")).is_err());
}

#[test]
fn a_global_temp_view_is_a_view_to_alter_table() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE GLOBAL TEMPORARY VIEW gv(a, b) AS SELECT 1, 'one'")).unwrap();
    let e = block_on(s.sql("ALTER TABLE global_temp.gv CHANGE a TYPE INT")).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::ExpectTableNotViewNoAlternative
    );
}

#[test]
fn comment_on_table_sets_and_clears_the_table_comment() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE tc (a INT) USING parquet")).unwrap();
    block_on(s.sql("COMMENT ON TABLE IDENTIFIER('t' || 'c') IS 'hello'")).unwrap();
    assert!(show_create(&s, "tc").contains("COMMENT 'hello'\n"));
    block_on(s.sql("COMMENT ON TABLE tc IS NULL")).unwrap();
    assert!(!show_create(&s, "tc").contains("COMMENT"));
    assert!(block_on(s.sql("COMMENT ON TABLE nope IS 'x'")).is_err());
    block_on(s.sql("CREATE VIEW vc AS SELECT a FROM tc")).unwrap();
    assert!(block_on(s.sql("COMMENT ON TABLE vc IS 'x'")).is_err());
}

#[test]
fn a_three_part_name_of_an_unknown_catalog_needs_a_single_part_namespace() {
    let s = Session::new().unwrap();
    for sql in [
        "CREATE TABLE IDENTIFIER('a.b.c')(c1 INT) USING csv",
        "DROP TABLE IDENTIFIER('a.b.c')",
        "DROP VIEW IDENTIFIER('a.b.c')",
    ] {
        let e = block_on(s.sql(sql)).unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::RequiresSinglePartNamespace,
            "{sql}"
        );
    }
}
