// SPDX-License-Identifier: Apache-2.0
//! `COMMENT ON SCHEMA` and `ALTER SCHEMA ... SET PROPERTIES | SET LOCATION` change the namespace's metadata.
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn describe(s: &Session, name: &str) -> Vec<String> {
    let df = block_on(s.sql(&format!("DESCRIBE SCHEMA EXTENDED {name}"))).unwrap();
    let mut rows = Vec::new();
    for b in block_on(df.collect()).unwrap() {
        let (k, v) = (b.column(0).as_string_view(), b.column(1).as_string_view());
        for i in 0..b.num_rows() {
            rows.push(format!("{}={}", k.value(i), v.value(i)));
        }
    }
    rows
}

#[test]
fn a_namespace_takes_a_comment_properties_and_a_location() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE SCHEMA ns1")).unwrap();
    block_on(s.sql("COMMENT ON SCHEMA IDENTIFIER('ns' || '1') IS 'some comment'")).unwrap();
    block_on(s.sql("ALTER SCHEMA ns1 SET PROPERTIES ('a' = 'b')")).unwrap();
    block_on(s.sql("ALTER NAMESPACE ns1 SET LOCATION '/tmp/ns1-moved'")).unwrap();
    let rows = describe(&s, "ns1");
    assert!(
        rows.contains(&"Comment=some comment".to_owned()),
        "{rows:?}"
    );
    assert!(
        rows.iter()
            .any(|r| r.starts_with("Properties=") && r.contains("a=b")),
        "{rows:?}"
    );
    assert!(
        rows.iter()
            .any(|r| r.starts_with("Location=") && r.ends_with("/tmp/ns1-moved")),
        "{rows:?}"
    );
    block_on(s.sql("COMMENT ON DATABASE ns1 IS NULL")).unwrap();
    assert!(describe(&s, "ns1").contains(&"Comment=".to_owned()));
    let err = block_on(s.sql("COMMENT ON SCHEMA missing IS 'x'"))
        .map(|_| ())
        .unwrap_err();
    assert!(err.to_string().contains("SCHEMA_NOT_FOUND"), "{err}");
}

/// Spark 4.2.0 (`research/spark/row-format-file-format/probe13.py` a1, a2, a6, a7):
/// `UNSET PROPERTIES|DBPROPERTIES` removes the keys, an absent key is fine, and the
/// reserved `comment` is `CANNOT_REMOVE_RESERVED_PROPERTY` when the command runs.
#[test]
fn unset_properties_removes_the_keys() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE SCHEMA ns2 WITH DBPROPERTIES ('a' = '1', 'b' = '2', 'c' = '3')"))
        .unwrap();
    block_on(s.sql("ALTER SCHEMA ns2 UNSET PROPERTIES ('a', 'absent')")).unwrap();
    block_on(s.sql("ALTER DATABASE ns2 UNSET DBPROPERTIES ('c')")).unwrap();
    let rows = describe(&s, "ns2");
    let props = rows
        .iter()
        .find(|r| r.starts_with("Properties="))
        .cloned()
        .unwrap_or_default();
    assert!(props.contains("b=2"), "{rows:?}");
    assert!(!props.contains("a=1") && !props.contains("c=3"), "{rows:?}");
    let err = block_on(s.sql("ALTER NAMESPACE ns2 UNSET PROPERTIES ('comment')"))
        .map(|_| ())
        .unwrap_err();
    assert_eq!(
        err.error_class().name(),
        "CANNOT_REMOVE_RESERVED_PROPERTY",
        "{err}"
    );
    let err = block_on(s.sql("ALTER NAMESPACE missing UNSET PROPERTIES ('a')"))
        .map(|_| ())
        .unwrap_err();
    assert!(err.to_string().contains("SCHEMA_NOT_FOUND"), "{err}");
}
