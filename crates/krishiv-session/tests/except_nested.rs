// SPDX-License-Identifier: Apache-2.0
//! `* EXCEPT` and `s.* EXCEPT` over struct fields (Spark's `DropField`): the
//! struct is rebuilt without the named fields, in place, and unknown or
//! overlapping entries are refused.
#![cfg(test)]

use arrow_array::Array;
use futures::executor::block_on;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql(
        "CREATE TEMPORARY VIEW t AS SELECT * FROM VALUES \
         (10, 'n1', named_struct('f1', 1, 's2', named_struct('f2', 101, 'f3', 'a'))), \
         (20, 'n2', named_struct('f1', 2, 's2', named_struct('f2', 202, 'f3', 'b'))) \
         AS t(id, name, data)",
    ))
    .unwrap();
    s
}

fn schema(s: &Session, sql: &str) -> String {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| format!("{}:{}", f.name(), f.data_type()))
        .collect::<Vec<_>>()
        .join(",")
}

fn json(s: &Session, sql: &str) -> Vec<String> {
    let q = format!("SELECT to_json(data) FROM ({sql}) ORDER BY id");
    let batches = block_on(block_on(s.sql(&q)).unwrap().collect()).unwrap();
    batches
        .iter()
        .flat_map(|b| {
            let a = arrow_array::cast::AsArray::as_string_view(b.column(0)).clone();
            (0..a.len()).map(move |i| a.value(i).to_owned())
        })
        .collect()
}

#[test]
fn a_top_level_struct_field_is_dropped() {
    let s = session();
    let q = "SELECT * EXCEPT (data.f1) FROM t";
    assert!(schema(&s, q).starts_with("id:"));
    assert_eq!(
        json(&s, q),
        [
            r#"{"s2":{"f2":101,"f3":"a"}}"#,
            r#"{"s2":{"f2":202,"f3":"b"}}"#
        ]
    );
}

#[test]
fn a_nested_field_and_a_whole_substruct_are_dropped() {
    let s = session();
    assert_eq!(
        json(&s, "SELECT * EXCEPT (data.s2.f2) FROM t"),
        [r#"{"f1":1,"s2":{"f3":"a"}}"#, r#"{"f1":2,"s2":{"f3":"b"}}"#]
    );
}

#[test]
fn dropping_every_field_leaves_an_empty_struct() {
    let s = session();
    assert_eq!(
        json(&s, "SELECT * EXCEPT (data.f1, data.s2) FROM t"),
        ["{}", "{}"]
    );
}

#[test]
fn a_struct_star_drops_fields_and_names_the_rest() {
    let s = session();
    assert_eq!(schema(&s, "SELECT data.* EXCEPT (s2) FROM t"), "f1:Int32");
    assert_eq!(
        schema(&s, "SELECT data.s2.* EXCEPT (f2) FROM t"),
        "f3:Utf8View"
    );
    let q = "SELECT data.* EXCEPT (s2.f2) FROM t";
    assert!(schema(&s, q).starts_with("f1:Int32,s2:Struct"));
}

#[test]
fn unknown_and_overlapping_entries_are_refused() {
    let s = session();
    for q in [
        "SELECT * EXCEPT (data.nope) FROM t",
        "SELECT * EXCEPT (data.s2, data.s2.f2) FROM t",
        "SELECT data.* EXCEPT (nope) FROM t",
    ] {
        assert!(block_on(s.sql(q)).is_err(), "{q}");
    }
}
