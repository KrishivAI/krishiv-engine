// SPDX-License-Identifier: Apache-2.0
//! Array, map and struct columns end to end: unions, casts, comparisons and
//! the checks Spark makes on them. Every expectation was checked against
//! Spark 4.2.0 (`SELECT ...` over the same values; types as Spark prints
//! them, values as `CAST(.. AS STRING)` renders them).
#![cfg(test)]

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, ListBuilder, MapBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType as ArrowType, Field, Fields, Schema};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;
use krishiv_types::DataType;

/// `n(s struct<a:int>, s2 struct<b:int>, l array<int> (elements not null),
/// m map<string,int>)`, three rows; row 2 is NULL in every column but `l`.
fn session() -> Session {
    let s = Session::new().unwrap();
    let a = Field::new("a", ArrowType::Int32, true);
    let b = Field::new("b", ArrowType::Int32, true);
    let nulls = Some(NullBuffer::from(vec![true, true, false]));
    let st = |f: &Field, v: Vec<Option<i32>>| -> ArrayRef {
        Arc::new(StructArray::new(
            Fields::from(vec![f.clone()]),
            vec![Arc::new(Int32Array::from(v)) as ArrayRef],
            nulls.clone(),
        ))
    };
    // The NULL struct row holds a value no query may read.
    let s_col = st(&a, vec![Some(1), None, Some(99)]);
    let s2_col = st(&b, vec![Some(2), Some(3), Some(4)]);
    let element = Field::new("element", ArrowType::Int32, false);
    let mut l = ListBuilder::new(Int32Builder::new()).with_field(Arc::new(element));
    l.append_value([Some(1), Some(2)]);
    l.append_value([]);
    l.append_value([Some(3)]);
    let l_col: ArrayRef = Arc::new(l.finish());
    let mut m = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    m.keys().append_value("k");
    m.values().append_value(1);
    m.keys().append_value("z");
    m.values().append_null();
    m.append(true).unwrap();
    m.append(true).unwrap();
    m.append(false).unwrap();
    let m_col: ArrayRef = Arc::new(m.finish());
    let schema = Arc::new(Schema::new(vec![
        Field::new("s", s_col.data_type().clone(), true),
        Field::new("s2", s2_col.data_type().clone(), true),
        Field::new("l", l_col.data_type().clone(), false),
        Field::new("m", m_col.data_type().clone(), true),
    ]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![s_col, s2_col, l_col, m_col]).unwrap();
    s.register_batches("n", schema, vec![batch]).unwrap();
    s
}

/// The Spark type and nullability of each output column of `sql`.
fn types(s: &Session, sql: &str) -> Vec<(String, bool)> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let schema = block_on(df.schema()).unwrap();
    schema
        .fields()
        .iter()
        .map(|f| {
            let t = DataType::from_arrow_field(f).unwrap();
            (t.json(), f.is_nullable())
        })
        .collect()
}

/// Every row of `sql`'s first column, cast to STRING by the engine.
fn strings(s: &Session, sql: &str) -> Vec<Option<String>> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    block_on(df.collect())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_string_view()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn error(s: &Session, sql: &str) -> (ErrorClass, String) {
    match block_on(s.sql(sql)) {
        Ok(_) => panic!("{sql} resolved"),
        Err(e) => (e.error_class(), e.to_string()),
    }
}

fn some(v: &[&str]) -> Vec<Option<String>> {
    v.iter().map(|s| Some((*s).to_owned())).collect()
}

#[test]
fn unions_of_types_that_differ_in_names_or_nullability_run() {
    let s = session();
    // The first input's names; values from both.
    let sql = "SELECT CAST(x AS STRING) FROM (SELECT s AS x FROM n UNION ALL SELECT s2 FROM n)";
    assert_eq!(
        types(&s, "SELECT s AS x FROM n UNION ALL SELECT s2 FROM n"),
        [(
            r#"{"type":"struct","fields":[{"name":"a","type":"integer","nullable":true,"metadata":{}}]}"#
                .to_owned(),
            true
        )]
    );
    let mut rows = strings(&s, sql);
    rows.sort();
    assert_eq!(
        rows,
        [
            None,
            None,
            Some("{1}".to_owned()),
            Some("{2}".to_owned()),
            Some("{3}".to_owned()),
            Some("{null}".to_owned())
        ]
    );
    // An array whose elements are never NULL meets one whose may be.
    let sql = "SELECT CAST(x AS STRING) FROM \
               (SELECT l AS x FROM n UNION ALL SELECT CAST(NULL AS ARRAY<INT>) FROM n)";
    let mut rows = strings(&s, sql);
    rows.sort();
    assert_eq!(rows[..3], [None, None, None]);
    assert_eq!(rows[3..], some(&["[1, 2]", "[3]", "[]"]));
}

#[test]
fn nested_casts_run_as_spark_runs_them() {
    let s = session();
    assert_eq!(
        strings(&s, "SELECT CAST(l AS STRING) FROM n"),
        some(&["[1, 2]", "[]", "[3]"])
    );
    assert_eq!(
        strings(&s, "SELECT CAST(m AS STRING) FROM n"),
        [
            Some("{k -> 1, z -> null}".to_owned()),
            Some("{}".to_owned()),
            None
        ]
    );
    assert_eq!(
        strings(&s, "SELECT CAST(CAST(l AS ARRAY<STRING>) AS STRING) FROM n"),
        some(&["[1, 2]", "[]", "[3]"])
    );
    // Renamed and widened by position; the NULL row's hidden value is not read.
    assert_eq!(
        strings(
            &s,
            "SELECT CAST(CAST(s AS STRUCT<x: BIGINT>) AS STRING) FROM n"
        ),
        [Some("{1}".to_owned()), Some("{null}".to_owned()), None]
    );
    assert_eq!(
        strings(
            &s,
            "SELECT CAST(CAST(m AS MAP<STRING, BIGINT>) AS STRING) FROM n"
        ),
        [
            Some("{k -> 1, z -> null}".to_owned()),
            Some("{}".to_owned()),
            None
        ]
    );
    // `TRY_CAST` that only widens keeps the column's nullability; one that
    // narrows may fail, so may be NULL.
    let t = types(
        &s,
        "SELECT TRY_CAST(l AS ARRAY<BIGINT>), TRY_CAST(l AS ARRAY<STRING>), TRY_CAST(l AS ARRAY<TINYINT>) FROM n",
    );
    assert_eq!(
        t.iter().map(|(_, n)| *n).collect::<Vec<_>>(),
        [false, false, true]
    );
}

#[test]
fn nested_types_are_not_strings() {
    let s = session();
    for sql in [
        "SELECT upper(l) FROM n",
        "SELECT length(s) FROM n",
        "SELECT l LIKE 'x' FROM n",
        "SELECT concat(m, m) FROM n",
    ] {
        assert_eq!(
            error(&s, sql).0,
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{sql}"
        );
    }
    assert_eq!(
        error(&s, "SELECT concat(l, 'x') FROM n").0,
        ErrorClass::DatatypeMismatchDataDiffTypes
    );
    assert_eq!(
        types(&s, "SELECT concat(l, l) FROM n")[0].0,
        r#"{"type":"array","elementType":"integer","containsNull":false}"#
    );
}

#[test]
fn maps_do_not_order_and_structs_compare_by_shape() {
    let s = session();
    for sql in [
        "SELECT m = m FROM n",
        "SELECT m IN (m) FROM n",
        "SELECT * FROM n ORDER BY m",
    ] {
        assert_eq!(
            error(&s, sql).0,
            ErrorClass::DatatypeMismatchInvalidOrderingType,
            "{sql}"
        );
    }
    let (_, message) = error(&s, "SELECT * FROM n ORDER BY m");
    assert!(
        message.contains("`sortorder`") && message.contains("\"m ASC NULLS FIRST\""),
        "{message}"
    );
    for sql in [
        "SELECT DISTINCT m FROM n",
        "SELECT m FROM n INTERSECT SELECT m FROM n",
        "SELECT m FROM n EXCEPT SELECT m FROM n",
        "SELECT m FROM n UNION SELECT m FROM n",
    ] {
        assert_eq!(
            error(&s, sql).0,
            ErrorClass::UnsupportedFeatureSetOperationOnMapType,
            "{sql}"
        );
    }
    // Spark groups by a map and counts its distinct values.
    types(&s, "SELECT count(*) FROM n GROUP BY m");
    // Field names do not matter to `=` or `IN`; types do.
    assert_eq!(types(&s, "SELECT s = s2, s IN (s2) FROM n").len(), 2);
    assert_eq!(
        error(&s, "SELECT s = CAST(s2 AS STRUCT<b: BIGINT>) FROM n").0,
        ErrorClass::DatatypeMismatchBinaryOpDiffTypes
    );
}

#[test]
fn struct_field_names_follow_case_sensitivity() {
    let s = session();
    let sql = "SELECT coalesce(s, CAST(NULL AS STRUCT<A: BIGINT>)) FROM n";
    assert_eq!(
        types(&s, sql)[0].0,
        r#"{"type":"struct","fields":[{"name":"a","type":"long","nullable":true,"metadata":{}}]}"#
    );
    // Java's `equalsIgnoreCase`, beyond ASCII.
    types(
        &s,
        "SELECT coalesce(CAST(NULL AS STRUCT<`é`: INT>), CAST(NULL AS STRUCT<`É`: INT>))",
    );
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    assert_eq!(error(&s, sql).0, ErrorClass::DatatypeMismatchDataDiffTypes);
    assert_eq!(
        error(
            &s,
            "SELECT s FROM n INTERSECT SELECT CAST(s AS STRUCT<A: INT>) FROM n"
        )
        .0,
        ErrorClass::IncompatibleColumnType
    );
}
