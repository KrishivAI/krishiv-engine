// SPDX-License-Identifier: Apache-2.0
//! `json_tuple` end to end: the Spark 4.2.0 golden statements
//! (`tests/spark-golden/v4.2.0/results/json-functions.sql.out` and
//! `generators.sql.out`), the view-backed statement over data registered
//! through the Rust API, and the out-of-scope `FROM` forms.
//!
//! The `SELECT * FROM json_tuple(...)` forms stay
//! `KRISHIV_UNSUPPORTED_FEATURE`: the engine has no table-valued-function
//! machinery (every `FROM` function is rejected at lowering), so the
//! correlated `LATERAL json_tuple(...)` forms cannot run either.
#![cfg(test)]
#![forbid(unsafe_code)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn session() -> R<Session> {
    Ok(Session::new()?)
}

fn cell(c: &ArrayRef, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".to_owned();
    }
    match c.data_type() {
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        DataType::Int32 => c
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(row)
            .to_string(),
        other => panic!("unrendered test value of {other}"),
    }
}

/// Output column names and rows of `sql` (`NULL` for nulls).
fn query(s: &Session, sql: &str) -> Result<(Vec<String>, Vec<String>), KrishivError> {
    block_on(async {
        let df = s.sql(sql).await?;
        let schema = df.schema().await?;
        let names = schema.fields().iter().map(|f| f.name().clone()).collect();
        let batches = df.collect().await?;
        let rows = batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows()).map(|r| {
                    b.columns()
                        .iter()
                        .map(|c| cell(c, r))
                        .collect::<Vec<_>>()
                        .join("|")
                })
            })
            .collect();
        Ok((names, rows))
    })
}

/// The error class and message parameters of `sql`, which must fail.
fn error(s: &Session, sql: &str) -> R<(String, Vec<(String, String)>)> {
    match query(s, sql) {
        Ok(_) => Err(format!("{sql} did not fail").into()),
        Err(e) => Ok((
            e.error_class().name().to_owned(),
            e.message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
        )),
    }
}

fn param<'a>(params: &'a [(String, String)], key: &str) -> Option<&'a str> {
    params
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// The golden `json-functions.sql` and `generators.sql` statements that
/// read no table, with exactly the golden schemas and rows.
#[test]
fn golden_select_forms_match_spark() -> R {
    let s = session()?;
    // `json-functions.sql`: NULL field names give NULL columns.
    let (names, got) = query(
        &s,
        "SELECT json_tuple('{\"a\" : 1, \"b\" : 2}', CAST(NULL AS STRING), 'b', CAST(NULL AS STRING), 'a')",
    )?;
    assert_eq!(names, ["c0", "c1", "c2", "c3"]);
    assert_eq!(got, ["NULL|2|NULL|1"]);
    // `json-functions.sql`: non-foldable NULL field names.
    let (names, got) = query(
        &s,
        "SELECT json_tuple('{\"a\":\"1\"}', if(c1 < 1, null, 'a')) FROM ( SELECT rand() AS c1 )",
    )?;
    assert_eq!(names, ["c0"]);
    assert_eq!(got, ["NULL"]);
    let (names, got) = query(
        &s,
        "SELECT json_tuple('{\"a\":\"1\"}', if(c1 < 1, null, 'a'), if(c2 < 1, null, 'a')) FROM ( SELECT 0 AS c1, rand() AS c2 )",
    )?;
    assert_eq!(names, ["c0", "c1"]);
    assert_eq!(got, ["NULL|NULL"]);
    // `generators.sql`: rows and errors.
    let (names, got) = query(&s, "select json_tuple('{\"a\": 1, \"b\": 2}', 'a', 'b')")?;
    assert_eq!(names, ["c0", "c1"]);
    assert_eq!(got, ["1|2"]);
    let (class, params) = error(&s, "select json_tuple()")?;
    assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    assert_eq!(param(&params, "expectedNum"), Some("> 1"));
    assert_eq!(param(&params, "functionName"), Some("`json_tuple`"));
    let (class, _) = error(&s, "select json_tuple('{\"a\": 1}')")?;
    assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    let (class, params) = error(&s, "select json_tuple('{\"a\": 1}', 1)")?;
    assert_eq!(class, "DATATYPE_MISMATCH.NON_STRING_TYPE");
    assert_eq!(param(&params, "funcName"), Some("`json_tuple`"));
    let (class, _) = error(&s, "select json_tuple('{\"a\": 1}', null)")?;
    assert_eq!(class, "DATATYPE_MISMATCH.NON_STRING_TYPE");
    Ok(())
}

/// The golden view-backed statement: `CREATE TEMPORARY VIEW jsonTable` is
/// a command the session cannot run yet (`depends-on-command`), so the
/// same data is registered through the Rust API and the golden rows are
/// asserted.
#[test]
fn golden_view_statement_runs_over_registered_data() -> R {
    let s = session()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("jsonField", DataType::Utf8View, true),
        Field::new("a", DataType::Utf8View, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::StringViewArray::from(vec![Some(
                "{\"a\": 1, \"b\": 2}",
            )])) as ArrayRef,
            Arc::new(arrow_array::StringViewArray::from(vec![Some("a")])) as ArrayRef,
        ],
    )?;
    s.register_batches("jsonTable", schema, vec![batch])?;
    let (names, got) = query(
        &s,
        "SELECT json_tuple(jsonField, 'b', CAST(NULL AS STRING), a) FROM jsonTable",
    )?;
    assert_eq!(names, ["c0", "c1", "c2"]);
    assert_eq!(got, ["2|NULL|1"]);
    Ok(())
}

/// Value semantics through the whole engine: compacted objects and arrays,
/// verbatim number tokens, decoded strings, duplicate keys and per-row
/// independence over several rows.
#[test]
fn rows_evaluate_per_row() -> R {
    let s = session()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("j", DataType::Utf8, true),
        Field::new("k", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some("{\"a\": 1.50, \"b\": [1, \"x\"], \"a\": true}"),
                None,
                Some("{\"a\": \"\\u00e9\"}"),
                Some("broken"),
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("a"),
                None,
                Some("b"),
            ])) as ArrayRef,
        ],
    )?;
    s.register_batches("t", schema, vec![batch])?;
    let (names, got) = query(&s, "SELECT json_tuple(j, k, 'b') FROM t ORDER BY id")?;
    assert_eq!(names, ["c0", "c1"]);
    assert_eq!(
        got,
        ["true|[1,\"x\"]", "NULL|NULL", "NULL|NULL", "NULL|NULL",]
    );
    Ok(())
}

/// The `FROM json_tuple(...)` forms (constant and erroneous arguments, and
/// the column-aliased table alias) stay `KRISHIV_UNSUPPORTED_FEATURE`:
/// the engine has no table-valued-function machinery.
#[test]
fn table_valued_from_forms_stay_unsupported() -> R {
    let s = session()?;
    for sql in [
        "select * from json_tuple('{\"a\": 1, \"b\": 2}', 'a', 'b')",
        "select * from json_tuple('{\"a\": 1, \"b\": 2}', 'a', 'c')",
        "select * from json_tuple('{\"a\": 1, \"b\": 2}', 'a', 'a')",
        "select * from json_tuple('{\"a\": 1, \"b\": 2}', 'a', 'b') AS t(x, y)",
        "select * from json_tuple('{\"a\": bad, \"b\": string}', 'a', 'b')",
        "select * from json_tuple()",
        "select * from json_tuple('{\"a\": 1}')",
        "select * from json_tuple('{\"a\": 1}', 1)",
        "select * from json_tuple('{\"a\": 1}', null)",
        "select * from json_tuple('{\"a\": 1, \"b\": 2}', 'a', 'b') AS t(x)",
    ] {
        let (class, params) = error(&s, sql)?;
        assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE", "{sql}");
        assert!(
            param(&params, "feature").is_some_and(|f| f.contains("table-valued functions")),
            "{sql}: {params:?}"
        );
    }
    Ok(())
}

/// Select-list shapes around a generator, against a table: an alias renames
/// the one output column (and several outputs under one name are Spark's
/// `UDTF_ALIAS_NUMBER_MISMATCH`), the name is case-insensitive, and the
/// shapes the one-struct rewrite cannot express stay unsupported (Spark 4.2
/// runs several generators as a cross product, and fails the aggregate
/// shapes with its own classes).
#[test]
fn select_list_shapes_around_a_generator() -> R {
    let s = session()?;
    let schema = Arc::new(Schema::new(vec![Field::new("j", DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec![Some("{\"a\":1,\"b\":2}")])) as ArrayRef],
    )?;
    s.register_batches("v", schema, vec![batch])?;
    let (names, rows) = query(&s, "SELECT json_tuple(j, 'a') AS x FROM v")?;
    assert_eq!((names, rows), (vec!["x".to_owned()], vec!["1".to_owned()]));
    let (names, rows) = query(&s, "SELECT JSON_TUPLE(j, 'a', 'b') FROM v")?;
    assert_eq!(names, ["c0", "c1"]);
    assert_eq!(rows, ["1|2"]);
    let (class, params) = error(&s, "SELECT json_tuple(j, 'a', 'b') AS x FROM v")?;
    assert_eq!(class, "UDTF_ALIAS_NUMBER_MISMATCH");
    assert_eq!(param(&params, "aliasesSize"), Some("2"));
    for sql in [
        "SELECT json_tuple(j, 'a'), json_tuple(j, 'b') FROM v",
        "SELECT json_tuple(j, 'a'), count(*) FROM v",
        "SELECT j AS jj, json_tuple(jj, 'a') FROM v",
    ] {
        let (class, _) = error(&s, sql)?;
        assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE", "{sql}");
    }
    Ok(())
}
