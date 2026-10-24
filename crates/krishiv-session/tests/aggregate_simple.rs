// SPDX-License-Identifier: Apache-2.0
//! `count_if`, `any_value`, `max_by` and `min_by` through SQL: the planner's
//! rewrite of `count_if` to `count`, the typing errors, grouped, global and
//! window forms. Expected values follow Spark 4.2's definitions (the
//! boolean aggregates, `every`/`some`/`any`/`bool_and`/`bool_or`, are
//! checked against its goldens in `aggregate_goldens.rs`).
#![forbid(unsafe_code)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Seven rows `(id, k, v, x, y, d)`: `k` groups them (1: ids 1–4, 2: ids
/// 5–6, 3: id 7); `y` and `d` hold NULLs, ties (`0.0 = -0.0`, `y = 7`),
/// NaN and infinity.
fn session() -> R<Session> {
    let s = Session::new()?;
    let nan = f64::NAN;
    let columns: Vec<(&str, DataType, ArrayRef)> = vec![
        (
            "id",
            DataType::Int32,
            Arc::new(Int32Array::from((1..=7).collect::<Vec<_>>())),
        ),
        (
            "k",
            DataType::Int32,
            Arc::new(Int32Array::from(vec![1, 1, 1, 1, 2, 2, 3])),
        ),
        (
            "v",
            DataType::Boolean,
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                None,
                Some(true),
                None,
                None,
                Some(false),
            ])),
        ),
        (
            "x",
            DataType::Utf8,
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                Some("c"),
                Some("d"),
                None,
                Some("p"),
                Some("q"),
            ])),
        ),
        (
            "y",
            DataType::Int32,
            Arc::new(Int32Array::from(vec![
                Some(3),
                None,
                Some(7),
                Some(7),
                None,
                Some(2),
                Some(9),
            ])),
        ),
        (
            "d",
            DataType::Float64,
            Arc::new(Float64Array::from(vec![
                Some(0.0),
                Some(-0.0),
                Some(nan),
                Some(f64::INFINITY),
                None,
                Some(nan),
                Some(nan),
            ])),
        ),
    ];
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|(n, t, _)| Field::new(*n, t.clone(), true))
            .collect::<Vec<_>>(),
    ));
    let arrays = columns.into_iter().map(|c| c.2).collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    s.register_batches("t", schema, vec![batch])?;
    Ok(s)
}

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        DataType::Float64 => format!("{:?}", c.as_primitive::<Float64Type>().value(row)),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        other => format!("<{other}>"),
    }
}

/// The column names and types, and the rows (`a|b|c`) of `sql`, in order.
fn query(s: &Session, sql: &str) -> Result<(Vec<String>, Vec<String>), KrishivError> {
    block_on(async {
        let df = s.sql(sql).await?;
        let schema = df.schema().await?;
        let batches = df.collect().await?;
        let fields = schema
            .fields()
            .iter()
            .map(|f| format!("{}:{}", f.name(), f.data_type()))
            .collect();
        let rows = batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows()).map(|r| {
                    let cells: Vec<String> =
                        b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
                    cells.join("|")
                })
            })
            .collect();
        Ok((fields, rows))
    })
}

fn rows(s: &Session, sql: &str) -> R<Vec<String>> {
    Ok(query(s, sql)?.1)
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

/// `count_if(c)` counts the rows where `c` is true: `bigint`, never NULL,
/// 0 for no rows and for NULL conditions; `DISTINCT` counts at most one.
#[test]
fn count_if_counts_true_rows() -> R {
    let s = session()?;
    let (names, got) = query(
        &s,
        "SELECT k, count_if(v), count_if(v IS NULL), count_if(k = 1) \
         FROM t GROUP BY k ORDER BY k",
    )?;
    assert_eq!(
        names,
        [
            "k:Int32",
            "count_if(v):Int64",
            "count_if((v IS NULL)):Int64",
            "count_if((k = 1)):Int64"
        ]
    );
    assert_eq!(got, ["1|2|1|4", "2|0|2|0", "3|0|0|0"]);
    // Global: one row, 0 without rows or without a true one.
    assert_eq!(rows(&s, "SELECT count_if(v) FROM t")?, ["2"]);
    assert_eq!(rows(&s, "SELECT count_if(v) FROM t WHERE 1 = 0")?, ["0"]);
    assert_eq!(rows(&s, "SELECT count_if(v) FROM t WHERE k = 2")?, ["0"]);
    assert_eq!(
        rows(&s, "SELECT count_if(true), count_if(false) FROM t")?,
        ["7|0"]
    );
    // A grouped aggregate over no rows has no group.
    assert!(rows(&s, "SELECT k, count_if(v) FROM t WHERE 1 = 0 GROUP BY k")?.is_empty());
    assert_eq!(
        rows(
            &s,
            "SELECT k, count_if(DISTINCT v) FROM t GROUP BY k ORDER BY k"
        )?,
        ["1|1", "2|0", "3|0"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT k FROM t GROUP BY k HAVING count_if(v) > 1 ORDER BY k"
        )?,
        ["1"]
    );
    // Not nullable: the result of the aggregate is `count`'s.
    assert_eq!(
        rows(&s, "SELECT count_if(v) + 1 FROM t WHERE k = 3")?,
        ["1"]
    );
    Ok(())
}

#[test]
fn count_if_in_windows() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT id, count_if(v) OVER (PARTITION BY k ORDER BY id) FROM t ORDER BY id"
        )?,
        ["1|1", "2|1", "3|1", "4|2", "5|0", "6|0", "7|0"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT id, count_if(v) OVER (PARTITION BY k) FROM t ORDER BY id"
        )?,
        ["1|2", "2|2", "3|2", "4|2", "5|0", "6|0", "7|0"]
    );
    Ok(())
}

/// The argument must be a `boolean` (a string is cast, a number is an
/// error), as `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` with the registry's
/// required type.
#[test]
fn count_if_checks_its_argument_type() -> R {
    let s = session()?;
    for (sql, input_sql, input_type) in [
        ("SELECT count_if(1)", "\"1\"", "\"INT\""),
        ("SELECT count_if(y) FROM t", "\"y\"", "\"INT\""),
        ("SELECT count_if(d) FROM t", "\"d\"", "\"DOUBLE\""),
    ] {
        let (class, params) = error(&s, sql)?;
        assert_eq!(class, "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE", "{sql}");
        assert_eq!(param(&params, "requiredType"), Some("\"BOOLEAN\""));
        assert_eq!(param(&params, "paramIndex"), Some("first"));
        assert_eq!(param(&params, "inputSql"), Some(input_sql));
        assert_eq!(param(&params, "inputType"), Some(input_type));
        assert!(param(&params, "sqlExpr").is_some_and(|e| e.contains("count_if(")));
    }
    let (class, _) = error(&s, "SELECT count_if(v, v) FROM t")?;
    assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    // The boolean aggregates take the same check.
    let (class, params) = error(&s, "SELECT bool_and(y) FROM t")?;
    assert_eq!(class, "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE");
    assert_eq!(param(&params, "requiredType"), Some("\"BOOLEAN\""));
    Ok(())
}

/// Spark 4.2: the value of the row with the extreme non-NULL key, the first
/// such row on a tie; NULL keys are skipped; NaN is the largest `double`,
/// `-0.0 = 0.0`.
#[test]
fn max_by_and_min_by() -> R {
    let s = session()?;
    let (names, got) = query(
        &s,
        "SELECT k, max_by(x, y), min_by(x, y) FROM t GROUP BY k ORDER BY k",
    )?;
    assert_eq!(
        names,
        ["k:Int32", "max_by(x, y):Utf8View", "min_by(x, y):Utf8View"]
    );
    // Group 1: keys 3, NULL, 7, 7: the largest is the first 7 (`c`).
    // Group 2: the NULL key of the first row is skipped.
    assert_eq!(got, ["1|c|a", "2|p|p", "3|q|q"]);
    assert_eq!(
        rows(&s, "SELECT max_by(x, y), min_by(x, y) FROM t")?,
        ["q|p"]
    );
    // Doubles: NaN beats infinity; `0.0` and `-0.0` tie, the first wins.
    assert_eq!(
        rows(
            &s,
            "SELECT k, max_by(y, d), min_by(y, d) FROM t GROUP BY k ORDER BY k"
        )?,
        ["1|7|3", "2|2|2", "3|9|9"]
    );
    // Expressions as arguments; a NULL value is a result.
    assert_eq!(rows(&s, "SELECT max_by(y, id) FROM t WHERE k = 1")?, ["7"]);
    assert_eq!(
        rows(
            &s,
            "SELECT max_by(y, id), min_by(y, id) FROM t WHERE k = 1 AND id < 3"
        )?,
        ["NULL|3"]
    );
    // No rows, or only NULL keys: NULL.
    assert_eq!(
        rows(&s, "SELECT max_by(x, y), min_by(x, y) FROM t WHERE 1 = 0")?,
        ["NULL|NULL"]
    );
    assert_eq!(
        rows(&s, "SELECT max_by(x, y), min_by(x, y) FROM t WHERE id = 5")?,
        ["NULL|NULL"]
    );
    assert!(rows(&s, "SELECT k, max_by(x, y) FROM t WHERE 1 = 0 GROUP BY k")?.is_empty());
    Ok(())
}

#[test]
fn max_by_and_min_by_in_windows() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT id, max_by(x, y) OVER (PARTITION BY k), \
             min_by(x, y) OVER (PARTITION BY k) FROM t ORDER BY id"
        )?,
        [
            "1|c|a", "2|c|a", "3|c|a", "4|c|a", "5|p|p", "6|p|p", "7|q|q"
        ]
    );
    // A running frame: the first row wins ties (`-0.0` after `0.0`, `inf`
    // after NaN), a NULL key leaves the state as it was.
    assert_eq!(
        rows(
            &s,
            "SELECT id, max_by(y, d) OVER (PARTITION BY k ORDER BY id) FROM t ORDER BY id"
        )?,
        ["1|3", "2|3", "3|7", "4|7", "5|NULL", "6|2", "7|9"]
    );
    // Strings would be copied for every row of a running frame.
    let (class, _) = error(
        &s,
        "SELECT max_by(x, y) OVER (PARTITION BY k ORDER BY id) FROM t",
    )?;
    assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE");
    Ok(())
}

/// `max_by(x, y, k)` returns an array: not supported, and the wrong number
/// of arguments is Spark's error.
#[test]
fn max_by_with_a_limit_is_unsupported() -> R {
    let s = session()?;
    for f in ["max_by", "min_by"] {
        let (class, params) = error(&s, &format!("SELECT {f}(x, y, 2) FROM t"))?;
        assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE");
        assert!(param(&params, "feature").is_some_and(|m| m.contains("array")));
        let (class, _) = error(&s, &format!("SELECT {f}(x) FROM t"))?;
        assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    }
    // The other aggregates that return arrays stay unsupported.
    for f in ["collect_list", "collect_set"] {
        let (class, _) = error(&s, &format!("SELECT {f}(x) FROM t"))?;
        assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE", "{f}");
    }
    Ok(())
}

/// `any_value(x)` is the first row's value, `any_value(x, true)` the first
/// non-NULL one; the flag must be a boolean literal and is not part of the
/// column's name (Spark: `any_value(col)`).
#[test]
fn any_value_returns_the_first_value() -> R {
    let s = session()?;
    let (names, got) = query(
        &s,
        "SELECT k, any_value(x), any_value(x, true), any_value(x, false) \
         FROM t GROUP BY k ORDER BY k",
    )?;
    assert_eq!(
        names,
        [
            "k:Int32",
            "any_value(x):Utf8View",
            "any_value(x):Utf8View",
            "any_value(x):Utf8View"
        ]
    );
    assert_eq!(got, ["1|a|a|a", "2|NULL|p|NULL", "3|q|q|q"]);
    assert_eq!(
        rows(
            &s,
            "SELECT any_value(x), any_value(y, true) FROM t WHERE 1 = 0"
        )?,
        ["NULL|NULL"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT id, any_value(x, true) OVER (PARTITION BY k ORDER BY id) FROM t ORDER BY id"
        )?,
        ["1|a", "2|a", "3|a", "4|a", "5|NULL", "6|p", "7|q"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT id, any_value(x) IGNORE NULLS OVER (PARTITION BY k ORDER BY id) \
             FROM t ORDER BY id"
        )?,
        ["1|a", "2|a", "3|a", "4|a", "5|NULL", "6|p", "7|q"]
    );
    let (class, params) = error(&s, "SELECT any_value(x, 1 = 1) FROM t")?;
    assert_eq!(class, "_LEGACY_ERROR_TEMP_1210");
    assert_eq!(param(&params, "funcName"), Some("any_value"));
    Ok(())
}

/// The planner maps every registry name of an aggregate, aliases included:
/// `mean` is `avg`, grouped and in a window (one mapping serves both).
#[test]
fn aliases_run_as_their_aggregate() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "SELECT k, mean(y) FROM t WHERE k > 1 GROUP BY k ORDER BY k"
        )?,
        ["2|2.0", "3|9.0"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT id, mean(y) OVER (PARTITION BY k) FROM t WHERE k > 1 ORDER BY id"
        )?,
        ["5|2.0", "6|2.0", "7|9.0"]
    );
    Ok(())
}
