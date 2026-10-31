// SPDX-License-Identifier: Apache-2.0
//! The `percentiles.sql` statements over `aggr` (classed
//! `depends-on-command` in the golden baseline, since the file creates the
//! view first): the same data registered through the Rust API, asserting
//! the golden's rows and error class for each statement this change
//! covers. Interval statements are not covered (the engine has no
//! interval type); the window tests are below.
#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use serde_json::Value;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

struct Case {
    query: String,
    schema: String,
    output: String,
}

/// The golden statements at `indexes` of `percentiles.sql.out`.
fn load(indexes: &[usize]) -> R<Vec<Case>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/spark-golden/v4.2.0/results/percentiles.sql.out");
    let text = std::fs::read_to_string(path)?;
    let blocks: Vec<&str> = text.split("-- !query\n").skip(1).collect();
    let mut out = Vec::new();
    for &i in indexes {
        let (query, rest) = blocks[i]
            .split_once("\n-- !query schema\n")
            .ok_or("schema")?;
        let (schema, output) = rest.split_once("\n-- !query output\n").ok_or("output")?;
        out.push(Case {
            query: query.trim().to_owned(),
            schema: schema.trim().to_owned(),
            output: output.trim_end().to_owned(),
        });
    }
    Ok(out)
}

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Float64 => format!("{:?}", c.as_primitive::<Float64Type>().value(row)),
        other => format!("<{other}>"),
    }
}

fn check(s: &Session, case: &Case) -> R {
    let result = block_on(async {
        let df = s.sql(&case.query).await?;
        let schema = df.schema().await?;
        Ok::<_, KrishivError>((schema, df.collect().await?))
    });
    if case.schema == "struct<>" {
        let body = case
            .output
            .split_once('\n')
            .ok_or("an error without a body")?
            .1;
        let json: Value = serde_json::from_str(body)?;
        let text = |k: &str| json[k].as_str().unwrap_or_default().to_owned();
        let Err(e) = result else {
            return Err(format!("{}\n  expected error {}", case.query, json["errorClass"]).into());
        };
        assert_eq!(e.error_class().name(), text("errorClass"), "{}", case.query);
        assert_eq!(
            e.sql_state(),
            Some(text("sqlState").as_str()),
            "{}",
            case.query
        );
        let got: serde_json::Map<String, Value> = e
            .message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), Value::String(v.clone())))
            .collect();
        assert_eq!(
            Value::Object(got),
            json["messageParameters"],
            "{}",
            case.query
        );
        return Ok(());
    }
    let (schema, batches) = result.map_err(|e| format!("{}\n  error {e}", case.query))?;
    let mut names: Vec<String> = Vec::with_capacity(schema.fields().len());
    for f in schema.fields() {
        let t = match f.data_type() {
            DataType::Int32 => "int",
            DataType::Float64 => "double",
            // Only `int` and `double` columns occur here.
            other => return Err(format!("unexpected {other}").into()),
        };
        names.push(format!("{}:{t}", f.name()));
    }
    assert_eq!(
        format!("struct<{}>", names.join(",")),
        case.schema,
        "{}",
        case.query
    );
    let mut rows: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(|r| {
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r))
                    .collect::<Vec<_>>()
                    .join("\t")
            })
        })
        .collect();
    let mut want: Vec<String> = case.output.lines().map(str::to_owned).collect();
    rows.sort();
    want.sort();
    assert_eq!(rows, want, "{}", case.query);
    Ok(())
}

#[test]
fn percentiles_over_aggr_match_the_golden() -> R {
    let s = Session::new()?;
    // `CREATE OR REPLACE TEMPORARY VIEW aggr` from `percentiles.sql`.
    let (k, v) = (
        vec![
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(2),
            Some(2),
            Some(2),
            Some(2),
            Some(3),
            Some(4),
        ],
        vec![
            Some(0),
            Some(10),
            Some(20),
            Some(30),
            Some(40),
            Some(10),
            Some(20),
            Some(10),
            Some(20),
            Some(25),
            Some(30),
            Some(60),
            None,
        ],
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(k)) as ArrayRef,
            Arc::new(Int32Array::from(v)) as ArrayRef,
        ],
    )?;
    s.register_batches("aggr", schema, vec![batch])?;
    // Global and grouped exact percentiles, the `WITHIN GROUP` error
    // cases, and the discrete series (statements 6, 11, 13, 17, 18, 19,
    // 45 of the golden file).
    for case in load(&[6, 11, 13, 17, 18, 19, 45])? {
        check(&s, &case)?;
    }
    Ok(())
}

/// The quantile error classes: each malformed call fails with Spark's
/// class.
#[test]
fn quantile_failures_carry_sparks_classes() -> R {
    let s = Session::new()?;
    for (sql, class) in [
        (
            "SELECT percentile_cont(0.5) FROM (VALUES (1)) AS t(v)",
            "INVALID_WITHIN_GROUP_EXPRESSION.WITHIN_GROUP_MISSING",
        ),
        (
            "SELECT percentile_cont(DISTINCT 0.5) WITHIN GROUP (ORDER BY v) FROM (VALUES (1)) AS t(v)",
            "INVALID_WITHIN_GROUP_EXPRESSION.DISTINCT_UNSUPPORTED",
        ),
        (
            "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY v, v) FROM (VALUES (1)) AS t(v)",
            "INVALID_WITHIN_GROUP_EXPRESSION.WRONG_NUM_ORDERINGS",
        ),
        (
            "SELECT approx_count_distinct(v, 'x') FROM (VALUES (1)) AS t(v)",
            "_LEGACY_ERROR_TEMP_1104",
        ),
    ] {
        let err = block_on(s.sql(sql)).expect_err(sql);
        assert_eq!(err.error_class().name(), class, "{sql}");
    }
    // A negative frequency fails when the rows fold, with Spark's class.
    let df = block_on(s.sql("SELECT percentile(v, 0.5, f) FROM (VALUES (1, -1)) AS t(v, f)"))?;
    let err = block_on(df.collect()).expect_err("negative frequency");
    assert_eq!(
        err.error_class().name(),
        "NEGATIVE_VALUES_IN_FREQUENCY_EXPRESSION"
    );
    Ok(())
}

/// `median` and the percentiles run over a whole partition; an ordered or
/// framed window is refused with Spark's class and the aggregate's text.
#[test]
fn percentile_windows_take_only_the_whole_partition() -> R {
    let s = Session::new()?;
    let data = "(VALUES ('a', 10), ('a', 20), ('a', 40), ('b', 5)) AS t(k, v)";
    let whole = shown(
        &s,
        &format!(
            "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY v) OVER (PARTITION BY k) \
             FROM {data} ORDER BY k, v"
        ),
    )?;
    assert_eq!(whole, ["20", "20", "20", "5"]);
    let median = shown(
        &s,
        &format!("SELECT median(v) OVER (PARTITION BY k) FROM {data} ORDER BY k, v"),
    )?;
    assert_eq!(median, ["20", "20", "20", "5"]);
    for (over, text) in [
        (
            "percentile_disc(0.25) WITHIN GROUP (ORDER BY v DESC) OVER (PARTITION BY k ORDER BY v)",
            "\"percentile_disc(v, 0.25)\"",
        ),
        (
            "median(v) OVER (PARTITION BY k ORDER BY v)",
            "\"median(v)\"",
        ),
        (
            "percentile_cont(0.5) WITHIN GROUP (ORDER BY v) OVER (PARTITION BY k ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING)",
            "\"percentile_cont(v, 0.5)\"",
        ),
    ] {
        let sql = format!("SELECT {over} FROM {data}");
        let err = block_on(s.sql(&sql)).expect_err(&sql);
        assert_eq!(
            err.error_class().name(),
            "INVALID_WINDOW_SPEC_FOR_AGGREGATION_FUNC",
            "{sql}"
        );
        let params = err.message_parameters();
        assert_eq!(params.first().map(|(_, v)| v.as_str()), Some(text), "{sql}");
    }
    Ok(())
}

/// Column 0 of every row of `sql`, rendered: `NULL`, a number, `[a, b]` for a
/// list of doubles, or the decimal text.
fn shown(s: &Session, sql: &str) -> R<Vec<String>> {
    use arrow_array::types::Int64Type;
    use arrow_array::{Decimal64Array, Decimal128Array};
    let df = block_on(s.sql(sql))?;
    let mut out = Vec::new();
    for batch in block_on(df.collect())? {
        let c = batch.column(0);
        for row in 0..c.len() {
            out.push(if c.is_null(row) {
                "NULL".to_owned()
            } else {
                match c.data_type() {
                    DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
                    DataType::Float64 => c.as_primitive::<Float64Type>().value(row).to_string(),
                    DataType::List(_) => {
                        let list = c.as_list::<i32>().value(row);
                        let doubles = list.as_primitive::<Float64Type>();
                        let items: Vec<String> = doubles
                            .iter()
                            .map(|v| v.unwrap_or(f64::NAN).to_string())
                            .collect();
                        format!("[{}]", items.join(", "))
                    }
                    DataType::Decimal128(..) => c
                        .as_any()
                        .downcast_ref::<Decimal128Array>()
                        .map(|a| a.value_as_string(row))
                        .unwrap_or_default(),
                    DataType::Decimal64(..) => c
                        .as_any()
                        .downcast_ref::<Decimal64Array>()
                        .map(|a| a.value_as_string(row))
                        .unwrap_or_default(),
                    other => format!("<{other}>"),
                }
            });
        }
    }
    Ok(out)
}

/// Statements checked on a Spark 4.2.0 run: a `FILTER` on `approx_count_distinct`,
/// NULL (not an empty list) for a group without values or percentages, a
/// decimal that keeps its text, and a NULL frequency.
#[test]
fn quantile_edge_cases_follow_spark() -> R {
    let s = Session::new()?;
    for (sql, want) in [
        (
            "SELECT approx_count_distinct(v) FILTER (WHERE false) FROM VALUES (1),(2) t(v)",
            vec!["0"],
        ),
        (
            "SELECT approx_count_distinct(v) FILTER (WHERE v > 1) FROM VALUES (1),(2),(3) t(v)",
            vec!["2"],
        ),
        (
            "SELECT percentile(v, array(0.5)) FROM VALUES (CAST(NULL AS DOUBLE)) t(v)",
            vec!["NULL"],
        ),
        (
            "SELECT percentile_approx(v, array(0.5)) FROM VALUES (CAST(NULL AS DOUBLE)) t(v)",
            vec!["NULL"],
        ),
        (
            "SELECT percentile(v, array()) FROM VALUES (1.0) t(v)",
            vec!["NULL"],
        ),
        (
            "SELECT percentile_approx(c, 0.5) FROM VALUES (CAST(1.15 AS DECIMAL(10,2))) t(c)",
            vec!["1.15"],
        ),
        (
            "SELECT percentile(v, 0.5, NULL) FROM VALUES (1.0) t(v)",
            vec!["NULL"],
        ),
    ] {
        assert_eq!(shown(&s, sql)?, want, "{sql}");
    }
    // A group of only NULLs is NULL, not an empty list.
    let grouped = "SELECT percentile(v, array(0.5)) FROM \
                   VALUES (1, 1.0), (1, 2.0), (2, CAST(NULL AS DOUBLE)) t(k, v) GROUP BY k ORDER BY k";
    assert_eq!(shown(&s, grouped)?, vec!["[1.5]", "NULL"]);
    Ok(())
}

/// A negative frequency and a too-large `relativeSD` carry Spark's parameters.
#[test]
fn quantile_error_parameters_follow_spark() -> R {
    let s = Session::new()?;
    let param = |e: &KrishivError, key: &str| -> Option<String> {
        e.message_parameters()
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.clone())
    };
    let sql = "SELECT percentile(v, 0.5, -1) FROM VALUES (1.0) t(v)";
    let err = match block_on(s.sql(sql)) {
        Ok(df) => block_on(df.collect()).expect_err("negative frequency"),
        Err(e) => e,
    };
    assert_eq!(
        err.error_class().name(),
        "NEGATIVE_VALUES_IN_FREQUENCY_EXPRESSION"
    );
    assert_eq!(
        param(&err, "frequencyExpression").as_deref(),
        Some("\"-1\"")
    );
    assert_eq!(param(&err, "negativeValue").as_deref(), Some("-1L"));
    let err = block_on(s.sql("SELECT approx_count_distinct(v, 0.5) FROM VALUES (1) t(v)"))
        .expect_err("relativeSD above 39%");
    assert_eq!(err.error_class().name(), "FAILED_FUNCTION_CALL");
    assert_eq!(
        param(&err, "funcName").as_deref(),
        Some("`approx_count_distinct`")
    );
    Ok(())
}

/// `percentile` (unlike `median` and the `percentile_cont/disc` forms)
/// accepts an ordered or framed window, as Spark does: each row sees the
/// percentile of its frame. A running frame freezes its answer per row
/// instead of copying the value map.
#[test]
fn percentile_over_running_and_sliding_frames() -> R {
    let s = Session::new()?;
    let data = "FROM VALUES (1, 1.0), (2, 5.0), (3, 9.0), (4, CAST(NULL AS DOUBLE)) t(id, v)";
    assert_eq!(
        shown(
            &s,
            &format!("SELECT percentile(v, 0.5) OVER (ORDER BY id) {data} ORDER BY id")
        )?,
        vec!["1", "3", "5", "5"]
    );
    assert_eq!(
        shown(
            &s,
            &format!(
                "SELECT percentile(v, 0.5) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND \
                 CURRENT ROW) {data} ORDER BY id"
            )
        )?,
        vec!["1", "3", "7", "9"]
    );
    Ok(())
}
