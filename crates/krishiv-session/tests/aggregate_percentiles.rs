// SPDX-License-Identifier: Apache-2.0
//! The `percentiles.sql` statements over `aggr` (classed
//! `depends-on-command` in the golden baseline, since the file creates the
//! view first): the same data registered through the Rust API, asserting
//! the golden's rows and error class for each statement this change
//! covers. `FILTER`, window and interval statements are not covered (the
//! engine refuses `FILTER` and window percentile aggregates, and has no
//! interval type).
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
