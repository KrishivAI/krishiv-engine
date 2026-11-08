// SPDX-License-Identifier: Apache-2.0
//! `PIVOT` cells with no matching row against Spark 4.2.0: every case of
//! `tests/fixtures/spark-4.2.0/pivot-empty-cells.json` (real Spark runs, generator
//! `tests/scripts/gen_pivot_empty_cells.py`) runs through a session over the fixture's temporary views and must give
//! Spark's output names, types, nullability and rows (compared as sorted `|`-joined strings, as the generator writes
//! them), or Spark's error class. `KNOWN_GAPS` lists the cases not matched yet; each must still fail, so the list
//! only shrinks.
#![cfg(test)]

use std::fmt::Write as _;
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

/// Cases not matched yet:
/// - pivot columns of complex types (a map is not orderable, an array value is a `array(...)` call): the engine
///   does not take them yet (P2b row 11);
/// - several timestamp values, one of them a string: the analyzer cannot tell whether two values are equal, which
///   decides Spark's result, so the engine refuses them (`docs/explanation/known-differences.md`).
const KNOWN_GAPS: &[&str] = &[
    "err-unorderable-pivot-col",
    "pf-array-pivot-col",
    "pf-ts-dup-cast-value",
    "pf-ts-distinct-values",
];

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/pivot-empty-cells.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Days since the epoch as `yyyy-mm-dd`.
fn date(z: i64) -> String {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{m:02}-{d:02}", yoe + era * 400 + i64::from(m <= 2))
}

/// A float as Python's `str` prints it (`5.0`, `-0.5`, `nan`).
fn float(v: f64) -> String {
    if v.is_nan() {
        "nan".to_owned()
    } else if v.fract() == 0.0 && v.abs() < 1e16 {
        format!("{v:.1}")
    } else {
        v.to_string()
    }
}

/// One value as the generator writes it (Python's `str` of the collected value, `NULL` for null).
fn render(a: &ArrayRef, i: usize) -> String {
    if a.is_null(i) {
        return "NULL".to_owned();
    }
    match a.data_type() {
        DataType::Int8 => a.as_primitive::<Int8Type>().value(i).to_string(),
        DataType::Int16 => a.as_primitive::<Int16Type>().value(i).to_string(),
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Float32 => float(f64::from(a.as_primitive::<Float32Type>().value(i))),
        DataType::Float64 => float(a.as_primitive::<Float64Type>().value(i)),
        DataType::Boolean => if a.as_boolean().value(i) {
            "True"
        } else {
            "False"
        }
        .to_owned(),
        DataType::Decimal128(_, scale) => {
            let v = a.as_primitive::<Decimal128Type>().value(i);
            let scale = u32::try_from(*scale).unwrap();
            let p = 10_i128.pow(scale);
            let sign = if v < 0 { "-" } else { "" };
            if scale == 0 {
                format!("{v}")
            } else {
                format!(
                    "{sign}{}.{:0width$}",
                    (v / p).abs(),
                    (v % p).abs(),
                    width = scale as usize
                )
            }
        }
        DataType::Date32 => date(i64::from(a.as_primitive::<Date32Type>().value(i))),
        DataType::Utf8 => a.as_string::<i32>().value(i).to_owned(),
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
        DataType::List(_) => {
            let inner = a.as_list::<i32>().value(i);
            let parts: Vec<String> = (0..inner.len()).map(|j| render(&inner, j)).collect();
            format!("[{}]", parts.join(", "))
        }
        t => panic!("result type {t}"),
    }
}

/// The fixture's type name of an Arrow result type.
fn spark_type(t: &DataType) -> String {
    match t {
        DataType::Int8 => "tinyint".into(),
        DataType::Int16 => "smallint".into(),
        DataType::Int32 => "int".into(),
        DataType::Int64 => "bigint".into(),
        DataType::Float32 => "float".into(),
        DataType::Float64 => "double".into(),
        DataType::Boolean => "boolean".into(),
        DataType::Decimal128(p, s) => format!("decimal({p},{s})"),
        DataType::Date32 => "date".into(),
        DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => "string".into(),
        DataType::List(f) => format!("array<{}>", spark_type(f.data_type())),
        t => format!("{t}"),
    }
}

fn check(s: &Session, case: &Value, report: &mut String) {
    let id = case["id"].as_str().unwrap();
    let sql = case["sql"].as_str().unwrap();
    let conf = s.conf();
    let settings = case["conf"].as_object().unwrap();
    for (k, v) in settings {
        conf.set(k, v.as_str().unwrap()).unwrap();
    }
    let result = block_on(async {
        let df = s.sql(sql).await?;
        let schema = df.schema().await?;
        let batches = df.collect().await?;
        Ok::<_, krishiv_common::error::KrishivError>((schema, batches))
    });
    for k in settings.keys() {
        conf.unset(k).unwrap();
    }
    if case["ok"].as_bool() != Some(true) {
        let class = case["error_class"].as_str().unwrap();
        match result {
            Err(e) if e.error_class().name() == class => {}
            Err(e) => {
                let _ = writeln!(
                    report,
                    "{id}: error {} vs Spark {class}: {e}",
                    e.error_class().name()
                );
            }
            Ok(_) => {
                let _ = writeln!(report, "{id}: ran; Spark raised {class}");
            }
        }
        return;
    }
    let (schema, batches) = match result {
        Ok(r) => r,
        Err(e) => {
            let _ = writeln!(report, "{id}: {e}");
            return;
        }
    };
    let got_output: Vec<(String, String, bool)> = schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), spark_type(f.data_type()), f.is_nullable()))
        .collect();
    let want_output: Vec<(String, String, bool)> = case["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            (
                o["name"].as_str().unwrap().to_owned(),
                o["type"].as_str().unwrap().to_owned(),
                o["nullable"].as_bool().unwrap(),
            )
        })
        .collect();
    if got_output != want_output {
        let _ = writeln!(
            report,
            "{id}: output\n  got  {got_output:?}\n  want {want_output:?}"
        );
    }
    let mut got: Vec<String> = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| render(c, r)).collect();
            got.push(cells.join("|"));
        }
    }
    got.sort();
    let mut want: Vec<String> = case["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_str().unwrap().to_owned())
        .collect();
    want.sort();
    if got != want {
        let _ = writeln!(report, "{id}: rows\n  got  {got:?}\n  want {want:?}");
    }
}

#[test]
fn pivot_cells_match_spark() {
    let fx = fixture();
    let s = Session::builder().build().unwrap();
    for (name, sql) in fx["views"].as_object().unwrap() {
        block_on(async {
            s.sql(&format!(
                "CREATE TEMPORARY VIEW {name} AS {}",
                sql.as_str().unwrap()
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        });
    }
    let mut report = String::new();
    let cases = fx["cases"].as_array().unwrap();
    for case in cases {
        let mut one = String::new();
        check(&s, case, &mut one);
        let id = case["id"].as_str().unwrap();
        match (KNOWN_GAPS.contains(&id), one.is_empty()) {
            (false, _) => report.push_str(&one),
            (true, true) => {
                let _ = writeln!(report, "{id}: matches Spark now; remove it from KNOWN_GAPS");
            }
            (true, false) => {}
        }
    }
    assert!(
        report.is_empty(),
        "{} cases, mismatches:\n{report}",
        cases.len()
    );
}

/// Rows of `sql` over the fixture's views in a session of `profile`, as the generator writes them.
fn rows_on(
    profile: krishiv_plan::analyzer::SparkProfile,
    sql: &str,
) -> Result<Vec<String>, String> {
    let fx = fixture();
    let s = Session::builder()
        .profile(profile)
        .build()
        .map_err(|e| e.to_string())?;
    block_on(async {
        for (name, view) in fx["views"].as_object().unwrap() {
            s.sql(&format!(
                "CREATE TEMPORARY VIEW {name} AS {}",
                view.as_str().unwrap()
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        }
        let batches = s
            .sql(sql)
            .await
            .map_err(|e| e.to_string())?
            .collect()
            .await
            .map_err(|e| e.to_string())?;
        let mut rows: Vec<String> = Vec::new();
        for b in &batches {
            for r in 0..b.num_rows() {
                rows.push(
                    b.columns()
                        .iter()
                        .map(|c| render(c, r))
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }
        }
        rows.sort();
        Ok(rows)
    })
}

/// Before Spark 4.2, `PivotFirst` keys a binary pivot column's values in a `HashMap` of byte arrays, which compares
/// arrays by identity (`PivotFirst.scala:89` at v3.5.6, v4.0.0, v4.1.0): no row is found, so every cell is NULL, and
/// repeated values are separate keys (no `INVALID_ARRAY_INDEX`). 4.2 matches by content (`pf-binary-pivot-col`).
#[test]
fn binary_pivot_columns_match_by_identity_before_4_2() {
    use krishiv_plan::analyzer::SparkProfile;
    for profile in [SparkProfile::V3_5, SparkProfile::V4_0, SparkProfile::V4_1] {
        assert_eq!(
            rows_on(
                profile,
                "SELECT * FROM b PIVOT (sum(v) FOR p IN (X'01', X'03'))"
            )
            .unwrap(),
            ["1|NULL|NULL", "2|NULL|NULL"],
            "{profile:?}"
        );
        assert_eq!(
            rows_on(
                profile,
                "SELECT * FROM b PIVOT (sum(v) FOR p IN (X'01', X'01'))"
            )
            .unwrap(),
            ["1|NULL|NULL", "2|NULL|NULL"],
            "{profile:?}"
        );
    }
    assert_eq!(
        rows_on(
            SparkProfile::V4_2,
            "SELECT * FROM b PIVOT (sum(v) FOR p IN (X'01', X'03'))"
        )
        .unwrap(),
        ["1|1|NULL", "2|3|NULL"]
    );
}

/// Duplicate values with ANSI off and several aggregates: Spark writes the repeated value into the next aggregate's
/// slots (its last index is past `indexSize`), which the engine refuses instead of returning.
#[test]
fn duplicate_values_with_ansi_off_and_several_aggregates_are_refused() {
    let s = Session::builder().build().unwrap();
    s.conf().set("spark.sql.ansi.enabled", "false").unwrap();
    let e = block_on(async {
        s.sql("SELECT * FROM VALUES (1, 10, 5) AS g(g, p, v) PIVOT (sum(v) AS s, max(v) AS m FOR p IN (10, 20, 10))")
            .await?
            .collect()
            .await
    })
    .unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE", "{e}");
}

/// A NULL value for a binary pivot column: Spark 4.2 compares it with the other values in its `TreeMap` (a
/// `NullPointerException` from source), earlier versions find the NULL rows in their `HashMap` while matching no
/// other binary value by content. Neither is reproduced: the engine refuses it on every profile.
#[test]
fn a_null_value_for_a_binary_pivot_column_is_refused() {
    use krishiv_plan::analyzer::SparkProfile;
    for profile in SparkProfile::ALL {
        let e = rows_on(
            profile,
            "SELECT * FROM b PIVOT (sum(v) FOR p IN (X'01', NULL))",
        )
        .unwrap_err();
        assert!(
            e.contains("KRISHIV_UNSUPPORTED_FEATURE"),
            "{profile:?}: {e}"
        );
    }
}
