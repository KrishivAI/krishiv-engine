// SPDX-License-Identifier: Apache-2.0
//! The Spark 4.3 coverage list (P3 PR-05, the 4.3 coverage list of P3-T090) end to end against a real Spark 4.3.0
//! run: `tests/fixtures/spark-4.3.0/spark43-coverage.json` (`tests/scripts/gen_spark43_coverage_goldens.py`) holds
//! new functions, new syntax, behaviour changes and the correctness fixes that upstream's goldens do not exercise.
//! Each case records what Spark 4.3.0 did: the rows (every cell as Spark's `cast(c as string)`), the column
//! names, or the error class. Cases with setup (temporary views, a table in the session catalog) run in order.
//!
//! A case the engine does not match yet is listed in `PENDING` with how it differs (`"rows"`, `"columns"`,
//! `"no error"`, or the error class the engine raised instead), and the test requires exactly that list, so a fix
//! shrinks it and a regression grows it. Set `KRISHIV_SPARK43_COVERAGE_DUMP=1` to print the current list.
#![cfg(test)]

use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

/// Cases that differ from Spark today: `(id, how)`. They are the 4.3 work the golden tail (plan row GT) takes, by area:
/// new functions (`fn_*`), the nanosecond and TIME(p) types, `@` time travel, `WITH` options on DML, `COMMENT ON COLUMN`,
/// `SHOW ... AS JSON`, and the ANSI overflow of `ceil`/`floor`.
const PENDING: &[(&str, &str)] = &[
    ("fn_collect_union", "UNRESOLVED_ROUTINE"),
    ("fn_collect_union_nulls", "UNRESOLVED_ROUTINE"),
    ("fn_collect_union_empty", "UNRESOLVED_ROUTINE"),
    ("fn_to_base32", "UNRESOLVED_ROUTINE"),
    ("fn_from_base32", "UNRESOLVED_ROUTINE"),
    ("fn_from_base32_invalid", "UNRESOLVED_ROUTINE"),
    ("fn_variant_insert", "UNRESOLVED_ROUTINE"),
    ("fn_variant_insert_nested_missing", "UNRESOLVED_ROUTINE"),
    ("fn_try_variant_insert", "UNRESOLVED_ROUTINE"),
    ("fn_variant_array_append", "UNRESOLVED_ROUTINE"),
    ("fn_variant_array_append_nested", "UNRESOLVED_ROUTINE"),
    ("fn_try_variant_array_append", "UNRESOLVED_ROUTINE"),
    ("fn_variant_strip_nulls", "UNRESOLVED_ROUTINE"),
    ("fn_variant_strip_nulls_scalar", "UNRESOLVED_ROUTINE"),
    ("fn_try_variant_set", "UNRESOLVED_ROUTINE"),
    ("fn_localtime", "UNRESOLVED_ROUTINE"),
    ("fn_localtime_precision_error", "UNRESOLVED_ROUTINE"),
    ("fn_time_disabled", "UNSUPPORTED_TIME_TYPE"),
    ("fn_time_precision_9_disabled", "UNSUPPORTED_TIME_PRECISION"),
    (
        "fn_timestamp_nanos_disabled",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    ("fn_time_precision_9", "UNSUPPORTED_TIME_PRECISION"),
    ("fn_time_typeof_precisions", "UNSUPPORTED_TIME_PRECISION"),
    (
        "fn_timestamp_ntz_to_time",
        "DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION",
    ),
    ("fn_timestamp_ntz_to_time_9", "UNSUPPORTED_TIME_PRECISION"),
    ("fn_time_to_timestamp_ntz_nanos", "UNSUPPORTED_DATATYPE"),
    (
        "fn_current_timestamp_precision",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    (
        "fn_current_timestamp_precision_9",
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
    ),
    ("fn_localtime_9", "UNRESOLVED_ROUTINE"),
    ("beh_ceil_double_overflow", "no error"),
    ("beh_floor_double_overflow", "no error"),
    ("fix_pivot_duplicate_values", "INVALID_ARRAY_INDEX"),
    (
        "fix_insert_struct_by_position_legacy",
        "setup: INCOMPATIBLE_DATA_FOR_TABLE.CANNOT_SAFELY_CAST",
    ),
    ("syn_comment_on_column", "PARSE_SYNTAX_ERROR"),
    (
        "syn_comment_on_column_describe",
        "setup: PARSE_SYNTAX_ERROR",
    ),
    ("syn_comment_on_column_missing", "PARSE_SYNTAX_ERROR"),
    ("syn_show_tables_as_json", "PARSE_SYNTAX_ERROR"),
    ("syn_show_table_extended_as_json", "PARSE_SYNTAX_ERROR"),
    ("syn_show_partitions_as_json", "PARSE_SYNTAX_ERROR"),
    ("syn_time_travel_at_version", "PARSE_SYNTAX_ERROR"),
    ("syn_time_travel_at_timestamp", "PARSE_SYNTAX_ERROR"),
    ("syn_time_travel_clause_and_at", "PARSE_SYNTAX_ERROR"),
    ("syn_insert_replace_where", "KRISHIV_UNSUPPORTED_FEATURE"),
    ("syn_insert_replace_where_columns", "PARSE_SYNTAX_ERROR"),
    ("syn_update_with_options", "PARSE_SYNTAX_ERROR"),
    ("syn_delete_with_options", "PARSE_SYNTAX_ERROR"),
    ("syn_merge_with_options", "PARSE_SYNTAX_ERROR"),
];

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.3.0/spark43-coverage.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("valid JSON")
}

fn cell(a: &ArrayRef, i: usize) -> String {
    if a.is_null(i) {
        return "NULL".to_owned();
    }
    match a.data_type() {
        DataType::Utf8 => a.as_string::<i32>().value(i).to_owned(),
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
        t => format!("<{t}>"),
    }
}

/// The rows of the batches as text, `NULL` for a null.
fn rows_of(batches: &[arrow_array::RecordBatch]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            rows.push(b.columns().iter().map(|c| cell(c, i)).collect());
        }
    }
    rows
}

/// Empty when the case matches Spark, else how it differs.
fn run(case: &Value) -> String {
    let s = Session::new().expect("session");
    let setup = |stmt: &str| block_on(async { s.sql(stmt).await?.collect().await });
    let _ = block_on(async {
        s.sql("SET spark.sql.ansi.enabled=true")
            .await?
            .collect()
            .await
    });
    for stmt in case["setup"].as_array().expect("setup") {
        if let Err(e) = setup(stmt.as_str().expect("statement")) {
            // Spark failed its own setup too (a recorded `setup_error_class`): the case is then that error.
            return match case.get("setup_error_class").and_then(Value::as_str) {
                Some(want) if want == e.error_class().name() => String::new(),
                _ => format!("setup: {}", e.error_class().name()),
            };
        }
    }
    if case.get("setup_error_class").is_some() {
        return "setup: no error".to_owned();
    }
    let sql = case["sql"].as_str().expect("sql");
    if let Some(want) = case.get("error_class").and_then(Value::as_str) {
        return match setup(sql) {
            Ok(_) => "no error".to_owned(),
            Err(e) if e.error_class().name() == want => String::new(),
            Err(e) => e.error_class().name().to_owned(),
        };
    }
    if let Some(want) = case.get("columns").and_then(Value::as_array) {
        let df = match block_on(s.sql(sql)) {
            Ok(df) => df,
            Err(e) => return e.error_class().name().to_owned(),
        };
        let schema = match block_on(df.schema()) {
            Ok(schema) => schema,
            Err(e) => return e.error_class().name().to_owned(),
        };
        let got: Vec<String> = schema
            .fields()
            .iter()
            .map(|f| f.name().to_owned())
            .collect();
        let want: Vec<String> = want
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_owned())
            .collect();
        return if got == want {
            String::new()
        } else {
            "columns".to_owned()
        };
    }
    // Rows. A query is wrapped so every cell is Spark's text of the value; a command is run as it is.
    let ncols = usize::try_from(case["ncols"].as_u64().expect("ncols")).expect("small");
    let wrapped = case["wrapped"].as_bool().unwrap_or(true);
    let query = if ncols == 0 || !wrapped {
        sql.to_owned()
    } else {
        let names: Vec<String> = (0..ncols).map(|i| format!("c{i}")).collect();
        let casts: Vec<String> = names
            .iter()
            .map(|n| format!("cast({n} as string)"))
            .collect();
        format!(
            "select {} from ({sql}) as q({})",
            casts.join(", "),
            names.join(", ")
        )
    };
    let batches = match setup(&query) {
        Ok(b) => b,
        Err(e) => return e.error_class().name().to_owned(),
    };
    let mut got = rows_of(&batches);
    let mut want: Vec<Vec<String>> = case["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .map(|r| {
            r.as_array()
                .expect("row")
                .iter()
                .map(|v| v.as_str().unwrap_or("NULL").to_owned())
                .collect()
        })
        .collect();
    if !sql.to_lowercase().contains(" order by ") {
        got.sort();
        want.sort();
    }
    if got == want {
        String::new()
    } else {
        "rows".to_owned()
    }
}

#[test]
fn spark_4_3_coverage_list() {
    let doc = fixture();
    assert_eq!(doc["spark_version"], "4.3.0");
    let cases = doc["cases"].as_array().expect("cases");
    assert!(cases.len() > 70, "the fixture is the whole list");
    let mut failing = Vec::new();
    for case in cases {
        let how = run(case);
        if !how.is_empty() {
            failing.push((case["id"].as_str().expect("id").to_owned(), how));
        }
    }
    if std::env::var_os("KRISHIV_SPARK43_COVERAGE_DUMP").is_some() {
        for (id, how) in &failing {
            eprintln!("    (\"{id}\", \"{how}\"),");
        }
    }
    let want: Vec<(String, String)> = PENDING
        .iter()
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect();
    assert_eq!(failing, want, "cases failing today");
}
