// SPDX-License-Identifier: Apache-2.0
//! The P2-T180 function-call forms end to end: the Spark 4.2.0 goldens
//! `named-function-arguments.sql` (the builtin-function statements),
//! `like-any.sql`, `like-all.sql`, `ilike.sql`, `ilike-any.sql` and
//! `ilike-all.sql`, replayed through a session.
//!
//! Statements that build their tables with `CREATE TEMPORARY VIEW ... AS
//! VALUES` run against equivalent tables registered with
//! `Session::register_batches` (another change adds `VALUES`); the UDTF,
//! Python-UDF and `count_min_sketch` statements are out of scope and are
//! not replayed. Row order without an `ORDER BY` is unspecified, so rows
//! compare as sets; schemas (including generated names), error classes and
//! message parameters compare exactly.
#![cfg(test)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use serde_json::Value;

/// One golden statement and its expected outcome.
struct Case {
    sql: String,
    schema: Option<String>,
    output: Outcome,
}

enum Outcome {
    Rows(Vec<String>),
    Error {
        class: String,
        params: BTreeMap<String, String>,
    },
}

/// The golden `.out` file parsed into statements.
fn golden(name: &str) -> Vec<Case> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../tests/spark-golden/v4.2.0/results/{name}.out"
    ));
    let text = std::fs::read_to_string(&path).unwrap();
    let mut cases = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line != "-- !query" {
            continue;
        }
        let mut sql = Vec::new();
        for line in lines.by_ref() {
            if line == "-- !query schema" {
                break;
            }
            sql.push(line);
        }
        let mut schema = Vec::new();
        for line in lines.by_ref() {
            if line == "-- !query output" {
                break;
            }
            schema.push(line);
        }
        let mut output = Vec::new();
        while let Some(line) = lines.peek() {
            if *line == "-- !query" {
                break;
            }
            output.push(lines.next().unwrap());
        }
        let schema = schema.join("\n");
        let schema = schema
            .strip_prefix("struct<")
            .and_then(|s| s.strip_suffix(">"));
        let output_text = output.join("\n");
        let output = output_text.trim();
        let outcome = if output.starts_with("org.apache.spark") {
            let json_start = output.find('{').unwrap();
            let error: Value = serde_json::from_str(&output[json_start..]).unwrap();
            let params = error["messageParameters"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
                .collect();
            Outcome::Error {
                class: error["errorClass"].as_str().unwrap().to_owned(),
                params,
            }
        } else if output.is_empty() {
            Outcome::Rows(Vec::new())
        } else {
            Outcome::Rows(output.lines().map(str::to_owned).collect())
        };
        cases.push(Case {
            sql: sql.join("\n"),
            schema: schema.map(str::to_owned),
            output: outcome,
        });
    }
    cases
}

fn strings_table(s: &Session, name: &str, rows: &[(&str, &str)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("company", ArrowType::Utf8, true),
        Field::new("pat", ArrowType::Utf8, true),
    ]));
    let (companies, pats): (Vec<_>, Vec<_>) = rows.iter().copied().unzip();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(companies)),
            Arc::new(StringArray::from(pats)),
        ],
    )
    .unwrap();
    s.register_batches(name, schema, vec![batch]).unwrap();
}

fn session() -> Session {
    let s = Session::new().unwrap();
    strings_table(
        &s,
        "like_any_table",
        &[
            ("google", "%oo%"),
            ("facebook", "%oo%"),
            ("linkedin", "%in"),
        ],
    );
    strings_table(
        &s,
        "like_all_table",
        &[
            ("google", "%oo%"),
            ("facebook", "%oo%"),
            ("linkedin", "%in"),
        ],
    );
    strings_table(
        &s,
        "ilike_any_table",
        &[
            ("Google", "%Oo%"),
            ("FaceBook", "%oO%"),
            ("linkedIn", "%IN"),
        ],
    );
    strings_table(
        &s,
        "ilike_all_table",
        &[
            ("gOOgle", "%oo%"),
            ("facebook", "%OO%"),
            ("liNkedin", "%In"),
        ],
    );
    s
}

fn cell(column: &arrow_array::ArrayRef, i: usize) -> String {
    if column.is_null(i) {
        return "NULL".to_owned();
    }
    match column.data_type() {
        ArrowType::Boolean => column.as_boolean().value(i).to_string(),
        ArrowType::Utf8View => column.as_string_view().value(i).to_owned(),
        ArrowType::LargeUtf8 => column.as_string::<i64>().value(i).to_owned(),
        _ => column.as_string::<i32>().value(i).to_owned(),
    }
}

fn arrow_simple(t: &ArrowType) -> &str {
    match t {
        ArrowType::Utf8 | ArrowType::Utf8View | ArrowType::LargeUtf8 => "string",
        ArrowType::Boolean => "boolean",
        _ => panic!("unexpected type {t}"),
    }
}

/// Whether `sql` is replayed: `VALUES`-built tables (registered instead),
/// UDTF/generator statements, and `count_min_sketch` (unimplemented).
fn replayed(file: &str, sql: &str) -> bool {
    if sql.starts_with("CREATE OR REPLACE TEMPORARY VIEW")
        || sql.starts_with("create temporary view")
        || sql.starts_with("CREATE TEMPORARY VIEW")
    {
        return false;
    }
    if file == "named-function-arguments.sql"
        && (sql.contains("explode")
            || sql.contains("inline")
            || sql.contains("posexplode")
            || sql.contains("count_min_sketch")
            || sql.starts_with("SELECT * FROM"))
    {
        return false;
    }
    true
}

fn check(s: &Session, file: &str, case: &Case) -> String {
    if !replayed(file, &case.sql) {
        return String::new();
    }
    let result = block_on(async {
        let df = s.sql(&case.sql).await?;
        let schema = df.schema().await?;
        let batches = df.collect().await?;
        Ok::<_, KrishivError>((schema, batches))
    });
    match (&case.output, result) {
        (Outcome::Rows(want), Ok((schema, batches))) => {
            let mut out = String::new();
            let fields: Vec<String> = schema
                .fields()
                .iter()
                .map(|f| format!("{}:{}", f.name(), arrow_simple(f.data_type())))
                .collect();
            let got_schema = format!("<{}>", fields.join(","));
            if Some(got_schema.clone()) != case.schema.clone().map(|s| format!("<{s}>")) {
                out.push_str(&format!(
                    "{}: schema {got_schema:?} vs golden {:?}\n",
                    case.sql, case.schema
                ));
            }
            let mut got: Vec<String> = Vec::new();
            for b in &batches {
                for i in 0..b.num_rows() {
                    got.push(
                        b.columns()
                            .iter()
                            .map(|c| cell(c, i))
                            .collect::<Vec<_>>()
                            .join("\t"),
                    );
                }
            }
            got.sort();
            let mut want = want.clone();
            want.sort();
            if got != want {
                out.push_str(&format!("{}: rows {got:?} vs golden {want:?}\n", case.sql));
            }
            out
        }
        (Outcome::Error { class, params }, Err(e)) => {
            let mut out = String::new();
            if e.error_class().name() != *class {
                out.push_str(&format!(
                    "{}: class {} vs golden {class}\n",
                    case.sql,
                    e.error_class().name()
                ));
            }
            let got: BTreeMap<String, String> = e
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect();
            if &got != params {
                out.push_str(&format!(
                    "{}: params {got:?} vs golden {params:?}\n",
                    case.sql
                ));
            }
            out
        }
        (Outcome::Rows(_), Err(e)) => format!("{}: unexpected error {e}\n", case.sql),
        (Outcome::Error { class, .. }, Ok(_)) => {
            format!("{}: expected error {class}, succeeded\n", case.sql)
        }
    }
}

#[test]
fn function_call_forms_match_spark() {
    let s = session();
    let mut failures = String::new();
    let mut count = 0;
    for file in [
        "named-function-arguments.sql",
        "like-any.sql",
        "like-all.sql",
        "ilike.sql",
        "ilike-any.sql",
        "ilike-all.sql",
    ] {
        for case in golden(file) {
            if replayed(file, &case.sql) {
                count += 1;
            }
            failures.push_str(&check(&s, file, &case));
        }
    }
    // Every in-scope statement replays: 4 mask calls and 7 mask errors, and
    // every `like`/`ilike` statement but the `VALUES`-built table.
    assert_eq!(count, 89, "skipped statements changed");
    // The session-qualified forms Spark accepts fail as unresolved routines
    // (no temporary functions exist), while `builtin.` and
    // `system.builtin.` resolve.
    for sql in [
        "SELECT builtin.abs(-1)",
        "SELECT system.builtin.abs(-1)",
        "SELECT BUILTIN.ABS(-1)",
        "SELECT System.Builtin.Abs(-1)",
    ] {
        let rows = block_on(async {
            let df = s.sql(sql).await?;
            df.collect().await
        })
        .unwrap();
        assert_eq!(rows.len(), 1, "{sql}");
    }
    for sql in ["SELECT session.abs(-1)", "SELECT system.session.abs(-1)"] {
        let e = block_on(s.sql(sql)).unwrap_err();
        assert_eq!(e.error_class().name(), "UNRESOLVED_ROUTINE", "{sql}: {e}");
    }
    assert!(failures.is_empty(), "\n{failures}");
}

/// Spark 4.2: `*` expands to positional arguments, so a named argument
/// after it fills the parameters from the expanded count on, and naming a
/// parameter the expansion already covers is an error.
#[test]
fn named_arguments_after_star_follow_the_expanded_count() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", ArrowType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["Ab1"]))],
    )
    .unwrap();
    s.register_batches("one_col", schema, vec![batch]).unwrap();
    strings_table(&s, "two_cols", &[("Ab1", "xY")]);
    let first = |sql: &str| {
        let rows = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
        cell(rows[0].column(0), 0)
    };
    assert_eq!(
        first("SELECT mask(*, upperChar => 'Q') FROM one_col"),
        "Qxn"
    );
    assert_eq!(
        first("SELECT mask(*, upperChar => 'Q', digitChar => 'D') FROM one_col"),
        "QxD"
    );
    for sql in [
        "SELECT mask(*, upperChar => 'Q') FROM two_cols",
        "SELECT mask(*, lowerChar => 'q', upperChar => 'Q') FROM two_cols",
    ] {
        let e = block_on(s.sql(sql)).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "DUPLICATE_ROUTINE_PARAMETER_ASSIGNMENT.BOTH_POSITIONAL_AND_NAMED",
            "{sql}: {e}"
        );
    }
}

/// Spark 4.2: a `builtin.` qualifier does not make an unknown function
/// resolve; the routine is named with its qualifier.
#[test]
fn qualified_unknown_builtin_is_an_unresolved_routine() {
    let s = Session::new().unwrap();
    let e = block_on(s.sql("SELECT builtin.nope(1)")).unwrap_err();
    assert_eq!(e.error_class().name(), "UNRESOLVED_ROUTINE", "{e}");
    assert!(
        e.message_parameters()
            .iter()
            .any(|(k, v)| *k == "routineName" && v == "`builtin`.`nope`"),
        "{:?}",
        e.message_parameters()
    );
}
