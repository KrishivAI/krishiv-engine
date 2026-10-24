// SPDX-License-Identifier: Apache-2.0
//! Spark 4.2 `SQLQueryTestSuite` goldens for `PIVOT` and `UNPIVOT`: each
//! replayable statement of the vendored `pivot` and `unpivot` result files
//! runs through a session and its schema (including the generated pivot
//! column names), rows (compared as sets of rows, as Spark's suite sorts
//! rows of queries without `ORDER BY`) or error class and message
//! parameters must match.
//!
//! The golden files create their inputs with `CREATE TEMPORARY VIEW ...
//! AS VALUES`, which the engine does not run yet; the driver registers
//! equivalent tables instead and skips the `CREATE` statements. Statements
//! pivoting on several columns need struct values, which the engine cannot
//! value yet: they fail with `KRISHIV_UNSUPPORTED_FEATURE` by design (never
//! a wrong result), and statements reading array, map or struct columns
//! cannot run at all (the engine has no complex types), so those are
//! skipped with their reason.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// One golden statement: the query and its expected schema and output.
struct Case {
    query: String,
    schema: String,
    output: String,
}

fn parse_cases(text: &str, file: &str) -> R<Vec<Case>> {
    let total = text.split("-- !query\n").skip(1).count();
    let cases: Vec<Case> = text
        .split("-- !query\n")
        .skip(1)
        .filter_map(|block| {
            let (query, rest) = block.split_once("\n-- !query schema\n")?;
            let (schema, output) = rest.split_once("\n-- !query output\n")?;
            Some(Case {
                query: query.trim().to_owned(),
                schema: schema.trim().to_owned(),
                output: output.trim_end().to_owned(),
            })
        })
        .collect();
    if cases.is_empty() {
        return Err(format!("no statements in {file}").into());
    }
    if cases.len() != total {
        return Err(format!(
            "dropped {} of {total} statements in {file}",
            total - cases.len()
        )
        .into());
    }
    Ok(cases)
}

fn load(file: &str) -> R<Vec<Case>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/spark-golden/v4.2.0/results")
        .join(file);
    let text = std::fs::read_to_string(&path)?;
    parse_cases(&text, file)
}

fn spark_type(t: &DataType) -> String {
    match t {
        DataType::Int8 => "tinyint".into(),
        DataType::Int16 => "smallint".into(),
        DataType::Int32 => "int".into(),
        DataType::Int64 => "bigint".into(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "string".into(),
        DataType::Boolean => "boolean".into(),
        DataType::Float64 => "double".into(),
        other => format!("{other}"),
    }
}

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        DataType::LargeUtf8 => c.as_string::<i64>().value(row).to_owned(),
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        DataType::Float64 => format!("{:?}", c.as_primitive::<Float64Type>().value(row)),
        other => format!("<{other}>"),
    }
}

fn render(schema: &Schema, batches: &[RecordBatch]) -> (String, Vec<String>) {
    let schema = schema
        .fields()
        .iter()
        .map(|f| format!("{}:{}", f.name(), spark_type(f.data_type())))
        .collect::<Vec<_>>()
        .join(",");
    let rows = batches
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
    (format!("struct<{schema}>"), rows)
}

fn strings(values: Vec<Option<&str>>) -> ArrayRef {
    // The engine's canonical string layout is `Utf8View` (what literals
    // evaluate to); a `Utf8` column would trip the equality kernels.
    Arc::new(arrow_array::StringViewArray::from(values))
}

fn ints(values: Vec<Option<i32>>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

fn register(s: &Session, name: &str, columns: Vec<(&str, ArrayRef, bool)>) -> R {
    let fields = columns
        .iter()
        .map(|(name, array, nullable)| Field::new(*name, array.data_type().clone(), *nullable))
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let arrays = columns.into_iter().map(|(_, array, _)| array).collect();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    s.register_batches(name, schema, vec![batch])?;
    Ok(())
}

/// The session with the golden inputs registered: `courseSales`, `years`,
/// `courseEarnings` and `courseEarningsAndSales`.
fn session() -> R<Session> {
    let s = Session::new()?;
    register(
        &s,
        "courseSales",
        vec![
            (
                "course",
                strings(vec![
                    Some("dotNET"),
                    Some("Java"),
                    Some("dotNET"),
                    Some("dotNET"),
                    Some("Java"),
                ]),
                false,
            ),
            (
                "year",
                ints(vec![
                    Some(2012),
                    Some(2012),
                    Some(2012),
                    Some(2013),
                    Some(2013),
                ]),
                false,
            ),
            (
                "earnings",
                ints(vec![
                    Some(10000),
                    Some(20000),
                    Some(5000),
                    Some(48000),
                    Some(30000),
                ]),
                false,
            ),
        ],
    )?;
    register(
        &s,
        "years",
        vec![
            ("y", ints(vec![Some(2012), Some(2013)]), false),
            ("s", ints(vec![Some(1), Some(2)]), false),
        ],
    )?;
    register(
        &s,
        "courseEarnings",
        vec![
            ("course", strings(vec![Some("dotNET"), Some("Java")]), false),
            ("2012", ints(vec![Some(15000), Some(20000)]), false),
            ("2013", ints(vec![Some(48000), Some(30000)]), false),
            ("2014", ints(vec![Some(22500), None]), true),
        ],
    )?;
    register(
        &s,
        "courseEarningsAndSales",
        vec![
            ("course", strings(vec![Some("dotNET"), Some("Java")]), false),
            ("earnings2012", ints(vec![Some(15000), Some(20000)]), false),
            ("sales2012", ints(vec![None, Some(1)]), true),
            ("earnings2013", ints(vec![Some(48000), Some(30000)]), false),
            ("sales2013", ints(vec![Some(1), Some(2)]), false),
            ("earnings2014", ints(vec![Some(22500), None]), true),
            ("sales2014", ints(vec![Some(1), None]), true),
        ],
    )?;
    Ok(s)
}

/// The error class and message parameters a golden expects, from its JSON
/// error body (`None` for rows).
fn expected_error(output: &str) -> Option<(String, BTreeMap<String, String>)> {
    let class = output
        .lines()
        .find_map(|l| l.trim().strip_prefix("\"errorClass\" : \""))
        .map(|l| l.trim_end_matches(['"', ',']).to_owned())?;
    let params = output
        .split_once("\"messageParameters\"")
        .and_then(|(_, tail)| {
            let start = tail.find('{')?;
            // The parameters object ends where its braces balance (a
            // `queryContext` array may follow).
            let mut depth = 0;
            let mut in_string = false;
            let mut end = None;
            for (i, b) in tail.as_bytes().iter().enumerate().skip(start) {
                match b {
                    b'"' => in_string = !in_string,
                    b'{' if !in_string => depth += 1,
                    b'}' if !in_string => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            serde_json::from_str::<BTreeMap<String, String>>(&tail[start..=end?]).ok()
        })
        .unwrap_or_default();
    Some((class, params))
}

fn error_params(e: &KrishivError) -> BTreeMap<String, String> {
    e.message_parameters()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).clone()))
        .collect()
}

/// What a skipped statement needs that the engine lacks.
fn skip_reason(file: &str, query: &str) -> Option<&'static str> {
    if query.to_ascii_uppercase().starts_with("CREATE") {
        return Some("inputs are registered instead");
    }
    if file == "pivot.sql.out" {
        // Several pivot columns stand for struct values, which the engine
        // cannot value yet.
        for marker in [
            "FOR (course, year) IN (('dotNET', 2012), ('Java', 2013))",
            "FOR (course, s) IN (('dotNET', 2) as c1, ('Java', 1) as c2)",
            "FOR (y, course) IN ((2012, 'dotNET'), (2013, 'Java'))",
            "FOR (course, a) IN (('dotNET', array(1, 1)), ('Java', array(2, 2)))",
            "FOR (course, s) IN (('dotNET', (1, 'a')), ('Java', (2, 'b')))",
            "FOR (course, m) IN (('dotNET', map('1', 1)), ('Java', map('2', 2)))",
        ] {
            if query.contains(marker) {
                return Some("multi-column PIVOT needs struct values");
            }
        }
        // Array, struct and map columns: the engine has no complex types.
        for marker in [
            "min(a)\n  FOR course",
            "FOR a IN (array(1, 1), array(2, 2))",
            "FOR s IN ((1, 'a'), (2, 'b'))",
            "FOR m IN (map('1', 1), map('2', 2))",
        ] {
            if query.contains(marker) {
                return Some("needs a complex-typed column");
            }
        }
    }
    None
}

/// Statements pivoting on several columns with well-typed values: by
/// design they fail with `KRISHIV_UNSUPPORTED_FEATURE`, never a wrong
/// result.
fn expect_unsupported(file: &str, query: &str) -> bool {
    file == "pivot.sql.out"
        && (query.contains("FOR (course, year) IN (('dotNET', 2012), ('Java', 2013))")
            || query.contains("FOR (course, s) IN (('dotNET', 2) as c1, ('Java', 1) as c2)"))
}

/// Run every statement of `file`; return the statements that disagree with
/// Spark, and the skipped ones with their reason.
fn run(file: &str) -> R<(Vec<String>, Vec<String>)> {
    let s = session()?;
    let mut bad = Vec::new();
    let mut skipped = Vec::new();
    for case in load(file)? {
        if let Some(reason) = skip_reason(file, &case.query) {
            if expect_unsupported(file, &case.query) {
                match block_on(s.sql(&case.query)) {
                    Err(e) => {
                        if e.error_class().name() != "KRISHIV_UNSUPPORTED_FEATURE" {
                            bad.push(format!("{}\n  error {e}", case.query));
                        }
                    }
                    Ok(df) => match block_on(df.collect()) {
                        Err(e) => {
                            if e.error_class().name() != "KRISHIV_UNSUPPORTED_FEATURE" {
                                bad.push(format!("{}\n  error {e}", case.query));
                            }
                        }
                        Ok(_) => bad.push(format!(
                            "{}\n  expected KRISHIV_UNSUPPORTED_FEATURE",
                            case.query
                        )),
                    },
                }
                continue;
            }
            skipped.push(format!(
                "{}\n  ({reason})",
                case.query.lines().next().unwrap_or("")
            ));
            continue;
        }
        if case.schema == "struct<>" {
            let Some((class, params)) = expected_error(&case.output) else {
                bad.push(format!("{}\n  no error body", case.query));
                continue;
            };
            let result =
                block_on(s.sql(&case.query)).and_then(|df| block_on(df.collect()).map(|_| ()));
            match result {
                Err(e) if e.error_class().name() == class && error_params(&e) == params => {}
                Err(e) => bad.push(format!(
                    "{}\n  error {} params {:?} want {class} {params:?}",
                    case.query,
                    e.error_class().name(),
                    error_params(&e),
                )),
                Ok(()) => bad.push(format!("{}\n  expected error {class}", case.query)),
            }
            continue;
        }
        let result = block_on(async {
            let df = s.sql(&case.query).await?;
            let schema = df.schema().await?;
            Ok::<_, KrishivError>((schema, df.collect().await?))
        });
        match result {
            Err(e) => bad.push(format!("{}\n  error {e}", case.query)),
            Ok((schema, batches)) => {
                let (schema, mut rows) = render(&schema, &batches);
                let mut want: Vec<String> = case.output.lines().map(str::to_owned).collect();
                // Spark's suite sorts rows of queries without `ORDER BY`.
                rows.sort();
                want.sort();
                if schema != case.schema || rows != want {
                    bad.push(format!(
                        "{}\n  schema {schema} want {}\n  rows {rows:?}\n  want {want:?}",
                        case.query, case.schema
                    ));
                }
            }
        }
    }
    Ok((bad, skipped))
}

#[test]
fn pivot_matches_spark() {
    let (bad, skipped) = run("pivot.sql.out").unwrap();
    assert!(
        bad.is_empty(),
        "{} mismatches:\n{}",
        bad.len(),
        bad.join("\n")
    );
    // The complex-type statements the engine cannot run yet.
    assert_eq!(skipped.len(), 11, "{skipped:?}");
}

#[test]
fn unpivot_matches_spark() {
    let (bad, skipped) = run("unpivot.sql.out").unwrap();
    assert!(
        bad.is_empty(),
        "{} mismatches:\n{}",
        bad.len(),
        bad.join("\n")
    );
    assert_eq!(skipped.len(), 2, "{skipped:?}");
}
