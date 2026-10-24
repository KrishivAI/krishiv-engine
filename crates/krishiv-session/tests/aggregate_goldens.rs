// SPDX-License-Identifier: Apache-2.0
//! Spark 4.2 `SQLQueryTestSuite` goldens for the simple aggregates
//! (`every`, `some`, `any`, `bool_and`, `bool_or`, `any_value`): each claimed
//! statement of the vendored `group-by` and `window` result files runs
//! through a session over the tables the file creates, and its schema, rows
//! (in order when the statement orders them) or error class, SQLSTATE and
//! message parameters must be Spark's.
//!
//! Claimed: the `group-by.sql` statements on `test_agg` (the boolean
//! aggregates grouped, global, with `HAVING`, in subqueries, in windows and
//! with wrong argument types, but not `every("true")`, which needs a string
//! to boolean cast) and the `window.sql` statement of
//! `first_value`/`any_value`/`last_value` over `()`.
//!
//! Not claimed (they need other functions or syntax): the other `group-by`
//! statements, which test `count`, `sum`, maps and arrays, and the
//! `first`/`last`/`any_value` ones over `t1` (they use `GROUP BY ALL` and a
//! correlated `EXISTS`);
//! the `window.sql` statements that also use `nth_value`, `lead`, `lag`,
//! `stddev`, `collect_list` and the like; the `try_aggregates.sql` ones
//! (`try_sum`, `try_avg`); every `max-min-by-k.sql` statement (all use the
//! three-argument form, which returns an array).
#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use serde_json::Value;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// One golden statement: the query and its expected schema and output.
struct Case {
    query: String,
    schema: String,
    output: String,
}

fn load(file: &str) -> R<Vec<Case>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/spark-golden/v4.2.0/results")
        .join(file);
    let text = std::fs::read_to_string(path)?;
    let blocks: Vec<&str> = text.split("-- !query\n").skip(1).collect();
    let cases: Vec<Case> = blocks
        .iter()
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
    if cases.len() != blocks.len() {
        return Err(format!("dropped statements of {file}").into());
    }
    Ok(cases)
}

/// The tuples of a `VALUES` list as text fields (top-level commas only).
fn tuples(values: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let (mut depth, mut quote) = (0usize, None::<char>);
    let mut field = String::new();
    let mut row: Vec<String> = Vec::new();
    for ch in values.chars() {
        match (quote, ch) {
            (Some(q), c) => {
                field.push(c);
                if c == q {
                    quote = None;
                }
            }
            (None, '"' | '\'') => {
                field.push(ch);
                quote = Some(ch);
            }
            (None, '(') => {
                depth += 1;
                if depth > 1 {
                    field.push(ch);
                }
            }
            (None, ')') => {
                depth -= 1;
                if depth == 0 {
                    row.push(field.trim().to_owned());
                    field.clear();
                    rows.push(std::mem::take(&mut row));
                } else {
                    field.push(ch);
                }
            }
            (None, ',') if depth == 1 => {
                row.push(field.trim().to_owned());
                field.clear();
            }
            (None, c) if depth >= 1 => field.push(c),
            _ => {}
        }
    }
    rows
}

/// Registers the view `name` the setup statement `create` makes, with the
/// columns `keep` (indices of its column list) typed from their values.
fn register(s: &Session, create: &str, name: &str, keep: &[usize]) -> R {
    let from = create.find("FROM VALUES").ok_or("no VALUES")? + "FROM VALUES".len();
    let as_at = create.rfind(&format!("AS {name}(")).ok_or("no alias")?;
    let columns: Vec<&str> = create[as_at..]
        .split_once('(')
        .and_then(|(_, c)| c.split_once(')'))
        .ok_or("no column list")?
        .0
        .split(',')
        .map(str::trim)
        .collect();
    let rows = tuples(&create[from..as_at]);
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for &c in keep {
        let col: Vec<Option<&str>> = rows
            .iter()
            .map(|r| r.get(c).map(String::as_str).filter(|v| *v != "null"))
            .collect();
        let nullable = col.iter().any(Option::is_none);
        let name = columns.get(c).ok_or("column out of range")?;
        let (t, a): (DataType, ArrayRef) =
            if col.iter().flatten().all(|v| *v == "true" || *v == "false") {
                let a = BooleanArray::from_iter(col.iter().map(|v| v.map(|v| v == "true")));
                (DataType::Boolean, Arc::new(a))
            } else if col.iter().flatten().all(|v| v.parse::<i32>().is_ok()) {
                let a = Int32Array::from_iter(col.iter().map(|v| v.and_then(|v| v.parse().ok())));
                (DataType::Int32, Arc::new(a))
            } else {
                let a = StringArray::from_iter(
                    col.iter().map(|v| v.map(|v| v.trim_matches(['"', '\'']))),
                );
                (DataType::Utf8, Arc::new(a))
            };
        fields.push(Field::new(*name, t, nullable));
        arrays.push(a);
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    s.register_batches(name, schema, vec![batch])?;
    Ok(())
}

fn spark_type(t: &DataType) -> String {
    match t {
        DataType::Int32 => "int".into(),
        DataType::Int64 => "bigint".into(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "string".into(),
        DataType::Boolean => "boolean".into(),
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
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        other => format!("<{other}>"),
    }
}

fn render(schema: &Schema, batches: &[RecordBatch]) -> (String, Vec<String>) {
    let columns: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| format!("{}:{}", f.name(), spark_type(f.data_type())))
        .collect();
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
    (format!("struct<{}>", columns.join(",")), rows)
}

/// The JSON body of a golden error: class, SQLSTATE, message parameters.
fn expected_error(output: &str) -> R<(String, String, Value)> {
    let body = output.split_once('\n').ok_or("an error without a body")?.1;
    let json: Value = serde_json::Deserializer::from_str(body)
        .into_iter::<Value>()
        .next()
        .ok_or("no error JSON")??;
    let text = |k: &str| json[k].as_str().unwrap_or_default().to_owned();
    Ok((
        text("errorClass"),
        text("sqlState"),
        json["messageParameters"].clone(),
    ))
}

/// Runs `case` over `s`; the disagreement with Spark, if any.
fn run(s: &Session, case: &Case) -> R<Option<String>> {
    let result = block_on(async {
        let df = s.sql(&case.query).await?;
        let schema = df.schema().await?;
        Ok::<_, KrishivError>((schema, df.collect().await?))
    });
    if case.schema == "struct<>" {
        let (class, state, params) = expected_error(&case.output)?;
        let Err(e) = result else {
            return Ok(Some(format!("{}\n  expected error {class}", case.query)));
        };
        let got: serde_json::Map<String, Value> = e
            .message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), Value::String(v.clone())))
            .collect();
        let same = e.error_class().name() == class
            && e.sql_state() == Some(state.as_str())
            && Value::Object(got.clone()) == params;
        return Ok((!same).then(|| {
            format!(
                "{}\n  error {} {:?} {got:?}\n  want {class} {state} {params}",
                case.query,
                e.error_class().name(),
                e.sql_state()
            )
        }));
    }
    let (schema, batches) = match result {
        Ok(r) => r,
        Err(e) => return Ok(Some(format!("{}\n  error {e}", case.query))),
    };
    let (schema, mut rows) = render(&schema, &batches);
    let mut want: Vec<String> = case.output.lines().map(str::to_owned).collect();
    // Only an `ORDER BY` fixes the order of Spark's rows.
    if !case.query.to_ascii_uppercase().contains("ORDER BY") {
        rows.sort();
        want.sort();
    }
    Ok((schema != case.schema || rows != want).then(|| {
        format!(
            "{}\n  schema {schema} want {}\n  rows {rows:?}\n  want {want:?}",
            case.query, case.schema
        )
    }))
}

/// The setup statement creating view `name`.
fn setup<'a>(cases: &'a [Case], name: &str) -> R<&'a str> {
    cases
        .iter()
        .find(|c| c.query.contains(&format!("VIEW {name} AS")))
        .map(|c| c.query.as_str())
        .ok_or_else(|| format!("no setup of {name}").into())
}

fn check(bad: &[String], claimed: usize, want: usize) -> R {
    if claimed != want {
        return Err(format!("{claimed} statements claimed, expected {want}").into());
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!("{} mismatches:\n{}", bad.len(), bad.join("\n")).into())
    }
}

/// The boolean aggregates of `group-by.sql`: every statement that calls one.
#[test]
fn boolean_aggregates_match_spark() -> R {
    let cases = load("group-by.sql.out")?;
    let create = setup(&cases, "test_agg")?;
    let names = ["every(", "some(", "any(", "bool_and(", "bool_or("];
    let mut bad = Vec::new();
    let mut claimed = 0;
    for case in cases.iter().filter(|c| {
        let q = c.query.to_ascii_lowercase();
        c.query != create && names.iter().any(|n| q.contains(n)) && !c.query.contains("\"true\"")
    }) {
        claimed += 1;
        let s = Session::new()?;
        register(&s, create, "test_agg", &[0, 1])?;
        bad.extend(run(&s, case)?);
    }
    check(&bad, claimed, 18)
}

/// `first_value`, `any_value` and `last_value` over an empty window
/// definition (`WINDOW w AS ()`), with and without `ignoreNulls`.
#[test]
fn any_value_over_the_whole_partition_matches_spark() -> R {
    let cases = load("window.sql.out")?;
    let create = setup(&cases, "testData")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for case in cases
        .iter()
        .filter(|c| c.query.contains("any_value(false) OVER w AS any_value"))
    {
        claimed += 1;
        let s = Session::new()?;
        // Only `val` and `cate` are read: columns 0 and 5.
        register(&s, create, "testData", &[0, 5])?;
        bad.extend(run(&s, case)?);
    }
    check(&bad, claimed, 1)
}
