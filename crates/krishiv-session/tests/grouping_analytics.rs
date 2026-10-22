// SPDX-License-Identifier: Apache-2.0
//! Spark 4.2 `SQLQueryTestSuite` goldens for grouping analytics (`ROLLUP`, `CUBE`,
//! `GROUPING SETS`, `grouping`, `grouping_id`): each statement of the vendored
//! `group-analytics` and `grouping_set` result files runs through a session and
//! its schema, rows (compared as sets of rows) or error class must match.
//!
//! The session has no `CREATE VIEW` or inline `VALUES` yet, so the driver turns
//! each literal `VALUES` table into a registered table of the same name.
#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, StringArray};
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

fn load(file: &str) -> R<Vec<Case>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/spark-golden/v4.2.0/results")
        .join(file);
    let text = std::fs::read_to_string(&path)?;
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
    Ok(cases)
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
        DataType::Int8 => c.as_primitive::<Int8Type>().value(row).to_string(),
        DataType::Int16 => c.as_primitive::<Int16Type>().value(row).to_string(),
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
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

/// A parsed `VALUES` literal table.
struct Table {
    name: String,
    columns: Vec<String>,
    rows: Vec<Vec<Option<String>>>,
}

fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn skip_ws(bytes: &[u8], i: &mut usize) {
    while bytes.get(*i).is_some_and(u8::is_ascii_whitespace) {
        *i += 1;
    }
}

/// Find an inline `VALUES (..), (..) [AS] alias[(cols)]` (optionally wrapped in
/// parentheses) in `sql`; return the text with it replaced by the alias, and the table.
fn extract_values(sql: &str) -> Option<(String, Table)> {
    let upper = sql.to_ascii_uppercase();
    let bytes = sql.as_bytes();
    let at = upper.match_indices("VALUES").map(|(i, _)| i).find(|&i| {
        (i == 0 || bytes.get(i - 1).is_none_or(|&b| !is_word(b)))
            && bytes.get(i + 6).is_none_or(|&b| !is_word(b))
    })?;
    let before = sql[..at].trim_end();
    let wrapped = before.ends_with('(');
    let start = if wrapped { before.len() - 1 } else { at };
    let mut i = at + 6;
    let mut rows = Vec::new();
    loop {
        skip_ws(bytes, &mut i);
        if bytes.get(i) != Some(&b'(') {
            return None;
        }
        let close = i + sql[i..].find(')')?;
        rows.push(
            sql[i + 1..close]
                .split(',')
                .map(|v| {
                    let v = v.trim();
                    (!v.eq_ignore_ascii_case("NULL")).then(|| v.to_owned())
                })
                .collect::<Vec<_>>(),
        );
        i = close + 1;
        skip_ws(bytes, &mut i);
        if bytes.get(i) == Some(&b',') {
            i += 1;
        } else {
            break;
        }
    }
    if wrapped {
        if bytes.get(i) != Some(&b')') {
            return None;
        }
        i += 1;
        skip_ws(bytes, &mut i);
    }
    if upper[i..].starts_with("AS ") {
        i += 3;
        skip_ws(bytes, &mut i);
    }
    let name_start = i;
    while bytes.get(i).is_some_and(|&b| is_word(b)) {
        i += 1;
    }
    let name = sql[name_start..i].to_owned();
    skip_ws(bytes, &mut i);
    let mut columns = Vec::new();
    if bytes.get(i) == Some(&b'(') {
        let close = i + sql[i..].find(')')?;
        columns = sql[i + 1..close]
            .split(',')
            .map(|c| c.trim().to_owned())
            .collect();
        i = close + 1;
    }
    let text = format!("{}{}{}", &sql[..start], name, &sql[i..]);
    Some((
        text,
        Table {
            name,
            columns,
            rows,
        },
    ))
}

fn register(s: &Session, t: &Table, name: &str) -> R {
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for (c, column) in t.columns.iter().enumerate() {
        let col: Vec<Option<&String>> = t
            .rows
            .iter()
            .map(|r| r.get(c).and_then(Option::as_ref))
            .collect();
        let numeric = col.iter().flatten().all(|v| v.parse::<i32>().is_ok());
        let nullable = col.iter().any(Option::is_none);
        if numeric {
            fields.push(Field::new(column, DataType::Int32, nullable));
            arrays.push(Arc::new(Int32Array::from(
                col.iter()
                    .map(|v| v.and_then(|v| v.parse().ok()))
                    .collect::<Vec<_>>(),
            )));
        } else {
            fields.push(Field::new(column, DataType::Utf8, nullable));
            arrays.push(Arc::new(StringArray::from(
                col.iter()
                    .map(|v| v.map(|v| v.trim_matches(['"', '\''])))
                    .collect::<Vec<_>>(),
            )));
        }
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    s.register_batches(name, schema, vec![batch])?;
    Ok(())
}

/// The error class a golden expects, from its JSON error body.
fn expected_class(output: &str) -> Option<String> {
    output
        .lines()
        .find_map(|l| l.trim().strip_prefix("\"errorClass\" : \""))
        .map(|l| l.trim_end_matches(['"', ',']).to_owned())
}

/// Run every statement of `file`; return the statements that disagree with Spark.
fn run(file: &str) -> R<Vec<String>> {
    let mut views: Vec<(String, Table)> = Vec::new();
    let mut bad = Vec::new();
    for case in load(file)? {
        let upper = case.query.to_ascii_uppercase();
        if upper.starts_with("CREATE") {
            let name = case
                .query
                .split_whitespace()
                .skip_while(|w| !w.eq_ignore_ascii_case("VIEW"))
                .nth(1);
            let (Some(name), Some((_, table))) = (name, extract_values(&case.query)) else {
                bad.push(format!("{}\n  unsupported setup", case.query));
                continue;
            };
            views.retain(|(n, _)| n != name);
            views.push((name.to_owned(), table));
            continue;
        }
        if let Some(name) = upper.strip_prefix("DROP VIEW ") {
            views.retain(|(n, _)| !n.eq_ignore_ascii_case(name.trim()));
            continue;
        }
        let s = Session::new()?;
        for (name, t) in &views {
            register(&s, t, name)?;
        }
        let mut query = case.query.clone();
        if let Some((text, table)) = extract_values(&query) {
            register(&s, &table, &table.name)?;
            query = text;
        }
        let result = block_on(async {
            let df = s.sql(&query).await?;
            let schema = df.schema().await?;
            Ok::<_, KrishivError>((schema, df.collect().await?))
        });
        if case.schema == "struct<>" {
            let class = expected_class(&case.output);
            match (result, class) {
                (Err(e), Some(c)) if e.error_class().name() == c => {}
                (Err(e), c) => bad.push(format!(
                    "{}\n  error class {} expected {c:?}",
                    case.query,
                    e.error_class().name()
                )),
                (Ok(_), c) => bad.push(format!("{}\n  expected error {c:?}", case.query)),
            }
            continue;
        }
        match result {
            Err(e) => bad.push(format!("{}\n  error {e}", case.query)),
            Ok((schema, batches)) => {
                let (schema, mut rows) = render(&schema, &batches);
                let mut want: Vec<String> = case.output.lines().map(str::to_owned).collect();
                // Rows tied under `ORDER BY` come in no fixed order.
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
    Ok(bad)
}

fn check(file: &str) -> R {
    let bad = run(file)?;
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!("{} mismatches:\n{}", bad.len(), bad.join("\n")).into())
    }
}

#[test]
fn group_analytics_golden() -> R {
    check("group-analytics.sql.out")
}

#[test]
fn grouping_set_golden() -> R {
    check("grouping_set.sql.out")
}
