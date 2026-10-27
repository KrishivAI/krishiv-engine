// SPDX-License-Identifier: Apache-2.0
//! Spark 4.2 `SQLQueryTestSuite` goldens for the simple aggregates
//! (`every`, `some`, `any`, `bool_and`, `bool_or`, `any_value`), the collect
//! aggregates (`collect_list`, `collect_set`, `array_agg`) and the vector
//! aggregates (`vector_sum`, `vector_avg`): each claimed statement of the
//! vendored `group-by`, `vector-agg` and `window` result files runs through
//! a session over the tables the file creates, and its schema, rows (in
//! order when the statement orders them) or error class, SQLSTATE and
//! message parameters must be Spark's.
//!
//! Claimed: the `group-by.sql` statements on `test_agg` (the boolean
//! aggregates grouped, global, with `HAVING`, in subqueries, in windows and
//! with wrong argument types, but not `every("true")`, which needs a string
//! to boolean cast), the `collect_list`/`array_agg` statements of
//! `group-by.sql`, every `vector-agg.sql` statement (grouped, global, NULL
//! and empty vectors, window partitions, dimension mismatches and wrong
//! argument types) and the `window.sql` statement of
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
use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
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
        DataType::Float32 => "float".into(),
        DataType::Float64 => "double".into(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "string".into(),
        DataType::Boolean => "boolean".into(),
        DataType::List(f) | DataType::LargeList(f) => {
            format!("array<{}>", spark_type(f.data_type()))
        }
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
        DataType::Float32 => format!("{:?}", c.as_primitive::<Float32Type>().value(row)),
        DataType::Float64 => format!("{:?}", c.as_primitive::<Float64Type>().value(row)),
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        DataType::LargeUtf8 => c.as_string::<i64>().value(row).to_owned(),
        DataType::List(_) => {
            let l = c.as_list::<i32>();
            let at = |i: usize| {
                l.value_offsets()
                    .get(i)
                    .copied()
                    .map_or(0, |o| usize::try_from(o).unwrap_or(0))
            };
            let (start, end) = (at(row), at(row + 1).max(at(row)));
            let items = (start..end)
                .map(|i| cell(l.values().as_ref(), i))
                .collect::<Vec<_>>()
                .join(",");
            format!("[{items}]")
        }
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

/// Whether `query` has an `ORDER BY` at parenthesis depth 0, outside
/// string literals, quoted identifiers and comments. An `ORDER BY` inside
/// `OVER (...)` orders only the window frame, as in Spark, and does not
/// fix the order of the query's rows: the engine runs window partitions
/// in parallel across hash buckets, so rows come back in bucket order.
/// (Same approximation as `krishiv_testkit::golden::parse::text_sorted`.)
fn outer_order_by(query: &str) -> bool {
    let mut top = String::with_capacity(query.len());
    let mut depth = 0_usize;
    let mut chars = query.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' | '"' | '`' => {
                while let Some(n) = chars.next() {
                    if n == '\\' {
                        chars.next();
                    } else if n == c {
                        break;
                    }
                }
                top.push(' ');
            }
            '-' if chars.peek() == Some(&'-') => {
                for n in chars.by_ref() {
                    if n == '\n' {
                        break;
                    }
                }
                top.push(' ');
            }
            '/' if chars.peek() == Some(&'*') => {
                let mut prev = ' ';
                for n in chars.by_ref() {
                    if prev == '*' && n == '/' {
                        break;
                    }
                    prev = n;
                }
                top.push(' ');
            }
            '(' => {
                depth += 1;
                top.push(' ');
            }
            ')' => {
                depth = depth.saturating_sub(1);
                top.push(' ');
            }
            _ if depth == 0 => top.push(c.to_ascii_lowercase()),
            _ => {}
        }
    }
    let mut words = top
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .peekable();
    while let Some(w) = words.next() {
        if w == "order" && words.peek() == Some(&"by") {
            return true;
        }
    }
    false
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
    // Only an outer `ORDER BY` fixes the order of Spark's rows; an
    // `ORDER BY` inside `OVER (...)` orders only the window frame.
    if !outer_order_by(&case.query) {
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

/// Every `vector-agg.sql` statement: element-wise sums and means grouped,
/// global, over NULL and empty vectors, over window partitions, and the
/// dimension-mismatch and wrong-argument-type errors with Spark's classes
/// and parameters. Inline `VALUES` need no registered table.
#[test]
fn vector_aggregates_match_spark() -> R {
    let cases = load("vector-agg.sql.out")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for case in &cases {
        claimed += 1;
        bad.extend(run(&Session::new()?, case)?);
    }
    check(&bad, claimed, 16)
}

/// The `collect_list`/`array_agg` statements of `group-by.sql`: Spark names
/// `array_agg`'s column `collect_list(...)`, so both spellings give the
/// same column. Inline `VALUES` need no registered table.
#[test]
fn collect_and_array_agg_match_spark() -> R {
    let cases = load("group-by.sql.out")?;
    let mut bad = Vec::new();
    let mut claimed = 0;
    for case in cases.iter().filter(|c| {
        c.query.contains("collect_list(col)")
            || c.query.contains("array_agg")
            || c.query.contains("collect_list(b)")
    }) {
        claimed += 1;
        bad.extend(run(&Session::new()?, case)?);
    }
    check(&bad, claimed, 2)
}

/// `collect_list` and `collect_set` skip NULLs; the set dedups in
/// first-seen order. `DISTINCT` collects skip NULLs and repeats through the
/// distinct rewrite.
#[test]
fn collect_skips_nulls_and_distinct_dedups() -> R {
    let s = Session::new()?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("x", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(1), None])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                Some("b"),
                Some("a"),
                None,
                Some("b"),
            ])) as ArrayRef,
        ],
    )?;
    s.register_batches("t", schema, vec![batch])?;
    // Grouped and global, in input order within the single partition.
    let one = |sql: &str| -> R<Vec<String>> {
        let df = block_on(s.sql(sql))?;
        let batches = block_on(df.collect())?;
        Ok(batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows()).map(|r| {
                    b.columns()
                        .iter()
                        .map(|c| cell(c.as_ref(), r))
                        .collect::<Vec<_>>()
                        .join("|")
                })
            })
            .collect())
    };
    assert_eq!(
        one("SELECT k, collect_list(x), collect_set(x) FROM t GROUP BY k ORDER BY k")?,
        ["NULL|[b]|[b]", "1|[b,a]|[b,a]"]
    );
    assert_eq!(one("SELECT collect_list(x) FROM t")?, ["[b,a,b]"]);
    assert_eq!(
        one("SELECT collect_list(DISTINCT x), collect_set(DISTINCT x) FROM t")?,
        ["[b,a]|[b,a]"]
    );
    // `FILTER (WHERE ...)` is refused for every aggregate: the planner
    // keeps filters out of accumulators, never evaluates them inside.
    let df = block_on(s.sql("SELECT collect_list(x) FILTER (WHERE k = 1) FROM t"))?;
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    Ok(())
}
