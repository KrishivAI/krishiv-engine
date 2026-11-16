// SPDX-License-Identifier: Apache-2.0
//! Shared by the analyze tests: the parser corpus (`krishiv-sql`'s statement-class goldens), the session
//! its statements name, and the TPC-DS tables.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

/// The repository root.
pub(crate) fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The parser corpus: (branch label, SQL) per line.
pub(crate) fn corpus() -> Vec<(String, String)> {
    let path = root().join("crates/krishiv-sql/tests/fixtures/statement-classes.tsv");
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut f = l.splitn(3, '\t');
            let label = f.next().unwrap().to_owned();
            let _class = f.next().unwrap();
            (label, f.next().unwrap().to_owned())
        })
        .collect()
}

/// The objects the corpus statements name: tables `t`, `s`, `u` (`a`, `b`, `k`), temporary views `v0`,
/// `v1`, `v2`, namespaces `d` and `db`, the permanent view `v`.
pub(crate) fn corpus_session(s: &Session) {
    for sql in [
        "CREATE TABLE t (a INT, b INT, k INT)",
        "CREATE TABLE s (a INT, b INT, k INT)",
        "CREATE TABLE u (a INT, b INT, k INT)",
        "CREATE TEMPORARY VIEW v0 AS SELECT * FROM t",
        "CREATE TEMPORARY VIEW v1 AS SELECT k, a FROM t",
        "CREATE TEMPORARY VIEW v2 AS SELECT k, b FROM s",
        "CREATE DATABASE d",
        "CREATE DATABASE db",
        "CREATE VIEW v AS SELECT a FROM t",
    ] {
        block_on(s.sql(sql)).unwrap();
    }
}

/// The TPC-DS tables (the Spark 4.2.0 golden schemas), empty; `CHAR`/`VARCHAR` as strings.
pub(crate) fn tpcds_session() -> Session {
    let s = Session::new().unwrap();
    let path = root().join("tests/fixtures/spark-4.2.0/golden-tables/tpcds-schemas.json");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for table in doc["tables"].as_array().unwrap() {
        let fields: Vec<Field> = table["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let ty = c["type"].as_str().unwrap().to_ascii_lowercase();
                Field::new(
                    c["name"].as_str().unwrap(),
                    arrow_type(&ty),
                    c["nullable"].as_bool().unwrap(),
                )
            })
            .collect();
        s.register_batches(
            table["name"].as_str().unwrap(),
            Arc::new(Schema::new(fields)),
            vec![],
        )
        .unwrap();
    }
    s
}

fn arrow_type(ty: &str) -> ArrowType {
    if let Some(ps) = ty.strip_prefix("decimal(") {
        let (p, s) = ps.trim_end_matches(')').split_once(',').unwrap();
        return ArrowType::Decimal128(p.trim().parse().unwrap(), s.trim().parse().unwrap());
    }
    match ty {
        "int" | "integer" => ArrowType::Int32,
        "bigint" => ArrowType::Int64,
        "double" => ArrowType::Float64,
        "date" => ArrowType::Date32,
        _ if ty.starts_with("char") || ty.starts_with("varchar") || ty == "string" => {
            ArrowType::Utf8
        }
        other => panic!("TPC-DS type {other}"),
    }
}

/// The TPC-DS query `q` (1..=99).
pub(crate) fn tpcds_query(q: usize) -> String {
    std::fs::read_to_string(root().join(format!("bench/queries/tpcds/q{q:02}.sql"))).unwrap()
}

/// The compact analyze JSON with a line break after each top-level member and each item of a
/// top-level array, for readable goldens (the text is otherwise unchanged).
pub(crate) fn layout(json: &str) -> String {
    let (mut out, mut depth, mut in_str, mut escaped) = (String::new(), 0usize, false, false);
    for c in json.chars() {
        out.push(c);
        if in_str {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            ',' if depth <= 2 => out.push_str(&format!("\n{}", "  ".repeat(depth - 1))),
            _ => {}
        }
    }
    out
}

/// Checks `value` against the hand-written description `shape` (`krishiv_plan::analyzer::lineage::schema`):
/// the errors, by JSON path.
pub(crate) fn validate(
    shape: &krishiv_plan::analyzer::lineage::schema::Shape,
    value: &serde_json::Value,
    path: &str,
) -> Vec<String> {
    use krishiv_plan::analyzer::lineage::schema::Shape;
    use serde_json::Value;
    let bad = |what: &str| vec![format!("{path}: expected {what}, got {value}")];
    match (shape, value) {
        (Shape::String, Value::String(_))
        | (Shape::Boolean, Value::Bool(_))
        | (Shape::Nullable(_), Value::Null) => Vec::new(),
        (Shape::Count, Value::Number(n)) if n.is_u64() => Vec::new(),
        (Shape::Const(c), Value::String(s)) if s == c => Vec::new(),
        (Shape::Enum(values), Value::String(s)) if values.contains(&s.as_str()) => Vec::new(),
        (Shape::Nullable(inner), v) => validate(inner, v, path),
        (Shape::Array(inner), Value::Array(items)) => items
            .iter()
            .enumerate()
            .flat_map(|(i, v)| validate(inner, v, &format!("{path}[{i}]")))
            .collect(),
        (Shape::Object(fields), Value::Object(members)) => {
            let mut errors = Vec::new();
            for f in fields {
                match members.get(f.name) {
                    Some(v) => errors.extend(validate(&f.shape, v, &format!("{path}.{}", f.name))),
                    None => errors.push(format!("{path}: missing {}", f.name)),
                }
            }
            for k in members.keys() {
                if !fields.iter().any(|f| f.name == k) {
                    errors.push(format!("{path}: unknown member {k}"));
                }
            }
            errors
        }
        (Shape::OneOf(shapes), v) => {
            let matching = shapes
                .iter()
                .filter(|s| validate(s, v, path).is_empty())
                .count();
            if matching == 1 {
                Vec::new()
            } else {
                bad(&format!(
                    "exactly one of {} forms ({matching} match)",
                    shapes.len()
                ))
            }
        }
        (other, _) => bad(&format!("{other:?}")),
    }
}

/// [`validate`] against the `krishiv.analyze.v1` description, panicking with the errors.
pub(crate) fn assert_valid(json: &str) {
    let value: serde_json::Value = serde_json::from_str(json).unwrap();
    let errors = validate(
        &krishiv_plan::analyzer::lineage::schema::analyze_v1(),
        &value,
        "$",
    );
    assert!(errors.is_empty(), "{json}\n{errors:#?}");
}
