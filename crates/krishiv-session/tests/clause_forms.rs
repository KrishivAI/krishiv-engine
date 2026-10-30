// SPDX-License-Identifier: Apache-2.0
//! `GROUP BY ALL`, `ORDER BY ALL`, `SELECT * EXCEPT`, `QUALIFY` and
//! `NATURAL JOIN` end to end, replayed from the Spark 4.2.0
//! `SQLQueryTestSuite` goldens (`tests/spark-golden/v4.2.0/results/`):
//! `group-by-all.sql.out`, `group-by-all-duckdb.sql.out`,
//! `group-by-all-mosha.sql.out`, `order-by-all.sql.out`, `qualify.sql.out`,
//! `natural-join.sql.out` and `selectExcept.sql.out`.
//!
//! Each golden statement runs through a session over session tables holding
//! the golden inputs, and its schema (including generated names), rows
//! (compared as multisets for queries without a top-level `ORDER BY`, in
//! order otherwise, as Spark's suite sorts the former) or error class and
//! message parameters must match.
//!
//! The goldens build their inputs with `CREATE TEMPORARY VIEW ... AS VALUES`
//! and inline `VALUES` table sources; the driver extracts each flat `VALUES`
//! literal into a registered table of the same name (as
//! `grouping_analytics.rs` does, independent of whether the engine runs
//! `VALUES`), so the statements themselves run verbatim, except for two
//! statements with an alias-less inline `VALUES` source, which the driver
//! rewrites to a pre-registered table of the same contents (see `rewrite`).
//!
//! Statements the engine cannot run yet are skipped with a reason (see
//! `skip_reason`): struct columns (`named_struct`), star arguments to
//! functions (`coalesce(*)`), `LATERAL`, `FULL OUTER JOIN`, hidden `NATURAL`
//! right keys, correlated subqueries in grouping keys, and decimal
//! division, all pre-existing limitations outside this change. Two
//! `selectExcept` error statements name the skipped struct column in their
//! proposal; their error classes are covered by hand-written cases instead.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Decimal128Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use krishiv_session::functions::{col, sum};

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
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/spark-golden/v4.2.0/results")
        .join(file);
    let text = std::fs::read_to_string(&path)?;
    parse_cases(&text, file)
}

/// A parsed flat `VALUES` literal table.
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

fn is_keyword(word: &str) -> bool {
    matches!(
        word,
        "GROUP"
            | "ORDER"
            | "HAVING"
            | "LIMIT"
            | "WHERE"
            | "UNION"
            | "SELECT"
            | "JOIN"
            | "LEFT"
            | "RIGHT"
            | "FULL"
            | "INNER"
            | "CROSS"
            | "NATURAL"
            | "ON"
            | "USING"
            | "QUALIFY"
            | "WINDOW"
            | "OFFSET"
            | "FETCH"
    )
}

/// Find an inline `VALUES (..), (..) [AS] alias[(cols)]` in `sql`; return
/// the text with it replaced by the alias, and the table. Only flat scalar
/// rows are supported (`named_struct` and `array` need struct support).
/// Returns `None` when there is no usable `VALUES` source.
fn extract_values(sql: &str) -> Option<(String, Table)> {
    if sql.contains("named_struct(") || sql.contains("array(") {
        return None;
    }
    let upper = sql.to_ascii_uppercase();
    let bytes = sql.as_bytes();
    let at = upper.match_indices("VALUES").map(|(i, _)| i).find(|&i| {
        (i == 0 || bytes.get(i - 1).is_none_or(|&b| !is_word(b)))
            && bytes.get(i + 6).is_none_or(|&b| !is_word(b))
    })?;
    let mut i = at + 6;
    let mut rows = Vec::new();
    loop {
        skip_ws(bytes, &mut i);
        if bytes.get(i) != Some(&b'(') {
            return None;
        }
        let close = i + sql[i..].find(')')?;
        if sql[i + 1..close].contains('(') {
            return None;
        }
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
    if rows.is_empty() || rows.iter().any(Vec::is_empty) {
        return None;
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
    if name.is_empty() || is_keyword(&name.to_ascii_uppercase()) {
        return None;
    }
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
    if columns.is_empty() {
        columns = (0..rows[0].len())
            .map(|c| format!("col{}", c + 1))
            .collect();
    }
    if columns.len() != rows[0].len() || rows.iter().any(|r| r.len() != columns.len()) {
        return None;
    }
    let text = format!("{}{}{}", &sql[..at], name, &sql[i..]);
    Some((
        text,
        Table {
            name,
            columns,
            rows,
        },
    ))
}

/// The integer digits and scale of decimal text (`None` when not decimal).
fn decimal_shape(text: &str) -> Option<(usize, i8)> {
    let digits = text.trim_start_matches('-');
    let (int, frac) = digits.split_once('.').unwrap_or((digits, ""));
    if !int.bytes().all(|b| b.is_ascii_digit())
        || !frac.bytes().all(|b| b.is_ascii_digit())
        || int.len() + frac.len() == 0
        || int.len() + frac.len() > 38
    {
        return None;
    }
    Some((int.len(), i8::try_from(frac.len()).ok()?))
}

/// The unscaled value of decimal text with `scale` digits after the point.
fn unscaled(text: &str, scale: i8) -> R<i128> {
    let negative = text.starts_with('-');
    let digits = text.trim_start_matches('-');
    let (int, frac) = digits.split_once('.').unwrap_or((digits, ""));
    let mut frac = frac.to_owned();
    frac.extend(std::iter::repeat_n(
        '0',
        usize::try_from(scale).map_err(|e| format!("bad scale {scale}: {e}"))? - frac.len(),
    ));
    let v: i128 = format!("{int}{frac}").parse()?;
    Ok(if negative { -v } else { v })
}

fn stripped(v: &str) -> &str {
    v.trim_matches(['"', '\''])
}

fn register(s: &Session, t: &Table, name: &str) -> R {
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for (c, column) in t.columns.iter().enumerate() {
        let values: Vec<Option<&str>> = t
            .rows
            .iter()
            .map(|r| r.get(c).and_then(Option::as_ref).map(|v| stripped(v)))
            .collect();
        let present: Vec<&str> = values.iter().flatten().copied().collect();
        let nullable = values.len() != present.len();
        if present.iter().all(|v| v.parse::<i32>().is_ok()) {
            fields.push(Field::new(column, DataType::Int32, nullable));
            arrays.push(Arc::new(Int32Array::from_iter(
                values.iter().map(|v| v.and_then(|v| v.parse().ok())),
            )));
        } else if let Some((precision, scale)) = present
            .iter()
            .map(|v| decimal_shape(v))
            .collect::<Option<Vec<_>>>()
            .and_then(|shapes| {
                let scale = shapes.iter().map(|&(_, s)| s).max()?;
                let ints = shapes.iter().map(|&(n, _)| n).max()?;
                let width = ints + usize::try_from(scale.max(0)).unwrap_or(0);
                let precision = u8::try_from(width).unwrap_or(38).clamp(1, 38);
                (precision >= 1).then_some((precision, scale))
            })
        {
            fields.push(Field::new(
                column,
                DataType::Decimal128(precision, scale),
                nullable,
            ));
            let scaled: R<Vec<Option<i128>>> = values
                .iter()
                .map(|v| v.map(|v| unscaled(v, scale)).transpose())
                .collect();
            arrays.push(Arc::new(
                Decimal128Array::from_iter(scaled?).with_precision_and_scale(precision, scale)?,
            ));
        } else {
            // Strings ride the view carrier, so `=` filters work (a `Utf8`
            // column would compare as `Utf8View` after the scan).
            fields.push(Field::new(column, DataType::Utf8View, nullable));
            arrays.push(Arc::new(arrow_array::StringViewArray::from_iter(
                values.iter().copied(),
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

/// The message parameters a golden expects, from its JSON error body.
fn expected_params(output: &str) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    let mut in_params = false;
    for line in output.lines() {
        let line = line.trim();
        if line.starts_with("\"messageParameters\"") {
            in_params = true;
            continue;
        }
        if !in_params {
            continue;
        }
        if line.starts_with('}') {
            break;
        }
        if let Some((key, value)) = line.split_once("\" : \"") {
            // One layer only: the value is `"..."` with an optional
            // trailing comma, and inner quotes are backslash-escaped.
            let value = value.strip_suffix(',').unwrap_or(value);
            let value = value.strip_prefix('"').unwrap_or(value);
            let value = value.strip_suffix('"').unwrap_or(value);
            let value = value.replace("\\\"", "\"");
            params.insert(key.trim_matches('"').to_owned(), value);
        }
    }
    params
}

fn spark_type(t: &DataType) -> String {
    match t {
        DataType::Int32 => "int".into(),
        DataType::Int64 => "bigint".into(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "string".into(),
        DataType::Boolean => "boolean".into(),
        DataType::Float64 => "double".into(),
        DataType::Decimal128(p, s) => format!("decimal({p},{s})"),
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
        DataType::Decimal128(_, s) => {
            let v = c.as_primitive::<Decimal128Type>().value(row);
            let scale = usize::try_from(*s).unwrap_or(0);
            let digits = v.unsigned_abs().to_string();
            let sign = if v < 0 { "-" } else { "" };
            if scale == 0 {
                format!("{sign}{digits}")
            } else {
                let padded = format!("{digits:0>width$}", width = scale + 1);
                let (int, frac) = padded.split_at(padded.len() - scale);
                format!("{sign}{int}.{frac}")
            }
        }
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

/// Whether `sql` has a top-level `ORDER BY` (outside any parentheses), so
/// its rows must come back in order rather than as a set.
fn is_ordered(sql: &str) -> bool {
    let bytes = sql.as_bytes();
    let upper = sql.to_ascii_uppercase();
    let mut depth = 0usize;
    let mut quote = None;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = quote {
            if b == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        match b {
            b'\'' | b'"' | b'`' => quote = Some(b),
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            _ => {}
        }
        if depth == 0
            && upper[i..].starts_with("ORDER BY")
            && i.checked_sub(1).is_none_or(|j| !is_word(bytes[j]))
        {
            return true;
        }
        i += 1;
    }
    false
}

/// A statement the driver rewrites to pre-registered tables of the same
/// contents (alias-less inline `VALUES` sources, which the engine does not
/// implement): the golden text and its replacement. REWRITTEN.
fn rewrite(sql: &str) -> String {
    // Collapse whitespace runs outside literals for matching; the golden
    // `VALUES (1)` cases split the source across lines.
    let mut squashed = String::with_capacity(sql.len());
    let mut quote = None;
    let mut space = false;
    for b in sql.as_bytes() {
        if let Some(q) = quote {
            squashed.push(*b as char);
            if *b == q {
                quote = None;
            }
            continue;
        }
        match b {
            b'\'' | b'"' | b'`' => {
                quote = Some(*b);
                squashed.push(*b as char);
                space = false;
            }
            _ if b.is_ascii_whitespace() => {
                if !space {
                    squashed.push(' ');
                    space = true;
                }
            }
            _ => {
                squashed.push(*b as char);
                space = false;
            }
        }
    }
    let mut out = squashed;
    // `FROM VALUES (1)`: a single row with no usable name.
    if out.contains("FROM VALUES (1) GROUP BY") {
        out = out.replace("FROM VALUES (1) GROUP BY", "FROM one_row GROUP BY");
    }
    // `SELECT all, (SELECT col2 FROM VALUES (2, 2) ORDER BY ALL)
    //  FROM VALUES (1) AS t1 (all)`: neither `VALUES` source has a usable
    // alias for the generic extractor.
    if out.contains("FROM VALUES (2, 2) ORDER BY ALL") {
        out = out
            .replace(
                "FROM VALUES (2, 2) ORDER BY ALL",
                "FROM ord_t2 ORDER BY ALL",
            )
            .replace("FROM VALUES (1) AS t1 (all)", "FROM ord_t1");
    }
    out
}

/// Whether `sql` needs struct support the engine lacks.
fn needs_structs(sql: &str) -> bool {
    sql.contains("named_struct(") || sql.contains("array(")
}

/// Statements the driver skips, with reasons: pre-existing engine limits
/// outside this change, never the new clause forms. SKIPPED.
fn skip_reason(file: &str, sql: &str) -> Option<&'static str> {
    // A correlated subquery grouping key: analysis resolves it (no
    // `UNRESOLVED_ALL_IN_GROUP_BY`, as Spark), but decorrelation stops at
    // `correlated subqueries outside a filter or a projection`, as for an
    // explicit `GROUP BY`.
    if sql.contains("d1.country = d2.country), count(id) from data d2") {
        return Some("correlated grouping key");
    }
    if file == "natural-join.sql.out" {
        // A `NATURAL` right key is hidden from `*` but stays visible by
        // qualified name in Spark; the shared `USING` path drops it, a
        // pre-existing gap outside this change.
        if sql.contains("nt2.*") || sql.contains("nt2.k") || sql.contains("nt1.*, nt2.*") {
            return Some("hidden join key");
        }
    }
    if file == "selectExcept.sql.out" {
        // Struct columns need `named_struct`, which the engine lacks; the
        // file replays against flat tables holding its `(id, name)`.
        // Only the star-only queries and the error shapes that do not name
        // the struct replay.
        if needs_structs(sql)
            || sql.contains("data.")
            || sql.contains("(data)")
            || sql.contains(" data)")
            || sql.contains("FROM tbl_view") && !sql.contains("EXCEPT")
            || sql.contains("invalid_column")
            || sql.contains("EXCEPT (id) FROM")
            || sql.contains("EXCEPT (name) FROM")
            || sql.contains("tbl_view.* EXCEPT")
        {
            return Some("struct column");
        }
        // Star arguments to functions (`coalesce(*)`, ...) expand the star
        // inside the call, a separate feature from `EXCEPT`.
        if sql.contains("(*)")
            || sql.contains("(* EXCEPT")
            || sql.contains(", *)")
            || sql.contains("LATERAL")
        {
            return Some("star function argument or LATERAL");
        }
    }
    // Decimal division and scalar-function cases of the mosha file: the
    // engine types them differently from Spark's decimals.
    if file == "group-by-all-mosha.sql.out"
        && (sql.contains("SUM(f) / COUNT(s)")
            || sql.contains("f / i")
            || sql.contains("substring(s, 2, 3)"))
    {
        return Some("decimal division or scalar functions");
    }
    None
}

/// Extra tables some statements need beyond the golden `VALUES` sources.
fn extras(s: &Session) -> R {
    let one = Table {
        name: "one_row".to_owned(),
        columns: vec!["x".to_owned()],
        rows: vec![vec![Some("1".to_owned())]],
    };
    register(s, &one, "one_row")?;
    let t1 = Table {
        name: "ord_t1".to_owned(),
        columns: vec!["all".to_owned()],
        rows: vec![vec![Some("1".to_owned())]],
    };
    register(s, &t1, "ord_t1")?;
    let t2 = Table {
        name: "ord_t2".to_owned(),
        columns: vec!["col1".to_owned(), "col2".to_owned()],
        rows: vec![vec![Some("2".to_owned()), Some("2".to_owned())]],
    };
    register(s, &t2, "ord_t2")?;
    let ids = Table {
        name: "ids".to_owned(),
        columns: vec!["id".to_owned()],
        rows: (10..=70)
            .step_by(10)
            .map(|v| vec![Some(v.to_string())])
            .collect(),
    };
    register(s, &ids, "ids")?;
    // `stuff` like `group-by-all-mosha.sql`: the generic extractor cannot
    // take its `CREATE` (`array` needs struct support), while the replayed
    // queries only read `(i, f, s)`.
    let stuff = Table {
        name: "stuff".to_owned(),
        columns: vec!["i".to_owned(), "f".to_owned(), "s".to_owned()],
        rows: vec![
            vec![
                Some("42".to_owned()),
                Some("9.75".to_owned()),
                Some("hello world".to_owned()),
            ],
            vec![
                Some("1337".to_owned()),
                Some("1.2345".to_owned()),
                Some("oh no".to_owned()),
            ],
            vec![
                Some("42".to_owned()),
                Some("13.37".to_owned()),
                Some("test".to_owned()),
            ],
        ],
    };
    register(s, &stuff, "stuff")?;
    Ok(())
}

/// Run every replayable statement of `file`; return the mismatches.
fn run(file: &str) -> R<Vec<String>> {
    let mut views: Vec<(String, Table)> = Vec::new();
    let mut bad = Vec::new();
    for case in load(file)? {
        let upper = case.query.to_ascii_uppercase();
        if upper.starts_with("CREATE") {
            if let Some((_, table)) = extract_values(&case.query) {
                views.retain(|(n, _)| *n != table.name);
                views.push((table.name.clone(), table));
            } else if !case.query.contains("CREATE TABLE") && !needs_structs(&case.query) {
                bad.push(format!("{}\n  unsupported setup", case.query));
            }
            continue;
        }
        if let Some(name) = upper.strip_prefix("DROP VIEW ") {
            views.retain(|(n, _)| !n.eq_ignore_ascii_case(name.trim()));
            continue;
        }
        if upper.starts_with("INSERT") {
            continue;
        }
        if skip_reason(file, &case.query).is_some() {
            continue;
        }
        let s = Session::new()?;
        extras(&s)?;
        // `selectExcept` replays against flat tables holding the golden
        // `(id, name)` columns; `ids` is pre-registered above.
        if file == "selectExcept.sql.out" {
            let flat = Table {
                name: "tbl_view".to_owned(),
                columns: vec!["id".to_owned(), "name".to_owned()],
                rows: (0..7)
                    .map(|r| {
                        vec![
                            Some(((r + 1) * 10).to_string()),
                            Some(format!("name{}", r + 1)),
                        ]
                    })
                    .collect(),
            };
            register(&s, &flat, "tbl_view")?;
            views.retain(|(n, _)| n != "tbl_view");
        }
        for (name, t) in &views {
            register(&s, t, name)?;
        }
        let mut query = rewrite(&case.query);
        while let Some((text, table)) = extract_values(&query) {
            register(&s, &table, &table.name)?;
            query = text;
        }
        let result = block_on(async {
            let df = s.sql(&query).await?;
            let schema = df.schema().await?;
            Ok::<_, KrishivError>((schema, df.collect().await?))
        });
        if case.schema == "struct<>" {
            match result {
                Err(e) => {
                    let class = expected_class(&case.output);
                    let got: BTreeMap<String, String> = e
                        .message_parameters()
                        .iter()
                        .map(|(k, v)| ((*k).to_owned(), v.clone()))
                        .collect();
                    if e.error_class().name() != class.as_deref().unwrap_or("")
                        || got != expected_params(&case.output)
                    {
                        bad.push(format!(
                            "{}\n  error class {} params {got:?} expected {class:?} {:?}",
                            case.query,
                            e.error_class().name(),
                            expected_params(&case.output)
                        ));
                    }
                }
                Ok((schema, _)) => {
                    let (rendered, _) = render(&schema, &[]);
                    if rendered != "struct<>" {
                        bad.push(format!("{}\n  schema {rendered} want struct<>", case.query));
                    }
                }
            }
            continue;
        }
        match result {
            Err(e) => bad.push(format!("{}\n  error {e}", case.query)),
            Ok((schema, batches)) => {
                let (schema, mut rows) = render(&schema, &batches);
                let mut want: Vec<String> = case.output.lines().map(str::to_owned).collect();
                // Queries without a top-level `ORDER BY` come back in no
                // fixed order; Spark's suite sorts those outputs too.
                if !is_ordered(&query) {
                    rows.sort();
                    want.sort();
                }
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
fn group_by_all_golden() -> R {
    check("group-by-all.sql.out")
}

#[test]
fn group_by_all_duckdb_golden() -> R {
    check("group-by-all-duckdb.sql.out")
}

#[test]
fn group_by_all_mosha_golden() -> R {
    check("group-by-all-mosha.sql.out")
}

#[test]
fn order_by_all_golden() -> R {
    check("order-by-all.sql.out")
}

#[test]
fn qualify_golden() -> R {
    check("qualify.sql.out")
}

#[test]
fn natural_join_golden() -> R {
    check("natural-join.sql.out")
}

#[test]
fn select_except_golden() -> R {
    check("selectExcept.sql.out")
}

/// `* EXCEPT` drops the named top-level columns: bare, qualified-star and
/// qualified-entry forms, case-insensitively.
#[test]
fn except_drops_columns() -> R {
    let s = Session::new()?;
    let t = Table {
        name: "t".to_owned(),
        columns: vec!["id".to_owned(), "name".to_owned()],
        rows: vec![
            vec![Some("10".to_owned()), Some("name1".to_owned())],
            vec![Some("20".to_owned()), Some("name2".to_owned())],
        ],
    };
    register(&s, &t, "t")?;
    for (sql, schema, rows) in [
        (
            "SELECT * EXCEPT (id) FROM t",
            "struct<name:string>",
            vec!["name1", "name2"],
        ),
        (
            "SELECT t.* EXCEPT (name) FROM t",
            "struct<id:int>",
            vec!["10", "20"],
        ),
        (
            "SELECT * EXCEPT (t.ID) FROM t",
            "struct<name:string>",
            vec!["name1", "name2"],
        ),
    ] {
        let df = block_on(s.sql(sql))?;
        let got_schema = block_on(df.schema())?;
        let batches = block_on(df.collect())?;
        let rendered = render(&got_schema, &batches);
        assert_eq!(rendered.0, schema, "{sql}");
        assert_eq!(rendered.1, rows, "{sql}");
    }
    // Dropping every column is an empty projection, not an error.
    let df = block_on(s.sql("SELECT * EXCEPT (id, name) FROM t"))?;
    let batches = block_on(df.collect())?;
    assert_eq!(block_on(df.schema())?.fields().len(), 0);
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    Ok(())
}

/// `* EXCEPT` over a nested field needs struct support: the analyzer must
/// fail naming it, never drop a wrong column.
#[test]
fn except_nested_field_is_unsupported() -> R {
    let s = Session::new()?;
    let t = Table {
        name: "t".to_owned(),
        columns: vec!["id".to_owned(), "name".to_owned()],
        rows: vec![vec![Some("1".to_owned()), Some("a".to_owned())]],
    };
    register(&s, &t, "t")?;
    for (sql, feature) in [
        (
            "SELECT * EXCEPT (id.sub) FROM t",
            "SELECT * EXCEPT with nested column `id.sub`",
        ),
        (
            "SELECT t.* EXCEPT (t.id.sub) FROM t",
            "SELECT * EXCEPT with nested column `t.id.sub`",
        ),
    ] {
        let e = match block_on(s.sql(sql)) {
            Err(e) => e,
            Ok(df) => block_on(df.collect()).expect_err("nested EXCEPT must fail"),
        };
        assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
        assert!(
            e.message_parameters()
                .iter()
                .any(|(k, v)| *k == "feature" && v == feature),
            "{sql}: {e}"
        );
    }
    Ok(())
}

/// Overlapping `EXCEPT` entries fail as Spark's `EXCEPT_OVERLAPPING_COLUMNS`
/// does, even through nesting prefixes on flat tables.
#[test]
fn except_overlapping_columns() -> R {
    let s = Session::new()?;
    let t = Table {
        name: "t".to_owned(),
        columns: vec!["id".to_owned(), "name".to_owned()],
        rows: vec![vec![Some("1".to_owned()), Some("a".to_owned())]],
    };
    register(&s, &t, "t")?;
    for (sql, columns) in [
        ("SELECT * EXCEPT (id, id) FROM t", "id, id"),
        (
            "SELECT * EXCEPT (name, name.first) FROM t",
            "name, name.first",
        ),
    ] {
        let e = match block_on(s.sql(sql)) {
            Err(e) => e,
            Ok(df) => block_on(df.collect()).expect_err("overlapping EXCEPT must fail"),
        };
        assert_eq!(e.error_class().name(), "EXCEPT_OVERLAPPING_COLUMNS");
        assert!(
            e.message_parameters()
                .iter()
                .any(|(k, v)| *k == "columns" && v == columns),
            "{sql}: {e}"
        );
    }
    Ok(())
}

/// `NATURAL JOIN` with no common column is a cross join, as Spark's is.
#[test]
fn natural_join_without_common_columns_is_a_cross_join() -> R {
    let s = Session::new()?;
    for (name, column, values) in [("l", "a", vec!["1", "2"]), ("r", "b", vec!["3", "4"])] {
        let t = Table {
            name: name.to_owned(),
            columns: vec![column.to_owned()],
            rows: values
                .into_iter()
                .map(|v| vec![Some(v.to_owned())])
                .collect(),
        };
        register(&s, &t, name)?;
    }
    let df = block_on(s.sql("SELECT * FROM l NATURAL JOIN r"))?;
    let batches = block_on(df.collect())?;
    let rows: Vec<String> = batches
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
    assert_eq!(rows.len(), 4, "{rows:?}");
    Ok(())
}

/// `QUALIFY` over a global aggregate filters the select list's windows.
#[test]
fn qualify_over_global_aggregate() -> R {
    let s = Session::new()?;
    let t = Table {
        name: "t".to_owned(),
        columns: vec!["a".to_owned()],
        rows: vec![
            vec![Some("2".to_owned())],
            vec![Some("1".to_owned())],
            vec![Some("3".to_owned())],
        ],
    };
    register(&s, &t, "t")?;
    let df = block_on(
        s.sql("SELECT SUM(a) AS s, RANK() OVER (ORDER BY SUM(a)) AS r FROM t QUALIFY r = 1"),
    )?;
    let got_schema = block_on(df.schema())?;
    let batches = block_on(df.collect())?;
    let (schema, rows) = render(&got_schema, &batches);
    assert_eq!(schema, "struct<s:bigint,r:int>");
    assert_eq!(rows, ["6\t1"]);
    Ok(())
}

/// A stray `qualify` call outside `QUALIFY` is not a routine.
#[test]
fn qualify_outside_qualify_is_unresolved() -> R {
    let s = Session::new()?;
    let t = Table {
        name: "t".to_owned(),
        columns: vec!["a".to_owned()],
        rows: vec![vec![Some("1".to_owned())]],
    };
    register(&s, &t, "t")?;
    let e = match block_on(s.sql("SELECT qualify(a) FROM t")) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).expect_err("stray qualify must fail"),
    };
    assert_eq!(e.error_class().name(), "UNRESOLVED_ROUTINE");
    Ok(())
}

/// Two small integer tables for the cases below: `l(a, b)` and `r(a, q)`.
fn join_tables(s: &Session) -> R {
    for (name, columns, rows) in [
        ("l", ["a", "b"], [["1", "2"], ["3", "4"]]),
        ("r", ["a", "q"], [["1", "2"], ["5", "6"]]),
    ] {
        let t = Table {
            name: name.to_owned(),
            columns: columns.map(str::to_owned).to_vec(),
            rows: rows
                .iter()
                .map(|r| r.iter().map(|v| Some((*v).to_owned())).collect())
                .collect(),
        };
        register(s, &t, name)?;
    }
    Ok(())
}

/// The error of a query Spark rejects (at planning or at execution).
fn error_of(s: &Session, sql: &str) -> R<KrishivError> {
    match block_on(s.sql(sql)) {
        Err(e) => Ok(e),
        Ok(df) => match block_on(df.collect()) {
            Err(e) => Ok(e),
            Ok(_) => Err(format!("{sql} must fail").into()),
        },
    }
}

/// `SORT BY ALL` is not `ORDER BY ALL`: `all` is an unknown column there
/// (`UNRESOLVED_COLUMN`, checked against Spark 4.2.0), while `ORDER BY ALL`
/// and the DataFrame `orderBy("all")` expand.
#[test]
fn sort_by_all_is_an_unresolved_column() -> R {
    let s = Session::new()?;
    join_tables(&s)?;
    for sql in [
        "SELECT a, b FROM l SORT BY ALL",
        "SELECT a, b FROM l SORT BY all",
    ] {
        let e = error_of(&s, sql)?;
        assert!(
            e.error_class().name().starts_with("UNRESOLVED_COLUMN"),
            "{sql}: {e}"
        );
    }
    let ordered = block_on(s.sql("SELECT a, b FROM l ORDER BY ALL DESC"))?;
    assert_eq!(
        block_on(ordered.collect())?
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
    let ordered = s.table("l").order_by([col("all")]);
    assert_eq!(
        block_on(ordered.collect())?
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
    Ok(())
}

/// `GROUP BY ALL` is SQL only: a DataFrame `groupBy("all")` is an unknown
/// column, as in Spark.
#[test]
fn dataframe_group_by_all_does_not_expand() -> R {
    let s = Session::new()?;
    join_tables(&s)?;
    let df = s.table("l").group_by([col("all")]).agg([sum(col("b"))]);
    let e = block_on(df.collect()).expect_err("groupBy(\"all\") must fail");
    assert!(
        e.error_class().name().starts_with("UNRESOLVED_COLUMN"),
        "{e}"
    );
    // The SQL clause still groups by the select list, and a column named
    // `all` takes the name.
    let df = block_on(s.sql("SELECT a, sum(b) FROM l GROUP BY ALL"))?;
    assert_eq!(
        block_on(df.collect())?
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
    Ok(())
}

/// `NATURAL LEFT/RIGHT JOIN` with no common column keeps its kind with
/// no condition (Spark returns the product), and `NATURAL LEFT SEMI/ANTI`
/// is rejected (Spark: `requirement failed: Unsupported natural join type`).
#[test]
fn natural_outer_joins_without_common_columns_and_semi_joins() -> R {
    let s = Session::new()?;
    for (name, column, values) in [("p", "x", vec!["1", "2"]), ("q", "y", vec!["3", "4"])] {
        let t = Table {
            name: name.to_owned(),
            columns: vec![column.to_owned()],
            rows: values
                .into_iter()
                .map(|v| vec![Some(v.to_owned())])
                .collect(),
        };
        register(&s, &t, name)?;
    }
    for kind in ["LEFT", "RIGHT"] {
        let df = block_on(s.sql(&format!("SELECT * FROM p NATURAL {kind} JOIN q")))?;
        let rows: usize = block_on(df.collect())?
            .iter()
            .map(RecordBatch::num_rows)
            .sum();
        assert_eq!(rows, 4, "NATURAL {kind} JOIN");
    }
    join_tables(&s)?;
    for kind in ["LEFT SEMI", "LEFT ANTI"] {
        let e = error_of(&s, &format!("SELECT * FROM l NATURAL {kind} JOIN r"))?;
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{kind}"
        );
    }
    Ok(())
}

/// `SELECT * EXCEPT (a)` over a join that outputs `a` twice is ambiguous.
#[test]
fn except_over_a_repeated_name_is_ambiguous() -> R {
    let s = Session::new()?;
    join_tables(&s)?;
    let e = error_of(&s, "SELECT * EXCEPT (a) FROM l JOIN r ON l.b = r.q")?;
    assert_eq!(e.error_class().name(), "AMBIGUOUS_REFERENCE");
    Ok(())
}

/// The `UNRESOLVED_COLUMN` proposals rank quoted candidates by edit
/// distance to the written name (case ignored, qualified or not), with
/// ties in column order (each expected list was taken from Spark 4.2.0).
#[test]
fn unresolved_column_proposals_rank_like_spark() -> R {
    let s = Session::new()?;
    join_tables(&s)?;
    for (sql, proposal) in [
        ("SELECT aa FROM l", "`a`, `b`"),
        ("SELECT l.bb FROM l", "`l`.`b`, `l`.`a`"),
        ("SELECT AA FROM l", "`a`, `b`"),
        ("SELECT zz FROM l", "`a`, `b`"),
    ] {
        let e = error_of(&s, sql)?;
        assert_eq!(
            e.error_class().name(),
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            "{sql}"
        );
        assert!(
            e.message_parameters()
                .iter()
                .any(|(k, v)| *k == "proposal" && v == proposal),
            "{sql}: {e}"
        );
    }
    Ok(())
}

/// `QUALIFY` needs a window function in the select list and rejects an
/// aggregate in its predicate (the goldens' two error statements, named).
#[test]
fn qualify_errors() -> R {
    let s = Session::new()?;
    join_tables(&s)?;
    for (sql, class) in [
        (
            "SELECT a FROM l QUALIFY a = 1",
            "QUALIFY_REQUIRES_WINDOW_FUNCTION",
        ),
        (
            "SELECT a, RANK() OVER (ORDER BY b) AS rank FROM l QUALIFY COUNT(1) > 1",
            "QUALIFY_AGGREGATE_NOT_ALLOWED",
        ),
    ] {
        assert_eq!(error_of(&s, sql)?.error_class().name(), class, "{sql}");
    }
    Ok(())
}
