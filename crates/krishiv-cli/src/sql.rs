// SPDX-License-Identifier: Apache-2.0
//! `krishiv sql`: runs SQL from `-e`, `-f` or stdin and prints the rows.

use std::fmt::Write as _;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, ValueEnum};
use krishiv::arrow::array::cast::AsArray;
use krishiv::arrow::array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use krishiv::arrow::array::{Array, RecordBatch};
use krishiv::arrow::datatypes::{DataType, TimeUnit};
use krishiv::{DataFrame, Session};

/// Output formats of a result set.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum Format {
    /// Aligned text table with a row count.
    Table,
    /// RFC 4180 CSV with a header row.
    Csv,
    /// One JSON object per row (JSON Lines).
    Json,
}

/// Arguments of `krishiv sql`.
#[derive(Args, Debug)]
pub(crate) struct SqlArgs {
    /// SQL to run (one or more `;`-separated statements).
    #[arg(
        short = 'e',
        long = "execute",
        value_name = "SQL",
        conflicts_with = "file"
    )]
    execute: Option<String>,
    /// Read SQL from a file (`-` is stdin). With neither `-e` nor `-f`, stdin is read.
    #[arg(short = 'f', long = "file", value_name = "PATH")]
    file: Option<PathBuf>,
    /// Output format.
    #[arg(short = 'o', long = "format", value_enum, default_value = "table")]
    format: Format,
    /// Session configuration entry; repeatable.
    #[arg(long = "conf", value_name = "KEY=VALUE", value_parser = key_value)]
    conf: Vec<(String, String)>,
    /// Register a Parquet file or directory as a temp view; repeatable.
    #[arg(long = "table", value_name = "NAME=PATH", value_parser = key_value)]
    table: Vec<(String, String)>,
    /// Print each statement's elapsed time on stderr.
    #[arg(long)]
    timing: bool,
}

fn key_value(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((k, v)) if !k.trim().is_empty() => Ok((k.trim().to_owned(), v.to_owned())),
        _ => Err(format!("expected KEY=VALUE, got `{s}`")),
    }
}

/// Runs the command; the process exit code (0 ok, 1 query or I/O error).
pub(crate) fn run(args: &SqlArgs) -> i32 {
    match run_inner(args) {
        Ok(()) => 0,
        Err(msg) => {
            let _ = writeln!(io::stderr(), "{msg}");
            1
        }
    }
}

fn read_input(args: &SqlArgs) -> Result<String, String> {
    if let Some(sql) = &args.execute {
        return Ok(sql.clone());
    }
    match &args.file {
        Some(p) if p.as_os_str() != "-" => {
            std::fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display()))
        }
        _ => {
            let mut s = String::new();
            io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| format!("cannot read stdin: {e}"))?;
            Ok(s)
        }
    }
}

fn run_inner(args: &SqlArgs) -> Result<(), String> {
    let input = read_input(args)?;
    let mut builder = Session::builder();
    for (k, v) in &args.conf {
        builder = builder.config(k, v);
    }
    let session = builder.build().map_err(|e| e.to_string())?;
    for (name, path) in &args.table {
        let df = session.read().parquet([path.as_str()]);
        futures::executor::block_on(df.create_or_replace_temp_view(name))
            .map_err(|e| format!("--table {name}={path}: {e}"))?;
    }
    for (line_offset, stmt) in split_statements(&input) {
        let start = Instant::now();
        let df = futures::executor::block_on(session.sql(&stmt))
            .map_err(|e| format!("{e}{}", caret(&stmt, line_offset, &e)))?;
        print_result(&df, args.format)?;
        if args.timing {
            let _ = writeln!(io::stderr(), "Time: {:.3}s", start.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

/// Spark-style position and caret for a parse error in `stmt`; empty for any
/// other error. The session drops the parser's span, so the statement is
/// parsed again here to recover it.
fn caret(stmt: &str, line_offset: usize, error: &krishiv::Error) -> String {
    if !error.error_class().name().starts_with("PARSE_")
        && !error.error_class().name().starts_with("UNCLOSED_")
    {
        return String::new();
    }
    let Err(parse) = krishiv_sql::parse_statement(stmt, &krishiv_sql::ParserOptions::default())
    else {
        return String::new();
    };
    let start = parse.span.start.min(stmt.len());
    let before = &stmt[..stmt.floor_char_boundary(start)];
    let line_no = line_offset + before.matches('\n').count() + 1;
    let line_start = before.rfind('\n').map_or(0, |i| i + 1);
    let line = stmt[line_start..].lines().next().unwrap_or("");
    let col = before[line_start..].chars().count();
    let width = stmt[before.len()..parse.span.end.max(start).min(stmt.len())]
        .chars()
        .take_while(|c| *c != '\n')
        .count()
        .max(1);
    format!(
        "\n(line {line_no}, pos {col})\n\n== SQL ==\n{line}\n{}{}",
        " ".repeat(col),
        "^".repeat(width)
    )
}

fn print_result(df: &DataFrame, format: Format) -> Result<(), String> {
    let batches = futures::executor::block_on(df.collect()).map_err(|e| e.to_string())?;
    let schema = futures::executor::block_on(df.schema()).map_err(|e| e.to_string())?;
    if let Ok(warnings) = futures::executor::block_on(df.warnings()) {
        for w in warnings {
            let _ = writeln!(io::stderr(), "warning: {w}");
        }
    }
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let numeric: Vec<bool> = schema
        .fields()
        .iter()
        .map(|f| is_json_number(f.data_type()))
        .collect();
    let mut out = io::BufWriter::new(io::stdout().lock());
    let io_err = |e: io::Error| format!("cannot write output: {e}");
    match format {
        Format::Table => write_table(&mut out, &names, &batches).map_err(io_err)?,
        Format::Csv => write_csv(&mut out, &names, &batches).map_err(io_err)?,
        Format::Json => write_json(&mut out, &names, &numeric, &batches).map_err(io_err)?,
    }
    out.flush().map_err(io_err)
}

fn write_table(out: &mut impl Write, names: &[String], batches: &[RecordBatch]) -> io::Result<()> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            rows.push(
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r, "NULL"))
                    .collect(),
            );
        }
    }
    let mut widths: Vec<usize> = names.iter().map(|n| n.chars().count()).collect();
    for row in &rows {
        for (w, c) in widths.iter_mut().zip(row) {
            *w = (*w).max(c.chars().count());
        }
    }
    let rule: String = widths.iter().fold(String::from("+"), |mut s, w| {
        let _ = write!(s, "{}+", "-".repeat(w + 2));
        s
    });
    let line = |cells: &[String]| -> String {
        let mut s = String::from("|");
        for (c, w) in cells.iter().zip(&widths) {
            let pad = w - c.chars().count();
            let _ = write!(s, " {c}{} |", " ".repeat(pad));
        }
        s
    };
    writeln!(out, "{rule}\n{}\n{rule}", line(names))?;
    for row in &rows {
        writeln!(out, "{}", line(row))?;
    }
    writeln!(out, "{rule}")?;
    writeln!(
        out,
        "{} row{}",
        rows.len(),
        if rows.len() == 1 { "" } else { "s" }
    )
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

fn write_csv(out: &mut impl Write, names: &[String], batches: &[RecordBatch]) -> io::Result<()> {
    let header: Vec<String> = names.iter().map(|n| csv_field(n)).collect();
    writeln!(out, "{}", header.join(","))?;
    for b in batches {
        for r in 0..b.num_rows() {
            let row: Vec<String> = b
                .columns()
                .iter()
                .map(|c| csv_field(&cell(c.as_ref(), r, "")))
                .collect();
            writeln!(out, "{}", row.join(","))?;
        }
    }
    Ok(())
}

fn json_string(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn is_json_number(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Decimal128(..)
    )
}

fn write_json(
    out: &mut impl Write,
    names: &[String],
    numeric: &[bool],
    batches: &[RecordBatch],
) -> io::Result<()> {
    for b in batches {
        for r in 0..b.num_rows() {
            let mut line = String::from("{");
            for (i, c) in b.columns().iter().enumerate() {
                if i > 0 {
                    line.push(',');
                }
                line.push_str(&json_string(&names[i]));
                line.push(':');
                let t = c.data_type();
                if c.is_null(r) {
                    line.push_str("null");
                } else if numeric[i] || *t == DataType::Boolean {
                    line.push_str(&cell(c.as_ref(), r, ""));
                } else if matches!(t, DataType::Float32 | DataType::Float64) {
                    let v = cell(c.as_ref(), r, "");
                    // JSON has no NaN or infinity.
                    if v.parse::<f64>().is_ok_and(f64::is_finite) {
                        line.push_str(&v);
                    } else {
                        line.push_str(&json_string(&v));
                    }
                } else {
                    line.push_str(&json_string(&cell(c.as_ref(), r, "")));
                }
            }
            line.push('}');
            writeln!(out, "{line}")?;
        }
    }
    Ok(())
}

/// One value as text; `null` stands in for SQL NULL.
fn cell(a: &dyn Array, i: usize, null: &str) -> String {
    if a.data_type() == &DataType::Null || a.is_null(i) {
        return null.to_owned();
    }
    match a.data_type() {
        DataType::Boolean => a.as_boolean().value(i).to_string(),
        DataType::Int8 => a.as_primitive::<Int8Type>().value(i).to_string(),
        DataType::Int16 => a.as_primitive::<Int16Type>().value(i).to_string(),
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::UInt8 => a.as_primitive::<UInt8Type>().value(i).to_string(),
        DataType::UInt16 => a.as_primitive::<UInt16Type>().value(i).to_string(),
        DataType::UInt32 => a.as_primitive::<UInt32Type>().value(i).to_string(),
        DataType::UInt64 => a.as_primitive::<UInt64Type>().value(i).to_string(),
        DataType::Float32 => format!("{:?}", a.as_primitive::<Float32Type>().value(i)),
        DataType::Float64 => format!("{:?}", a.as_primitive::<Float64Type>().value(i)),
        DataType::Decimal128(..) => a.as_primitive::<Decimal128Type>().value_as_string(i),
        DataType::Utf8 => a.as_string::<i32>().value(i).to_owned(),
        DataType::LargeUtf8 => a.as_string::<i64>().value(i).to_owned(),
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
        DataType::Binary => hex(a.as_binary::<i32>().value(i)),
        DataType::Date32 => a
            .as_primitive::<Date32Type>()
            .value_as_date(i)
            .map_or_else(String::new, |d| d.to_string()),
        DataType::Timestamp(TimeUnit::Microsecond, _) => a
            .as_primitive::<TimestampMicrosecondType>()
            .value_as_datetime(i)
            .map_or_else(String::new, |d| d.to_string()),
        DataType::List(_) => {
            let inner = a.as_list::<i32>().value(i);
            let items: Vec<String> = (0..inner.len())
                .map(|j| cell(inner.as_ref(), j, "NULL"))
                .collect();
            format!("[{}]", items.join(", "))
        }
        DataType::Struct(fields) => {
            let s = a.as_struct();
            let items: Vec<String> = fields
                .iter()
                .zip(s.columns())
                .map(|(f, c)| format!("{}: {}", f.name(), cell(c.as_ref(), i, "NULL")))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        other => format!("<{other}>"),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Splits `input` at `;` outside quotes and comments; blank statements are
/// dropped. Each statement comes with the number of input lines before it.
fn split_statements(input: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut done_lines = 0;
    let mut cur = String::new();
    let mut chars = input.chars().peekable();
    let mut quote: Option<char> = None;
    while let Some(c) = chars.next() {
        cur.push(c);
        match quote {
            Some(q) => {
                if c == '\\' && q != '`' {
                    if let Some(n) = chars.next() {
                        cur.push(n);
                    }
                } else if c == q {
                    quote = None;
                }
            }
            None => match c {
                '\'' | '"' | '`' => quote = Some(c),
                '-' if chars.peek() == Some(&'-') => {
                    for n in chars.by_ref() {
                        cur.push(n);
                        if n == '\n' {
                            break;
                        }
                    }
                }
                '/' if chars.peek() == Some(&'*') => {
                    let mut prev = '\0';
                    for n in chars.by_ref() {
                        cur.push(n);
                        if prev == '*' && n == '/' {
                            break;
                        }
                        prev = n;
                    }
                }
                ';' => {
                    cur.pop();
                    push_stmt(&mut out, &mut done_lines, &mut cur);
                }
                _ => {}
            },
        }
    }
    push_stmt(&mut out, &mut done_lines, &mut cur);
    out
}

fn push_stmt(out: &mut Vec<(usize, String)>, done_lines: &mut usize, cur: &mut String) {
    if !cur.trim().is_empty() {
        let leading = cur.len() - cur.trim_start().len();
        let skipped = cur[..leading].matches('\n').count();
        out.push((*done_lines + skipped, cur.trim().to_owned()));
    }
    *done_lines += cur.matches('\n').count();
    cur.clear();
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv::arrow::array::{
        ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float64Array,
        Int32Array, ListArray, NullArray, RecordBatch, StringArray, StructArray,
        TimestampMicrosecondArray, UInt8Array,
    };
    use krishiv::arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    #[test]
    fn splits_outside_quotes_and_comments() {
        let s = split_statements("select 'a;b'; -- x;y\nselect 2; /* ; */ select `c;`;;");
        assert_eq!(s.len(), 3, "{s:?}");
        assert_eq!(s[0].1, "select 'a;b'");
        assert_eq!(s[1].0, 0);
        let t = split_statements("select 1;\n\n  select 2");
        assert_eq!(t[1], (2, "select 2".to_owned()));
        let esc = split_statements("select 'it\\'s;' ; select 2");
        assert_eq!(esc.len(), 2, "{esc:?}");
    }

    #[test]
    fn key_value_parsing() {
        assert_eq!(key_value("a=b=c"), Ok(("a".into(), "b=c".into())));
        assert_eq!(key_value(" k = v"), Ok(("k".into(), " v".into())));
        assert!(key_value("novalue").is_err());
        assert!(key_value("=v").is_err());
    }

    #[test]
    fn csv_and_json_escaping() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a\nb"), "\"a\nb\"");
        assert_eq!(
            json_string("a\\b\n\r\t\u{1}\""),
            "\"a\\\\b\\n\\r\\t\\u0001\\\"\""
        );
        assert_eq!(hex(&[0, 255, 16]), "00ff10");
    }

    fn arr<A: Array + 'static>(a: A) -> ArrayRef {
        Arc::new(a)
    }

    #[test]
    fn cell_renders_every_supported_type() {
        let dec = Decimal128Array::from(vec![Some(12345), None])
            .with_precision_and_scale(10, 2)
            .expect("decimal");
        let list =
            ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![Some(1), Some(2)])]);
        let st = StructArray::from(vec![(
            Arc::new(Field::new("x", DataType::Int32, true)),
            arr(Int32Array::from(vec![7])),
        )]);
        let cases: Vec<(ArrayRef, &str)> = vec![
            (arr(BooleanArray::from(vec![true])), "true"),
            (arr(UInt8Array::from(vec![5])), "5"),
            (arr(Float64Array::from(vec![1.0])), "1.0"),
            (arr(dec), "123.45"),
            (arr(StringArray::from(vec!["s"])), "s"),
            (arr(BinaryArray::from(vec![&[1_u8, 171][..]])), "01ab"),
            (arr(Date32Array::from(vec![0])), "1970-01-01"),
            (
                arr(TimestampMicrosecondArray::from(vec![1_000_000])),
                "1970-01-01 00:00:01",
            ),
            (arr(list), "[1, 2]"),
            (arr(st), "{x: 7}"),
            (arr(NullArray::new(1)), "NULL"),
        ];
        for (a, want) in cases {
            assert_eq!(cell(a.as_ref(), 0, "NULL"), want, "{:?}", a.data_type());
        }
        let d = Decimal128Array::from(vec![None::<i128>]);
        assert_eq!(cell(&d, 0, "-"), "-");
        assert_eq!(cell(&arr(NullArray::new(1)), 0, "-"), "-");
    }

    fn batch() -> (Vec<String>, Vec<RecordBatch>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int32, true),
            Field::new("f", DataType::Float64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("b", DataType::Boolean, true),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                arr(Int32Array::from(vec![Some(1), None])),
                arr(Float64Array::from(vec![f64::NAN, 2.5])),
                arr(StringArray::from(vec![Some("a\"b"), None])),
                arr(BooleanArray::from(vec![Some(true), None])),
            ],
        )
        .expect("batch");
        (
            vec!["i".into(), "f".into(), "s".into(), "b".into()],
            vec![b],
        )
    }

    #[test]
    fn writers_format_nulls_nan_and_widths() {
        let (names, batches) = batch();
        let numeric = [true, false, false, false];
        let mut o = Vec::new();
        write_json(&mut o, &names, &numeric, &batches).expect("json");
        let t = String::from_utf8(o).expect("utf-8");
        assert_eq!(
            t,
            "{\"i\":1,\"f\":\"NaN\",\"s\":\"a\\\"b\",\"b\":true}\n{\"i\":null,\"f\":2.5,\"s\":null,\"b\":null}\n"
        );
        let mut o = Vec::new();
        write_csv(&mut o, &names, &batches).expect("csv");
        assert_eq!(
            String::from_utf8(o).expect("utf-8"),
            "i,f,s,b\n1,NaN,\"a\"\"b\",true\n,2.5,,\n"
        );
        let mut o = Vec::new();
        write_table(&mut o, &names, &batches).expect("table");
        let t = String::from_utf8(o).expect("utf-8");
        assert!(t.contains("| NULL | 2.5 | NULL | NULL |"), "{t}");
        assert!(t.ends_with("2 rows\n"), "{t}");
    }

    #[test]
    fn caret_is_empty_for_non_parse_errors() {
        let e = Session::new()
            .and_then(|s| futures::executor::block_on(s.sql("select * from nope")).map(|_| ()))
            .expect_err("unknown table");
        assert_eq!(caret("select * from nope", 0, &e), "");
    }

    #[test]
    fn json_number_types() {
        assert!(is_json_number(&DataType::Int64));
        assert!(is_json_number(&DataType::Decimal128(10, 2)));
        assert!(!is_json_number(&DataType::Utf8));
    }
}
