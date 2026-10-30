// SPDX-License-Identifier: Apache-2.0
//! `krishiv sql`: runs SQL from `-e`, `-f` or stdin and prints the rows.

use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::time::Instant;

use crate::common::{build_session, key_value, read_input, split_statements};
use crate::format::{OutputMode, render_mode};
use clap::{Args, ValueEnum};
use krishiv::DataFrame;
use krishiv::agent::{cell, session_tz_micros};
use krishiv::arrow::array::Array;
use krishiv::arrow::datatypes::DataType;

/// Output formats of a result set.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum Format {
    /// Aligned text table with a row count.
    Table,
    /// RFC 4180 CSV with a header row.
    Csv,
    /// Tab-separated values with a header row.
    Tsv,
    /// One JSON object per row (JSON Lines).
    Json,
    /// Alias of `json`.
    Jsonl,
    /// GitHub-flavoured Markdown table.
    Markdown,
    /// `column = value` lines, blank-separated rows.
    Line,
}

impl Format {
    /// The renderer behind the flag value.
    fn mode(self) -> OutputMode {
        match self {
            Format::Table => OutputMode::Table,
            Format::Csv => OutputMode::Csv,
            Format::Tsv => OutputMode::Tsv,
            Format::Json | Format::Jsonl => OutputMode::Json,
            Format::Markdown => OutputMode::Markdown,
            Format::Line => OutputMode::Line,
        }
    }
}

/// Output formats of `--explain-analyze` and `--explain-verbose`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum ExplainFormat {
    /// Plain text (the default).
    Text,
    /// The versioned JSON document (experimental: may change before 1.0).
    Json,
    /// A Mermaid `flowchart TD` of the plan or profile DAG.
    Mermaid,
    /// A Graphviz `digraph` of the plan or profile DAG.
    Dot,
}

impl ExplainFormat {
    /// The [`krishiv::GraphFormat`] of a non-text CLI format; `None` for
    /// [`Self::Text`]. Reuses [`krishiv::GraphFormat::as_str`] for the shared
    /// names instead of a second string table.
    fn graph_format(self) -> Option<krishiv::GraphFormat> {
        match self {
            Self::Text => None,
            Self::Json => Some(krishiv::GraphFormat::Json),
            Self::Mermaid => Some(krishiv::GraphFormat::Mermaid),
            Self::Dot => Some(krishiv::GraphFormat::Dot),
        }
    }

    fn name(self) -> &'static str {
        self.graph_format()
            .map_or("text", krishiv::GraphFormat::as_str)
    }
}

/// Arguments of `krishiv sql`.
#[derive(Args, Debug)]
#[command(group(clap::ArgGroup::new("explain").args(["explain_analyze", "explain_verbose"])))]
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
    /// Output format (default `table`; with stdout piped and no `--format`,
    /// agent mode instead: the `krishiv.agent-result` v1 table, errors
    /// as JSON).
    #[arg(short = 'o', long = "format", value_enum)]
    format: Option<Format>,
    /// Agent-facing output: the `krishiv.agent-result` v1 table (compact,
    /// injection-safe, with a `-- rows:` footer) and errors as one JSON
    /// object each. Implied when stdout is not a TTY unless `--format` is
    /// given.
    #[arg(long)]
    agent: bool,
    /// Most result rows printed per statement. Default 1,000 in agent
    /// mode and the REPL, unlimited for `--format` output to a file or
    /// pipe. Agent mode stops reading the query one row past the cap and
    /// reports `stopped (row cap)` with a `≥ N` lower bound.
    #[arg(long)]
    max_rows: Option<usize>,
    /// Wall-clock budget per statement in seconds (agent mode): checked
    /// between result batches; when spent, the query is dropped and the
    /// table reports `stopped (time cap)` with the rows read so far. A
    /// single batch that never arrives is not interrupted.
    #[arg(long, value_name = "SECS", value_parser = positive_seconds)]
    max_seconds: Option<f64>,
    /// Session configuration entry; repeatable.
    #[arg(long = "conf", value_name = "KEY=VALUE", value_parser = key_value)]
    conf: Vec<(String, String)>,
    /// Register a Parquet file or directory as a temp view; repeatable.
    #[arg(long = "table", value_name = "NAME=PATH", value_parser = key_value)]
    table: Vec<(String, String)>,
    /// Print each statement's elapsed time on stderr.
    #[arg(long)]
    timing: bool,
    /// Run each query and print its profile (`EXPLAIN ANALYZE`) instead of its rows.
    #[arg(long = "explain-analyze", conflicts_with = "explain_verbose")]
    explain_analyze: bool,
    /// Print each query's plan after every optimizer rule that changed it
    /// (`EXPLAIN VERBOSE`) instead of running it.
    #[arg(long = "explain-verbose")]
    explain_verbose: bool,
    /// With --explain-analyze or --explain-verbose: print the versioned JSON
    /// document (one line per statement) instead of text (experimental: may
    /// change before 1.0). A synonym for `--explain-format json`.
    #[arg(long, requires = "explain")]
    json: bool,
    /// With --explain-analyze or --explain-verbose: the output format
    /// (`text`, `json`, `mermaid` or `dot`). `mermaid` and `dot` graph the
    /// physical plan (for --explain-verbose the optimizer steps stay in
    /// `text`/`json` only).
    #[arg(long = "explain-format", value_enum, requires = "explain")]
    explain_format: Option<ExplainFormat>,
}

/// Default row cap of agent mode and the REPL.
const DEFAULT_MAX_ROWS: usize = 1000;

/// A finite, positive number of seconds.
fn positive_seconds(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("`{s}` is not a number"))?;
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(format!("`{s}` must be a finite number of seconds above 0"))
    }
}

/// Runs the command; the process exit code (0 ok, 1 query or I/O error).
pub(crate) fn run(args: &SqlArgs) -> i32 {
    if args.json
        && let Some(format) = args.explain_format
        && format != ExplainFormat::Json
    {
        let _ = writeln!(
            io::stderr(),
            "error: the argument '--json' cannot be used with '--explain-format {}'\n\nFor more information, try '--help'.",
            format.name(),
        );
        return 2;
    }
    if args.execute.is_none() && args.file.is_none() && io::stdin().is_terminal() {
        return run_repl_command(args);
    }
    match run_inner(args) {
        Ok(()) => 0,
        Err(msg) => {
            if !msg.is_empty() {
                let _ = writeln!(io::stderr(), "{msg}");
            }
            1
        }
    }
}

/// The interactive REPL (`krishiv sql` with no `-e`/`-f` on a terminal).
fn run_repl_command(args: &SqlArgs) -> i32 {
    match build_session(&args.conf, &args.table) {
        Ok(session) => crate::repl::run_repl(&session, args.max_rows.unwrap_or(DEFAULT_MAX_ROWS)),
        Err(msg) => {
            let _ = writeln!(io::stderr(), "{msg}");
            1
        }
    }
}

/// Whether agent mode applies: `--agent`, or stdout piped/redirected
/// without an explicit `--format`.
fn is_agent(args: &SqlArgs) -> bool {
    args.agent || (!io::stdout().is_terminal() && args.format.is_none())
}

fn run_inner(args: &SqlArgs) -> Result<(), String> {
    let input = read_input(args.execute.as_ref(), args.file.as_ref())?;
    let session = build_session(&args.conf, &args.table)?;
    let agent = is_agent(args);
    let mode = args.format.map_or(OutputMode::Table, Format::mode);
    let mut out = io::BufWriter::new(io::stdout().lock());
    for (idx, (line_offset, stmt)) in split_statements(&input).into_iter().enumerate() {
        let start = Instant::now();
        let query_id = format!("q{}", idx.saturating_add(1));
        let df = futures::executor::block_on(session.sql(&stmt)).map_err(|e| {
            if agent {
                print_agent_error(
                    &mut out,
                    &AgentStatement {
                        text: &stmt,
                        line_offset,
                        query_id: &query_id,
                        start: &start,
                    },
                    &e,
                );
                let _ = out.flush();
            }
            describe(&stmt, line_offset, &e)
        })?;
        if args.explain_analyze || args.explain_verbose {
            print_explain(&df, args)?;
        } else if agent {
            print_agent_result(
                &mut out,
                &AgentStatement {
                    text: &stmt,
                    line_offset,
                    query_id: &query_id,
                    start: &start,
                },
                &df,
                &session,
                args,
            )?;
        } else {
            print_result(&mut out, &df, &session, mode, args)?;
        }
        // Flush per statement: on the next statement's failure the rows
        // and agent errors printed so far must already be out.
        out.flush()
            .map_err(|e| format!("cannot write output: {e}"))?;
        if args.timing {
            let _ = writeln!(io::stderr(), "Time: {:.3}s", start.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

/// Runs one REPL statement and prints its result; the REPL exit-code
/// contribution (0 ok, 1 query or I/O error). Timing prints on stderr
/// when `timer` (`.timer on`).
pub(crate) fn print_batches(
    session: &krishiv::Session,
    stmt: &str,
    mode: OutputMode,
    max_rows: usize,
    timer: bool,
    out: &mut dyn Write,
) -> i32 {
    let start = Instant::now();
    let outcome = run_statement(session, stmt, mode, max_rows, out);
    if timer {
        let _ = writeln!(io::stderr(), "Time: {:.3}s", start.elapsed().as_secs_f64());
    }
    outcome
}

/// Collects `stmt` and renders it in `mode`, capped to `max_rows` rows
/// for display; 0 on success, else the error text.
fn run_statement(
    session: &krishiv::Session,
    stmt: &str,
    mode: OutputMode,
    max_rows: usize,
    out: &mut dyn Write,
) -> i32 {
    let df = match futures::executor::block_on(session.sql(stmt)) {
        Ok(df) => df,
        Err(e) => {
            let _ = writeln!(io::stderr(), "{}", describe(stmt, 0, &e));
            return 1;
        }
    };
    match collect_rows(&df, session, max_rows) {
        Ok(c) => {
            let _ = write!(
                out,
                "{}",
                render_mode(
                    mode, &c.names, &c.rows, &c.numeric, &c.float, &c.bools, &c.nulls
                )
            );
            let _ = out.flush();
            0
        }
        Err(msg) => {
            let _ = writeln!(io::stderr(), "{msg}");
            1
        }
    }
}

/// The output format of `--explain-analyze` and `--explain-verbose`:
/// `--json` is a synonym for `--explain-format json`.
fn explain_format(args: &SqlArgs) -> ExplainFormat {
    if args.json {
        ExplainFormat::Json
    } else {
        args.explain_format.unwrap_or(ExplainFormat::Text)
    }
}

/// Prints the profile (`--explain-analyze`) or the optimizer's stages
/// (`--explain-verbose`) of the query `df`, in the requested format.
fn print_explain(df: &DataFrame, args: &SqlArgs) -> Result<(), String> {
    // One run per branch: the `Json`/`Mermaid`/`Dot` arms share
    // `explain_*_with`, so there is a single route to each document and no
    // second rendering that can drift.
    let format = explain_format(args);
    let text = if args.explain_analyze {
        match format.graph_format() {
            None => {
                let p =
                    futures::executor::block_on(df.explain_analyze()).map_err(|e| e.to_string())?;
                p.to_text(krishiv::TextOptions::default())
            }
            Some(graph) => futures::executor::block_on(df.explain_analyze_with(graph))
                .map_err(|e| e.to_string())?,
        }
    } else {
        match format.graph_format() {
            None => {
                let v =
                    futures::executor::block_on(df.explain_verbose()).map_err(|e| e.to_string())?;
                v.to_text()
            }
            Some(graph) => futures::executor::block_on(df.explain_verbose_with(graph))
                .map_err(|e| e.to_string())?,
        }
    };
    let mut out = io::stdout().lock();
    writeln!(out, "{}", text.trim_end()).map_err(|e| format!("cannot write output: {e}"))
}

/// `error` of the `line_offset`-th input line's statement `stmt` as shown
/// to the user: a syntax error in Spark's `ParseException` layout
/// ([`krishiv_sql::parse_exception_suffix`]), otherwise the message with the SQL span
/// that raised it marked. The text shown is the statement with its secrets
/// redacted, and line numbers count from the start of the input on both
/// paths: the summary's headers are shifted by `line_offset`, whose lines
/// the contexts count from the statement's own first line.
fn describe(stmt: &str, line_offset: usize, error: &krishiv::Error) -> String {
    let redacted = krishiv_sql::redact_sql(stmt);
    let text = redacted.as_str();
    let parse = krishiv_sql::parse_exception_suffix(text, line_offset, error);
    if !parse.is_empty() {
        return format!("{error}{parse}");
    }
    shift_lines(
        &krishiv_common::error::render_with_context(error, text),
        line_offset,
    )
}

/// `rendered` with every `== SQL (line N, ...)` header moved `line_offset`
/// lines down, so a statement past the first shows file-relative lines.
fn shift_lines(rendered: &str, line_offset: usize) -> String {
    if line_offset == 0 {
        return rendered.to_owned();
    }
    let mut out = String::with_capacity(rendered.len());
    let mut first = true;
    for line in rendered.split('\n') {
        if !first {
            out.push('\n');
        }
        first = false;
        out.push_str(&shift_header(line, line_offset));
    }
    out
}

/// `line` with its `== SQL (line N, ...)` header moved `line_offset` lines
/// down; any other line unchanged.
fn shift_header(line: &str, line_offset: usize) -> String {
    let Some(after) = line.strip_prefix("== SQL") else {
        return line.to_owned();
    };
    let Some(found) = after.find("(line ") else {
        return line.to_owned();
    };
    let digits: String = after[found + "(line ".len()..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    let Ok(no) = digits.parse::<usize>() else {
        return line.to_owned();
    };
    let head = &line[..line.len() - after.len() + found + "(line ".len()];
    let tail = &after[found + "(line ".len() + digits.len()..];
    format!("{head}{}{tail}", no.saturating_add(line_offset))
}

fn print_result(
    out: &mut dyn Write,
    df: &DataFrame,
    session: &krishiv::Session,
    mode: OutputMode,
    args: &SqlArgs,
) -> Result<(), String> {
    let c = collect_rows(df, session, args.max_rows.unwrap_or(usize::MAX))?;
    write!(
        out,
        "{}",
        render_mode(
            mode, &c.names, &c.rows, &c.numeric, &c.float, &c.bools, &c.nulls
        )
    )
    .map_err(|e| format!("cannot write output: {e}"))
}

/// The statement an agent result is for.
struct AgentStatement<'a> {
    /// The statement text.
    text: &'a str,
    /// Lines before it in its file (for error positions).
    line_offset: usize,
    /// Its query id (`q1`, `q2`, ...).
    query_id: &'a str,
    /// When it started.
    start: &'a Instant,
}

/// Runs `df` and prints the v1 agent table from `krishiv-session::agent`.
/// Reads the result stream itself: it stops one row past `--max-rows` or
/// when `--max-seconds` is spent (checked between batches) and drops the
/// stream, which cancels the query. A failure while running prints the
/// agent JSON error like a failure while planning does.
fn print_agent_result(
    out: &mut dyn Write,
    stmt: &AgentStatement<'_>,
    df: &DataFrame,
    session: &krishiv::Session,
    args: &SqlArgs,
) -> Result<(), String> {
    use futures::StreamExt as _;

    let fail = |out: &mut dyn Write, e: &krishiv::Error| {
        print_agent_error(out, stmt, e);
        let _ = out.flush();
        describe(stmt.text, stmt.line_offset, e)
    };
    let tz = session_tz_micros(session.conf().get("spark.sql.session.timeZone").as_deref())?;
    let max_rows = args.max_rows.unwrap_or(DEFAULT_MAX_ROWS);
    let mut stream = futures::executor::block_on(df.execute_stream()).map_err(|e| fail(out, &e))?;
    let schema = futures::executor::block_on(df.schema()).map_err(|e| fail(out, &e))?;
    let mut batches = Vec::new();
    let mut seen = 0usize;
    let mut stop = None;
    while let Some(next) = futures::executor::block_on(stream.next()) {
        let batch = next.map_err(|e| fail(out, &e))?;
        seen = seen.saturating_add(batch.num_rows());
        batches.push(batch);
        if seen > max_rows {
            break;
        }
        if args
            .max_seconds
            .is_some_and(|budget| stmt.start.elapsed().as_secs_f64() > budget)
        {
            stop = Some(krishiv::agent::AgentStop::TimeCap);
            break;
        }
    }
    drop(stream);
    if let Ok(warnings) = futures::executor::block_on(df.warnings()) {
        for w in warnings {
            let _ = writeln!(io::stderr(), "warning: {w}");
        }
    }
    let tables = df.input_tables();
    let limits = krishiv::agent::AgentLimits {
        max_rows,
        max_bytes: krishiv::agent::DEFAULT_MAX_BYTES,
        max_cell_chars: krishiv::agent::DEFAULT_MAX_CELL_CHARS,
        max_seconds: args.max_seconds,
    };
    let elapsed_ms = stmt
        .start
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    let rendered = krishiv::agent::render_batches(
        &schema,
        &batches,
        &krishiv::agent::AgentRender {
            tables: &tables,
            limits: &limits,
            elapsed_ms,
            query_id: Some(stmt.query_id),
            tz_micros: tz,
            stop,
        },
    );
    write!(out, "{}", rendered.text).map_err(|e| format!("cannot write output: {e}"))
}

/// One failure as agent JSON on stdout (the human rendering still goes
/// to stderr through the caller's `describe`).
fn print_agent_error(out: &mut dyn Write, stmt: &AgentStatement<'_>, error: &krishiv::Error) {
    // Positions are counted in the statement as written; no statement text is echoed.
    let source = krishiv::agent::ErrorSource {
        text: stmt.text,
        first_line: stmt.line_offset.saturating_add(1),
    };
    let _ = write!(
        out,
        "{}",
        krishiv::agent::render_error(error, Some(&source), Some(stmt.query_id))
    );
}

/// A collected result set: column names, shown rows (SQL NULL as
/// `NULL`), numeric/float column flags and NULL cells. The non-agent modes
/// render what is shown; the agent path renders through
/// `krishiv-session::agent` instead.
struct Collected {
    /// Column names.
    names: Vec<String>,
    /// Shown rows (cut to the caps).
    rows: Vec<Vec<String>>,
    /// Integer and decimal columns (bare in JSON).
    numeric: Vec<bool>,
    /// Float columns (bare when finite in JSON).
    float: Vec<bool>,
    /// Boolean columns (bare in JSON).
    bools: Vec<bool>,
    /// NULL cells.
    nulls: Vec<Vec<bool>>,
}

/// Collects `df` into typed string rows, cut to `max_rows` for display.
fn collect_rows(
    df: &DataFrame,
    session: &krishiv::Session,
    max_rows: usize,
) -> Result<Collected, String> {
    let tz = session_tz_micros(session.conf().get("spark.sql.session.timeZone").as_deref())?;
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
    let float: Vec<bool> = schema
        .fields()
        .iter()
        .map(|f| matches!(f.data_type(), DataType::Float32 | DataType::Float64))
        .collect();
    let bools: Vec<bool> = schema
        .fields()
        .iter()
        .map(|f| *f.data_type() == DataType::Boolean)
        .collect();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut nulls: Vec<Vec<bool>> = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            rows.push(
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r, "NULL", tz))
                    .collect(),
            );
            nulls.push(b.columns().iter().map(|c| c.is_null(r)).collect());
        }
    }
    if rows.len() > max_rows {
        rows.truncate(max_rows);
        nulls.truncate(max_rows);
    }
    Ok(Collected {
        names,
        rows,
        numeric,
        float,
        bools,
        nulls,
    })
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv::arrow::array::{
        ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray,
    };
    use krishiv::arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    #[test]
    fn explain_format_names_reuse_graph_format() {
        assert_eq!(ExplainFormat::Text.graph_format(), None);
        assert_eq!(
            ExplainFormat::Json.graph_format(),
            Some(krishiv::GraphFormat::Json)
        );
        assert_eq!(
            ExplainFormat::Mermaid.graph_format(),
            Some(krishiv::GraphFormat::Mermaid)
        );
        assert_eq!(
            ExplainFormat::Dot.graph_format(),
            Some(krishiv::GraphFormat::Dot)
        );
        assert_eq!(ExplainFormat::Text.name(), "text");
        assert_eq!(
            ExplainFormat::Mermaid.name(),
            krishiv::GraphFormat::Mermaid.as_str()
        );
    }

    fn arr<A: Array + 'static>(a: A) -> ArrayRef {
        std::sync::Arc::new(a)
    }

    fn batch_rows() -> Collected {
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
        let names = vec!["i".into(), "f".into(), "s".into(), "b".into()];
        let numeric = vec![true, false, false, false];
        let float = vec![false, true, false, false];
        let bools = vec![false, false, false, true];
        let mut rows = Vec::new();
        let mut nulls = Vec::new();
        for r in 0..b.num_rows() {
            rows.push(
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r, "NULL", 0))
                    .collect(),
            );
            nulls.push(b.columns().iter().map(|c| c.is_null(r)).collect());
        }
        Collected {
            names,
            rows,
            numeric,
            float,
            bools,
            nulls,
        }
    }

    #[test]
    fn modes_format_nulls_nan_and_widths() {
        let c = batch_rows();
        let json = render_mode(
            OutputMode::Json,
            &c.names,
            &c.rows,
            &c.numeric,
            &c.float,
            &c.bools,
            &c.nulls,
        );
        assert_eq!(
            json,
            "{\"i\":1,\"f\":\"NaN\",\"s\":\"a\\\"b\",\"b\":true}\n{\"i\":null,\"f\":2.5,\"s\":null,\"b\":null}\n"
        );
        let csv = render_mode(
            OutputMode::Csv,
            &c.names,
            &c.rows,
            &c.numeric,
            &c.float,
            &c.bools,
            &c.nulls,
        );
        assert_eq!(csv, "i,f,s,b\n1,NaN,\"a\"\"b\",true\n,2.5,,\n");
        let tsv = render_mode(
            OutputMode::Tsv,
            &c.names,
            &c.rows,
            &c.numeric,
            &c.float,
            &c.bools,
            &c.nulls,
        );
        assert_eq!(tsv, "i\tf\ts\tb\n1\tNaN\ta\"b\ttrue\n\t2.5\t\t\n");
        let table = render_mode(
            OutputMode::Table,
            &c.names,
            &c.rows,
            &c.numeric,
            &c.float,
            &c.bools,
            &c.nulls,
        );
        assert!(table.contains("| NULL | 2.5 | NULL | NULL |"), "{table}");
        assert!(table.ends_with("2 rows\n"), "{table}");
    }

    #[test]
    fn parse_suffix_is_empty_for_non_parse_errors() {
        let e = krishiv::Session::new()
            .and_then(|s| futures::executor::block_on(s.sql("select * from nope")).map(|_| ()))
            .expect_err("unknown table");
        assert_eq!(
            krishiv_sql::parse_exception_suffix("select * from nope", 0, &e),
            ""
        );
    }

    #[test]
    fn describe_counts_lines_from_the_file_start() {
        let stmt = "select x\nfrom (select 1 as a) t";
        let e = krishiv::Session::new()
            .and_then(|s| futures::executor::block_on(s.sql(stmt)).map(|_| ()))
            .expect_err("unknown column");
        let shown = describe(stmt, 5, &e);
        assert!(shown.starts_with(&e.to_string()), "{shown}");
        assert!(
            shown.ends_with("\n== SQL (line 6, position 8) ==\nselect x\n       ^"),
            "{shown}"
        );
        // A later statement's syntax error already counted from the file.
        let parse = krishiv::Session::new()
            .and_then(|s| futures::executor::block_on(s.sql("xyz 1")).map(|_| ()))
            .expect_err("syntax error");
        let shown = describe("xyz 1", 2, &parse);
        assert!(
            shown.ends_with(" (line 3, pos 0)\n\n== SQL ==\nxyz 1\n^^^\n"),
            "{shown}"
        );
    }

    #[test]
    fn describe_keeps_the_parse_exception_layout_for_syntax_errors() {
        let e = krishiv::Session::new()
            .and_then(|s| futures::executor::block_on(s.sql("xyz 1")).map(|_| ()))
            .expect_err("syntax error");
        let shown = describe("xyz 1", 0, &e);
        assert!(
            shown.ends_with(" (line 1, pos 0)\n\n== SQL ==\nxyz 1\n^^^\n"),
            "{shown}"
        );
        assert_eq!(shown.matches("== SQL").count(), 1, "{shown}");
    }

    #[test]
    fn json_number_types() {
        assert!(is_json_number(&DataType::Int64));
        assert!(is_json_number(&DataType::Decimal128(10, 2)));
        assert!(!is_json_number(&DataType::Utf8));
    }
}
