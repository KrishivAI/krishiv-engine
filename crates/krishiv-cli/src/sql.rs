// SPDX-License-Identifier: Apache-2.0
//! `krishiv sql`: runs SQL from `-e`, `-f` or stdin and prints the rows.

use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::time::Instant;

use crate::common::{build_session, key_value, read_input, split_statements};
use crate::deadline::Deadline;
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
#[derive(Args, Clone, Debug)]
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
    pub(crate) max_rows: Option<usize>,
    /// Wall-clock budget per statement in seconds (agent mode), counted from
    /// the statement's start: when spent, the query is dropped and the table
    /// reports `stopped (time cap)` with the rows read so far, even if the
    /// next batch never arrives. Planning is not interrupted.
    #[arg(long, value_name = "SECS", value_parser = positive_seconds)]
    max_seconds: Option<f64>,
    /// Session configuration entry; repeatable.
    #[arg(long = "conf", value_name = "KEY=VALUE", value_parser = key_value)]
    conf: Vec<(String, String)>,
    /// Register a Parquet file or directory as a temp view; repeatable.
    #[arg(long = "table", value_name = "NAME=PATH", value_parser = key_value)]
    table: Vec<(String, String)>,
    /// Run statements the agent cost gate would stop (estimated table scans of `krishiv.agent.gateBytes`, 1 GiB by
    /// default). Without it such a statement is not run: the estimate and plan are printed and the exit status is 3.
    #[arg(long)]
    allow_heavy: bool,
    /// Print each statement's elapsed time on stderr.
    #[arg(long)]
    pub(crate) timing: bool,
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

impl SqlArgs {
    /// The result format `-o` names (`table` without it).
    pub(crate) fn output_mode(&self) -> OutputMode {
        self.format.map_or(OutputMode::Table, Format::mode)
    }
}

/// Default row cap of agent mode and the REPL.
pub(crate) const DEFAULT_MAX_ROWS: usize = 1000;

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
        Ok(code) => code,
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
        Ok(session) => crate::repl::run_repl(&session, args),
        Err(msg) => {
            let _ = writeln!(io::stderr(), "{msg}");
            1
        }
    }
}

/// Whether agent mode applies: `--agent`, or stdout piped/redirected
/// without an explicit `--format`.
pub(crate) fn is_agent(args: &SqlArgs) -> bool {
    args.agent || (!io::stdout().is_terminal() && args.format.is_none())
}

/// Exit status of a run in which the cost gate stopped a statement.
const EXIT_NOT_RUN: i32 = 3;

fn run_inner(args: &SqlArgs) -> Result<i32, String> {
    let mut code = 0;
    let input = read_input(args.execute.as_ref(), args.file.as_ref())?;
    let session = build_session(&args.conf, &args.table)?;
    let agent = is_agent(args);
    let mode = args.output_mode();
    let mut out = io::BufWriter::new(io::stdout().lock());
    for (idx, (line_offset, stmt)) in split_statements(&input).into_iter().enumerate() {
        let start = Instant::now();
        let query_id = format!("q{}", idx.saturating_add(1));
        let statement = AgentStatement {
            text: &stmt,
            line_offset,
            query_id: &query_id,
            start: &start,
        };
        if run_one(&session, args, agent, mode, &statement, &mut out)? == Ran::NotRun {
            code = EXIT_NOT_RUN;
        }
        // Flush per statement: on the next statement's failure the rows
        // and agent errors printed so far must already be out.
        out.flush()
            .map_err(|e| format!("cannot write output: {e}"))?;
        if args.timing {
            let _ = writeln!(io::stderr(), "Time: {:.3}s", start.elapsed().as_secs_f64());
        }
    }
    Ok(code)
}

/// Plans, runs and prints one statement the way `krishiv sql -e` does: the explain flags, agent output or the
/// chosen `mode`. Shared by the batch path and the REPL, so both honour the same flags.
pub(crate) fn run_one(
    session: &krishiv::Session,
    args: &SqlArgs,
    agent: bool,
    mode: OutputMode,
    statement: &AgentStatement<'_>,
    out: &mut dyn Write,
) -> Result<Ran, String> {
    let stmt = statement.text;
    let df = futures::executor::block_on(session.sql(stmt)).map_err(|e| {
        if agent {
            print_agent_error(out, statement, &e);
            let _ = out.flush();
        }
        describe(stmt, statement.line_offset, &e)
    })?;
    // The gate guards what would run: agent output and `--explain-analyze` (which runs the query).
    if agent && !args.allow_heavy && !args.explain_verbose && gated(session, &df, out)? {
        return Ok(Ran::NotRun);
    }
    if args.explain_analyze || args.explain_verbose {
        print_explain(out, &df, args).map(|()| Ran::Yes)
    } else if agent {
        print_agent_result(out, statement, &df, session, args).map(|()| Ran::Yes)
    } else {
        print_result(out, &df, session, mode, args).map(|()| Ran::Yes)
    }
}

/// Whether a statement ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ran {
    /// It ran (or failed, which is an error).
    Yes,
    /// The cost gate stopped it and its estimate was printed.
    NotRun,
}

/// The agent cost gate (ADR 0041): when the estimated table scans of `df` reach `krishiv.agent.gateBytes`, prints
/// the estimate and the plan outline and returns `true`. An unknown estimate, a statement whose estimate fails (the
/// run raises the same error) and a gate of 0 let it run.
fn gated(session: &krishiv::Session, df: &DataFrame, out: &mut dyn Write) -> Result<bool, String> {
    let gate = gate_bytes(session);
    if gate == 0 {
        return Ok(false);
    }
    let Ok(estimate) = futures::executor::block_on(df.estimate()) else {
        return Ok(false);
    };
    let Some(scan) = over_gate(estimate.scan_bytes, gate) else {
        return Ok(false);
    };
    write!(
        out,
        "{}",
        krishiv::agent::render_estimate(scan, gate, estimate.rows.unwrap_or(0), &estimate.plan)
    )
    .map_err(|e| format!("cannot write output: {e}"))?;
    Ok(true)
}

/// The estimate when it reaches the gate; an unknown estimate and a gate of 0 never do.
fn over_gate(scan_bytes: Option<u64>, gate: u64) -> Option<u64> {
    scan_bytes.filter(|b| gate > 0 && *b >= gate)
}

/// `krishiv.agent.gateBytes` of the session (1 GiB when unset or unreadable).
fn gate_bytes(session: &krishiv::Session) -> u64 {
    session
        .conf()
        .get("krishiv.agent.gateBytes")
        .and_then(|raw| {
            krishiv_common::config::AGENT_GATE_BYTES
                .parse("krishiv.agent.gateBytes", &raw)
                .ok()
        })
        .and_then(|v| v.as_size())
        .unwrap_or(1 << 30)
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
fn print_explain(out: &mut dyn Write, df: &DataFrame, args: &SqlArgs) -> Result<(), String> {
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
pub(crate) struct AgentStatement<'a> {
    /// The statement text.
    pub(crate) text: &'a str,
    /// Lines before it in its file (for error positions).
    pub(crate) line_offset: usize,
    /// Its query id (`q1`, `q2`, ...).
    pub(crate) query_id: &'a str,
    /// When it started.
    pub(crate) start: &'a Instant,
}

/// Reads `stream` until it ends, holds more than `max_rows` rows, or the `max_seconds` budget (counted from
/// `start`) is spent; the batches read and why it stopped early. A batch that never arrives is abandoned at the
/// budget. A budget too large to be a point in time means no cap.
fn read_batches<S>(
    stream: &mut S,
    max_rows: usize,
    max_seconds: Option<f64>,
    start: Instant,
) -> Result<
    (
        Vec<krishiv::arrow::array::RecordBatch>,
        Option<krishiv::agent::AgentStop>,
    ),
    krishiv::Error,
>
where
    S: futures::Stream<Item = Result<krishiv::arrow::array::RecordBatch, krishiv::Error>> + Unpin,
{
    use futures::StreamExt as _;

    let mut batches = Vec::new();
    let mut seen = 0usize;
    let mut deadline = max_seconds.and_then(|budget| Deadline::after(start, budget));
    loop {
        let next = match deadline.as_mut() {
            Some(d) => match d.next_item(stream) {
                Some(next) => next,
                None => return Ok((batches, Some(krishiv::agent::AgentStop::TimeCap))),
            },
            None => futures::executor::block_on(stream.next()),
        };
        let Some(next) = next else {
            return Ok((batches, None));
        };
        let batch = next?;
        seen = seen.saturating_add(batch.num_rows());
        batches.push(batch);
        if seen > max_rows {
            return Ok((batches, None));
        }
        if max_seconds.is_some_and(|budget| start.elapsed().as_secs_f64() > budget) {
            return Ok((batches, Some(krishiv::agent::AgentStop::TimeCap)));
        }
    }
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
    let fail = |out: &mut dyn Write, e: &krishiv::Error| {
        print_agent_error(out, stmt, e);
        let _ = out.flush();
        describe(stmt.text, stmt.line_offset, e)
    };
    let tz = session_tz_micros(session.conf().get("spark.sql.session.timeZone").as_deref())?;
    let max_rows = args.max_rows.unwrap_or(DEFAULT_MAX_ROWS);
    let mut stream = futures::executor::block_on(df.execute_stream()).map_err(|e| fail(out, &e))?;
    let schema = futures::executor::block_on(df.schema()).map_err(|e| fail(out, &e))?;
    let (batches, stop) = read_batches(&mut stream, max_rows, args.max_seconds, *stmt.start)
        .map_err(|e| fail(out, &e))?;
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
    use futures::StreamExt as _;

    let tz = session_tz_micros(session.conf().get("spark.sql.session.timeZone").as_deref())?;
    let schema = futures::executor::block_on(df.schema()).map_err(|e| e.to_string())?;
    // Streams and stops at the cap, so a huge result is never buffered: dropping the stream cancels the query.
    let mut stream = futures::executor::block_on(df.execute_stream()).map_err(|e| e.to_string())?;
    let mut batches = Vec::new();
    let mut seen = 0usize;
    while seen < max_rows {
        let Some(next) = futures::executor::block_on(stream.next()) else {
            break;
        };
        let batch = next.map_err(|e| e.to_string())?;
        seen = seen.saturating_add(batch.num_rows());
        batches.push(batch);
    }
    drop(stream);
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
            if rows.len() >= max_rows {
                break;
            }
            rows.push(
                b.columns()
                    .iter()
                    .map(|c| cell(c.as_ref(), r, "NULL", tz))
                    .collect(),
            );
            nulls.push(b.columns().iter().map(|c| c.is_null(r)).collect());
        }
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

    use futures::StreamExt as _;
    use krishiv::arrow::array::{
        ArrayRef, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray,
    };
    use krishiv::arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn int_batch(rows: i32) -> RecordBatch {
        let col = Int32Array::from_iter_values(0..rows);
        RecordBatch::try_from_iter([("a", Arc::new(col) as ArrayRef)]).expect("batch")
    }

    #[test]
    fn a_batch_that_never_arrives_stops_the_read_at_the_budget() {
        // One batch, then a read that never completes: the old between-batches check never ran again.
        let mut stream =
            futures::stream::iter([Ok(int_batch(3))]).chain(futures::stream::pending());
        let started = Instant::now();
        let (batches, stop) = read_batches(&mut stream, 1000, Some(0.2), started).expect("read");
        assert_eq!(batches.len(), 1);
        assert!(matches!(stop, Some(krishiv::agent::AgentStop::TimeCap)));
        assert!(started.elapsed() < std::time::Duration::from_secs(30));
    }

    #[test]
    fn a_huge_budget_is_no_cap_and_the_row_cap_still_applies() {
        let batches = (0..3).map(|_| Ok(int_batch(2)));
        let mut stream = futures::stream::iter(batches);
        let (read, stop) = read_batches(&mut stream, 3, Some(1e30), Instant::now()).expect("read");
        assert_eq!(read.len(), 2, "stops one batch past the row cap");
        assert!(stop.is_none());
    }

    #[test]
    fn the_gate_stops_at_the_threshold_and_never_on_unknown() {
        assert_eq!(over_gate(Some(1000), 1000), Some(1000));
        assert_eq!(over_gate(Some(999), 1000), None);
        assert_eq!(over_gate(Some(5000), 1000), Some(5000));
        assert_eq!(over_gate(None, 1000), None);
        assert_eq!(over_gate(Some(u64::MAX), 0), None);
    }

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
