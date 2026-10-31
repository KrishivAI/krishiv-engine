// SPDX-License-Identifier: Apache-2.0
//! The interactive REPL: `krishiv sql` with no `-e`/`-f` and a terminal on
//! stdin.
//!
//! The loop is deliberately line-based (no line-editor dependency): it
//! prints a `krishiv>` prompt, reads lines, runs dot-commands
//! (`.timer`, `.format`, `.help`, ...) and SQL statements, and persists
//! history through [`crate::history`]. Completion ([`crate::complete`])
//! and highlighting ([`crate::highlight`]) are pure functions over the
//! current line so that a line editor can adopt them later without
//! changing their behavior or tests.

use std::io::{self, BufRead, Write};

use crate::common::split_statements;
use crate::complete::CompletionContext;
use crate::format::OutputMode;
use crate::sql::{AgentStatement, DEFAULT_MAX_ROWS, SqlArgs, is_agent, run_one};

/// Dot-commands the REPL understands.
const HELP: &str = "\
dot-commands:
  .help              show this help
  .quit | .exit      leave the REPL
  .timer on|off      print wall time after each statement (default off)
  .agent on|off      agent output (krishiv.agent-result v1 table) instead of the mode
  .format NAME       output mode: table csv tsv json markdown line (default table)
  .complete PREFIX   list completions for PREFIX (keywords, functions, tables, …)
  .history [N]       show the last N history entries (default 20)
SQL statements end with `;`; an empty line runs nothing.";

/// REPL state: timer flag, output mode and agent output.
#[derive(Debug)]
pub(crate) struct Repl {
    /// Print wall time after each statement.
    pub(crate) timer: bool,
    /// Output mode for result sets.
    pub(crate) mode: OutputMode,
    /// Most result rows printed per statement.
    pub(crate) max_rows: usize,
    /// Print results as the `krishiv.agent-result` v1 table instead of `mode`.
    pub(crate) agent: bool,
}

impl Repl {
    /// Default state: timer off, table output (tests; the REPL starts from the flags).
    #[cfg(test)]
    pub(crate) fn new(max_rows: usize) -> Self {
        Self {
            timer: false,
            mode: OutputMode::Table,
            max_rows,
            agent: false,
        }
    }

    /// The state the command-line flags ask for: `-o` picks the mode, `--timing` turns the timer on, `--agent`
    /// (or a piped stdout without `-o`) selects agent output.
    pub(crate) fn from_args(args: &SqlArgs) -> Self {
        Self {
            timer: args.timing,
            mode: args.output_mode(),
            max_rows: args.max_rows.unwrap_or(DEFAULT_MAX_ROWS),
            agent: is_agent(args),
        }
    }

    /// Handles one dot-command line (leading `.`); returns `None` for
    /// non-dot lines. The `bool` is whether the REPL should quit.
    /// Completion candidates come from `ctx` (`.complete`).
    pub(crate) fn handle_dot(
        &mut self,
        line: &str,
        ctx: &CompletionContext,
        out: &mut dyn Write,
    ) -> Option<bool> {
        let trimmed = line.trim();
        if !trimmed.starts_with('.') {
            return None;
        }
        let mut words = trimmed.split_whitespace();
        let quit = match words.next().unwrap_or("") {
            ".quit" | ".exit" => true,
            ".help" => {
                let _ = writeln!(out, "{HELP}");
                false
            }
            ".timer" => {
                match words.next().unwrap_or("") {
                    "on" => self.timer = true,
                    "off" => self.timer = false,
                    other => {
                        let _ = writeln!(out, "usage: .timer on|off (got `{other}`)");
                        return Some(false);
                    }
                }
                let _ = writeln!(out, "timer {}", if self.timer { "on" } else { "off" });
                false
            }
            ".agent" => {
                match words.next().unwrap_or("") {
                    "on" => self.agent = true,
                    "off" => self.agent = false,
                    other => {
                        let _ = writeln!(out, "usage: .agent on|off (got `{other}`)");
                        return Some(false);
                    }
                }
                let _ = writeln!(out, "agent {}", if self.agent { "on" } else { "off" });
                false
            }
            ".format" | ".mode" => {
                match parse_mode(words.next().unwrap_or("")) {
                    Some(mode) => {
                        self.mode = mode;
                        // Choosing a format means plain rows: agent output would hide it.
                        self.agent = false;
                        let _ = writeln!(out, "format {}", self.mode.name());
                    }
                    None => {
                        let _ = writeln!(out, "usage: .format table|csv|tsv|json|markdown|line");
                        return Some(false);
                    }
                }
                false
            }
            ".complete" => {
                let prefix = words.next().unwrap_or("");
                for c in crate::complete::complete(prefix, prefix.len(), ctx) {
                    let _ = writeln!(out, "{c}");
                }
                false
            }
            ".history" => {
                let count = words
                    .next()
                    .and_then(|n| n.parse::<usize>().ok())
                    .unwrap_or(20);
                if let Some(path) = crate::history::history_path() {
                    let entries = crate::history::load(&path);
                    let from = entries.len().saturating_sub(count);
                    for e in &entries[from..] {
                        let _ = writeln!(out, "{e}");
                    }
                }
                false
            }
            unknown => {
                let _ = writeln!(out, "unknown command `{unknown}` (try .help)");
                false
            }
        };
        Some(quit)
    }
}

/// Parses an output-mode name (case-insensitive); `jsonl` is `json`.
pub(crate) fn parse_mode(name: &str) -> Option<OutputMode> {
    match name.to_ascii_lowercase().as_str() {
        "table" => Some(OutputMode::Table),
        "csv" => Some(OutputMode::Csv),
        "tsv" => Some(OutputMode::Tsv),
        "json" | "jsonl" => Some(OutputMode::Json),
        "markdown" | "md" => Some(OutputMode::Markdown),
        "line" => Some(OutputMode::Line),
        _ => None,
    }
}

/// Runs the REPL on `session`; returns the process exit code.
pub(crate) fn run_repl(session: &krishiv::Session, args: &SqlArgs) -> i32 {
    let mut repl = Repl::from_args(args);
    // Every statement runs with the command-line flags (`--max-seconds`, the explain flags) and the REPL's cap.
    let mut run_args = args.clone();
    let mut query_count = 0_usize;
    let ctx = CompletionContext::gather(session);
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let mut stdout = io::stdout().lock();
    let mut buffer = String::new();
    let history_file = crate::history::history_path();
    let mut history_failed = false;
    let mut failures = 0;
    loop {
        if buffer.trim().is_empty() {
            let _ = write!(stdout, "krishiv> ");
        } else {
            let _ = write!(stdout, "     ...> ");
        }
        let _ = stdout.flush();
        let next = lines.next();
        let Some(read) = next else {
            break;
        };
        let Ok(line) = read else {
            break;
        };
        if buffer.trim().is_empty() {
            if let Some(quit) = repl.handle_dot(&line, &ctx, &mut stdout) {
                if quit {
                    break;
                }
                continue;
            }
            if line.trim().is_empty() {
                continue;
            }
        }
        buffer.push_str(&line);
        buffer.push('\n');
        if !statement_complete(&buffer) {
            continue;
        }
        // The whole statement at once: a secret on a continuation line is
        // judged together with the `SET` that names it.
        if let Some(path) = &history_file
            && let Err(e) = crate::history::append(path, &buffer)
            && !history_failed
        {
            history_failed = true;
            let _ = writeln!(io::stderr(), "history not saved: {e}");
        }
        let stmts = split_statements(&buffer);
        buffer.clear();
        for (_, stmt) in stmts {
            if stmt.trim_start().starts_with('.')
                && let Some(quit) = repl.handle_dot(&stmt, &ctx, &mut stdout)
                && quit
            {
                return i32::from(failures > 0);
            }
            if stmt.trim_start().starts_with('.') {
                continue;
            }
            if crate::highlight::colors_enabled() {
                let _ = writeln!(stdout, "{}", crate::highlight::highlight(&stmt));
            }
            query_count = query_count.saturating_add(1);
            if !run_statement(
                session,
                &repl,
                &mut run_args,
                query_count,
                &stmt,
                &mut stdout,
            ) {
                failures += 1;
            }
        }
    }
    i32::from(failures > 0)
}

/// Runs one statement with the REPL's state and the command-line flags; prints a failure on stderr and the
/// elapsed time when the timer is on. Whether it succeeded.
fn run_statement(
    session: &krishiv::Session,
    repl: &Repl,
    args: &mut SqlArgs,
    number: usize,
    stmt: &str,
    out: &mut dyn Write,
) -> bool {
    args.max_rows = Some(repl.max_rows);
    let start = std::time::Instant::now();
    let query_id = format!("q{number}");
    let statement = AgentStatement {
        text: stmt,
        line_offset: 0,
        query_id: &query_id,
        start: &start,
    };
    let outcome = run_one(session, args, repl.agent, repl.mode, &statement, out);
    let _ = out.flush();
    if repl.timer {
        let _ = writeln!(io::stderr(), "Time: {:.3}s", start.elapsed().as_secs_f64());
    }
    match outcome {
        Ok(()) => true,
        Err(msg) => {
            if !msg.is_empty() {
                let _ = writeln!(io::stderr(), "{msg}");
            }
            false
        }
    }
}

/// Whether `buffer` holds a complete statement: a `;` outside quotes and
/// comments, or a dot-command line.
fn statement_complete(buffer: &str) -> bool {
    if buffer.trim_start().starts_with('.') && !buffer.contains('\n') {
        return true;
    }
    let mut quote: Option<char> = None;
    let mut chars = buffer.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) => {
                if c == '\\' && q != '`' {
                    chars.next();
                } else if c == q {
                    quote = None;
                }
            }
            None => match c {
                '\'' | '"' | '`' => quote = Some(c),
                '-' if chars.peek() == Some(&'-') => {
                    // A line comment ends at the line, not at the buffer.
                    for n in chars.by_ref() {
                        if n == '\n' {
                            break;
                        }
                    }
                }
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    let mut depth = 1_usize;
                    let mut prev = '\0';
                    for n in chars.by_ref() {
                        if prev == '*' && n == '/' {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                            prev = '\0';
                        } else if prev == '/' && n == '*' {
                            depth += 1;
                            prev = '\0';
                        } else {
                            prev = n;
                        }
                    }
                }
                ';' => return true,
                _ => {}
            },
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use crate::format::render_mode;

    use clap::Parser as _;

    use super::*;

    #[derive(clap::Parser)]
    struct Flags {
        #[command(flatten)]
        args: SqlArgs,
    }

    fn flags(argv: &[&str]) -> SqlArgs {
        Flags::try_parse_from(std::iter::once("t").chain(argv.iter().copied()))
            .expect("flags")
            .args
    }

    fn session() -> krishiv::Session {
        crate::common::build_session(&[], &[]).expect("session")
    }

    fn repl() -> (Repl, Vec<u8>) {
        (Repl::new(1000), Vec::new())
    }

    fn ctx() -> CompletionContext {
        CompletionContext::default()
    }

    #[test]
    fn dot_commands_switch_timer_and_format() {
        let (mut r, mut out) = repl();
        let ctx = ctx();
        assert_eq!(r.handle_dot("select 1", &ctx, &mut out), None);
        assert_eq!(r.handle_dot(".timer on", &ctx, &mut out), Some(false));
        assert!(r.timer);
        assert_eq!(r.handle_dot(".timer off", &ctx, &mut out), Some(false));
        assert!(!r.timer);
        assert_eq!(r.handle_dot(".timer bogus", &ctx, &mut out), Some(false));
        assert!(!r.timer);
        assert_eq!(r.handle_dot(".format csv", &ctx, &mut out), Some(false));
        assert_eq!(r.mode, OutputMode::Csv);
        assert_eq!(r.handle_dot(".mode markdown", &ctx, &mut out), Some(false));
        assert_eq!(r.mode, OutputMode::Markdown);
        assert_eq!(r.handle_dot(".format bogus", &ctx, &mut out), Some(false));
        assert_eq!(r.handle_dot(".quit", &ctx, &mut out), Some(true));
        assert_eq!(r.handle_dot(".exit", &ctx, &mut out), Some(true));
        assert_eq!(r.handle_dot(".bogus", &ctx, &mut out), Some(false));
        assert_eq!(r.handle_dot(".complete SEL", &ctx, &mut out), Some(false));
        assert_eq!(r.handle_dot(".history", &ctx, &mut out), Some(false));
        assert_eq!(r.handle_dot(".history 5", &ctx, &mut out), Some(false));
        let text = String::from_utf8(out).expect("utf-8");
        assert!(
            text.contains("timer on") && text.contains("format csv"),
            "{text}"
        );
        assert!(text.contains("SELECT"), "{text}");
    }

    #[test]
    fn mode_names_parse() {
        assert_eq!(parse_mode("TABLE"), Some(OutputMode::Table));
        assert_eq!(parse_mode("jsonl"), Some(OutputMode::Json));
        assert_eq!(parse_mode("md"), Some(OutputMode::Markdown));
        assert_eq!(parse_mode("line"), Some(OutputMode::Line));
        assert_eq!(parse_mode("tsv"), Some(OutputMode::Tsv));
        assert_eq!(parse_mode("nope"), None);
    }

    #[test]
    fn statements_complete_on_top_level_semicolons() {
        assert!(statement_complete("select 1;"));
        assert!(statement_complete("select 'a;b';"));
        assert!(!statement_complete("select 1"));
        assert!(!statement_complete("select 'a;"));
        assert!(!statement_complete("select 1 /* ;"));
        assert!(statement_complete(".help"));
        // A line comment ends at its line: a statement after it completes.
        assert!(statement_complete("-- note\nselect 1;"));
        assert!(!statement_complete("select 1 -- ;\n"));
        assert!(statement_complete("select 1 -- c\n;"));
    }

    #[test]
    fn render_mode_matches_each_renderer() {
        let names = vec!["a".to_owned()];
        let rows = vec![vec!["1".to_owned()]];
        let numeric = [true];
        let float = [false];
        let bools = [false];
        let nulls = vec![vec![false]];
        let table = render_mode(
            OutputMode::Table,
            &names,
            &rows,
            &numeric,
            &float,
            &bools,
            &nulls,
        );
        assert!(table.ends_with("1 row\n"), "{table}");
        let csv = render_mode(
            OutputMode::Csv,
            &names,
            &rows,
            &numeric,
            &float,
            &bools,
            &nulls,
        );
        assert_eq!(csv, "a\n1\n");
        let tsv = render_mode(
            OutputMode::Tsv,
            &names,
            &rows,
            &numeric,
            &float,
            &bools,
            &nulls,
        );
        assert_eq!(tsv, "a\n1\n");
        let json = render_mode(
            OutputMode::Json,
            &names,
            &rows,
            &numeric,
            &float,
            &bools,
            &nulls,
        );
        assert_eq!(json, "{\"a\":1}\n");
        let md = render_mode(
            OutputMode::Markdown,
            &names,
            &rows,
            &numeric,
            &float,
            &bools,
            &nulls,
        );
        assert!(md.contains("| 1 |"), "{md}");
        let line = render_mode(
            OutputMode::Line,
            &names,
            &rows,
            &numeric,
            &float,
            &bools,
            &nulls,
        );
        assert_eq!(line, "a = 1\n\n");
    }

    #[test]
    fn flags_set_the_starting_state() {
        let r = Repl::from_args(&flags(&[
            "-o",
            "csv",
            "--timing",
            "--max-rows",
            "7",
            "--agent",
        ]));
        assert_eq!(
            (r.mode, r.timer, r.max_rows, r.agent),
            (OutputMode::Csv, true, 7, true)
        );
        let plain = Repl::from_args(&flags(&["-o", "json"]));
        assert_eq!(plain.mode, OutputMode::Json);
        assert!(!plain.timer);
        assert_eq!(plain.max_rows, DEFAULT_MAX_ROWS);
    }

    #[test]
    fn the_agent_command_toggles_agent_output() {
        let (mut r, mut out) = repl();
        let ctx = ctx();
        assert_eq!(r.handle_dot(".agent on", &ctx, &mut out), Some(false));
        assert!(r.agent);
        assert_eq!(r.handle_dot(".agent off", &ctx, &mut out), Some(false));
        assert!(!r.agent);
        assert_eq!(r.handle_dot(".agent on", &ctx, &mut out), Some(false));
        assert_eq!(r.handle_dot(".format csv", &ctx, &mut out), Some(false));
        assert!(!r.agent, "choosing a format leaves agent output");
        assert_eq!(r.handle_dot(".agent maybe", &ctx, &mut out), Some(false));
        let text = String::from_utf8(out).expect("utf-8");
        assert!(
            text.contains("agent on")
                && text.contains("agent off")
                && text.contains("usage: .agent"),
            "{text}"
        );
    }

    #[test]
    fn statements_use_the_repl_mode_agent_flag_and_row_cap() {
        let session = session();
        let mut args = flags(&["--max-seconds", "30"]);
        let mut r = Repl::new(2);
        let mut out = Vec::new();
        assert!(run_statement(
            &session,
            &r,
            &mut args,
            1,
            "SELECT id FROM range(5)",
            &mut out
        ));
        let table = String::from_utf8(std::mem::take(&mut out)).expect("utf-8");
        assert!(
            table.contains("2 rows") && !table.contains("-- rows:"),
            "{table}"
        );
        r.agent = true;
        assert!(run_statement(
            &session,
            &r,
            &mut args,
            2,
            "SELECT id FROM range(5)",
            &mut out
        ));
        let agent = String::from_utf8(std::mem::take(&mut out)).expect("utf-8");
        assert!(
            agent.contains("| id:bigint |")
                && agent.contains("-- rows: 2 of ≥ 3 · stopped (row cap)"),
            "{agent}"
        );
        r.agent = false;
        r.mode = OutputMode::Csv;
        assert!(run_statement(
            &session,
            &r,
            &mut args,
            3,
            "SELECT 1 AS a",
            &mut out
        ));
        assert_eq!(
            String::from_utf8(std::mem::take(&mut out)).expect("utf-8"),
            "a\n1\n"
        );
        assert!(!run_statement(
            &session,
            &r,
            &mut args,
            4,
            "NOT SQL AT ALL",
            &mut out
        ));
        // In agent mode a failure prints one JSON error object on `out`.
        r.agent = true;
        assert!(!run_statement(
            &session,
            &r,
            &mut args,
            6,
            "NOT SQL AT ALL",
            &mut out
        ));
        let error = String::from_utf8(std::mem::take(&mut out)).expect("utf-8");
        assert!(
            error.starts_with('{') && error.contains("\"query_id\":\"q6\""),
            "{error}"
        );
        r.agent = false;
        let mut args = flags(&["--explain-verbose"]);
        assert!(run_statement(
            &session,
            &r,
            &mut args,
            5,
            "SELECT 1 AS a",
            &mut out
        ));
        let plan = String::from_utf8(std::mem::take(&mut out)).expect("utf-8");
        assert!(
            !plan.is_empty() && !plan.contains("-- rows:"),
            "explain output goes to the writer: {plan}"
        );
    }

    #[test]
    fn the_time_cap_stops_a_long_read_in_the_repl() {
        let session = session();
        let mut args = flags(&["--max-seconds", "0.05", "--max-rows", "1000000000"]);
        let mut r = Repl::new(1_000_000_000);
        r.agent = true;
        let mut out = Vec::new();
        let started = std::time::Instant::now();
        assert!(run_statement(
            &session,
            &r,
            &mut args,
            1,
            "SELECT id FROM range(2000000000)",
            &mut out
        ));
        let text = String::from_utf8(out).expect("utf-8");
        assert!(text.contains("stopped (time cap)"), "{text}");
        assert!(started.elapsed() < std::time::Duration::from_secs(60));
    }
}
