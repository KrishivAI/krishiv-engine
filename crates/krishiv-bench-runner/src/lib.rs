// SPDX-License-Identifier: Apache-2.0
//! Line-protocol query runner for the bench harness (P2-T142).
//!
//! The process registers every `<table>.parquet` of a data directory as a temporary view, prints
//! `READY <version> <tables>` and then serves one command per stdin line:
//!
//! * `RUN <query.sql>`: plans and executes the query, materialising every result batch. Replies
//!   `OK <rows> <seconds>`; the seconds cover parsing, analysis, planning and execution.
//! * `DUMP <query.sql> <out-dir>`: as `RUN`, then writes the result as Parquet to `out-dir`
//!   (replacing it). The write is not in the reported seconds.
//! * `QUIT`: exits.
//!
//! A failure replies `ERR <one-line message>`; the process keeps serving.

#![forbid(unsafe_code)]

use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

use arrow_array::RecordBatch;
use krishiv_common::error::KrishivError;
use krishiv_exec::{ExecutorConfig, TokioExecutor};
use krishiv_session::{DataFrame, Session};

/// Why a runner call failed.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    /// Bad arguments, a bad protocol line or an empty data directory.
    #[error("{0}")]
    Config(String),
    /// A file or directory could not be read.
    #[error("{0}")]
    Io(String),
    /// The engine failed; the Spark error class and SQLSTATE stay in the text.
    #[error("{0}")]
    Engine(#[from] KrishivError),
}

/// Name of the temporary view a `DUMP` writes from.
const DUMP_VIEW: &str = "__krishiv_bench_dump";

/// Settings of a runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Directory holding `<table>.parquet` files (a file or a directory of part files each).
    pub data: String,
    /// Compute threads of the executor.
    pub threads: usize,
    /// `krishiv.memory.limit` (for example `12g`).
    pub memory_limit: String,
}

impl RunnerConfig {
    /// The configuration from `args` (without the program name).
    ///
    /// # Errors
    /// A missing `--data`, an unknown flag or a bad value.
    pub fn from_args(args: impl IntoIterator<Item = String>) -> Result<Self, RunnerError> {
        let mut data = None;
        let mut threads = std::thread::available_parallelism().map_or(1, usize::from);
        let mut memory_limit = "12g".to_owned();
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let mut value = || {
                args.next()
                    .ok_or_else(|| RunnerError::Config(format!("{flag} needs a value")))
            };
            match flag.as_str() {
                "--data" => data = Some(value()?),
                "--threads" => {
                    threads = value()?
                        .parse()
                        .map_err(|e| RunnerError::Config(format!("--threads: {e}")))?;
                }
                "--memory-limit" => memory_limit = value()?,
                other => return Err(RunnerError::Config(format!("unknown argument {other}"))),
            }
        }
        Ok(Self {
            data: data.ok_or_else(|| RunnerError::Config("--data DIR is required".to_owned()))?,
            threads,
            memory_limit,
        })
    }
}

/// A session with the data directory's tables registered.
#[derive(Debug)]
pub struct Runner {
    session: Session,
    tables: Vec<String>,
}

impl Runner {
    /// Builds the session and registers the tables.
    ///
    /// # Errors
    /// The data directory is unreadable or holds no Parquet tables, or the session cannot start.
    pub fn new(config: &RunnerConfig) -> Result<Self, RunnerError> {
        let threads = config.threads.max(1);
        let executor = TokioExecutor::new(&ExecutorConfig {
            compute_threads: threads,
            io_threads: 2,
            blocking_threads: threads,
        })?;
        let session = Session::builder()
            .executor(executor)
            .config("krishiv.memory.limit", config.memory_limit.as_str())
            .build()?;
        let mut tables = Vec::new();
        let entries = std::fs::read_dir(&config.data)
            .map_err(|e| RunnerError::Io(format!("cannot read {}: {e}", config.data)))?;
        for entry in entries {
            let path = entry
                .map_err(|e| RunnerError::Io(format!("cannot list {}: {e}", config.data)))?
                .path();
            let name = table_name(&path);
            let (Some(name), Some(path_str)) = (name, path.to_str()) else {
                continue;
            };
            futures::executor::block_on(
                session
                    .read()
                    .parquet([path_str])
                    .create_or_replace_temp_view(&name),
            )?;
            tables.push(name);
        }
        tables.sort();
        if tables.is_empty() {
            return Err(RunnerError::Config(format!(
                "no <table>.parquet under {}",
                config.data
            )));
        }
        Ok(Self { session, tables })
    }

    /// The registered table names, sorted.
    #[must_use]
    pub fn tables(&self) -> &[String] {
        &self.tables
    }

    /// Runs the SQL text to completion; returns the row count and the seconds taken.
    ///
    /// # Errors
    /// Any planning or execution error.
    pub fn run_sql(&self, sql: &str) -> Result<(u64, f64), RunnerError> {
        let (batches, secs, _) = self.execute(sql)?;
        Ok((count_rows(&batches), secs))
    }

    /// Plans and runs `sql`, materialising every batch.
    fn execute(&self, sql: &str) -> Result<(Vec<RecordBatch>, f64, DataFrame), RunnerError> {
        let start = Instant::now();
        let df = futures::executor::block_on(self.session.sql(sql))?;
        let batches = futures::executor::block_on(df.collect())?;
        Ok((batches, start.elapsed().as_secs_f64(), df))
    }

    /// Writes already-collected `batches` as Parquet to `out`, without running the query again.
    fn dump(
        &self,
        df: &DataFrame,
        batches: Vec<RecordBatch>,
        out: &str,
    ) -> Result<(), RunnerError> {
        let schema = futures::executor::block_on(df.schema())?;
        self.session.register_batches(DUMP_VIEW, schema, batches)?;
        let written = futures::executor::block_on(
            self.session
                .table(DUMP_VIEW)
                .write()
                .mode("overwrite")
                .parquet(out),
        );
        let _existed = self.session.drop_temp_view(DUMP_VIEW);
        Ok(written?)
    }

    fn reply(&self, line: &str) -> Result<String, RunnerError> {
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some("RUN"), Some(file), None, _) => {
                let (batches, secs, _) = self.execute(&read_sql(&decode_path(file)?)?)?;
                Ok(format!("OK {} {secs:.6}", count_rows(&batches)))
            }
            (Some("DUMP"), Some(file), Some(out), None) => {
                let (batches, secs, df) = self.execute(&read_sql(&decode_path(file)?)?)?;
                let rows = count_rows(&batches);
                self.dump(&df, batches, &decode_path(out)?)?;
                Ok(format!("OK {rows} {secs:.6}"))
            }
            _ => Err(RunnerError::Config(format!("bad command: {line}"))),
        }
    }

    /// Serves one protocol line; `None` means quit.
    pub fn handle_line(&self, line: &str) -> Option<String> {
        match line.split_whitespace().next() {
            None => return Some(String::new()),
            Some("QUIT") => return None,
            Some(_) => {}
        }
        Some(match self.reply(line) {
            Ok(ok) => ok,
            Err(e) => format!("ERR {}", one_line(&e.to_string())),
        })
    }
}

fn count_rows(batches: &[RecordBatch]) -> u64 {
    batches.iter().map(|b| b.num_rows() as u64).sum()
}

/// Percent-encodes `path` for the protocol: every byte but ASCII alphanumerics and `-_./` becomes `%XX`, so a
/// path never contains whitespace and a line splits on whitespace into its arguments.
#[must_use]
pub fn encode_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for b in path.bytes() {
        if b.is_ascii_alphanumeric() || b"-_./".contains(&b) {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The inverse of [`encode_path`].
///
/// # Errors
/// A malformed `%XX` escape or bytes that are not UTF-8.
pub fn decode_path(encoded: &str) -> Result<String, RunnerError> {
    let bad = || RunnerError::Config(format!("bad path encoding: {encoded}"));
    let mut bytes = Vec::with_capacity(encoded.len());
    let mut it = encoded.bytes();
    while let Some(b) = it.next() {
        if b == b'%' {
            let hi = it
                .next()
                .and_then(|c| char::from(c).to_digit(16))
                .ok_or_else(bad)?;
            let lo = it
                .next()
                .and_then(|c| char::from(c).to_digit(16))
                .ok_or_else(bad)?;
            bytes.push(u8::try_from(hi * 16 + lo).map_err(|_| bad())?);
        } else {
            bytes.push(b);
        }
    }
    String::from_utf8(bytes).map_err(|_| bad())
}

/// Serves `input` line by line on `output` until `QUIT` or end of input.
///
/// # Errors
/// An I/O error on the streams.
pub fn serve(runner: &Runner, input: impl BufRead, output: &mut impl Write) -> std::io::Result<()> {
    writeln!(
        output,
        "READY {} {}",
        env!("CARGO_PKG_VERSION"),
        runner.tables().join(",")
    )?;
    output.flush()?;
    for line in input.lines() {
        let Some(reply) = runner.handle_line(&line?) else {
            break;
        };
        writeln!(output, "{reply}")?;
        output.flush()?;
    }
    Ok(())
}

/// The table a path stands for: `lineitem.parquet` or a `lineitem` directory.
fn table_name(path: &Path) -> Option<String> {
    let stem = if path.is_dir() {
        path.file_name()?.to_str()?
    } else {
        path.file_name()?.to_str()?.strip_suffix(".parquet")?
    };
    Some(stem.to_owned())
}

fn read_sql(file: &str) -> Result<String, RunnerError> {
    std::fs::read_to_string(file).map_err(|e| RunnerError::Io(format!("cannot read {file}: {e}")))
}

fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
