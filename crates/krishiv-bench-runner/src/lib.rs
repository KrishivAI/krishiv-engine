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

use krishiv_exec::{ExecutorConfig, TokioExecutor};
use krishiv_session::{DataFrame, Session};

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
    pub fn from_args(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut data = None;
        let mut threads = std::thread::available_parallelism().map_or(1, usize::from);
        let mut memory_limit = "12g".to_owned();
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--data" => data = Some(value()?),
                "--threads" => {
                    threads = value()?.parse().map_err(|e| format!("--threads: {e}"))?;
                }
                "--memory-limit" => memory_limit = value()?,
                other => return Err(format!("unknown argument {other}")),
            }
        }
        Ok(Self {
            data: data.ok_or("--data DIR is required")?,
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
    pub fn new(config: &RunnerConfig) -> Result<Self, String> {
        let threads = config.threads.max(1);
        let executor = TokioExecutor::new(&ExecutorConfig {
            compute_threads: threads,
            io_threads: 2,
            blocking_threads: threads,
        })
        .map_err(|e| e.to_string())?;
        let session = Session::builder()
            .executor(executor)
            .config("krishiv.memory.limit", config.memory_limit.as_str())
            .build()
            .map_err(|e| e.to_string())?;
        let mut tables = Vec::new();
        let entries = std::fs::read_dir(&config.data)
            .map_err(|e| format!("cannot read {}: {e}", config.data))?;
        for entry in entries {
            let path = entry
                .map_err(|e| format!("cannot list {}: {e}", config.data))?
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
            )
            .map_err(|e| e.to_string())?;
            tables.push(name);
        }
        tables.sort();
        if tables.is_empty() {
            return Err(format!("no <table>.parquet under {}", config.data));
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
    pub fn run_sql(&self, sql: &str) -> Result<(u64, f64), String> {
        let (rows, secs, _) = self.execute(sql)?;
        Ok((rows, secs))
    }

    fn execute(&self, sql: &str) -> Result<(u64, f64, DataFrame), String> {
        let start = Instant::now();
        let df = futures::executor::block_on(self.session.sql(sql)).map_err(|e| e.to_string())?;
        let batches = futures::executor::block_on(df.collect()).map_err(|e| e.to_string())?;
        let secs = start.elapsed().as_secs_f64();
        let rows = batches.iter().map(|b| b.num_rows() as u64).sum();
        Ok((rows, secs, df))
    }

    /// Serves one protocol line; `None` means quit.
    pub fn handle_line(&self, line: &str) -> Option<String> {
        let mut parts = line.split_whitespace();
        let reply = match (parts.next(), parts.next(), parts.next()) {
            (None, ..) => return Some(String::new()),
            (Some("QUIT"), ..) => return None,
            (Some("RUN"), Some(file), None) => read_sql(file)
                .and_then(|sql| self.run_sql(&sql))
                .map(|(rows, secs)| format!("OK {rows} {secs:.6}")),
            (Some("DUMP"), Some(file), Some(out)) => read_sql(file).and_then(|sql| {
                let (rows, secs, df) = self.execute(&sql)?;
                futures::executor::block_on(df.write().mode("overwrite").parquet(out))
                    .map_err(|e| e.to_string())?;
                Ok(format!("OK {rows} {secs:.6}"))
            }),
            _ => Err(format!("bad command: {line}")),
        };
        Some(match reply {
            Ok(ok) => ok,
            Err(e) => format!("ERR {}", one_line(&e)),
        })
    }
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

fn read_sql(file: &str) -> Result<String, String> {
    std::fs::read_to_string(file).map_err(|e| format!("cannot read {file}: {e}"))
}

fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
