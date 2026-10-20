// SPDX-License-Identifier: Apache-2.0
//! Line-protocol query runner for the bench harness (P2-T142).
//!
//! The process registers every `<table>.parquet` of a data directory as a temporary view, prints
//! `READY <version> <tables>` and then serves one command per stdin line:
//!
//! * `RUN <query.sql>`: plans and executes the query, materialising every result batch. Replies
//!   `OK <rows> <seconds>`; the seconds cover parsing, analysis, planning and execution.
//! * `PROFILE <query.sql>`: as `RUN`, replying `OK <rows> <seconds> <json>` where `json` is one line,
//!   `{"phases":[{"name":"parse_analyze","seconds":..},{"name":"plan_execute","seconds":..}]}`: the wall
//!   time of `Session::sql` (parse and analysis) and of `collect` (planning, execution and materialising).
//!   It adds `"operators":[{"id","name","rows_in","rows_out","compute_seconds","blocked_io_seconds"}]` from the
//!   executor's `QueryMetrics` (`DataFrame::collect_with_metrics`), in operator-id order, each also with
//!   `"peak_reserved_bytes"` (summed over drivers), `"spill_requests"` (arbitrator requests serviced),
//!   `"spill_files"`, `"spill_bytes"` (written, as stored), `"blocked_memory_seconds"` and `"extra"`, the
//!   operator's own counters (`OperatorEntry::extra`, e.g. a grace join's depth). The seconds are
//!   summed over the query's drivers, so with several threads they can exceed the wall time. A
//!   `"memory"` object gives `"limit_bytes"`, the query's `"peak_reserved_bytes"` (what the memory pool
//!   granted at once), `"spill_files"`/`"spill_bytes"` (all operators) and, on Linux, `"rss_before_bytes"`
//!   and `"peak_rss_bytes"`: the process's resident set before the query and its peak during it (the
//!   kernel's high-water mark, reset before the query through `/proc/self/clear_refs`), `null` where
//!   unavailable. The peak covers everything the process holds (allocator retention, untracked buffers),
//!   which is what a memory cap enforces; it starts from what the process already held.
//! * `DUMP <query.sql> <out-dir>`: as `RUN`, then writes the result as Parquet to `out-dir`
//!   (replacing it). The write is not in the reported seconds.
//! * `QUIT`: exits.
//!
//! `--conf key=value` (repeatable) sets extra session settings, for A/B runs of one binary.
//!
//! A failure replies `ERR <one-line message>`; the process keeps serving.

#![forbid(unsafe_code)]

use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

use arrow_array::RecordBatch;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_exec::{ExecutorConfig, QueryMetrics, TokioExecutor};
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

/// How this runner was built, as `profile=<cargo profile> target-cpu=<cpu>` (`--build-info`).
pub const BUILD_INFO: &str = concat!(
    "profile=",
    env!("KRISHIV_BUILD_PROFILE"),
    " target-cpu=",
    env!("KRISHIV_BUILD_TARGET_CPU")
);

/// Name of the temporary view a `DUMP` writes from.
const DUMP_VIEW: &str = "__krishiv_bench_dump";

/// The process's resident set and its high-water mark, in bytes (`VmRSS`, `VmHWM` of
/// `/proc/self/status`); `None` off Linux or if unreadable.
#[must_use]
pub fn rss() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib = |key: &str| -> Option<u64> {
        let line = status.lines().find(|l| l.starts_with(key))?;
        let n: u64 = line[key.len()..]
            .trim()
            .strip_suffix("kB")?
            .trim()
            .parse()
            .ok()?;
        n.checked_mul(1024)
    };
    Some((kib("VmRSS:")?, kib("VmHWM:")?))
}

/// Fixed part of the resident-set bound: the program, its runtimes and caches.
pub const RSS_FLOOR_BASE: u64 = 48 << 20;
/// Per compute thread: thread stacks and allocator caches, the batches in flight in each pipeline, and
/// one decoder working set a scan takes without a reservation when the pool is full.
pub const RSS_FLOOR_PER_THREAD: u64 = 16 << 20;

/// The most resident memory a runner with memory limit `limit` and `threads` compute threads may use:
/// `1.25 x limit + RSS_FLOOR_BASE + threads x RSS_FLOOR_PER_THREAD`. The 25% covers what the memory pool
/// cannot see exactly (allocator size classes and retention after a purge delay, estimates of decoder
/// pages); the floor what does not scale with the limit. The memory gate checks peak RSS against it
/// and runs the process in a cgroup with `memory.max` at the bound plus room for the spill files'
/// page cache, which the cgroup charges but which is not resident memory (64 MiB in the memory-bound
/// test).
///
/// It holds for a process whose global allocator is configured for it: mimalloc with
/// `krishiv_alloc` (this runner and the CLI, when the limit is known at process start: an argument or
/// `KRISHIV_MEMORY_LIMIT`). Memory an allocator retains is outside the
/// pools' control, so an embedding host with another allocator, or without a release hook
/// (`krishiv_memory::set_release_hook`), can exceed it by what its allocator keeps.
///
/// Known limitation: the row group a Parquet scan driver decodes is not reserved; the per-thread
/// floor covers row groups of a few MiB (TPC-H as generated), not files written with 128 MiB row
/// groups, which can exceed the bound by about one row group per driver.
#[must_use]
pub fn rss_bound(limit: u64, threads: u64) -> u64 {
    limit
        .saturating_add(limit / 4)
        .saturating_add(RSS_FLOOR_BASE)
        .saturating_add(threads.saturating_mul(RSS_FLOOR_PER_THREAD))
}

/// Resets the resident-set high-water mark to the current resident set (Linux 4.0+:
/// `/proc/self/clear_refs` value 5). Returns whether it was reset.
fn reset_peak_rss() -> bool {
    std::fs::write("/proc/self/clear_refs", "5").is_ok()
}

/// Settings of a runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Directory holding `<table>.parquet` files (a file or a directory of part files each).
    pub data: String,
    /// Compute threads of the executor.
    pub threads: usize,
    /// `krishiv.memory.limit` (for example `12g`).
    pub memory_limit: String,
    /// Extra session settings (`--conf key=value`, repeatable), applied after the memory limit so a
    /// repeated key (including `krishiv.memory.limit`) wins.
    pub conf: Vec<(String, String)>,
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
        let mut conf = Vec::new();
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
                "--conf" => {
                    let kv = value()?;
                    let (k, v) = kv
                        .split_once('=')
                        .filter(|(k, _)| !k.is_empty())
                        .ok_or_else(|| {
                            RunnerError::Config(format!("--conf expects key=value, got {kv:?}"))
                        })?;
                    conf.push((k.to_owned(), v.to_owned()));
                }
                other => return Err(RunnerError::Config(format!("unknown argument {other}"))),
            }
        }
        Ok(Self {
            data: data.ok_or_else(|| RunnerError::Config("--data DIR is required".to_owned()))?,
            threads,
            memory_limit,
            conf,
        })
    }
}

/// A session with the data directory's tables registered.
#[derive(Debug)]
pub struct Runner {
    session: Session,
    tables: Vec<String>,
    /// `krishiv.memory.limit` in bytes.
    limit: Option<u64>,
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
        // `SessionBuilder::build` drops configuration warnings; an unknown key would leave the A/B sides
        // identical without saying so, so reject it here.
        let pairs = std::iter::once(("krishiv.memory.limit", config.memory_limit.as_str()))
            .chain(config.conf.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        let (merged, warnings) = Config::from_pairs(pairs)?;
        if let Some(w) = warnings.first() {
            return Err(RunnerError::Config(format!("--conf not applied: {w}")));
        }
        let mut builder = Session::builder()
            .executor(executor)
            .config("krishiv.memory.limit", config.memory_limit.as_str());
        for (k, v) in &config.conf {
            builder = builder.config(k.as_str(), v.as_str());
        }
        let session = builder.build()?;
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
        // As the session sees it: a `--conf` limit wins over `--memory-limit`.
        let limit = merged
            .get(&krishiv_common::config::MEMORY_LIMIT)
            .and_then(krishiv_common::config::ConfigValue::as_size);
        Ok(Self {
            session,
            tables,
            limit,
        })
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
        let (batches, timing, df, _) = self.execute_timed(sql, false)?;
        Ok((batches, timing.total, df))
    }

    /// As [`Self::execute`], also timing the two phases.
    fn execute_timed(&self, sql: &str, profile: bool) -> Result<Executed, RunnerError> {
        let start = Instant::now();
        let df = futures::executor::block_on(self.session.sql(sql))?;
        let parse_analyze = start.elapsed().as_secs_f64();
        let (batches, metrics) = if profile {
            let (b, m) = futures::executor::block_on(df.collect_with_metrics())?;
            (b, Some(m))
        } else {
            (futures::executor::block_on(df.collect())?, None)
        };
        let total = start.elapsed().as_secs_f64();
        let timing = Timing {
            total,
            parse_analyze,
            plan_execute: total - parse_analyze,
        };
        Ok((batches, timing, df, metrics))
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
            (Some("PROFILE"), Some(file), None, _) => {
                let sql = read_sql(&decode_path(file)?)?;
                let before = rss().map(|(now, _)| now);
                let reset = reset_peak_rss();
                let (batches, t, _, metrics) = self.execute_timed(&sql, true)?;
                let peak = rss().filter(|_| reset).map(|(_, hwm)| hwm);
                let memory = Memory {
                    limit: self.limit,
                    rss_before: before,
                    peak_rss: peak,
                };
                Ok(format!(
                    "OK {} {:.6} {}",
                    count_rows(&batches),
                    t.total,
                    t.json(metrics.as_ref(), &memory)
                ))
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

type Executed = (Vec<RecordBatch>, Timing, DataFrame, Option<QueryMetrics>);

/// What `PROFILE` reports about memory besides the metrics.
#[derive(Debug, Clone, Copy)]
struct Memory {
    limit: Option<u64>,
    rss_before: Option<u64>,
    peak_rss: Option<u64>,
}

/// `v` as JSON (`null` for `None`).
fn json_opt(v: Option<u64>) -> String {
    v.map_or_else(|| "null".to_owned(), |n| n.to_string())
}

/// Wall time of one query, in seconds.
#[derive(Debug, Clone, Copy)]
struct Timing {
    total: f64,
    parse_analyze: f64,
    plan_execute: f64,
}

impl Timing {
    fn json(&self, metrics: Option<&QueryMetrics>, memory: &Memory) -> String {
        let operators = metrics.map_or_else(String::new, |m| {
            let ops: Vec<String> = m
                .operators
                .iter()
                .map(|(id, e)| {
                    let m = &e.metrics;
                    let extra: Vec<String> = e
                        .extra
                        .iter()
                        .map(|(k, v)| format!("\"{}\":{v}", k.escape_default()))
                        .collect();
                    format!(
                        "{{\"id\":{},\"name\":\"{}\",\"rows_in\":{},\"rows_out\":{},\"compute_seconds\":{:.6},\"blocked_io_seconds\":{:.6},\"peak_reserved_bytes\":{},\"spill_requests\":{},\"spill_files\":{},\"spill_bytes\":{},\"blocked_memory_seconds\":{:.6},\"extra\":{{{}}}}}",
                        id.0,
                        e.name.escape_default(),
                        m.rows_in,
                        m.rows_out,
                        m.compute_time.as_secs_f64(),
                        m.blocked_on_io.as_secs_f64(),
                        m.peak_reserved,
                        m.spills,
                        m.spill_files,
                        m.spill_write_bytes,
                        m.blocked_on_memory.as_secs_f64(),
                        extra.join(",")
                    )
                })
                .collect();
            format!(",\"operators\":[{}]", ops.join(","))
        });
        let ops = metrics.map(|m| m.operators.values().map(|e| &e.metrics));
        let spill_files = ops.clone().map(|o| o.map(|m| m.spill_files).sum::<u64>());
        let spill_bytes = ops.map(|o| o.map(|m| m.spill_write_bytes).sum::<u64>());
        let mem = format!(
            ",\"memory\":{{\"limit_bytes\":{},\"peak_reserved_bytes\":{},\"spill_files\":{},\"spill_bytes\":{},\"rss_before_bytes\":{},\"peak_rss_bytes\":{}}}",
            json_opt(memory.limit),
            json_opt(metrics.map(|m| m.memory.peak_reserved)),
            json_opt(spill_files),
            json_opt(spill_bytes),
            json_opt(memory.rss_before),
            json_opt(memory.peak_rss),
        );
        format!(
            "{{\"phases\":[{{\"name\":\"parse_analyze\",\"seconds\":{:.6}}},{{\"name\":\"plan_execute\",\"seconds\":{:.6}}}]{operators}{mem}}}",
            self.parse_analyze, self.plan_execute
        )
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
