// SPDX-License-Identifier: Apache-2.0
//! `LOAD`: concurrent queries over the runner's one session (P2-T104 §3, E01's A-only checks).
//!
//! `LOAD <heavy.sql> <heavies> <short.sql> <rate> <seconds>` runs, for `seconds`:
//!
//! * `heavies` closed loops, each running the heavy query back to back (`0` for none);
//! * the short query open-loop at `rate` arrivals per second (`0` for none), at fixed times
//!   `i / rate`, each on a thread of its own, so a slow query never delays the next arrival.
//!
//! At most 64 heavy loops and 10,000 short arrivals (`rate * seconds`): each is a thread of its own.
//! No query starts after `seconds`; the reply waits for those running. It is `OK <json>`, one line:
//! `{"seconds":..,"heavy":{..},"short":{..},"io":{..}}`. Each of `heavy`/`short` gives `"done"`, the
//! completed queries' `"latencies"` in seconds (a short query's from its scheduled arrival, so a
//! queue counts; a heavy one's from its start), `"max_slice_seconds"` (the longest driver poll of any
//! of them), `"stalls"` (polls over the stall threshold, summed; wall time, so a thread kept off a
//! CPU counts), `"stalled_pipelines"` (each pipeline that stalled, named by its operators joined with
//! `>`, with its stalls and longest poll), `"fetch_mean_seconds"` (each query's mean file-read time,
//! its scans' `scan_read_inflight_ns` over `scan_reads`: its own reads only), and `"errors"` (the first message
//! of each failed query). `io` is the change of the runtime's read-latency histogram over the run,
//! remote stores only (`krishiv_io::LatencyHistogram`): `"requests"` and `"p50_us"`, `"p99_us"`,
//! `"max_us"`, each the upper bound of its power-of-two microsecond bucket, `null` with no request.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use krishiv_io::{LATENCY_BUCKETS, LatencyHistogram};

use crate::{Runner, RunnerError, one_line};

/// What one class of queries (heavy or short) did.
#[derive(Default)]
struct Class {
    latencies: Vec<f64>,
    max_slice: Duration,
    stalls: u64,
    /// Per pipeline that stalled, named by its operators (in operator-id
    /// order, `>`-joined): its stalls and longest poll.
    stalled: std::collections::BTreeMap<String, (u64, Duration)>,
    /// Per query with file reads, its mean read time in seconds: its scans'
    /// `scan_read_inflight_ns` over their `scan_reads` (each read timed
    /// inside its own future, so another query's reads are not in it).
    fetch_means: Vec<f64>,
    errors: Vec<String>,
}

impl Class {
    fn json(&self) -> String {
        let lat: Vec<String> = self.latencies.iter().map(|s| format!("{s:.6}")).collect();
        let errors: Vec<String> = self
            .errors
            .iter()
            .map(|e| format!("\"{}\"", e.replace('\\', "\\\\").replace('"', "\\\"")))
            .collect();
        let stalled: Vec<String> = self
            .stalled
            .iter()
            .map(|(name, (n, max))| {
                format!(
                    "{{\"pipeline\":\"{}\",\"stalls\":{n},\"max_slice_seconds\":{:.6}}}",
                    name.replace('\\', "\\\\").replace('"', "\\\""),
                    max.as_secs_f64()
                )
            })
            .collect();
        let fetch: Vec<String> = self.fetch_means.iter().map(|s| format!("{s:.6}")).collect();
        format!(
            "{{\"done\":{},\"latencies\":[{}],\"max_slice_seconds\":{:.6},\"stalls\":{},\"stalled_pipelines\":[{}],\"fetch_mean_seconds\":[{}],\"errors\":[{}]}}",
            self.latencies.len(),
            lat.join(","),
            self.max_slice.as_secs_f64(),
            self.stalls,
            stalled.join(","),
            fetch.join(","),
            errors.join(",")
        )
    }
}

/// The parsed arguments of a `LOAD` line.
pub(crate) struct Load {
    pub(crate) heavy: String,
    pub(crate) heavies: usize,
    pub(crate) short: String,
    pub(crate) rate: f64,
    pub(crate) seconds: f64,
}

impl Load {
    /// Parses `<heavies> <rate> <seconds>`; the SQL texts are read by the caller.
    pub(crate) fn parse(
        heavy: String,
        heavies: &str,
        short: String,
        rate: &str,
        seconds: &str,
    ) -> Result<Self, RunnerError> {
        let bad = |what: &str, v: &str| RunnerError::Config(format!("LOAD: bad {what}: {v}"));
        let heavies: usize = heavies.parse().map_err(|_| bad("heavy count", heavies))?;
        let rate: f64 = rate.parse().map_err(|_| bad("rate", rate))?;
        let seconds: f64 = seconds.parse().map_err(|_| bad("seconds", seconds))?;
        if !(rate.is_finite() && rate >= 0.0 && seconds.is_finite() && seconds > 0.0) {
            return Err(bad("rate or seconds", &format!("{rate} {seconds}")));
        }
        if heavies > MAX_HEAVIES {
            return Err(bad("heavy count (at most 64)", &heavies.to_string()));
        }
        if rate * seconds > MAX_ARRIVALS {
            return Err(bad(
                "rate x seconds (at most 10000 arrivals)",
                &format!("{rate} {seconds}"),
            ));
        }
        Ok(Self {
            heavy,
            heavies,
            short,
            rate,
            seconds,
        })
    }
}

/// At most this many heavy loops, and at most this many short arrivals in one run
/// (`rate * seconds`): each is an OS thread holding a running query.
const MAX_HEAVIES: usize = 64;
const MAX_ARRIVALS: f64 = 10_000.0;

impl Class {
    /// Adds one query's outcome; `secs` is its latency.
    fn add(&mut self, secs: f64, outcome: Result<Option<krishiv_exec::QueryMetrics>, RunnerError>) {
        match outcome {
            Ok(m) => {
                self.latencies.push(secs);
                if let Some(m) = m {
                    self.add_metrics(&m);
                }
            }
            Err(e) => self.errors.push(one_line(&e.to_string())),
        }
    }

    /// Adds a finished query's stalls, stalled pipelines and mean file-read time.
    fn add_metrics(&mut self, m: &krishiv_exec::QueryMetrics) {
        self.max_slice = self.max_slice.max(m.drivers.max_slice);
        self.stalls = self.stalls.saturating_add(m.drivers.stalls);
        for (id, p) in m.pipelines.iter().filter(|(_, p)| p.stalls > 0) {
            let name: Vec<&str> = m
                .operators
                .values()
                .filter(|o| o.pipelines.contains(id))
                .map(|o| o.name.as_ref())
                .collect();
            let e = self.stalled.entry(name.join(">")).or_default();
            e.0 = e.0.saturating_add(p.stalls);
            e.1 = e.1.max(p.max_slice);
        }
        let count = |name: &str| -> u64 {
            m.operators
                .values()
                .flat_map(|o| &o.extra)
                .filter(|(k, _)| k == name)
                .map(|(_, v)| match v {
                    krishiv_exec::Metric::Count(n) | krishiv_exec::Metric::Max(n) => *n,
                    krishiv_exec::Metric::Text(_) => 0,
                })
                .sum()
        };
        let reads = count("scan_reads");
        if reads > 0 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "a mean in seconds of nanosecond sums"
            )]
            self.fetch_means
                .push(count("scan_read_inflight_ns") as f64 / reads as f64 / 1e9);
        }
    }
}

/// Runs one query, adding its outcome to `class`; `from` is when its latency starts.
fn one(runner: &Runner, sql: &str, from: Instant, class: &Mutex<Class>) {
    let outcome = runner.metered(sql);
    let secs = from.elapsed().as_secs_f64();
    class
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .add(secs, outcome);
}

fn io_json(before: &[u64; LATENCY_BUCKETS], after: &[u64; LATENCY_BUCKETS]) -> String {
    let d: [u64; LATENCY_BUCKETS] = std::array::from_fn(|i| after[i].saturating_sub(before[i]));
    let q =
        |p: f64| LatencyHistogram::quantile_us(&d, p).map_or("null".to_owned(), |v| v.to_string());
    format!(
        "{{\"requests\":{},\"p50_us\":{},\"p99_us\":{},\"max_us\":{}}}",
        d.iter().sum::<u64>(),
        q(0.5),
        q(0.99),
        q(1.0)
    )
}

/// Runs `load` and returns the reply's JSON.
pub(crate) fn run(runner: &Runner, load: &Load) -> String {
    let heavy = Mutex::new(Class::default());
    let short = Mutex::new(Class::default());
    let io_before = runner.io_latency();
    let start = Instant::now();
    let end = start + Duration::from_secs_f64(load.seconds);
    std::thread::scope(|s| {
        for _ in 0..load.heavies {
            s.spawn(|| {
                while Instant::now() < end {
                    one(runner, &load.heavy, Instant::now(), &heavy);
                }
            });
        }
        if load.rate > 0.0 {
            let mut i = 0u32;
            loop {
                let at = start + Duration::from_secs_f64(f64::from(i) / load.rate);
                if at >= end {
                    break;
                }
                std::thread::sleep(at.saturating_duration_since(Instant::now()));
                let short = &short;
                let sql = &load.short;
                s.spawn(move || one(runner, sql, at, short));
                i += 1;
            }
        }
    });
    let seconds = start.elapsed().as_secs_f64();
    let io_after = runner.io_latency();
    let take = |m: Mutex<Class>| {
        m.into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    };
    format!(
        "{{\"seconds\":{seconds:.6},\"heavy\":{},\"short\":{},\"io\":{}}}",
        take(heavy).json(),
        take(short).json(),
        io_json(&io_before, &io_after)
    )
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use krishiv_exec::{Metric, OperatorEntry, OperatorId, PipelineMetrics, QueryMetrics};

    use super::*;

    fn op(
        name: &'static str,
        pipelines: Vec<u32>,
        extra: Vec<(&'static str, u64)>,
    ) -> OperatorEntry {
        OperatorEntry {
            name: Cow::Borrowed(name),
            pipelines,
            extra: extra
                .into_iter()
                .map(|(k, v)| (Cow::Borrowed(k), Metric::Count(v)))
                .collect(),
            ..OperatorEntry::default()
        }
    }

    #[test]
    fn a_stalled_pipeline_is_named_by_its_operators_and_reads_give_a_mean() {
        let mut m = QueryMetrics::default();
        m.drivers.max_slice = Duration::from_millis(30);
        m.drivers.stalls = 3;
        m.operators.insert(
            OperatorId(0),
            op(
                "FileScan",
                vec![0],
                vec![("scan_reads", 4), ("scan_read_inflight_ns", 8_000_000)],
            ),
        );
        m.operators
            .insert(OperatorId(1), op("HashAggregate", vec![0, 1], vec![]));
        m.operators
            .insert(OperatorId(2), op("Project", vec![1], vec![]));
        m.pipelines.insert(
            0,
            PipelineMetrics {
                stalls: 3,
                max_slice: Duration::from_millis(30),
                ..PipelineMetrics::default()
            },
        );
        m.pipelines.insert(1, PipelineMetrics::default());
        let mut c = Class::default();
        c.add(0.5, Ok(Some(m.clone())));
        c.add(0.25, Ok(Some(m)));
        c.add(0.1, Ok(None));
        c.add(
            0.0,
            Err(RunnerError::Config("boom \"x\"\nsecond line".to_owned())),
        );
        assert_eq!(c.latencies, [0.5, 0.25, 0.1]);
        assert_eq!(c.stalls, 6);
        assert_eq!(c.stalled.len(), 1, "pipeline 1 did not stall");
        assert_eq!(
            c.stalled["FileScan>HashAggregate"],
            (6, Duration::from_millis(30))
        );
        assert_eq!(c.fetch_means, [0.002, 0.002]);
        let v: serde_json::Value = serde_json::from_str(&c.json()).unwrap();
        assert_eq!(v["done"], 3);
        assert_eq!(v["stalls"], 6);
        assert_eq!(
            v["stalled_pipelines"][0]["pipeline"],
            "FileScan>HashAggregate"
        );
        assert_eq!(v["stalled_pipelines"][0]["stalls"], 6);
        assert_eq!(v["errors"].as_array().unwrap().len(), 1);
        assert!(
            v["errors"][0].as_str().unwrap().contains("boom \"x\""),
            "{v}"
        );
    }

    #[test]
    fn io_json_reports_the_change_of_the_histogram() {
        let mut before = [0; LATENCY_BUCKETS];
        before[7] = 5;
        let mut after = before;
        after[7] += 99; // below 128 µs
        after[14] += 1; // below 16,384 µs
        let v: serde_json::Value = serde_json::from_str(&io_json(&before, &after)).unwrap();
        assert_eq!(v["requests"], 100);
        assert_eq!(v["p50_us"], 128);
        assert_eq!(v["p99_us"], 128);
        assert_eq!(v["max_us"], 16_384);
    }

    #[test]
    fn bad_arguments_name_what_is_wrong() {
        let parse = |h: &str, r: &str, s: &str| {
            Load::parse(String::new(), h, String::new(), r, s)
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        assert!(parse("1", "2", "3").is_ok());
        assert!(parse("1", "x", "3").unwrap_err().contains("bad rate"));
        assert!(parse("1", "2", "x").unwrap_err().contains("bad seconds"));
        assert!(
            parse("1", "-1", "3")
                .unwrap_err()
                .contains("rate or seconds")
        );
        assert!(parse("65", "0", "1").unwrap_err().contains("heavy count"));
        assert!(parse("0", "1000", "11").unwrap_err().contains("arrivals"));
    }
}
