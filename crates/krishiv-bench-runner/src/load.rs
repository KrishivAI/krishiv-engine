// SPDX-License-Identifier: Apache-2.0
//! `LOAD`: concurrent queries over the runner's one session (P2-T104 §3, E01's A-only checks).
//!
//! `LOAD <heavy.sql> <heavies> <short.sql> <rate> <seconds>` runs, for `seconds`:
//!
//! * `heavies` closed loops, each running the heavy query back to back (`0` for none);
//! * the short query open-loop at `rate` arrivals per second (`0` for none), at fixed times
//!   `i / rate`, each on a thread of its own, so a slow query never delays the next arrival.
//!
//! No query starts after `seconds`; the reply waits for those running. It is `OK <json>`, one line:
//! `{"seconds":..,"heavy":{..},"short":{..},"io":{..}}`. Each of `heavy`/`short` gives `"done"`, the
//! completed queries' `"latencies"` in seconds (a short query's from its scheduled arrival, so a
//! queue counts; a heavy one's from its start), `"max_slice_seconds"` (the longest driver poll of any
//! of them), `"stalls"` (polls over the stall threshold, summed; wall time, so a thread kept off a
//! CPU counts), `"stalled_pipelines"` (each pipeline that stalled, named by its operators joined with
//! `>`, with its stalls and longest poll), and `"errors"` (the first message
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
        format!(
            "{{\"done\":{},\"latencies\":[{}],\"max_slice_seconds\":{:.6},\"stalls\":{},\"stalled_pipelines\":[{}],\"errors\":[{}]}}",
            self.latencies.len(),
            lat.join(","),
            self.max_slice.as_secs_f64(),
            self.stalls,
            stalled.join(","),
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
        let heavies = heavies.parse().map_err(|_| bad("heavy count", heavies))?;
        let rate: f64 = rate.parse().map_err(|_| bad("rate", rate))?;
        let seconds: f64 = seconds.parse().map_err(|_| bad("seconds", seconds))?;
        if !(rate.is_finite() && rate >= 0.0 && seconds.is_finite() && seconds > 0.0) {
            return Err(bad("rate or seconds", &format!("{rate} {seconds}")));
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

/// Runs one query, adding its outcome to `class`; `from` is when its latency starts.
fn one(runner: &Runner, sql: &str, from: Instant, class: &Mutex<Class>) {
    let outcome = runner.metered(sql);
    let secs = from.elapsed().as_secs_f64();
    let mut c = class
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match outcome {
        Ok(m) => {
            c.latencies.push(secs);
            if let Some(m) = m {
                c.max_slice = c.max_slice.max(m.drivers.max_slice);
                c.stalls = c.stalls.saturating_add(m.drivers.stalls);
                for (id, p) in m.pipelines.iter().filter(|(_, p)| p.stalls > 0) {
                    let name: Vec<&str> = m
                        .operators
                        .values()
                        .filter(|o| o.pipelines.contains(id))
                        .map(|o| o.name.as_ref())
                        .collect();
                    let e = c.stalled.entry(name.join(">")).or_default();
                    e.0 = e.0.saturating_add(p.stalls);
                    e.1 = e.1.max(p.max_slice);
                }
            }
        }
        Err(e) => c.errors.push(one_line(&e.to_string())),
    }
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
