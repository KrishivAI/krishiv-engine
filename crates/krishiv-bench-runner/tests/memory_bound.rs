// SPDX-License-Identifier: Apache-2.0
//! The memory limit bounds the process: a runner child with a small `krishiv.memory.limit` runs
//! joins and aggregations that must spill, and its peak resident set stays within
//! [`rss_bound`]. Where `systemd-run --user` works (a developer machine), the child also runs in a
//! cgroup capped at that bound plus room for spill-file page cache, so a breach is an OOM kill, not
//! only a failed comparison.
// Peak RSS comes from /proc (Linux only).
#![cfg(test)]
#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_bench_runner::{encode_path, rss_bound};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

const LIMIT: u64 = 64 << 20;
const THREADS: u64 = 2;
const ROWS: i64 = 800_000;
/// Room in the cgroup cap for spill-file page cache (see the test).
const PAGE_CACHE_ALLOWANCE: u64 = 64 << 20;

/// `t(id, k, s)` and `u(id, v)`, `ROWS` rows each in 64K-row row groups.
fn write_tables(dir: &Path) {
    let props = || {
        WriterProperties::builder()
            .set_max_row_group_row_count(Some(65_536))
            .build()
    };
    let t = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("k", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let mut w = ArrowWriter::try_new(
        std::fs::File::create(dir.join("t.parquet")).unwrap(),
        Arc::clone(&t),
        Some(props()),
    )
    .unwrap();
    for start in (0..ROWS).step_by(50_000) {
        let ids: Vec<i64> = (start..(start + 50_000).min(ROWS)).collect();
        let k: Vec<i64> = ids.iter().map(|i| i % 20_000).collect();
        let s: Vec<String> = ids
            .iter()
            .map(|i| format!("{:016x}-{}", i.wrapping_mul(0x5851_f42d_4c95_7f2d), i % 97))
            .collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&t),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(k)),
                Arc::new(StringArray::from(s)),
            ],
        )
        .unwrap();
        w.write(&batch).unwrap();
    }
    w.close().unwrap();
    let u = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let mut w = ArrowWriter::try_new(
        std::fs::File::create(dir.join("u.parquet")).unwrap(),
        Arc::clone(&u),
        Some(props()),
    )
    .unwrap();
    let ids: Vec<i64> = (0..ROWS).rev().collect();
    let v: Vec<i64> = ids.iter().map(|i| i * 3).collect();
    let batch = RecordBatch::try_new(
        u,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(v)),
        ],
    )
    .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

/// Queries whose state far exceeds `LIMIT`, so each spills; each returns one row, so the result
/// itself holds no memory.
const QUERIES: [(&str, &str); 3] = [
    (
        "join",
        "SELECT count(*), sum(u.v - t.k) FROM t JOIN u ON t.id = u.id",
    ),
    (
        "aggregate",
        "SELECT count(*), sum(c), max(m) FROM (SELECT s, count(*) AS c, max(k) AS m FROM t GROUP BY s)",
    ),
    (
        "join_aggregate",
        "SELECT count(*), sum(x) FROM (SELECT t.s, sum(u.v) AS x FROM t JOIN u ON t.id = u.id GROUP BY t.s)",
    ),
];

/// Whether this machine can run a command in a capped user cgroup.
fn cgroup_available() -> bool {
    Command::new("systemd-run")
        .args([
            "--user",
            "--scope",
            "-q",
            "-p",
            "MemoryMax=1G",
            "--",
            "true",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// The runner child, killed (with its cgroup scope) when dropped.
struct Runner {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Runner {
    fn start(data: &Path, cap: Option<u64>) -> Self {
        let bin = env!("CARGO_BIN_EXE_krishiv-bench-runner");
        let args = [
            "--data",
            data.to_str().unwrap(),
            "--threads",
            &THREADS.to_string(),
            "--memory-limit",
            &LIMIT.to_string(),
        ]
        .map(str::to_owned);
        let mut cmd = match cap {
            Some(cap) => {
                let mut c = Command::new("systemd-run");
                c.args(["--user", "--scope", "-q", "-p"])
                    .arg(format!("MemoryMax={cap}"))
                    .args(["-p", "MemorySwapMax=0", "--", bin]);
                c
            }
            None => Command::new(bin),
        };
        let mut child = cmd
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("runner starts");
        let stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let mut ready = String::new();
        stdout.read_line(&mut ready).unwrap();
        assert!(ready.starts_with("READY "), "{ready}");
        Self {
            child,
            stdin,
            stdout,
        }
    }

    /// One protocol exchange; an empty reply means the child died (an OOM kill under the cap).
    fn command(&mut self, line: &str) -> String {
        writeln!(self.stdin, "{line}").unwrap();
        self.stdin.flush().unwrap();
        let mut reply = String::new();
        self.stdout.read_line(&mut reply).unwrap();
        reply
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        // Best effort: the child may have exited already.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The number after `"key":` in `json`'s `memory` object.
fn memory_field(json: &str, key: &str) -> Option<u64> {
    let mem = &json[json.find("\"memory\":{")?..];
    let at = mem.find(&format!("\"{key}\":"))? + key.len() + 3;
    let digits: String = mem[at..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

#[test]
fn peak_rss_stays_within_the_bound_while_queries_spill() {
    let dir = tempfile::tempdir().unwrap();
    write_tables(dir.path());
    let bound = rss_bound(LIMIT, THREADS);
    // A cgroup also charges the page cache of the spill files the child
    // writes (reclaimable once written back, but charged meanwhile), which is
    // not resident memory: the cap allows for it on top of the bound. The
    // bound itself is checked on the child's peak RSS below.
    let cap = cgroup_available().then_some(bound + PAGE_CACHE_ALLOWANCE);
    eprintln!(
        "memory bound {bound} B for a {LIMIT} B limit, cgroup cap: {}",
        cap.map_or_else(|| "unavailable".to_owned(), |c| c.to_string())
    );
    let mut runner = Runner::start(dir.path(), cap);
    for (name, sql) in QUERIES {
        let file = dir.path().join(format!("{name}.sql"));
        std::fs::write(&file, sql).unwrap();
        let reply = runner.command(&format!("PROFILE {}", encode_path(file.to_str().unwrap())));
        assert!(
            reply.starts_with("OK "),
            "{name}: {reply:?} (an empty reply is a killed runner: memory above {bound} B)"
        );
        let peak = memory_field(&reply, "peak_rss_bytes").expect("Linux reports the peak RSS");
        let reserved = memory_field(&reply, "peak_reserved_bytes").unwrap();
        let spilled = memory_field(&reply, "spill_files").unwrap();
        eprintln!("{name}: peak RSS {peak} B, peak reserved {reserved} B, {spilled} spill files");
        assert!(peak <= bound, "{name}: peak RSS {peak} B above {bound} B");
        assert!(
            reserved <= LIMIT,
            "{name}: reserved {reserved} B above the limit"
        );
        assert!(spilled > 0, "{name} must spill under {LIMIT} B: {reply}");
    }
    assert_eq!(runner.command("QUIT"), "");
}

#[test]
fn the_bound_is_a_quarter_over_the_limit_plus_the_floor() {
    use krishiv_bench_runner::{RSS_FLOOR_BASE, RSS_FLOOR_PER_THREAD};
    assert_eq!(
        rss_bound(2 << 30, 12),
        (2 << 30) + (512 << 20) + RSS_FLOOR_BASE + 12 * RSS_FLOOR_PER_THREAD
    );
    assert_eq!(rss_bound(u64::MAX, 1), u64::MAX);
    assert_eq!(rss_bound(0, 0), RSS_FLOOR_BASE);
}
