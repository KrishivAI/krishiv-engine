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
use std::sync::atomic::{AtomicBool, Ordering};

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_bench_runner::{encode_path, rss_bound};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

const LIMIT: u64 = 48 << 20;
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

/// The cgroup cap for a child with resident-set bound `bound`, or `None` where `systemd-run --user`
/// does not work (CI runners). Says which, since a run without the cap only compares the peak
/// RSS; `KRISHIV_TEST_REQUIRE_CGROUP=1` makes a missing cgroup a failure.
fn cgroup_cap(bound: u64) -> Option<u64> {
    let cap = cgroup_available().then_some(bound + PAGE_CACHE_ALLOWANCE);
    match cap {
        Some(c) => eprintln!("cgroup cap applied: MemoryMax={c} B (RSS bound {bound} B)"),
        None => {
            eprintln!("cgroup cap NOT applied (no systemd-run --user): RSS bound {bound} B only");
            assert!(
                std::env::var_os("KRISHIV_TEST_REQUIRE_CGROUP").is_none(),
                "KRISHIV_TEST_REQUIRE_CGROUP is set but systemd-run --user does not work"
            );
        }
    }
    cap
}

/// The runner child, killed (with its cgroup scope) when dropped.
struct Runner {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Runner {
    fn start(data: &Path, limit: u64, threads: u64, cap: Option<u64>, extra: &[&str]) -> Self {
        let bin = env!("CARGO_BIN_EXE_krishiv-bench-runner");
        let args = [
            "--data",
            data.to_str().unwrap(),
            "--threads",
            &threads.to_string(),
            "--memory-limit",
            &limit.to_string(),
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
            .args(extra)
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
    let cap = cgroup_cap(bound);
    eprintln!(
        "memory bound {bound} B for a {LIMIT} B limit, cgroup cap: {}",
        cap.map_or_else(|| "unavailable".to_owned(), |c| c.to_string())
    );
    let mut runner = Runner::start(dir.path(), LIMIT, THREADS, cap, &[]);
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

/// The sum of the operators' `"key":` counters in a PROFILE reply.
fn operator_counter(json: &str, key: &str) -> u64 {
    let pat = format!("\"{key}\":");
    json.match_indices(&pat)
        .filter_map(|(at, _)| {
            let digits: String = json[at + pat.len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse::<u64>().ok()
        })
        .sum()
}

/// Rows of the TPC-H-shaped `li` table of the q21-shaped test.
const LI_ROWS: i64 = 1_200_000;
/// The q21-shaped test's limit and compute threads: more threads than the test above, so the
/// scans keep many reads in flight at once (as TPC-H Q21 at SF10 does on 12 threads).
const LI_LIMIT: u64 = 96 << 20;
const LI_THREADS: u64 = 6;
/// `krishiv.io.blockingThreads` of the q21-shaped test: below the default, so the scans' reads
/// (more than this many in flight) must queue for the I/O runtime's threads.
const IO_BLOCKING: usize = 2;

/// One `li` row: `(orderkey, suppkey, late)`, `late` being `receipt > commit`.
fn li_row(i: i64) -> (i64, i64, bool) {
    // About four lines per order, suppliers spread over 1,000, about half the lines late:
    // deterministic, so the expected result is computed from the same function.
    // splitmix64's finalizer: every output bit depends on every bit of `i`.
    let mut z = u64::try_from(i)
        .unwrap()
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (i / 4, i64::try_from(z % 1_000).unwrap(), z >> 63 == 1)
}

/// `li(l_orderkey, l_suppkey, l_receipt, l_commit, l_comment)`: `LI_ROWS` rows in 64K-row
/// row groups.
fn write_li(dir: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int64, false),
        Field::new("l_suppkey", DataType::Int64, false),
        Field::new("l_receipt", DataType::Int64, false),
        Field::new("l_commit", DataType::Int64, false),
        Field::new("l_comment", DataType::Utf8, false),
    ]));
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(65_536))
        .build();
    let mut w = ArrowWriter::try_new(
        std::fs::File::create(dir.join("li.parquet")).unwrap(),
        Arc::clone(&schema),
        Some(props),
    )
    .unwrap();
    for start in (0..LI_ROWS).step_by(50_000) {
        let rows: Vec<_> = (start..(start + 50_000).min(LI_ROWS)).map(li_row).collect();
        let commit: Vec<i64> = (start..(start + 50_000).min(LI_ROWS))
            .map(|i| i % 365)
            .collect();
        let receipt: Vec<i64> = rows
            .iter()
            .zip(&commit)
            .map(|(r, c)| if r.2 { c + 1 } else { c - 1 })
            .collect();
        let comment: Vec<String> = rows
            .iter()
            .map(|r| format!("{:x} carefully final deposits {}", r.0, r.1))
            .collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(
                    rows.iter().map(|r| r.0).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(
                    rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(receipt)),
                Arc::new(Int64Array::from(commit)),
                Arc::new(StringArray::from(comment)),
            ],
        )
        .unwrap();
        w.write(&batch).unwrap();
    }
    w.close().unwrap();
}

/// TPC-H Q21's shape on `li`: late lines of a few suppliers (`l1`), kept if another supplier
/// has a line in the order (`EXISTS`, a semi join) and no other supplier's line is late
/// (`NOT EXISTS`, an anti join), both with a `<>` residual. Both build sides (`l2`, `l3`) are
/// far larger than the limit and spill; the probe side is small next to them, so their
/// spilled partitions are joined with roles reversed, as at SF10.
const Q21_SHAPED: &str = "SELECT count(*), sum(l1.l_suppkey), sum(length(l1.l_comment)) FROM li l1 \
    WHERE l1.l_receipt > l1.l_commit AND l1.l_suppkey % 25 = 0 \
    AND EXISTS (SELECT * FROM li l2 WHERE l2.l_orderkey = l1.l_orderkey \
        AND l2.l_suppkey <> l1.l_suppkey) \
    AND NOT EXISTS (SELECT * FROM li l3 WHERE l3.l_orderkey = l1.l_orderkey \
        AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receipt > l3.l_commit)";

/// `Q21_SHAPED`'s `(count, sum of suppkeys)`, computed directly from [`li_row`].
fn q21_shaped_expected() -> (i64, i64) {
    let (mut count, mut sum) = (0, 0);
    let mut order = Vec::new();
    let mut flush = |order: &mut Vec<(i64, i64, bool)>| {
        for &(_, supp, late) in order.iter() {
            let other = order.iter().any(|&(_, s, _)| s != supp);
            let other_late = order.iter().any(|&(_, s, l)| s != supp && l);
            if late && supp % 25 == 0 && other && !other_late {
                count += 1;
                sum += supp;
            }
        }
        order.clear();
    };
    for i in 0..LI_ROWS {
        let row = li_row(i);
        if order
            .first()
            .is_some_and(|f: &(i64, i64, bool)| f.0 != row.0)
        {
            flush(&mut order);
        }
        order.push(row);
    }
    flush(&mut order);
    (count, sum)
}

/// TPC-H Q21 at SF10 was OOM-killed under the memory gate's cgroup (2816M, its driver included):
/// the process held ~690 MiB more than the pool reserved, in the allocator caches of the I/O
/// runtime's blocking threads (one per local-file request in flight, up to Tokio's 512). That
/// failure is the cgroup's charge, at a scale this test does not reach (its RSS stays within
/// [`rss_bound`] either way), so the regression check is the cause: the I/O runtime's threads
/// stay within `krishiv.io.blockingThreads` plus its workers while a q21-shaped query whose
/// joins spill keeps many reads in flight. The test also checks the RSS bound (and runs under a
/// cgroup at that bound where `systemd-run --user` works) and the result. The joins keep Spark's
/// build side (`krishiv.exec.join.buildSide=false`): built from the small side they would not
/// spill at all, and this test covers the left semi and anti joins' spill and role reversal.
#[test]
fn a_q21_shaped_spilling_join_stays_within_the_bound() {
    let dir = tempfile::tempdir().unwrap();
    write_li(dir.path());
    let bound = rss_bound(LI_LIMIT, LI_THREADS);
    let cap = cgroup_cap(bound);
    let mut runner = Runner::start(
        dir.path(),
        LI_LIMIT,
        LI_THREADS,
        cap,
        &[
            "--conf",
            &format!("krishiv.io.blockingThreads={IO_BLOCKING}"),
            "--conf",
            "krishiv.exec.join.buildSide=false",
        ],
    );
    let file = dir.path().join("q21_shaped.sql");
    std::fs::write(&file, Q21_SHAPED).unwrap();
    // The most I/O-runtime threads (workers and blocking, all named `krishiv-io`) the runner has
    // while the query runs.
    let tasks = format!("/proc/{}/task", runner.child.id());
    let done = Arc::new(AtomicBool::new(false));
    let watch = {
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            let mut most = 0;
            while !done.load(Ordering::Acquire) {
                let io = std::fs::read_dir(&tasks).map_or(0, |d| {
                    d.filter_map(Result::ok)
                        .filter(|t| {
                            std::fs::read_to_string(t.path().join("comm"))
                                .is_ok_and(|c| c.trim_end() == "krishiv-io")
                        })
                        .count()
                });
                most = most.max(io);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            most
        })
    };
    let reply = runner.command(&format!("PROFILE {}", encode_path(file.to_str().unwrap())));
    done.store(true, Ordering::Release);
    let threads = watch.join().unwrap();
    assert!(
        reply.starts_with("OK "),
        "{reply:?} (an empty reply is a killed runner: memory above {bound} B)"
    );
    // Every thread that allocates keeps freed memory in its own allocator heap, so the I/O
    // runtime's are bounded: its two workers (the default) and `IO_BLOCKING` blocking threads.
    let allowed = 2 + IO_BLOCKING;
    assert!(
        threads >= 1,
        "the I/O runtime's threads are named krishiv-io"
    );
    assert!(
        threads <= allowed,
        "{threads} krishiv-io threads, at most {allowed}"
    );
    let peak = memory_field(&reply, "peak_rss_bytes").expect("Linux reports the peak RSS");
    let reserved = memory_field(&reply, "peak_reserved_bytes").unwrap();
    let spilled = memory_field(&reply, "spill_files").unwrap();
    eprintln!(
        "q21-shaped: peak RSS {peak} B (bound {bound} B), peak reserved {reserved} B, \
         {spilled} spill files, {threads} krishiv-io threads"
    );
    assert!(
        spilled > 0,
        "the joins must spill under {LI_LIMIT} B: {reply}"
    );
    assert!(
        operator_counter(&reply, "reversed_partitions") > 0,
        "spilled partitions are joined with roles reversed: {reply}"
    );
    assert!(peak <= bound, "peak RSS {peak} B above {bound} B");
    assert!(
        reserved <= LI_LIMIT,
        "reserved {reserved} B above the limit"
    );
    // The result, written by the same process.
    let out = dir.path().join("out");
    let reply = runner.command(&format!(
        "DUMP {} {}",
        encode_path(file.to_str().unwrap()),
        encode_path(out.to_str().unwrap())
    ));
    assert!(reply.starts_with("OK 1 "), "{reply:?}");
    let mut got = Vec::new();
    for entry in std::fs::read_dir(&out).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "parquet") {
            let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
                std::fs::File::open(path).unwrap(),
            )
            .unwrap()
            .build()
            .unwrap();
            for batch in reader {
                let batch = batch.unwrap();
                for c in 0..2 {
                    let col = batch
                        .column(c)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    got.extend(col.iter().map(Option::unwrap_or_default));
                }
            }
        }
    }
    let (count, sum) = q21_shaped_expected();
    assert!(count > 1_000, "the query keeps rows: {count}");
    assert_eq!(got, vec![count, sum]);
    assert_eq!(runner.command("QUIT"), "");
}
