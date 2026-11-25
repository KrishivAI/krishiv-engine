// SPDX-License-Identifier: Apache-2.0
//! Partition-indexed expressions around `UNION ALL` against
//! `tests/fixtures/spark-4.2.0/union-partition-ids.json` (real Spark 4.2.0
//! runs at `local[1]` and `local[4]`).
//!
//! Spark initialises a nondeterministic expression with the partition index
//! of the operator that hosts it: inside a branch that is the branch's own
//! index (`UnionRDD` iterates the child's own partition), above the union
//! the union's (branch 2 numbered after branch 1). A union whose children
//! are all `SinglePartition` (a one-slice `range`, a global aggregate) is one
//! partition (`spark.sql.unionOutputPartitioning`, `UnionExec`'s
//! `comparePartitioning`): its children run one after another at index 0,
//! and an expression above it sees one stream.
//!
//! Rows are compared exactly, in Spark's `collect()` order, at 1, 2, 4 and 8
//! compute threads, except the shapes listed in [`skip_reason`] and
//! [`one_thread_only`].
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_exec::{ExecutorConfig, TokioExecutor};
use krishiv_session::Session;
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/union-partition-ids.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

/// A session at `threads` compute threads with `script`'s views; the
/// one-file Parquet tables `p1a`, `p1b` written under `dir`.
fn session(doc: &Value, script: &str, threads: usize, dir: &std::path::Path) -> Session {
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: threads,
        io_threads: 1,
        blocking_threads: threads,
        io_blocking_threads: 2,
        release_when_idle: true,
    })
    .unwrap();
    let s = Session::builder().executor(executor).build().unwrap();
    for (name, range) in doc["views"][script].as_object().unwrap() {
        let range = range.as_str().unwrap();
        run(
            &s,
            &format!("CREATE TEMPORARY VIEW {name} AS SELECT id FROM {range}"),
        );
    }
    // Spark's `spark.range(a, b).coalesce(1).write.parquet`: one file.
    for (name, range) in [("p1a", "range(0, 6, 1, 1)"), ("p1b", "range(10, 14, 1, 1)")] {
        let path = dir.join(name);
        let path = path.to_str().unwrap();
        if !std::path::Path::new(path).exists() {
            let df = block_on(s.sql(&format!("SELECT id FROM {range}"))).unwrap();
            block_on(df.write().parquet(path)).unwrap();
        }
        run(
            &s,
            &format!(
                "CREATE TEMPORARY VIEW {name}(id BIGINT) USING parquet OPTIONS (path '{path}')"
            ),
        );
    }
    s
}

/// Java's `Double.toString`: plain digits in `[1e-3, 1e7)`, else
/// scientific with at least one fraction digit.
fn java_double(v: f64) -> String {
    let a = v.abs();
    if v == 0.0 || (1e-3..1e7).contains(&a) {
        let s = format!("{v}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        let s = format!("{v:e}");
        let (m, e) = s.split_once('e').unwrap();
        let m = if m.contains('.') {
            m.to_owned()
        } else {
            format!("{m}.0")
        };
        format!("{m}E{e}")
    }
}

/// One value as Spark's `Row.toSeq.map(_.toString)` prints it.
fn cell(c: &ArrayRef, i: usize) -> String {
    if c.is_null(i) {
        return "null".to_owned();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Float64 => {
            java_double(c.as_primitive::<arrow_array::types::Float64Type>().value(i))
        }
        DataType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
        DataType::LargeUtf8 => c.as_string::<i64>().value(i).to_owned(),
        DataType::Utf8View => c.as_string_view().value(i).to_owned(),
        DataType::List(_) => {
            let l = c.as_list::<i32>().value(i);
            let items: Vec<String> = (0..l.len()).map(|j| cell(&l, j)).collect();
            format!("ArraySeq({})", items.join(", "))
        }
        t => panic!("unexpected type {t}"),
    }
}

fn rows(s: &Session, sql: &str) -> Vec<String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let row: Vec<String> = b.columns().iter().map(|c| cell(c, i)).collect();
            out.push(row.join(","));
        }
    }
    out
}

/// Shapes not compared, with the reason.
fn skip_reason(tag: &str) -> Option<&'static str> {
    if tag.starts_with("c_pq3.") || tag.starts_with("g.pq3_") || tag.starts_with("g.pq1_pq3") {
        // Spark packs the three small files by `defaultParallelism` (and
        // `minPartitionNum`), and `repartition(3)` puts rows in files at
        // random: neither the partitions nor the rows of a file are
        // reproduced by Krishiv's writer and scan.
        return Some("three-file Parquet layout");
    }
    if tag.contains(".ifn.") || tag.contains(".ifbs.") || tag.contains(".ifbl.") {
        // The file names (random UUIDs) and sizes are Spark's own files.
        return Some("file name and size of Spark's files");
    }
    // Known gaps (the reviewer's Spark run of a70c2be3e, script
    // `review-probe.scala`): Spark's `GlobalLimitExec` and a sort are one
    // partition over any input, Krishiv's `LIMIT` and sort keep their
    // input's; a coalescible union under a hash-join probe runs its branches
    // together in Krishiv, so it keeps the plain numbering
    // (`a_coalescible_union_under_a_join_probe_stays_deterministic`).
    if matches!(
        tag,
        "r.limit_multi_above" | "r.sort_branch_above" | "r.bhj_above" | "r.bhj_above_nohint_parts"
    ) {
        return Some("known gap");
    }
    if tag == "r.range1_4slices_above" {
        // Krishiv clamps a range's slices to its rows (one row is slice 0 of
        // 1, Spark's slice 3 of 4); see
        // `a_range_of_more_slices_than_rows_is_not_one_partition`.
        return Some("range slice clamp");
    }
    if tag == "f2.repartition3_both" {
        // `DISTRIBUTE BY` on both sides: AQE coalesced each shuffle to one
        // partition in that run (Spark's hash-partitioned union is
        // unverified, semantics.md); on one worker Krishiv lowers the
        // exchange away and the range's slices keep their indexes.
        return Some("AQE-coalesced hash exchange");
    }
    None
}

/// Shapes compared at one compute thread only: Spark's partitions follow
/// its own layout rules here (`local[1]`), which Krishiv matches with one
/// driver per pipeline.
fn one_thread_only(tag: &str) -> bool {
    // A file scan's and `VALUES`' partitions follow the thread count (and
    // Spark's `defaultParallelism`); an aggregate's or exchange's output
    // partitions are AQE-coalesced to one in Spark, a driver each in Krishiv.
    tag.starts_with("c_pq1.")
        || tag.starts_with("g.pq1_pq1")
        || tag == "f.values"
        || tag == "f2.agg_children"
        // A shared CTE's stored batches carry no split: at more threads its
        // copies are read by several drivers, whose ids differ from Spark's
        // one partition (a known gap).
        || tag.starts_with("r.cte_two_reads")
}

/// Whether Spark's row order is not defined by the plan (an aggregate's
/// groups): compared sorted.
fn sorted(tag: &str) -> bool {
    tag == "f2.agg_children"
}

fn check(threads: &[usize], filter: impl Fn(&str) -> bool) {
    let doc = fixture();
    let dir = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    let mut checked = 0;
    for &threads in threads {
        let mut sessions: std::collections::HashMap<String, Session> =
            std::collections::HashMap::new();
        for case in doc["cases"].as_array().unwrap() {
            let tag = case["tag"].as_str().unwrap();
            if skip_reason(tag).is_some() || !filter(tag) {
                continue;
            }
            if threads > 1 && one_thread_only(tag) {
                continue;
            }
            let script = case["script"].as_str().unwrap();
            let s = sessions
                .entry(script.to_owned())
                .or_insert_with(|| session(&doc, script, threads, dir.path()));
            let conf = case["conf"].as_object().unwrap();
            for (k, v) in conf {
                run(s, &format!("SET {k}={}", v.as_str().unwrap()));
            }
            let sql = case["sql"].as_str().unwrap();
            let mut got = rows(s, sql);
            let mut want: Vec<String> = case["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_str().unwrap().to_owned())
                .collect();
            if sorted(tag) {
                got.sort();
                want.sort();
            }
            let same = got == want;
            if !same {
                failures.push(format!(
                    "{tag} at {threads} threads: {sql}\n  want {want:?}\n  got  {got:?}"
                ));
            }
            for k in conf.keys() {
                run(s, &format!("RESET {k}"));
            }
            checked += 1;
        }
    }
    assert!(checked > 0);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The expressions below and above a union of one-slice ranges, multi-slice
/// ranges and one-file Parquet tables.
#[test]
fn expressions_around_a_union_match_spark() {
    check(&[1, 2, 4, 8], |t| {
        t.starts_with("a_single.") || t.starts_with("b_") || t.starts_with("c_")
    });
}

/// Nested unions: a leaf of a flattened union and an expression over an
/// inner union inside an outer branch.
#[test]
fn nested_unions_match_spark() {
    check(&[1, 2, 4, 8], |t| t.starts_with("e_"));
}

/// When Spark makes a union one partition (`f`, `f2`), and the same with
/// `spark.sql.unionOutputPartitioning` off (`f2off`).
#[test]
fn union_coalescing_matches_spark() {
    check(&[1, 2, 4, 8], |t| t.starts_with("f."));
    check(&[1, 2, 4, 8], |t| t.starts_with("f2"));
}

/// The reviewer's shapes: global aggregates (conf on and off), a sample and
/// a generator in each branch.
#[test]
fn review_shapes_match_spark() {
    check(&[1, 2, 4, 8], |t| t.starts_with("r."));
}

/// `range(0, 1, 1, 4)` asks for four slices: Spark's `RangeExec` is not
/// `SinglePartition` (`numSlices` 4), so its union with another such range
/// stays plain (`r.range1_4slices_above`: 8 partitions, spid 3 and 7).
/// Krishiv clamps the slices to the one row, so the plain numbering gives
/// 0 and 1; coalescing would give 0 and 0.
#[test]
fn a_range_of_more_slices_than_rows_is_not_one_partition() {
    let doc = fixture();
    let dir = tempfile::tempdir().unwrap();
    for threads in [1, 4] {
        let s = session(&doc, "review-probe.scala", threads, dir.path());
        assert_eq!(
            rows(
                &s,
                "SELECT id, spark_partition_id() p FROM (SELECT id FROM range(0,1,1,4) \
                 UNION ALL SELECT id FROM range(5,6,1,4))"
            ),
            ["0,0", "5,1"],
            "{threads} threads"
        );
    }
}

/// `TABLESAMPLE ... REPEATABLE` below and above a union.
#[test]
fn samples_around_a_union_match_spark() {
    check(&[1, 2, 4, 8], |t| t.starts_with("g."));
}

/// Over two ranges of one slice each, a sample above the union draws one
/// stream (Spark 4.2.0, `local[1]`: 204 rows; run
/// `~/krishiv-data/tmp/opus-3/r4/p2b.out`).
#[test]
fn a_sample_over_a_union_of_single_ranges_draws_one_stream() {
    let doc = fixture();
    let dir = tempfile::tempdir().unwrap();
    let s = session(&doc, "probe.scala", 1, dir.path());
    run(
        &s,
        "CREATE TEMPORARY VIEW t AS SELECT id FROM range(0, 1000)",
    );
    let got = rows(
        &s,
        "SELECT * FROM (SELECT id FROM t UNION ALL SELECT id FROM t) \
         TABLESAMPLE (10 PERCENT) REPEATABLE (7)",
    );
    assert_eq!(got.len(), 204);
    let mut ids: Vec<i64> = got.iter().map(|r| r.parse().unwrap()).collect();
    ids.sort_unstable();
    assert_eq!(ids[..12], [11, 11, 13, 20, 25, 27, 31, 36, 37, 39, 42, 44]);
}

#[test]
fn java_doubles_print_as_spark_does() {
    assert_eq!(java_double(0.619_189_370_225_301), "0.619189370225301");
    assert_eq!(java_double(1.0), "1.0");
    assert_eq!(java_double(0.000_12), "1.2E-4");
    assert_eq!(java_double(-0.000_1), "-1.0E-4");
}

/// A union of two one-slice ranges probing a broadcast build: the probe needs
/// a flush, so the union's branches run together. Spark makes the union one
/// partition and draws one stream above the join (`bhj_above` in the
/// fixture: mono 0..7, one `rand(42)` stream); Krishiv keeps the plain
/// numbering for a group whose members run together (a known gap,
/// `numbering.rs`), so each branch draws its own stream and the rows are the
/// same on every run at any thread count.
#[test]
fn a_coalescible_union_under_a_join_probe_stays_deterministic() {
    let doc = fixture();
    let dir = tempfile::tempdir().unwrap();
    let sql = "SELECT /*+ BROADCAST(d) */ u.id, monotonically_increasing_id() m, rand(42) r \
               FROM (SELECT id FROM s1 UNION ALL SELECT id FROM s2) u JOIN d ON u.id = d.id";
    let mut first: Option<Vec<String>> = None;
    for threads in [1, 2, 4, 8] {
        let s = session(&doc, "probe.scala", threads, dir.path());
        run(
            &s,
            "CREATE TEMPORARY VIEW d AS SELECT id FROM range(0, 20, 1, 1)",
        );
        for attempt in 0..5 {
            let mut got = rows(&s, sql);
            got.sort();
            // The plain numbering: branch 2 is partition 1 above the union.
            assert_eq!(
                got,
                [
                    "0,0,0.619189370225301",
                    "1,1,0.5096018842446481",
                    "10,8589934592,0.8017532427858894",
                    "11,8589934593,0.6565552949992319",
                    "12,8589934594,0.2515595782593636",
                    "13,8589934595,0.2073428376111074",
                    "2,2,0.8325259388871524",
                    "3,3,0.26322809041172357",
                ],
                "{threads} threads, run {attempt}"
            );
            match &first {
                None => first = Some(got),
                Some(want) => assert_eq!(&got, want, "{threads} threads, run {attempt}"),
            }
        }
    }
}
