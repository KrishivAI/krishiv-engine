// SPDX-License-Identifier: Apache-2.0
//! Runtime filters (design P2-T163) never change results: TPC-H with
//! filters off, on and forced (a Bloom filter for every key, no adaptive
//! switch-off, or a switch-off after the first batch), and randomized joins
//! over Parquet files whose key types differ from the join's (`int` against
//! `bigint`, decimal widths, `float` against `double`, signed zeros and NaN,
//! strings, NULLs), with scans below aggregates and unions; and the
//! filters do prune: rows, row groups and whole splits.

use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringViewArray,
};
use arrow_schema::{DataType, Field, Schema};
use krishiv_exec::planner::PhysicalPlan;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{FileFormat as _, FormatOptions};
use krishiv_memory::{MemoryConfig, TreePool};
use proptest::prelude::*;

use super::tpch::{Tpch, answers, expected_rows, parquet_files, parquet_tables, rows, sql};

/// Configurations compared: off, the defaults, and forced forms.
const MODES: [(&str, &[(&str, &str)]); 5] = [
    ("off", &[("krishiv.runtimeFilter.enabled", "false")]),
    ("default", &[]),
    (
        "bloom",
        &[
            ("krishiv.runtimeFilter.bloom.minApplicationRows", "0"),
            ("krishiv.runtimeFilter.inList.maxValues", "0"),
            ("krishiv.runtimeFilter.adaptive.maxPassPercent", "100"),
        ],
    ),
    (
        "values",
        &[
            ("krishiv.runtimeFilter.inList.maxValues", "3"),
            ("spark.sql.optimizer.runtime.bloomFilter.enabled", "false"),
            ("krishiv.runtimeFilter.adaptive.maxPassPercent", "100"),
        ],
    ),
    (
        "adaptive-off",
        &[
            ("krishiv.runtimeFilter.bloom.minApplicationRows", "0"),
            ("krishiv.runtimeFilter.adaptive.minRows", "1"),
            ("krishiv.runtimeFilter.adaptive.maxPassPercent", "0"),
        ],
    ),
];

fn configured(tpch: &mut Tpch, mode: &[(&str, &str)]) {
    let mut config = krishiv_common::config::Config::new();
    config.set("spark.sql.ansi.enabled", "true").unwrap();
    for (k, v) in mode {
        config.set(k, v).unwrap();
    }
    tpch.config = Arc::new(config);
}

/// Rows and the runtime filter counters of one run.
fn run(plan: &PhysicalPlan) -> (Vec<RecordBatch>, Counters) {
    let root = TreePool::new(&MemoryConfig::default());
    let (out, query) = super::run_in(plan, &root);
    let out = out.unwrap_or_else(|e| panic!("{e}\n{plan}"));
    let mut c = Counters::default();
    for entry in query.metrics().operators.values() {
        for (name, v) in &entry.extra {
            match *name {
                "runtime_filter_rows_dropped" => c.rows_dropped += v,
                "runtime_filter_units_pruned" => c.units_pruned += v,
                "runtime_filter_splits_skipped" => c.splits_skipped += v,
                _ => {}
            }
        }
    }
    (out, c)
}

#[derive(Debug, Default, Clone, Copy)]
struct Counters {
    rows_dropped: u64,
    units_pruned: u64,
    splits_skipped: u64,
}

/// All 22 TPC-H queries return Spark's answers in every mode.
#[test]
fn tpch_results_do_not_depend_on_runtime_filters() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let mut tpch = parquet_tables(&dir, &rt);
    let answers = answers(&dir.join("spark-answers.json"));
    let mut failed = Vec::new();
    let mut dropped = [0u64; MODES.len()];
    let mut blooms = 0;
    for q in 1..=22 {
        let want = expected_rows(&answers, &format!("q{q:02}"));
        for (m, (name, mode)) in MODES.iter().enumerate() {
            configured(&mut tpch, mode);
            let plan = tpch.plan_sql(&sql(q)).unwrap();
            let planned = plan.to_string().matches("(bloom)").count();
            // `spark.sql.optimizer.runtime.bloomFilter.enabled=false` is
            // honoured; off plans no filter at all.
            assert!(!(*name == "values" && planned > 0), "q{q:02}: {plan}");
            assert!(
                !(*name == "off" && plan.to_string().contains("RuntimeFilter")),
                "q{q:02}: {plan}"
            );
            blooms += planned;
            let (got, c) = run(&plan);
            dropped[m] += c.rows_dropped;
            if rows(&got) != want {
                failed.push(format!("q{q:02} ({name}) differs\n{plan}"));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n\n"));
    assert_eq!(dropped[0], 0, "no filters when off");
    assert!(
        dropped[1] > 0 && dropped[2] > 0 && dropped[3] > 0,
        "{dropped:?}"
    );
    assert!(blooms >= 20, "{blooms} Bloom filters planned");
}

/// One table of the randomized joins: every key type, NULLs, and runs of
/// equal keys in row groups of 64 rows.
fn table(rng: &mut u64, rows: usize, domain: i32, nulls: u32) -> RecordBatch {
    let mut next = || {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng
    };
    let mut keys: Vec<Option<i32>> = (0..rows)
        .map(|_| {
            let r = next();
            let k = i32::try_from(r % u64::from(domain.unsigned_abs())).unwrap() - domain / 2;
            (r % 100 >= u64::from(nulls)).then_some(k)
        })
        .collect();
    // Clustered keys make row groups prunable.
    if next() % 2 == 0 {
        keys.sort_unstable();
    }
    let f = |k: i32| match k % 7 {
        0 => -0.0,
        3 => f64::NAN,
        _ => f64::from(k) / 4.0,
    };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(keys.iter().copied().collect::<Int32Array>()),
        Arc::new(
            keys.iter()
                .map(|k| k.map(i64::from))
                .collect::<Int64Array>(),
        ),
        Arc::new(
            keys.iter()
                .map(|k| k.map(|k| i128::from(k) * 10))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(5, 1)
                .unwrap(),
        ),
        Arc::new(
            keys.iter()
                .map(|k| k.map(|k| i128::from(k) * 100))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ),
        Arc::new(
            keys.iter()
                .map(|k| k.map(|k| narrow(f(k))))
                .collect::<Float32Array>(),
        ),
        Arc::new(keys.iter().map(|k| k.map(f)).collect::<Float64Array>()),
        Arc::new(
            keys.iter()
                .map(|k| {
                    k.map(|k| {
                        format!(
                            "key-{k}-{}",
                            "x".repeat(usize::try_from(k.rem_euclid(3)).unwrap() * 8)
                        )
                    })
                })
                .collect::<StringViewArray>(),
        ),
        Arc::new(
            (0..rows)
                .map(|i| i32::try_from(i).unwrap())
                .collect::<Int32Array>(),
        ),
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("k32", DataType::Int32, true),
        Field::new("k64", DataType::Int64, true),
        Field::new("d51", DataType::Decimal128(5, 1), true),
        Field::new("d102", DataType::Decimal128(10, 2), true),
        Field::new("f", DataType::Float32, true),
        Field::new("x", DataType::Float64, true),
        Field::new("s", DataType::Utf8View, true),
        Field::new("v", DataType::Int32, false),
    ]));
    RecordBatch::try_new(schema, columns).unwrap()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "test values: quarters, NaN and zero, exact in float"
)]
fn narrow(x: f64) -> f32 {
    x as f32
}

fn write(dir: &Path, name: &str, batch: &RecordBatch) {
    let o = FormatOptions::defaults(ParquetFormat.options());
    let mut enc = ParquetFormat::encoder_with(&batch.schema(), &o, |p| {
        p.set_max_row_group_row_count(Some(64))
    })
    .unwrap();
    let mut bytes = Vec::new();
    enc.write(batch).unwrap();
    while let Some(b) = enc.take_bytes() {
        bytes.extend_from_slice(&b);
    }
    bytes.extend_from_slice(&enc.finish().unwrap().tail);
    std::fs::write(dir.join(name), bytes).unwrap();
}

/// Joins of every key pairing and join type, a scan below an aggregate and
/// a union on the probe side, a widening cast in a projection, several
/// keys, null-safe keys.
const JOINS: [&str; 14] = [
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k32 = s.k64",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k64 = s.k32",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.d51 = s.d102",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.x = s.x",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.f = s.x",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.s = s.s",
    "SELECT b.v FROM big b LEFT SEMI JOIN small s ON b.k32 = s.k32",
    "SELECT b.v FROM big b LEFT ANTI JOIN small s ON b.k32 = s.k32",
    "SELECT b.v, s.v FROM big b LEFT JOIN small s ON b.k64 = s.k64",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k32 <=> s.k32",
    "SELECT x.k, x.c, s.v FROM (SELECT k32 AS k, count(*) AS c FROM big GROUP BY k32) x \
     JOIN small s ON x.k = s.k32",
    "SELECT u.k, s.v FROM (SELECT k32 AS k FROM big UNION ALL SELECT k32 FROM big WHERE v < 100) u \
     JOIN small s ON u.k = s.k32",
    "SELECT b.v, s.v FROM (SELECT CAST(k32 AS BIGINT) AS kk, v FROM big) b \
     JOIN small s ON b.kk = s.k64",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k32 = s.k32 AND b.s = s.s",
];

fn sorted(batches: &[RecordBatch]) -> Vec<String> {
    let mut r = rows(batches);
    r.sort_unstable();
    r
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    /// Every join returns the same rows with runtime filters off and in
    /// every forced mode; across the cases, filters drop rows, prune row
    /// groups, and skip splits of empty builds.
    #[test]
    fn randomized_joins_do_not_depend_on_runtime_filters(
        seed in 1u64..u64::MAX,
        domain in prop_oneof![Just(4i32), Just(40), Just(400)],
        nulls in 0u32..30,
        small_rows in prop_oneof![Just(0usize), Just(5), Just(40)],
        // A scan under an aggregate runs while the build does and adopts
        // the filter late (or never): results must not change either way.
        big_rows in prop_oneof![Just(100usize), Just(1500)],
    ) {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut rng = seed;
        write(dir.path(), "big.parquet", &table(&mut rng, big_rows, domain, nulls));
        write(dir.path(), "small.parquet", &table(&mut rng, small_rows, domain, nulls));
        let sources = [
            ("big", vec!["big.parquet".to_owned()]),
            ("small", vec!["small.parquet".to_owned()]),
        ];
        let mut t = parquet_files(dir.path(), &sources, true, &rt);
        let mut total = Counters::default();
        for q in JOINS {
            configured(&mut t, MODES[0].1);
            let off = sorted(&run(&t.plan_sql(q).unwrap()).0);
            for (name, mode) in &MODES[1..] {
                configured(&mut t, mode);
                let plan = t.plan_sql(q).unwrap();
                let (got, c) = run(&plan);
                total.rows_dropped += c.rows_dropped + c.units_pruned;
                total.splits_skipped += c.splits_skipped;
                prop_assert_eq!(&sorted(&got), &off, "{} ({}):\n{}", q, name, plan);
            }
        }
        // Not vacuous: an empty build skips the probe side unread; a few
        // build keys out of many drop probe rows.
        if small_rows == 0 {
            prop_assert!(total.splits_skipped > 0, "{:?}", total);
        }
        if small_rows == 5 && domain >= 40 {
            prop_assert!(total.rows_dropped > 0, "{:?}", total);
        }
    }
}

/// Filters prune at every stage they reach: an empty build skips the probe
/// side's splits unopened; a selective build prunes row groups of a
/// clustered key and drops rows of the others; a scan below an aggregate
/// runs while the build does (its filter may arrive late) and its join
/// still returns the right rows.
#[test]
fn filters_prune_splits_row_groups_and_rows() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = 7;
    let mut big = table(&mut rng, 4000, 4000, 0);
    // Sorted keys: row groups hold disjoint ranges.
    let mut keys: Vec<i32> = big
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .values()
        .to_vec();
    keys.sort_unstable();
    let mut cols = big.columns().to_vec();
    cols[0] = Arc::new(Int32Array::from(keys));
    big = RecordBatch::try_new(big.schema(), cols).unwrap();
    write(dir.path(), "big.parquet", &big);
    write(dir.path(), "small.parquet", &table(&mut rng, 3, 4000, 0));
    write(dir.path(), "none.parquet", &table(&mut rng, 0, 10, 0));
    let sources = [
        ("big", vec!["big.parquet".to_owned()]),
        ("small", vec!["small.parquet".to_owned()]),
        ("none", vec!["none.parquet".to_owned()]),
    ];
    let mut t = parquet_files(dir.path(), &sources, true, &rt);
    configured(&mut t, &[]);
    let q = "SELECT b.v FROM big b JOIN small s ON b.k32 = s.k32";
    let (_, c) = run(&t.plan_sql(q).unwrap());
    assert!(c.units_pruned >= 55, "{c:?}");
    let q = "SELECT b.v FROM big b JOIN small s ON b.k64 + 0 = s.k64 AND b.k32 = s.k32";
    let (_, c) = run(&t.plan_sql(q).unwrap());
    assert!(c.units_pruned >= 55, "{c:?}");
    let q = "SELECT b.v FROM big b JOIN none n ON b.k32 = n.k32";
    let (got, c) = run(&t.plan_sql(q).unwrap());
    assert!(got.iter().all(|b| b.num_rows() == 0));
    assert!(c.splits_skipped >= 1, "{c:?}");
    let q = "SELECT x.k, s.v FROM (SELECT k32 AS k, count(*) AS c FROM big GROUP BY k32) x \
             JOIN small s ON x.k = s.k32";
    let plan = t.plan_sql(q).unwrap();
    assert!(plan.to_string().contains("RuntimeFilter #"), "{plan}");
    configured(&mut t, MODES[0].1);
    let off = sorted(&run(&t.plan_sql(q).unwrap()).0);
    configured(&mut t, &[]);
    assert_eq!(sorted(&run(&plan).0), off);
    // Unclustered keys: rows are dropped at the row stage.
    let q = "SELECT b.v FROM big b JOIN small s ON b.s = s.s";
    let (_, c) = run(&t.plan_sql(q).unwrap());
    assert!(c.rows_dropped > 0, "{c:?}");
}

/// Runtime filters on and off over a TPC-H or TPC-DS data set
/// (`KRISHIV_RF_BENCH_DIR`, one `<table>.parquet` per table;
/// `KRISHIV_RF_BENCH_SUITE` = `tpch` or `tpcds`): per query, the results
/// must agree, and it prints the median time of each mode over
/// interleaved off-on-on-off blocks and what the filters pruned. Run with
/// `cargo test --release -p krishiv-exec --test planner runtime_filter_bench -- --ignored --nocapture`.
#[test]
#[ignore = "needs a benchmark data set"]
fn runtime_filter_bench() {
    let dir = std::env::var("KRISHIV_RF_BENCH_DIR").expect("KRISHIV_RF_BENCH_DIR");
    let suite = std::env::var("KRISHIV_RF_BENCH_SUITE").unwrap_or_else(|_| "tpch".to_owned());
    let reps: usize = std::env::var("KRISHIV_RF_BENCH_REPS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(2);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "parquet").then(|| p.file_stem()?.to_str().map(str::to_owned))?
        })
        .collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut t = super::tpch::parquet_dir(Path::new(&dir), &names, true, &rt);
    let queries = super::tpch::root().join(format!("bench/queries/{suite}"));
    let mut files: Vec<_> = std::fs::read_dir(&queries)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "sql"))
        .collect();
    files.sort();
    if let Ok(only) = std::env::var("KRISHIV_RF_BENCH_QUERIES") {
        files.retain(|f| {
            only.split(',')
                .any(|q| f.file_stem().is_some_and(|s| s == q))
        });
    }
    println!("query|off ms|on ms|on/off|rows dropped|row groups pruned|MiB pruned|splits skipped");
    let (mut log_sum, mut n) = (0.0, 0);
    let mut mismatched = Vec::new();
    for f in files {
        let q = f.file_stem().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(&f).unwrap();
        // A plan runs once: each run plans anew (not timed).
        let mut plan = |on: bool| {
            configured(&mut t, MODES[usize::from(on)].1);
            t.plan_sql(&text)
        };
        if std::env::var_os("KRISHIV_RF_BENCH_EXPLAIN").is_some()
            && let Ok(p) = plan(true)
        {
            println!("{p}");
        }
        if let Err(e) = plan(true) {
            println!(
                "{q}|unsupported: {}",
                e.to_string().lines().next().unwrap_or("")
            );
            continue;
        }
        let root = TreePool::new(&MemoryConfig::default());
        let mut timed = |on: bool| {
            let plan = plan(on).unwrap();
            let start = std::time::Instant::now();
            let (out, query) = super::run_in(&plan, &root);
            (out, query, start.elapsed().as_secs_f64())
        };
        let (base, _, _) = timed(false);
        let Ok(base) = base else {
            println!("{q}|failed");
            continue;
        };
        let (mut t_off, mut t_on) = (Vec::new(), Vec::new());
        let mut c = Counters::default();
        let mut bytes = 0;
        for _ in 0..reps {
            for on_first in [false, true, true, false] {
                let (out, query, secs) = timed(on_first);
                let out = out.unwrap_or_else(|e| panic!("{q}: {e}"));
                if on_first {
                    t_on.push(secs);
                    if sorted(&out) != sorted(&base) {
                        mismatched.push(q.clone());
                    }
                    c = Counters::default();
                    bytes = 0;
                    for entry in query.metrics().operators.values() {
                        for (name, v) in &entry.extra {
                            match *name {
                                "runtime_filter_rows_dropped" => c.rows_dropped += v,
                                "runtime_filter_units_pruned" => c.units_pruned += v,
                                "runtime_filter_splits_skipped" => c.splits_skipped += v,
                                "runtime_filter_bytes_pruned" => bytes += v,
                                _ => {}
                            }
                        }
                    }
                } else {
                    t_off.push(secs);
                }
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (a, b) = (median(&mut t_off), median(&mut t_on));
        log_sum += (b / a).ln();
        n += 1;
        #[expect(clippy::cast_precision_loss, reason = "a report")]
        let mib = bytes as f64 / f64::from(1 << 20);
        println!(
            "{q}|{:.1}|{:.1}|{:.3}|{}|{}|{mib:.1}|{}",
            a * 1e3,
            b * 1e3,
            b / a,
            c.rows_dropped,
            c.units_pruned,
            c.splits_skipped
        );
    }
    let geomean = (log_sum / f64::from(n.max(1))).exp();
    println!("geomean on/off over {n} queries: {geomean:.3}");
    assert!(mismatched.is_empty(), "results differ: {mismatched:?}");
}
