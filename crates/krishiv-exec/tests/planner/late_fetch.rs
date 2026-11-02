// SPDX-License-Identifier: Apache-2.0
//! Late fetch (design P2-T063): the same rows with it as without, over
//! several files and a file missing a deferred column; the planner uses it
//! only where its rules hold, and it reads the deferred columns of few rows.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Schema;
use krishiv_memory::{MemoryConfig, TreePool};
use proptest::prelude::*;

use super::runtime_filters::{configured, run, table, write};
use super::tpch::{Tpch, parquet_files, parquet_files_with, rows};

const OFF: &[(&str, &str)] = &[("krishiv.exec.lateFetch.maxRows", "0")];

/// `ORDER BY ... LIMIT` queries whose rows have one right answer: `v` breaks
/// every tie.
const QUERIES: [&str; 12] = [
    "SELECT * FROM big ORDER BY k32, v LIMIT 5",
    "SELECT * FROM big ORDER BY k32 NULLS LAST, v LIMIT 5",
    "SELECT * FROM big ORDER BY k32 DESC, v LIMIT 5",
    "SELECT * FROM big ORDER BY k32 DESC NULLS FIRST, v LIMIT 5",
    "SELECT * FROM big ORDER BY x DESC, v LIMIT 5",
    "SELECT * FROM big ORDER BY s, v LIMIT 4 OFFSET 3",
    "SELECT * FROM big WHERE v > 10 ORDER BY k64 DESC, v LIMIT 5",
    "SELECT * FROM big WHERE s LIKE '%x%' ORDER BY k32, v LIMIT 5",
    // The output's order isn't the scan's.
    "SELECT s, v, k32 FROM big ORDER BY k32, v LIMIT 5",
    "SELECT v, d102, x, f FROM big WHERE k32 > 0 ORDER BY k32, v LIMIT 5",
    // A computed key.
    "SELECT * FROM big ORDER BY k32 + 1, v LIMIT 5",
    // More rows than the filter leaves.
    "SELECT * FROM big WHERE v < 3 ORDER BY k32, v LIMIT 5",
];

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// The table `big` over three Parquet files of 64-row row groups.
fn tables(dir: &std::path::Path, batch: &RecordBatch, rt: &tokio::runtime::Runtime) -> Tpch {
    let n = batch.num_rows() / 3;
    let parts = [
        batch.slice(0, n),
        batch.slice(n, n),
        batch.slice(2 * n, batch.num_rows() - 2 * n),
    ];
    let mut names = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        let name = format!("part-{i}.parquet");
        write(dir, &name, part);
        names.push(name);
    }
    parquet_files(dir, &[("big", names)], true, rt)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    /// A late fetch returns what the plain scan returns.
    #[test]
    fn late_fetch_results_match_a_plain_scan(
        seed in 1u64..u64::MAX,
        domain in prop_oneof![Just(4i32), Just(400)],
        nulls in 0u32..30,
    ) {
        let rt = runtime();
        let dir = tempfile::tempdir().unwrap();
        let mut rng = seed;
        let mut t = tables(dir.path(), &table(&mut rng, 3000, domain, nulls), &rt);
        for q in QUERIES {
            configured(&mut t, OFF);
            let plan = t.plan_sql(q).unwrap();
            prop_assert!(!plan.to_string().contains("LateFetch"), "{}:\n{}", q, plan);
            let off = run(&plan).0;
            configured(&mut t, &[]);
            let plan = t.plan_sql(q).unwrap();
            prop_assert!(plan.to_string().contains("LateFetch"), "{}:\n{}", q, plan);
            let on = run(&plan).0;
            prop_assert_eq!(&rows(&on), &rows(&off), "{}:\n{}", q, plan);
            // Types and names as the plain scan has them, row ids gone.
            prop_assert_eq!(on[0].schema(), off[0].schema());
        }
    }
}

/// A file without a deferred column reads it as NULL, as a plain scan does.
#[test]
fn a_file_without_a_deferred_column_reads_it_as_null() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = 11;
    let batch = table(&mut rng, 2000, 400, 5);
    write(dir.path(), "full.parquet", &batch);
    let schema = batch.schema();
    let kept: Vec<usize> = (0..schema.fields().len())
        .filter(|&i| schema.field(i).name() != "s")
        .collect();
    let columns: Vec<ArrayRef> = kept.iter().map(|&i| Arc::clone(batch.column(i))).collect();
    let fields: Vec<_> = kept
        .iter()
        .map(|&i| Arc::clone(&schema.fields()[i]))
        .collect();
    let narrow = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    write(dir.path(), "narrow.parquet", &narrow);
    let names = vec!["full.parquet".to_owned(), "narrow.parquet".to_owned()];
    let mut t = parquet_files(dir.path(), &[("big", names)], true, &rt);
    // Each key's rows come from both files, so the first rows include both.
    let q = "SELECT s, x, v FROM big ORDER BY k32, v LIMIT 12";
    configured(&mut t, &[]);
    let plan = t.plan_sql(q).unwrap();
    assert!(plan.to_string().contains("LateFetch"), "{plan}");
    let on = run(&plan).0;
    let nulls: usize = on.iter().map(|b| b.column(0).null_count()).sum();
    assert!(nulls > 0, "{on:?}");
    configured(&mut t, OFF);
    assert_eq!(rows(&on), rows(&run(&t.plan_sql(q).unwrap()).0), "{plan}");
}

/// The planner's rules (design §4), and what the fetch reads.
#[test]
fn late_fetch_is_planned_only_where_it_pays_and_reads_few_rows() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = 3;
    // 47 row groups in all.
    let batch = table(&mut rng, 3000, 400, 5);
    let mut t = tables(dir.path(), &batch, &rt);
    configured(&mut t, &[]);
    for q in [
        // Nothing to defer: every column is a key or the filter's.
        "SELECT k32, s FROM big WHERE s LIKE '%x%' ORDER BY k32 LIMIT 5",
        "SELECT k32 FROM big ORDER BY k32 LIMIT 5",
        // No rows.
        "SELECT * FROM big ORDER BY k32 LIMIT 0",
        // Past the row limit, and past a quarter of the units.
        "SELECT * FROM big ORDER BY k32 LIMIT 51",
        "SELECT * FROM big ORDER BY k32 LIMIT 13",
        // The offset counts towards the row limit.
        "SELECT * FROM big ORDER BY k32 LIMIT 2 OFFSET 49",
        // A subquery in the filter.
        "SELECT * FROM big WHERE v > (SELECT min(v) FROM big) ORDER BY k32 LIMIT 3",
        // Not a scan under the sort.
        "SELECT k32, count(*) c FROM big GROUP BY k32 ORDER BY k32 LIMIT 3",
    ] {
        let plan = t.plan_sql(q).unwrap();
        assert!(!plan.to_string().contains("LateFetch"), "{q}:\n{plan}");
    }
    let q = "SELECT * FROM big ORDER BY k32, v LIMIT 3";
    let plan = t.plan_sql(q).unwrap();
    assert!(plan.to_string().contains("LateFetch"), "{plan}");
    let root = TreePool::new(&MemoryConfig {
        limit: Some(1 << 30),
        ..MemoryConfig::default()
    });
    let (out, query) = super::run_in(&plan, &root);
    let out = out.unwrap();
    assert_eq!(out.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    let decoded: Vec<u64> = query
        .metrics()
        .operators
        .values()
        .flat_map(|e| e.extra.iter())
        .filter(|(name, _)| name.as_ref() == "scan_rows_decoded")
        .map(|(_, v)| v.value().unwrap_or(0))
        .collect();
    // The narrow scan's rows and the fetch's three.
    assert!(decoded.contains(&3), "{decoded:?}\n{plan}");
}

/// A skipped file would make the second read differ from the first.
#[test]
fn late_fetch_stays_away_from_tables_that_skip_files() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = 9;
    write(dir.path(), "big.parquet", &table(&mut rng, 3000, 400, 5));
    let names = vec!["big.parquet".to_owned()];
    for setting in [
        "spark.sql.files.ignoreCorruptFiles",
        "spark.sql.files.ignoreMissingFiles",
    ] {
        let t = parquet_files_with(
            dir.path(),
            &[("big", names.clone())],
            true,
            &[(setting, "true")],
            &rt,
        );
        let plan = t
            .plan_sql("SELECT * FROM big ORDER BY k32 LIMIT 3")
            .unwrap();
        assert!(
            !plan.to_string().contains("LateFetch"),
            "{setting}:\n{plan}"
        );
    }
}

/// A shared body ending in a late fetch is ordered, so each reader keeps its
/// order.
#[test]
fn a_shared_body_ending_in_a_late_fetch_is_read_in_order() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = 13;
    let mut t = tables(dir.path(), &table(&mut rng, 3000, 400, 5), &rt);
    configured(&mut t, &[]);
    let plan = t
        .plan_sql(
            "WITH c AS (SELECT s, v, k32 FROM big WHERE rand() < 2 ORDER BY k32, v LIMIT 5) \
             SELECT k32, v FROM c UNION ALL SELECT k32, v FROM c",
        )
        .unwrap();
    let explain = plan.plan.explain_all().to_string();
    assert!(explain.contains("LateFetch"), "{explain}");
    assert_eq!(plan.plan.shared().len(), 1, "{explain}");
    assert!(plan.plan.shared()[0].ordered, "{explain}");
    let out = rows(&run(&plan).0);
    assert_eq!(out.len(), 10, "{out:?}");
    assert_eq!(out[..5], out[5..], "{out:?}");
}

/// Under a memory limit the fetch reserves what it holds and still finishes.
#[test]
fn late_fetch_runs_under_a_small_memory_limit() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = 5;
    let mut t = tables(dir.path(), &table(&mut rng, 3000, 400, 5), &rt);
    configured(&mut t, &[]);
    let plan = t
        .plan_sql("SELECT * FROM big ORDER BY k32, v LIMIT 5")
        .unwrap();
    assert!(plan.to_string().contains("LateFetch"), "{plan}");
    let root = TreePool::new(&MemoryConfig {
        limit: Some(16 << 20),
        ..MemoryConfig::default()
    });
    let (out, query) = super::run_in(&plan, &root);
    assert_eq!(
        out.unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        5
    );
    drop(query);
}
