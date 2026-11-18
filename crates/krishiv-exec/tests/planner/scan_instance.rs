// SPDX-License-Identifier: Apache-2.0
//! `krishiv.exec.scan.instanceProjection`: each instance of a table read several times
//! reads only the columns its parent uses (study 38, lever A), with the same rows.

use super::partial_aggregate::{physical, runtime, sorted};
use super::run;
use super::tpch::{Order, TABLES, Tpch, parquet_files_with, sql};

const OFF: (&str, &str) = ("krishiv.exec.scan.instanceProjection", "false");

/// The column names of a plan's scans of `table`, without their attribute ids (which
/// other planner changes renumber).
fn scans_of(plan: &str, table: &str) -> Vec<String> {
    let marker = format!("Scan spark_catalog.default.{table} [");
    plan.lines()
        .filter_map(|l| l.split_once(&marker))
        .map(|(_, rest)| {
            let cols = rest.split_once(']').unwrap().0;
            cols.split(", ")
                .map(|c| c.split_once('#').map_or(c, |(name, _)| name))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect()
}

/// The number of columns every scan of a plan reads.
fn scan_columns(plan: &str) -> usize {
    plan.lines()
        .filter_map(|l| l.split_once("Scan spark_catalog.default."))
        .map(|(_, rest)| {
            let cols = rest.split_once('[').unwrap().1.split_once(']').unwrap().0;
            cols.split(", ").count()
        })
        .sum()
}

/// All 22 TPC-H queries give the same rows with the narrowing on and off, no scan reads
/// more columns with it on, and q21's EXISTS leg reads the two columns it uses instead of
/// the four the other `lineitem` instances need.
#[test]
fn tpch_scan_instances_read_their_own_columns() {
    let rt = runtime();
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let sources: Vec<(&str, Vec<String>)> = TABLES
        .iter()
        .map(|t| (*t, vec![format!("{t}.parquet")]))
        .collect();
    let on = parquet_files_with(&dir, &sources, true, &[], &rt);
    let off = parquet_files_with(&dir, &sources, true, &[OFF], &rt);
    let run_with = |t: &Tpch, q: usize| {
        let plan = t.optimized(&sql(q), Order::Statistics).unwrap();
        let runnable = physical(t, &plan).unwrap();
        (sorted(&run(&runnable).unwrap()), runnable.to_string())
    };
    let mut narrowed = Vec::new();
    for q in 1..=22 {
        let (rows_on, plan_on) = run_with(&on, q);
        let (rows_off, plan_off) = run_with(&off, q);
        assert_eq!(rows_on, rows_off, "q{q:02}");
        let (n_on, n_off) = (scan_columns(&plan_on), scan_columns(&plan_off));
        assert!(n_on <= n_off, "q{q:02}: {n_on} > {n_off}\n{plan_on}");
        if n_on < n_off {
            narrowed.push(q);
        }
        if q == 21 {
            let four = "l_orderkey, l_suppkey, l_commitdate, l_receiptdate";
            assert_eq!(scans_of(&plan_off, "lineitem"), vec![four; 3], "{plan_off}");
            let mut on = scans_of(&plan_on, "lineitem");
            on.sort();
            assert_eq!(on, vec!["l_orderkey, l_suppkey", four, four], "{plan_on}");
        }
    }
    assert!(narrowed.contains(&21), "{narrowed:?}");
}
