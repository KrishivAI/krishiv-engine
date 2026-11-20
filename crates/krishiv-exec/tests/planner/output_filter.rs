// SPDX-License-Identifier: Apache-2.0
//! The aggregate output filter (`krishiv.exec.agg.outputFilter`): a
//! `HAVING` over a grouped aggregate's values is also evaluated by the
//! aggregate before it decodes keys. Results never change; where it applies
//! the `Filter` above receives only the groups that pass, and a predicate
//! that reads a key, or an expression projection between the two, leaves
//! the aggregate's output whole.

use std::sync::Arc;

use krishiv_memory::{MemoryConfig, TreePool};

use super::Runnable;
use super::runtime_filters::{configured, table, write};
use super::tpch::{parquet_files, rows};

/// Sorted rows and the rows the plan's `Filter` operators received.
fn run(plan: &Runnable) -> (Vec<String>, u64) {
    let root = TreePool::new(&MemoryConfig {
        limit: Some(1 << 30),
        ..MemoryConfig::default()
    });
    let (out, query) = super::run_in(plan, &root);
    let out = out.unwrap_or_else(|e| panic!("{e}\n{plan}"));
    let filtered = query
        .metrics()
        .operators
        .values()
        .filter(|e| e.name == "Filter")
        .map(|e| e.metrics.rows_in)
        .sum();
    let mut r = rows(&out);
    r.sort_unstable();
    (r, filtered)
}

#[test]
fn having_results_do_not_depend_on_the_output_filter() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = 31;
    write(dir.path(), "t.parquet", &table(&mut rng, 20_000, 4_000, 5));
    let sources = [("t", vec!["t.parquet".to_owned()])];
    let mut t = parquet_files(dir.path(), &sources, true, &rt);
    // (query, whether the aggregate takes the filter)
    let cases = [
        (
            "SELECT k64, sum(v) AS s FROM t GROUP BY k64 HAVING sum(v) > 60000",
            true,
        ),
        (
            "SELECT k64 FROM t GROUP BY k64 HAVING count(*) > 5 AND max(v) < 19000",
            true,
        ),
        (
            "SELECT k64, s FROM (SELECT k64, sum(v) AS s, count(*) AS c FROM t GROUP BY k64) \
             WHERE s > 60000 OR c IS NULL",
            true,
        ),
        // A key in the predicate (not pushed below: an OR with a value): the
        // aggregate keeps its output whole.
        (
            "SELECT k64, sum(v) FROM t GROUP BY k64 HAVING sum(v) > 60000 OR k64 > 1900",
            false,
        ),
        // A predicate that may raise (ANSI division; count(*) is never 0
        // here): refused, the Filter alone evaluates it (review NIT-A).
        (
            "SELECT k64 FROM t GROUP BY k64 HAVING sum(v) / count(*) > 2500",
            false,
        ),
        // An expression between the aggregate and the filter.
        (
            "SELECT k64, s2 FROM (SELECT k64, sum(v) * 2 AS s2 FROM t GROUP BY k64) WHERE s2 > 120000",
            false,
        ),
    ];
    configured(&mut t, &[]);
    // The worker's configuration (instantiation reads the key there).
    let mut off = krishiv_common::config::Config::new();
    off.set("krishiv.exec.agg.outputFilter", "false").unwrap();
    let off = Arc::new(off);
    for (q, applies) in cases {
        let (want, off_in) = run(&t
            .plan_sql(q)
            .unwrap()
            .with_worker(|w| w.with_config(Arc::clone(&off))));
        let plan = t.plan_sql(q).unwrap();
        let (got, on_in) = run(&plan);
        assert_eq!(got, want, "{q}\n{plan}");
        assert!(
            !want.is_empty() && off_in > want.len() as u64,
            "{q}: {off_in} {}",
            want.len()
        );
        if applies {
            assert_eq!(on_in, want.len() as u64, "{q}\n{plan}");
        } else {
            assert_eq!(on_in, off_in, "{q}\n{plan}");
        }
    }
}

/// A `HAVING` that may raise (ANSI division) is left to the `Filter`
/// alone, so a raising query fails with the same error with the key on and
/// off (review NIT-1, NIT-2).
#[test]
fn a_raising_having_fails_alike_with_the_output_filter_on_and_off() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = 37;
    write(dir.path(), "t.parquet", &table(&mut rng, 2_000, 400, 5));
    let sources = [("t", vec!["t.parquet".to_owned()])];
    let mut t = parquet_files(dir.path(), &sources, true, &rt);
    configured(&mut t, &[]);
    let q = "SELECT k64 FROM t GROUP BY k64 HAVING sum(v) / (count(*) - count(*)) > 1";
    let mut off = krishiv_common::config::Config::new();
    off.set("krishiv.exec.agg.outputFilter", "false").unwrap();
    let off = Arc::new(off);
    let root = || {
        TreePool::new(&MemoryConfig {
            limit: Some(1 << 30),
            ..MemoryConfig::default()
        })
    };
    let plan = t.plan_sql(q).unwrap();
    let on = super::run_in(&plan, &root())
        .0
        .expect_err("division by zero");
    let plan_off = t
        .plan_sql(q)
        .unwrap()
        .with_worker(|w| w.with_config(Arc::clone(&off)));
    let off_err = super::run_in(&plan_off, &root())
        .0
        .expect_err("division by zero");
    assert_eq!(on.error_class(), off_err.error_class(), "{on}\n{off_err}");
    assert_eq!(on.to_string(), off_err.to_string());
}
