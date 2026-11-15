// SPDX-License-Identifier: Apache-2.0
//! A filter and a scan's row stage fused into the hash probe above them
//! (design P2-E3 U2): the planner fuses only where nothing can raise on a
//! row the filter drops (ADR 0008 §6), the fused plan returns what the
//! same plan with a `Filter` node and the scan's own row stage returns,
//! and the row stage's counters (rows tested, dropped, switched off) are
//! the ones the scan counts.

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_exec::physical::{PhysNode, ProjectItemPlan};
use krishiv_exec::planner::PhysicalPlan;
use krishiv_memory::{MemoryConfig, TreePool};

use super::{Runnable, env_of};

/// The same plan with every probe selection written out as before: the
/// fused filter as a `Filter` node over the probe input, under the fused
/// projection, and the row stage left in the scan.
fn unfused(plan: &Runnable) -> Runnable {
    fn walk(n: &mut PhysNode) {
        if let PhysNode::HashJoin(j) = n
            && let Some(select) = j.select.take()
            && let Some(f) = select.filter
        {
            let probe = std::mem::replace(&mut *j.probe, PhysNode::OneRow);
            let filtered = PhysNode::Filter {
                input: Box::new(probe),
                layout: f.layout.clone(),
                predicate: f.predicate,
                subqueries: Vec::new(),
            };
            *j.probe = match select.columns {
                Some(columns) => PhysNode::Project {
                    input: Box::new(filtered),
                    layout: f.layout,
                    items: columns
                        .into_iter()
                        .map(|c| ProjectItemPlan::Column {
                            index: c.index,
                            name: Some(c.name),
                        })
                        .collect(),
                    nullable: None,
                    subqueries: Vec::new(),
                },
                None => filtered,
            };
        }
        for c in n.children_mut() {
            walk(c);
        }
    }
    let mut root = plan.root().clone();
    walk(&mut root);
    let rebuilt = PhysicalPlan::new(
        root,
        Arc::clone(plan.schema()),
        plan.explain().clone(),
        plan.env().clone(),
    );
    Runnable {
        plan: rebuilt,
        worker: plan.worker.clone(),
    }
}

fn selects(plan: &Runnable) -> (usize, usize) {
    fn walk(n: &PhysNode, out: &mut (usize, usize)) {
        if let PhysNode::HashJoin(j) = n
            && let Some(s) = &j.select
        {
            out.0 += usize::from(s.filter.is_some());
            out.1 += usize::from(s.row_stage.is_some());
        }
        for c in n.children() {
            walk(c, out);
        }
    }
    let mut out = (0, 0);
    walk(plan.root(), &mut out);
    out
}

fn sorted_rows(plan: &Runnable) -> Vec<String> {
    counters(plan).0
}

/// The runtime filter counters of one run: rows tested, rows dropped.
fn counters(plan: &Runnable) -> (Vec<String>, u64, u64) {
    let root = TreePool::new(&MemoryConfig::default());
    let (out, query) = super::run_in(plan, &root);
    let out = out.unwrap_or_else(|e| panic!("{e}\n{plan}"));
    let (mut tested, mut dropped) = (0, 0);
    for entry in query.metrics().operators.values() {
        for (name, v) in &entry.extra {
            let v = v.value().unwrap_or(0);
            match name.as_ref() {
                "runtime_filter_rows_tested" => tested += v,
                "runtime_filter_rows_dropped" => dropped += v,
                _ => {}
            }
        }
    }
    let mut rows: Vec<String> = out
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(|r| {
                b.columns()
                    .iter()
                    .map(|c| super::tpch::render(c.as_ref(), r))
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect();
    rows.sort();
    (rows, tested, dropped)
}

/// The TPC-H SF0.01 Parquet tables (file scans: their filters stay
/// `Filter` nodes above the scan) on a runtime of their own.
fn tpch() -> (super::tpch::Tpch, tokio::runtime::Runtime) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    (super::tpch::parquet_tables(&dir, &rt), rt)
}

/// A filter under the probe's projections is fused into the probe and
/// printed on the join, with the probe scan's row stage; the rows are the
/// unfused plan's, and so are the row stage's counters.
#[test]
fn a_filter_below_a_probe_is_fused_into_it() {
    let (tpch, _rt) = tpch();
    for sql in [
        // TPC-H Q19's shape: a filter and a runtime-filtered scan.
        "SELECT l_orderkey, p_brand FROM lineitem JOIN part ON l_partkey = p_partkey \
         WHERE l_quantity < 10 AND p_size < 5",
        // A filter reading every probe input column: no narrow form.
        "SELECT l_orderkey, p_brand FROM lineitem JOIN part ON l_partkey = p_partkey \
         WHERE l_quantity < 10 AND l_orderkey > 3 AND l_partkey > 2 AND p_size < 5",
        // A left outer join: the filter's NULL rows are not joined.
        "SELECT l_orderkey, o_orderdate FROM lineitem LEFT JOIN orders \
         ON l_orderkey = o_orderkey AND o_totalprice > 1000 WHERE l_discount > 0.05",
        // An anti join (`NOT EXISTS`) with a residual.
        "SELECT o_orderkey FROM orders WHERE o_orderpriority <> '1-URGENT' AND NOT EXISTS \
         (SELECT 1 FROM lineitem WHERE l_orderkey = o_orderkey AND l_quantity > 49)",
    ] {
        let plan = tpch.plan_sql(sql).unwrap();
        let text = plan.to_string();
        assert!(selects(&plan).0 >= 1, "{sql}\n{text}");
        assert!(text.contains("probe filter: "), "{text}");
        let (want, want_tested, want_dropped) = counters(&unfused(&plan));
        assert!(!want.is_empty(), "{sql}");
        let (got, tested, dropped) = counters(&plan);
        assert_eq!(got, want, "{sql}\n{text}");
        assert_eq!(
            (tested, dropped),
            (want_tested, want_dropped),
            "{sql}\n{text}"
        );
    }
}

/// A key computed by an expression projection above the filter stops the
/// fusion at that projection, so a key that raises under ANSI on a row the
/// filter drops never sees that row: no error.
#[test]
fn a_key_computed_above_the_filter_keeps_it_unfused() {
    let (tpch, _rt) = tpch();
    // `CAST(l_comment AS INT)` raises on every comment; the filter drops
    // every row before the key projection sees one.
    let sql = "SELECT l_orderkey FROM lineitem JOIN part \
               ON CAST(l_comment AS INT) = p_partkey WHERE l_quantity < 0";
    let plan = tpch.plan_sql(sql).unwrap();
    let text = plan.to_string();
    assert_eq!(selects(&plan).0, 0, "{text}");
    assert!(!text.contains("probe filter"), "{text}");
    assert!(text.contains("Filter "), "{text}");
    assert!(sorted_rows(&plan).is_empty());
}

/// A filter that may raise (ANSI decimal arithmetic) is not fused either.
#[test]
fn a_raising_filter_is_not_fused() {
    let (tpch, _rt) = tpch();
    let sql = "SELECT l_orderkey FROM lineitem JOIN part ON l_partkey = p_partkey \
               WHERE l_quantity * 1000 > 10";
    let plan = tpch.plan_sql(sql).unwrap();
    assert_eq!(selects(&plan).0, 0, "{plan}");
    assert!(plan.to_string().contains("Filter "), "{plan}");
}

/// The guard reads the session's ANSI mode: the filter that may raise under
/// ANSI is fused with ANSI off, and joins the same rows.
#[test]
fn a_filter_that_only_raises_under_ansi_is_fused_with_ansi_off() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let tpch = super::tpch::parquet_dir(&dir, &super::tpch::TABLES, false, &rt);
    let sql = "SELECT l_orderkey FROM lineitem JOIN part ON l_partkey = p_partkey \
               WHERE l_quantity * 1000 > 10";
    let plan = tpch.plan_sql(sql).unwrap();
    assert_eq!(selects(&plan).0, 1, "{plan}");
    assert_eq!(sorted_rows(&plan), sorted_rows(&unfused(&plan)));
}

/// A residual that may raise keeps a fusable filter unfused: the residual
/// must not see a pair the filter would have dropped.
#[test]
fn a_raising_residual_keeps_the_filter_unfused() {
    let (tpch, _rt) = tpch();
    let sql = "SELECT l_orderkey FROM lineitem JOIN part \
               ON l_partkey = p_partkey AND l_quantity * 1000 > p_size \
               WHERE l_discount > 0.05";
    let plan = tpch.plan_sql(sql).unwrap();
    assert_eq!(selects(&plan).0, 0, "{plan}");
    assert!(plan.to_string().contains("Filter "), "{plan}");
    assert!(!sorted_rows(&plan).is_empty());
}

/// `krishiv.exec.join.fuseProbeSelect=false` plans neither fusion: the row
/// stage stays in the scan and the rows do not change.
#[test]
fn the_switch_turns_the_fusion_off() {
    let env = env_of(300);
    let sql = "SELECT x.b, y.s FROM t x JOIN (SELECT a, s FROM t WHERE b < 300) y ON x.a = y.a";
    let on = env.plan_with(sql, Config::new()).unwrap();
    assert_eq!(selects(&on).1, 1, "{on}");
    let mut config = Config::new();
    config
        .set("krishiv.exec.join.fuseProbeSelect", "false")
        .unwrap();
    let off = env.plan_with(sql, config).unwrap();
    assert_eq!(selects(&off), (0, 0), "{off}");
    assert_eq!(sorted_rows(&off), sorted_rows(&on));
}

/// With the switch off a filter below a probe stays a `Filter` node.
#[test]
fn the_switch_keeps_a_filter_a_node() {
    let (mut tpch, _rt) = tpch();
    let sql = "SELECT l_orderkey FROM lineitem JOIN part ON l_partkey = p_partkey \
               WHERE l_quantity < 10";
    assert_eq!(selects(&tpch.plan_sql(sql).unwrap()).0, 1);
    let mut config = Config::new();
    config
        .set("krishiv.exec.join.fuseProbeSelect", "false")
        .unwrap();
    tpch.config = Arc::new(config);
    let plan = tpch.plan_sql(sql).unwrap();
    assert_eq!(selects(&plan).0, 0, "{plan}");
    assert!(plan.to_string().contains("Filter "), "{plan}");
}

/// The row stage of a runtime-filtered scan below the probe moves to the
/// probe: the rows, and the rows the stage tested and dropped, are the
/// scan's own, with the adaptive switch-off measuring windows of one row
/// (filters switch off and on as the scan restarts them).
#[test]
fn a_scans_row_stage_runs_in_the_probe_with_the_scans_counters() {
    let env = env_of(300);
    for (min_rows, max_pass) in [("65536", "90"), ("1", "50"), ("4", "0")] {
        let mut config = Config::new();
        config
            .set("krishiv.runtimeFilter.adaptive.minRows", min_rows)
            .unwrap();
        config
            .set("krishiv.runtimeFilter.adaptive.maxPassPercent", max_pass)
            .unwrap();
        let sql = "SELECT x.b, y.s FROM t x JOIN (SELECT a, s FROM t WHERE b < 300) y ON x.a = y.a";
        let plan = env.plan_with(sql, config).unwrap();
        let text = plan.to_string();
        assert_eq!(selects(&plan).1, 1, "{text}");
        let (want, want_tested, want_dropped) = counters(&unfused(&plan));
        let (got, tested, dropped) = counters(&plan);
        assert_eq!(got, want, "{text}");
        assert_eq!((tested, dropped), (want_tested, want_dropped), "{text}");
        if min_rows == "65536" {
            assert!(dropped > 0, "the filter dropped nothing: {text}");
        }
    }
}
