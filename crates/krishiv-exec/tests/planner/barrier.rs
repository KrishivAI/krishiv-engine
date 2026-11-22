// SPDX-License-Identifier: Apache-2.0
//! A security barrier at lowering (design `P3-E14-security-barrier.md` §6, §7.4): no physical node,
//! the rows of the plain plan, no runtime filter into a guarded scan, no probe fusion into one, the
//! guarded scans on the finished plan (kept by the write plans), and no compared column of a
//! guarded table in the profile's scan entries.

use std::sync::Arc;

use krishiv_common::{Principal, PrincipalKey, PrincipalKind};
use krishiv_exec::physical::{PhysNode, PhysicalPlan, ScanId};
use krishiv_exec::planner::ProviderStatistics;
use krishiv_exec::profile::profile_scans;
use krishiv_plan::analyzer::{Securable, TableIdent};
use krishiv_plan::{LogicalPlan, Plan, PlanNode, PolicyRef, ReadSource};

use super::Runnable;
use super::tpch::{Tpch, rows};

fn tpch() -> (Tpch, tokio::runtime::Runtime) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    (super::tpch::parquet_tables(&dir, &rt), rt)
}

fn policy(table: &str) -> PolicyRef {
    PolicyRef::new(
        Securable::Table(TableIdent {
            catalog: "spark_catalog".to_owned(),
            namespace: vec!["default".to_owned()],
            name: table.to_owned(),
        }),
        PrincipalKey::of(&Principal::new("ana", PrincipalKind::User)),
        1,
    )
}

/// `sql` analyzed, the first `guard` scans of `table` (all with `usize::MAX`) behind a barrier,
/// optimized by the tables' statistics and planned.
fn planned(tpch: &Tpch, sql: &str, table: &str, guard: usize) -> Runnable {
    let is_table = |p: &LogicalPlan| {
        matches!(&p.node, PlanNode::Scan(s) if matches!(&s.source, ReadSource::Table { name, .. }
            if name.last().is_some_and(|n| n == table)))
    };
    planned_at(tpch, sql, table, guard, &is_table)
}

/// As [`planned`], the barrier over the first `guard` nodes `at` picks.
fn planned_at(
    tpch: &Tpch,
    sql: &str,
    table: &str,
    guard: usize,
    at: &dyn Fn(&LogicalPlan) -> bool,
) -> Runnable {
    let plan = tpch.analyzed(sql).unwrap();
    let mut left = guard;
    let root = Arc::clone(plan.root())
        .transform_up(&mut |p| {
            Ok::<_, std::convert::Infallible>(if left > 0 && at(&p) {
                left -= 1;
                Arc::new(LogicalPlan::clone(&p).security_barrier(policy(table)))
            } else {
                p
            })
        })
        .unwrap_or_else(|e| match e {});
    let plan = Plan::try_new(root).unwrap();
    let stats = ProviderStatistics(Arc::clone(&tpch.tables) as _);
    let optimized = krishiv_optimizer::optimize_with(&plan, &stats).unwrap();
    tpch.runnable(&optimized).unwrap()
}

fn scans(n: &PhysNode, out: &mut Vec<(ScanId, String, usize)>) {
    if let PhysNode::Scan(s) = n
        && let ReadSource::Table { name, .. } = &s.scan.source
    {
        out.push((
            s.id,
            name.last().cloned().unwrap_or_default(),
            s.runtime_filters.len(),
        ));
    }
    for c in n.children() {
        scans(c, out);
    }
}

fn all_scans(plan: &PhysicalPlan) -> Vec<(ScanId, String, usize)> {
    let mut out = Vec::new();
    for root in plan.roots() {
        scans(root, &mut out);
    }
    out
}

fn all_scan_logical(n: &PhysNode) -> Option<krishiv_plan::Scan> {
    match n {
        PhysNode::Scan(s) => Some(s.scan.clone()),
        _ => n.children().into_iter().find_map(all_scan_logical),
    }
}

fn fused(n: &PhysNode) -> usize {
    let own = match n {
        PhysNode::HashJoin(j) => usize::from(j.select.as_ref().is_some_and(|s| s.filter.is_some())),
        _ => 0,
    };
    own + n.children().into_iter().map(fused).sum::<usize>()
}

fn result(plan: &Runnable) -> Vec<String> {
    let mut out = rows(&super::run(plan).unwrap_or_else(|e| panic!("{e}\n{plan}")));
    out.sort();
    out
}

const JOIN: &str = "SELECT l_orderkey, p_brand FROM lineitem JOIN part ON l_partkey = p_partkey \
                    WHERE l_quantity < 10 AND p_size < 5";

#[test]
fn no_runtime_filter_or_probe_fusion_reaches_a_guarded_scan() {
    let (tpch, _rt) = tpch();
    let plain = planned(&tpch, JOIN, "lineitem", 0);
    let guarded = planned(&tpch, JOIN, "lineitem", usize::MAX);
    // Without the barrier, the part build filters the lineitem scan and the filter is fused.
    let lineitem = |p: &PhysicalPlan| {
        all_scans(p)
            .into_iter()
            .filter(|(_, t, _)| t == "lineitem")
            .collect::<Vec<_>>()
    };
    assert!(lineitem(&plain).iter().any(|(_, _, f)| *f > 0), "{plain}");
    assert!(fused(plain.root()) >= 1, "{plain}");
    assert!(plain.guarded().is_empty());
    // With it, neither, and the guarded set is exactly the lineitem scans.
    let ids: Vec<ScanId> = lineitem(&guarded).iter().map(|(id, _, _)| *id).collect();
    assert!(!ids.is_empty());
    assert!(
        lineitem(&guarded).iter().all(|(_, _, f)| *f == 0),
        "{guarded}"
    );
    assert_eq!(fused(guarded.root()), 0, "{guarded}");
    let mut sorted = ids;
    sorted.sort_unstable();
    assert_eq!(guarded.guarded(), sorted.as_slice());
    // The same rows.
    assert_eq!(result(&guarded), result(&plain));
}

#[test]
fn a_hash_join_whose_build_is_guarded_still_filters_the_plain_probe() {
    // The barrier's rows may filter another scan: its keys are its output rows.
    let (tpch, _rt) = tpch();
    let plan = planned(&tpch, JOIN, "part", usize::MAX);
    assert!(
        all_scans(&plan)
            .iter()
            .any(|(_, t, f)| t == "lineitem" && *f > 0),
        "{plan}"
    );
    assert_eq!(result(&plan), result(&planned(&tpch, JOIN, "part", 0)));
}

#[test]
fn a_guarded_table_lists_no_compared_column() {
    let (tpch, _rt) = tpch();
    let entry = |plan: &PhysicalPlan, table: &str| {
        profile_scans(plan, &|_| None)
            .into_iter()
            .find(|s| s.table.ends_with(table))
            .unwrap_or_else(|| panic!("{table} in {plan}"))
    };
    // The comparison moves into the barrier, beside the scan.
    let sql = "SELECT count(*) FROM lineitem WHERE l_returnflag = 'R'";
    let plain = planned(&tpch, sql, "lineitem", 0);
    assert!(!entry(&plain, "lineitem").equality.is_empty(), "{plain}");
    let guarded = planned(&tpch, sql, "lineitem", usize::MAX);
    assert!(entry(&guarded, "lineitem").equality.is_empty(), "{guarded}");
    // The comparison stays above the barrier on a column it passes through: no filter crosses the
    // sample inside, so the conjunct is a residue and the filter above names the scan's column.
    let sql = "SELECT count(*) FROM lineitem TABLESAMPLE (50 PERCENT) REPEATABLE (7) \
               WHERE l_returnflag = 'R'";
    let sample = |p: &LogicalPlan| matches!(p.node, PlanNode::Sample { .. });
    let plain = planned_at(&tpch, sql, "lineitem", 0, &sample);
    assert!(!entry(&plain, "lineitem").equality.is_empty(), "{plain}");
    let guarded = planned_at(&tpch, sql, "lineitem", usize::MAX, &sample);
    assert!(!guarded.guarded().is_empty(), "{guarded}");
    assert!(entry(&guarded, "lineitem").equality.is_empty(), "{guarded}");
    // An unguarded read of the same table beside a guarded one: still none.
    let both = "SELECT l_orderkey FROM lineitem WHERE l_returnflag = 'R' \
                UNION ALL SELECT l_orderkey FROM lineitem WHERE l_linestatus = 'O'";
    // The branches share one scan relation, so the barrier goes over the first branch's filter.
    let first = |p: &LogicalPlan| matches!(&p.node, PlanNode::Filter { predicate, .. } if predicate.to_string().contains("'R'"));
    let plan = planned_at(&tpch, both, "lineitem", 1, &first);
    assert_eq!(plan.guarded().len(), 1, "{plan}");
    assert!(entry(&plan, "lineitem").equality.is_empty(), "{plan}");
}

#[test]
fn the_write_plans_keep_the_guarded_scans() {
    let (tpch, _rt) = tpch();
    let plan = planned(
        &tpch,
        "SELECT l_orderkey FROM lineitem",
        "lineitem",
        usize::MAX,
    );
    let set = plan.guarded().to_vec();
    assert!(!set.is_empty());
    let write = PhysicalPlan::clone(&plan)
        .into_write(krishiv_exec::physical::WritePlan {
            format: "parquet".to_owned(),
            options: std::collections::BTreeMap::new(),
            store: "file://out".to_owned(),
            dir: "result".to_owned(),
            job_uuid: "job-1".to_owned(),
            max_records_per_file: 0,
            target_file_bytes: 0,
            partition_by: Vec::new(),
            column_types: std::collections::BTreeMap::new(),
            bucket: None,
        })
        .unwrap();
    assert_eq!(write.guarded(), set.as_slice());
    let scan = all_scan_logical(plan.root()).expect("a scan");
    let request =
        krishiv_exec::physical::TableWriteRequest::Write(krishiv_exec::planner::WriteRequest::new(
            krishiv_exec::planner::WriteMode::Append,
            "`t`",
            Arc::clone(plan.schema()),
        ));
    let table_write = PhysicalPlan::clone(&plan).into_table_write(scan, request);
    assert_eq!(table_write.guarded(), set.as_slice());
    // The codec does not carry the set: a round trip compares with it cleared.
    let cleared = PhysicalPlan::clone(&plan).without_guarded();
    assert!(cleared.guarded().is_empty());
    assert!(cleared != *plan);
}

/// The test UDFs of design §7.4: `policy(k)` keeps a key that is not a multiple of 7; `leak(k)`
/// returns `k` and fails on a key the policy rejects (as a function that reports its argument
/// would leak it); neither is leakproof.
#[derive(Debug)]
struct KeyUdf(&'static str);

impl krishiv_udf_sdk::ScalarUdf for KeyUdf {
    fn name(&self) -> &str {
        self.0
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &krishiv_udf_sdk::Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<krishiv_udf_sdk::Signature> = OnceLock::new();
        SIG.get_or_init(|| {
            krishiv_udf_sdk::Signature::Exact(vec![krishiv_udf_sdk::UdfInputType::Any])
        })
    }

    fn return_field(
        &self,
        args: &[krishiv_udf_sdk::ArgField],
    ) -> Result<krishiv_udf_sdk::ReturnField, krishiv_udf_sdk::UdfError> {
        Ok(krishiv_udf_sdk::ReturnField {
            data_type: if self.0 == "policy" {
                krishiv_types::DataType::Boolean
            } else {
                args[0]
                    .data_type
                    .clone()
                    .unwrap_or(krishiv_types::DataType::Long)
            },
            nullable: true,
        })
    }

    fn invoke(
        &self,
        args: &[krishiv_udf_sdk::UdfArg<'_>],
        rows: usize,
        _: &krishiv_udf_sdk::InvokeContext,
    ) -> Result<arrow_array::ArrayRef, krishiv_udf_sdk::UdfError> {
        use arrow_array::Array;
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Int32Type, Int64Type};
        let a = args[0].array();
        let at = |i: usize| if a.len() == 1 { 0 } else { i };
        let key = |i: usize| -> Option<i64> {
            let i = at(i);
            if a.is_null(i) {
                return None;
            }
            match a.data_type() {
                arrow_schema::DataType::Int32 => {
                    Some(i64::from(a.as_primitive::<Int32Type>().value(i)))
                }
                arrow_schema::DataType::Int64 => Some(a.as_primitive::<Int64Type>().value(i)),
                _ => None,
            }
        };
        let kept = |i: usize| key(i).is_none_or(|k| k % 7 != 0);
        if self.0 == "policy" {
            return Ok(Arc::new(arrow_array::BooleanArray::from(
                (0..rows).map(kept).collect::<Vec<_>>(),
            )));
        }
        if let Some(i) = (0..rows).find(|&i| !kept(i)) {
            return Err(krishiv_udf_sdk::UdfError::new(format!(
                "leak({:?}) ran on a row the policy rejects",
                key(i)
            )));
        }
        Ok(a)
    }
}

fn udf(
    name: &str,
    args: Vec<Arc<krishiv_plan::Expr>>,
    t: krishiv_types::DataType,
) -> Arc<krishiv_plan::Expr> {
    use krishiv_plan::{Expr, ExprNode, FunctionId, FunctionKind, ResolvedExpr, ResolvedFunction};
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Udf {
                name: name.to_owned(),
                version: 1,
            },
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: t,
            nullable: true,
            deterministic: true,
        },
    ))))
}

/// `plan` with every table scan behind a barrier that keeps `policy(key)` rows (the key: the
/// scan's first column) and, with `plant`, `isnotnull(leak(key))` above each barrier.
fn guard_every_table(plan: &Plan, plant: bool) -> Plan {
    use krishiv_plan::{
        Builtin, Expr, ExprNode, FunctionId, FunctionKind, ResolvedExpr, ResolvedFunction,
    };
    let col = |c: &krishiv_plan::BoundColumn| {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
            c.clone(),
        ))))
    };
    let root = Arc::clone(plan.root())
        .transform_up(&mut |p| {
            let PlanNode::Scan(scan) = &p.node else {
                return Ok::<_, std::convert::Infallible>(p);
            };
            let (ReadSource::Table { name, .. }, Some(key)) = (&scan.source, scan.output.first())
            else {
                return Ok(p);
            };
            let table = name.last().cloned().unwrap_or_default();
            let inner = LogicalPlan::clone(&p).filter(udf(
                "policy",
                vec![col(key)],
                krishiv_types::DataType::Boolean,
            ));
            let barrier = inner.security_barrier(policy(&table));
            if !plant {
                return Ok(Arc::new(barrier));
            }
            let leak = udf("leak", vec![col(key)], key.data_type.clone());
            Ok(Arc::new(barrier.filter(Arc::new(Expr::new(
                ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                    id: FunctionId::Builtin(Builtin::IsNotNull),
                    kind: FunctionKind::Scalar,
                    args: vec![leak],
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                    data_type: krishiv_types::DataType::Boolean,
                    nullable: false,
                    deterministic: true,
                })),
            )))))
        })
        .unwrap_or_else(|e| match e {});
    Plan::try_new(root).unwrap()
}

/// §7.4: the planted-leak plans of every TPC-H query through the executor under a policy that
/// rejects a seventh of each table's keys. The optimized plan never runs `leak` on a rejected
/// row (it fails the query if it does) and returns the rows of the same barriers without the
/// leak.
#[test]
fn every_tpch_query_runs_the_barrier_rows_under_a_rejecting_policy() {
    let (tpch, _rt) = tpch();
    let udfs = Arc::new(krishiv_udf_sdk::UdfLibrary::new());
    for name in ["policy", "leak"] {
        udfs.register(Arc::new(KeyUdf(name))).unwrap();
    }
    let config = krishiv_common::config::Config::new();
    let stats = ProviderStatistics(Arc::clone(&tpch.tables) as _);
    let run = |plan: &Plan| {
        let optimized =
            krishiv_optimizer::optimize_with_udfs(plan, &stats, &udfs, &config).unwrap();
        let runnable = tpch
            .runnable_with_udfs(&optimized, Arc::clone(&udfs))
            .unwrap()
            .with_worker(|w| w.with_udfs(Arc::clone(&udfs)));
        let mut out = rows(&super::run(&runnable).unwrap_or_else(|e| panic!("{e}\n{optimized}")));
        out.sort();
        out
    };
    let mut nonempty = 0;
    for q in 1..=22 {
        let plan = tpch.analyzed(&super::tpch::sql(q)).unwrap();
        let want = run(&guard_every_table(&plan, false));
        let got = run(&guard_every_table(&plan, true));
        assert_eq!(got, want, "q{q}");
        nonempty += usize::from(!want.is_empty());
    }
    assert!(nonempty >= 15, "{nonempty} of 22 queries return rows");
    // The control: a leak below the policy's filter (what a broken pushdown arm would build) runs
    // on the rejected rows, and the query fails.
    let plan = tpch.analyzed("SELECT p_partkey FROM part").unwrap();
    let root = Arc::clone(plan.root())
        .transform_up(&mut |p| {
            Ok::<_, std::convert::Infallible>(match &p.node {
                PlanNode::Scan(scan) => {
                    let key = scan.output[0].clone();
                    let col = Arc::new(krishiv_plan::Expr::new(krishiv_plan::ExprNode::Resolved(
                        krishiv_plan::ResolvedExpr::Column(key.clone()),
                    )));
                    let leaked = udf("leak", vec![col], key.data_type);
                    let below = LogicalPlan::clone(&p).filter(udf(
                        "policy",
                        vec![leaked],
                        krishiv_types::DataType::Boolean,
                    ));
                    Arc::new(below.security_barrier(policy("part")))
                }
                _ => p,
            })
        })
        .unwrap_or_else(|e| match e {});
    let broken = Plan::try_new(root).unwrap();
    let runnable = tpch
        .runnable_with_udfs(&broken, Arc::clone(&udfs))
        .unwrap()
        .with_worker(|w| w.with_udfs(Arc::clone(&udfs)));
    let e = super::run(&runnable).unwrap_err();
    assert!(
        e.to_string().contains("ran on a row the policy rejects"),
        "{e}"
    );
}
