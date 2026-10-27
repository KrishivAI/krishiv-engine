// SPDX-License-Identifier: Apache-2.0
//! The physical plan as data (P2-T172): plans compare and clone as values,
//! instantiate any number of times, and instantiation checks that the
//! worker's tables still match the plan.

use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::physical::{
    ExprEnv, HashJoinPlan, JoinDistribution, JoinFilterPlan, PhysNode, PhysicalPlan, ScanId,
    ScanNode, ScanRuntimeFilter, WorkerContext, WriteSpec, assert_plan_data, instantiate,
};
use krishiv_exec::planner::{
    DefaultPlanner, ExplainNode, MemoryTable, PhysicalPlanner as _, PlanningContext, TableProvider,
    TableProviderResolver,
};
use krishiv_exec::values::{LocalValues, ValueId, ValueService};
use krishiv_exec::{FilterPushdown, JoinType, PushedFilter, ScanPushdown, ScanPushdowns, Source};
use krishiv_io::{ColumnPath, FilterId, FilterSnapshot, KeyFilterOptions};

use super::{Runnable, env, run, table};

/// Every node of `n`, depth first.
fn nodes(n: &PhysNode) -> Vec<&PhysNode> {
    let mut out = vec![n];
    let mut i = 0;
    while let Some(n) = out.get(i).copied() {
        out.extend(n.children());
        i += 1;
    }
    out
}

fn sorted(plan: &Runnable) -> Vec<String> {
    let mut rows: Vec<String> = run(plan)
        .unwrap()
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(move |r| {
                b.columns()
                    .iter()
                    .map(|c| super::cell(c.as_ref(), r))
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect();
    rows.sort();
    rows
}

#[test]
fn a_plan_is_a_value_that_instantiates_any_number_of_times() {
    assert_plan_data::<PhysicalPlan>();
    let env = env();
    // A join publishing runtime filters, an aggregate, a scalar subquery,
    // a union and a sort.
    let sql = "SELECT x.a, count(*) AS c FROM t AS x JOIN t AS y ON x.a = y.a \
               WHERE x.b > (SELECT min(b) FROM t) GROUP BY x.a \
               UNION ALL SELECT a, b FROM t ORDER BY 1, 2";
    let plan = env.plan(sql).unwrap();
    let copy = plan.plan.clone();
    assert_eq!(copy, plan.plan);
    assert!(format!("{copy:?}").starts_with("PhysicalPlan"));
    assert!(plan.env().ansi);
    let all = nodes(plan.root());
    let filters: Vec<FilterId> = all
        .iter()
        .filter_map(|n| match n {
            PhysNode::Scan(s) => Some(s.runtime_filters.iter().map(|f| f.id)),
            _ => None,
        })
        .flatten()
        .collect();
    assert!(!filters.is_empty(), "{plan}");
    assert!(
        all.iter()
            .any(|n| matches!(n, PhysNode::Filter { subqueries, .. } if subqueries.len() == 1)),
        "{plan}"
    );
    for kind in ["HashJoin", "Aggregate", "Union", "Sort"] {
        assert!(
            all.iter().any(|n| format!("{n:?}").starts_with(kind)),
            "{kind}: {plan}"
        );
    }
    // Each run instantiates its own operators and cells: same rows.
    let first = sorted(&plan);
    assert_eq!(first.len(), 17, "{first:?}");
    assert_eq!(sorted(&plan), first);
    let again = Runnable::new(copy, Arc::clone(&env.tables) as _);
    assert_eq!(sorted(&again), first);
    let worker = WorkerContext::new(Arc::clone(&env.tables) as _);
    instantiate(&plan, &worker).unwrap();
    instantiate(&plan, &worker).unwrap();
}

/// The table `t` of [`env`], whose scans honour pushed filters only
/// inexactly (a filter stays above them) and that does not say what it
/// returns without building a source.
struct Inexact(Arc<dyn ScanPushdown>);

impl ScanPushdown for Inexact {
    fn table_schema(&self) -> arrow_schema::SchemaRef {
        self.0.table_schema()
    }

    fn filter_pushdown(&self, _: &PushedFilter) -> FilterPushdown {
        FilterPushdown::Inexact
    }

    fn build(&self, pushdowns: ScanPushdowns) -> krishiv_exec::Result<Arc<dyn Source>> {
        self.0.build(pushdowns)
    }
}

#[derive(Debug)]
struct InexactTable(MemoryTable);

impl TableProvider for InexactTable {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::new(Inexact(self.0.scan()?)))
    }
}

#[derive(Debug)]
struct One(Arc<dyn TableProvider>);

impl TableProviderResolver for One {
    fn provider(&self, _: &krishiv_plan::Scan) -> krishiv_exec::Result<Arc<dyn TableProvider>> {
        Ok(Arc::clone(&self.0))
    }
}

#[test]
fn instantiation_checks_the_tables_still_match_the_plan() {
    let env = env();
    let (schema, batches) = table(10);
    let inexact: Arc<dyn TableProviderResolver> = Arc::new(One(Arc::new(InexactTable(
        MemoryTable::new(schema, batches),
    ))));
    let logical =
        krishiv_optimizer::optimize(&env.analyze("SELECT a FROM t WHERE b > 40")).unwrap();
    let cx = PlanningContext::new(Arc::clone(&inexact), Arc::new(Config::new()));
    let plan = DefaultPlanner.plan(&logical, &cx).unwrap();
    // Planned over the inexact scan: the filter stays above it, and the
    // scan's schema comes from a built source (the default).
    assert!(plan.to_string().starts_with("Project"), "{plan}");
    assert!(plan.to_string().contains("Filter"), "{plan}");
    let rows = sorted(&Runnable::new(plan.clone(), Arc::clone(&inexact)));
    assert_eq!(rows, ["5", "6", "8", "9", "NULL"]);
    // The in-memory table applies it exactly: the plan no longer matches.
    let e = instantiate(&plan, &WorkerContext::new(Arc::clone(&env.tables) as _)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.to_string().contains("pushed filter"), "{e}");
    // And the other way round.
    let exact = env.plan("SELECT a FROM t WHERE b > 40").unwrap();
    let e = instantiate(&exact, &WorkerContext::new(inexact)).unwrap_err();
    assert!(e.to_string().contains("pushed filter"), "{e}");
}

fn env_data() -> ExprEnv {
    ExprEnv {
        ansi: true,
        legacy_statistical_aggregate: false,
        zone: Some(0),
        query_start_micros: 0,
        user: Arc::from("u"),
    }
}

fn scan_of(plan: &PhysNode) -> ScanNode {
    nodes(plan)
        .into_iter()
        .find_map(|n| match n {
            PhysNode::Scan(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap()
}

fn instantiate_err(root: PhysNode, schema: Schema, worker: &WorkerContext) -> KrishivError {
    let plan = PhysicalPlan::new(
        root,
        Arc::new(schema),
        ExplainNode::leaf("Test", ""),
        env_data(),
    );
    instantiate(&plan, worker).unwrap_err()
}

#[test]
fn inconsistent_plans_are_internal_errors() {
    let env = env();
    let worker = WorkerContext::new(Arc::clone(&env.tables) as _);
    let scan = scan_of(env.plan("SELECT a, b FROM t").unwrap().root());
    let scan_schema = Schema::new(vec![
        Field::new("a", ArrowType::Int32, true),
        Field::new("b", ArrowType::Int64, false),
    ]);
    let check = |e: KrishivError, text: &str| {
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
        assert!(e.to_string().contains(text), "{e}");
    };
    // Another result schema than the plan states.
    check(
        instantiate_err(PhysNode::OneRow, scan_schema.clone(), &worker),
        "instantiates to",
    );
    // A join collecting a runtime filter no scan declares.
    let join = |filters: Vec<ScanRuntimeFilter>, id: u64| {
        let mut probe = scan.clone();
        probe.runtime_filters = filters;
        PhysNode::HashJoin(HashJoinPlan {
            probe: Box::new(PhysNode::Scan(probe)),
            build: Box::new(PhysNode::Scan(ScanNode {
                id: ScanId(1),
                ..scan.clone()
            })),
            join_type: JoinType::LeftSemi,
            probe_keys: vec![0],
            build_keys: vec![0],
            null_safe: vec![false],
            residual: None,
            single_match: false,
            distinct_build: false,
            runtime_filters: vec![JoinFilterPlan {
                id: FilterId(id),
                key: 0,
                options: KeyFilterOptions::default(),
            }],
            distribution: JoinDistribution::Broadcast,
        })
    };
    let declared = ScanRuntimeFilter {
        id: FilterId(7),
        column: ColumnPath::new("a"),
        key_type: ArrowType::Int32,
    };
    check(
        instantiate_err(join(Vec::new(), 7), scan_schema.clone(), &worker),
        "has no scan",
    );
    // A filter declared twice.
    check(
        instantiate_err(
            join(vec![declared.clone(), declared.clone()], 7),
            scan_schema.clone(),
            &worker,
        ),
        "declared twice",
    );
    // Declared once, it instantiates.
    let ok = PhysicalPlan::new(
        join(vec![declared], 7),
        Arc::new(scan_schema.clone()),
        ExplainNode::leaf("Test", ""),
        env_data(),
    );
    instantiate(&ok, &worker).unwrap();
    // Union inputs of other types, and none at all.
    let union = PhysNode::Union {
        inputs: vec![PhysNode::Scan(scan.clone()), PhysNode::OneRow],
    };
    check(
        instantiate_err(union, scan_schema.clone(), &worker),
        "different physical types",
    );
    check(
        instantiate_err(PhysNode::Union { inputs: Vec::new() }, scan_schema, &worker),
        "without inputs",
    );
    // A write on a worker without a write sink.
    let write = PhysNode::WriteFiles {
        input: Box::new(PhysNode::OneRow),
        spec: WriteSpec::default(),
    };
    check(
        instantiate_err(write, Schema::empty(), &worker),
        "without a write sink",
    );
}

/// Counts the providers it hands out.
#[derive(Debug)]
struct Counting(
    Arc<dyn TableProviderResolver>,
    std::sync::atomic::AtomicUsize,
);

impl TableProviderResolver for Counting {
    fn provider(&self, scan: &krishiv_plan::Scan) -> krishiv_exec::Result<Arc<dyn TableProvider>> {
        self.1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.0.provider(scan)
    }
}

/// `submit` does no instantiation work on the caller: the plan is
/// instantiated (on the blocking pool) when the returned future runs.
#[test]
fn submit_instantiates_inside_its_future() {
    use futures::StreamExt as _;
    use krishiv_exec::Scheduler as _;
    use std::sync::atomic::Ordering;

    let env = env();
    let counting = Arc::new(Counting(
        Arc::clone(&env.tables) as _,
        std::sync::atomic::AtomicUsize::new(0),
    ));
    let plan = env.plan("SELECT a FROM t").unwrap();
    let root = krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
    let query = krishiv_exec::QueryContext::new(
        krishiv_exec::QueryId(1),
        krishiv_exec::WorkloadClass::Interactive,
        krishiv_exec::ExecConfig::default(),
        krishiv_memory::MemoryPool::child(root.as_ref(), "q", None),
    );
    let runner = krishiv_exec::QueryRunner::new(
        krishiv_exec::TokioExecutor::new(&krishiv_exec::ExecutorConfig::default()).unwrap(),
    );
    let started = runner.submit(
        plan.plan,
        WorkerContext::new(Arc::clone(&counting) as _),
        Arc::clone(&query),
    );
    assert_eq!(counting.1.load(Ordering::SeqCst), 0);
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let rows = rt.block_on(async {
        let mut stream = started.await.unwrap();
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.unwrap().num_rows();
        }
        query.finished().await;
        rows
    });
    assert_eq!(rows, 10);
    assert_eq!(counting.1.load(Ordering::SeqCst), 1);
}

/// The runtime filters the scans of `plan` read.
fn filter_ids(plan: &PhysicalPlan) -> Vec<FilterId> {
    nodes(plan.root())
        .into_iter()
        .filter_map(|n| match n {
            PhysNode::Scan(s) => Some(s.runtime_filters.iter().map(|f| f.id)),
            _ => None,
        })
        .flatten()
        .collect()
}

/// A join whose build, behind a broadcast exchange, has fewer keys than its
/// probe and publishes a runtime filter to the probe's scan.
const FILTERED: &str = "SELECT x.a, x.b FROM t AS x JOIN (SELECT a FROM t WHERE b < 30) AS y \
                        ON x.a = y.a";

#[test]
fn instantiations_never_share_values() {
    let env = env();
    let plan = env.plan(FILTERED).unwrap();
    let ids = filter_ids(&plan);
    assert!(!ids.is_empty(), "{plan}");
    // Each instantiation of one context gets values of its own.
    let worker = WorkerContext::new(Arc::clone(&env.tables) as _);
    instantiate(&plan, &worker).unwrap();
    instantiate(&plan, &worker).unwrap();
    // A service set on the context serves one instantiation: it holds that
    // instantiation's cells, and a second one cannot declare them again.
    let (a, b) = (Arc::new(LocalValues::new()), Arc::new(LocalValues::new()));
    let on = |v: &Arc<LocalValues>| worker.clone().with_values(Arc::clone(v) as _);
    instantiate(&plan, &on(&a)).unwrap();
    instantiate(&plan, &on(&b)).unwrap();
    for id in &ids {
        assert!(!Arc::ptr_eq(
            &a.filter(*id).unwrap(),
            &b.filter(*id).unwrap()
        ));
    }
    let e = instantiate(&plan, &on(&a)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert!(e.to_string().contains("declared twice"), "{e}");
}

#[test]
fn a_filter_of_an_incomplete_value_applies_nothing() {
    let env = env();
    let plan = env.plan(FILTERED).unwrap();
    let ids = filter_ids(&plan);
    assert!(!ids.is_empty(), "{plan}");
    let expected = sorted(&env.plan(FILTERED).unwrap());
    assert_eq!(expected.len(), 3, "{expected:?}");
    let run_with = |values: &Arc<LocalValues>| {
        let runnable = env
            .plan(FILTERED)
            .unwrap()
            .with_worker(|w| w.with_values(Arc::clone(values) as _));
        sorted(&runnable)
    };
    // One producer partition: the join's filter is published to the scan.
    let one = Arc::new(LocalValues::new());
    assert_eq!(run_with(&one), expected);
    for id in &ids {
        let snapshot = one.filter(*id).unwrap().published().snapshot.clone();
        assert!(matches!(snapshot, FilterSnapshot::Final(_)), "{snapshot:?}");
        assert!(
            one.subscribe(ValueId::Filter(*id))
                .unwrap()
                .current()
                .is_some()
        );
    }
    // Two: this run publishes one partition's filter only, which would
    // drop the other partition's keys. The scan's filter stays pending, it
    // keeps every row, and the result is the same.
    let two = Arc::new(
        LocalValues::with_producers(ids.iter().map(|id| (ValueId::Filter(*id), 2)).collect())
            .unwrap(),
    );
    assert_eq!(run_with(&two), expected);
    for id in &ids {
        let cell = two.filter(*id).unwrap();
        assert!(matches!(cell.published().snapshot, FilterSnapshot::Pending));
        assert_eq!(cell.stats().counts().rows_dropped, 0);
        assert!(
            two.subscribe(ValueId::Filter(*id))
                .unwrap()
                .current()
                .is_none()
        );
    }
}

/// A broadcast build that is itself a join publishing a runtime filter:
/// its operators hold the instantiation's value service.
const NESTED: &str = "SELECT x.a FROM t AS x JOIN (SELECT y.a FROM t AS y JOIN \
                      (SELECT a FROM t WHERE b < 30) AS z ON y.a = z.a) AS w ON x.a = w.a";

#[test]
fn a_dropped_instantiation_frees_its_operators_and_values() {
    use krishiv_exec::physical::Partitioning;
    let env = env();
    let plan = env.plan(NESTED).unwrap();
    let broadcast_over_join = nodes(plan.root()).into_iter().any(|n| match n {
        PhysNode::Exchange(e) => {
            matches!(e.partitioning, Partitioning::Broadcast)
                && nodes(&e.input)
                    .into_iter()
                    .any(|n| matches!(n, PhysNode::HashJoin(_)))
        }
        _ => false,
    });
    assert!(broadcast_over_join, "{plan}");
    let values = Arc::new(LocalValues::new());
    let worker =
        WorkerContext::new(Arc::clone(&env.tables) as _).with_values(Arc::clone(&values) as _);
    let root = instantiate(&plan, &worker).unwrap();
    drop(worker);
    let (root_weak, values_weak) = (Arc::downgrade(&root), Arc::downgrade(&values));
    drop((root, values));
    // The service names the broadcast input without owning it: no cycle
    // through the join that publishes into it.
    assert!(root_weak.upgrade().is_none());
    assert!(values_weak.upgrade().is_none());
}

/// A hash exchange routes by the function its plan names: `KeyGroupV1`
/// sends each row to the partition owning its key group under the routing
/// hash, the same on every worker; an `Unpinned` exchange (a plan written
/// before the function was fixed) refuses to route at all.
#[test]
fn hash_exchanges_route_by_the_named_function() {
    use arrow_array::{Array, Int64Array, StringArray};
    use krishiv_exec::physical::HashScheme;
    use krishiv_kernels::hash::routing;

    let a = Int64Array::from_iter_values(0..1000);
    let b = StringArray::from_iter_values((0..1000).map(|i| format!("k{}", i % 13)));
    let keys: [&dyn Array; 2] = [&a, &b];
    let mut h = Vec::new();
    routing::hash_columns(&keys, &mut h).unwrap_or_else(|e| panic!("{e}"));
    let mut out = vec![7];
    for n in [1, 3, 64, 4096] {
        HashScheme::KeyGroupV1
            .partitions(&keys, n, &mut out)
            .unwrap_or_else(|e| panic!("{e}"));
        let want: Vec<u32> = h.iter().map(|&h| routing::partition_of(h, n)).collect();
        assert_eq!(out, want, "{n} partitions");
        assert!(out.iter().all(|&p| p < n));
    }
    let err = HashScheme::Unpinned
        .partitions(&keys, 4, &mut out)
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError, "{err}");
    let unsupported: [&dyn Array; 1] = [&arrow_array::ListArray::from_iter_primitive::<
        arrow_array::types::Int32Type,
        _,
        _,
    >(vec![Some(vec![Some(1)])])];
    assert!(
        HashScheme::KeyGroupV1
            .partitions(&unsupported, 4, &mut out)
            .is_err()
    );
}
