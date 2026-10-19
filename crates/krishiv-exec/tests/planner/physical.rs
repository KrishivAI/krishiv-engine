// SPDX-License-Identifier: Apache-2.0
//! The physical plan as data (P2-T172): plans compare and clone as values,
//! instantiate any number of times, and instantiation checks that the
//! worker's tables still match the plan.

use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::physical::{
    ExprEnv, HashJoinPlan, JoinFilterPlan, PhysNode, PhysicalPlan, ScanId, ScanNode,
    ScanRuntimeFilter, WorkerContext, assert_plan_data, instantiate,
};
use krishiv_exec::planner::{
    DefaultPlanner, ExplainNode, MemoryTable, PhysicalPlanner as _, PlanningContext, TableProvider,
    TableProviderResolver,
};
use krishiv_exec::{FilterPushdown, JoinType, PushedFilter, ScanPushdown, ScanPushdowns, Source};
use krishiv_io::{ColumnPath, FilterId, KeyFilterOptions};

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
            runtime_filters: vec![JoinFilterPlan {
                id: FilterId(id),
                key: 0,
                options: KeyFilterOptions::default(),
            }],
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
    };
    check(
        instantiate_err(write, Schema::empty(), &worker),
        "without a write sink",
    );
}
