// SPDX-License-Identifier: Apache-2.0
//! A scan that decides once (`ScanPushdown::decides_once`): it sees every conjunct of the filter over it
//! at once, claims some, and the plan keeps only Spark's residual above it. The claimed conjuncts are
//! trusted, never applied (a source that claims `a > 5` and returns every row keeps them all).

use std::sync::{Arc, Mutex};

use arrow_schema::SchemaRef;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_exec::physical::PhysNode;
use krishiv_exec::planner::{
    DefaultPlanner, MemoryTable, PhysicalPlanner as _, PlanningContext, TableProvider,
    TableProviderResolver,
};
use krishiv_exec::{
    BoundScan, BoundScanId, ExecutionReads, FilterPushdown, OfferedConjunct, PushedFilter,
    ScanDecision, ScanPushdown, ScanPushdowns, Source, SourceConjunct,
};
use krishiv_io::ColumnPath;
use krishiv_plan::Scan;

use super::{Env, Runnable, env, run};

/// What the fake source saw and did.
#[derive(Debug, Default)]
struct Log {
    decides: Mutex<Vec<Vec<SourceConjunct>>>,
    reads: Mutex<u32>,
}

/// A source that claims every serializable conjunct and applies none.
#[derive(Debug)]
struct Table {
    inner: Arc<MemoryTable>,
    log: Arc<Log>,
}

impl TableProvider for Table {
    fn scan(&self) -> Result<Arc<dyn ScanPushdown>, KrishivError> {
        Ok(Arc::new(Deciding {
            inner: self.inner.scan()?,
            log: Arc::clone(&self.log),
        }))
    }
}

struct Deciding {
    inner: Arc<dyn ScanPushdown>,
    log: Arc<Log>,
}

impl ScanPushdown for Deciding {
    fn table_schema(&self) -> SchemaRef {
        self.inner.table_schema()
    }
    fn filter_pushdown(&self, _filter: &PushedFilter) -> FilterPushdown {
        FilterPushdown::Exact
    }
    fn build(&self, pushdowns: ScanPushdowns) -> Result<Arc<dyn Source>, KrishivError> {
        self.inner.build(pushdowns)
    }
    fn decides_once(&self) -> bool {
        true
    }
    fn decide(
        &self,
        conjuncts: &[OfferedConjunct],
        _projection: &[ColumnPath],
    ) -> Result<ScanDecision, KrishivError> {
        let sources: Vec<SourceConjunct> = conjuncts.iter().map(|c| c.source.clone()).collect();
        let pushed = sources
            .iter()
            .map(|s| matches!(s, SourceConjunct::Serializable(_)))
            .collect();
        self.log.decides.lock().unwrap().push(sources);
        Ok(ScanDecision {
            pushed,
            bound: Arc::new(Bound {
                inner: Arc::clone(&self.inner),
                log: Arc::clone(&self.log),
            }),
        })
    }
}

struct Bound {
    inner: Arc<dyn ScanPushdown>,
    log: Arc<Log>,
}

impl BoundScan for Bound {
    fn table_schema(&self) -> SchemaRef {
        self.inner.table_schema()
    }
    fn output_schema(&self, pushdowns: &ScanPushdowns) -> Result<SchemaRef, KrishivError> {
        self.inner.output_schema(pushdowns)
    }
    fn explain(&self, _pushdowns: &ScanPushdowns) -> Option<String> {
        None
    }
    fn read(
        &self,
    ) -> futures::future::BoxFuture<'static, Result<Arc<dyn ScanPushdown>, KrishivError>> {
        *self.log.reads.lock().unwrap() += 1;
        let inner = Arc::clone(&self.inner);
        Box::pin(async move { Ok(inner) })
    }
}

/// Keeps the bound scans of one query.
#[derive(Debug)]
struct Resolver {
    table: Arc<Table>,
    bound: Mutex<Vec<Arc<dyn BoundScan>>>,
}

impl TableProviderResolver for Resolver {
    fn provider(&self, _scan: &Scan) -> Result<Arc<dyn TableProvider>, KrishivError> {
        Ok(Arc::clone(&self.table) as _)
    }
    fn bind_scan(&self, bound: Arc<dyn BoundScan>) -> Result<BoundScanId, KrishivError> {
        let mut list = self.bound.lock().unwrap();
        list.push(bound);
        Ok(BoundScanId(u32::try_from(list.len() - 1).unwrap()))
    }
    fn bound_scan(&self, id: BoundScanId) -> Result<Arc<dyn BoundScan>, KrishivError> {
        Ok(Arc::clone(&self.bound.lock().unwrap()[id.0 as usize]))
    }
}

struct Fixture {
    env: Env,
    resolver: Arc<Resolver>,
    log: Arc<Log>,
}

fn fixture() -> Fixture {
    let env = env();
    let inner = env.tables.provider(&scan_of_t()).unwrap();
    let log = Arc::new(Log::default());
    // The table's rows, through the in-memory provider.
    let memory: Arc<MemoryTable> = Arc::new(MemoryTable::new(
        inner.scan().unwrap().table_schema(),
        super::table(10).1,
    ));
    let resolver = Arc::new(Resolver {
        table: Arc::new(Table {
            inner: memory,
            log: Arc::clone(&log),
        }),
        bound: Mutex::new(Vec::new()),
    });
    Fixture { env, resolver, log }
}

fn scan_of_t() -> Scan {
    Scan {
        source: krishiv_plan::ReadSource::Table {
            name: ["spark_catalog", "default", "t"]
                .map(str::to_owned)
                .to_vec(),
            options: Default::default(),
        },
        output: Vec::new(),
        version: None,
    }
}

impl Fixture {
    fn plan(&self, sql: &str, decisions: bool) -> Runnable {
        self.plan_of(&self.env.analyze(sql), decisions)
    }

    /// `sql` with a security barrier (design E14) over the `Filter` directly above the scan, as a
    /// policy's filter is placed.
    fn plan_guarded(&self, sql: &str) -> Runnable {
        use krishiv_common::{Principal, PrincipalKey, PrincipalKind};
        use krishiv_plan::analyzer::{Securable, TableIdent};
        use krishiv_plan::{LogicalPlan, PlanNode, PolicyRef};
        let policy = PolicyRef::new(
            Securable::Table(TableIdent {
                catalog: "spark_catalog".to_owned(),
                namespace: vec!["default".to_owned()],
                name: "t".to_owned(),
            }),
            PrincipalKey::of(&Principal::new("ana", PrincipalKind::User)),
            1,
        );
        let mut placed = 0;
        let root = Arc::clone(self.env.analyze(sql).root())
            .transform_up(&mut |p| {
                let over_scan = matches!(&p.node, PlanNode::Filter { input, .. }
                    if matches!(input.node, PlanNode::Scan(_)));
                Ok::<_, std::convert::Infallible>(if over_scan {
                    placed += 1;
                    Arc::new(LogicalPlan::clone(&p).security_barrier(policy.clone()))
                } else {
                    p
                })
            })
            .unwrap_or_else(|e| match e {});
        assert_eq!(placed, 1, "no Filter directly over the scan in {sql}");
        self.plan_of(&krishiv_plan::Plan::try_new(root).unwrap(), true)
    }

    fn plan_of(&self, plan: &krishiv_plan::Plan, decisions: bool) -> Runnable {
        let plan = krishiv_optimizer::optimize(plan).unwrap();
        let cx = PlanningContext::new(Arc::clone(&self.resolver) as _, Arc::new(Config::new()))
            .with_decisions(decisions);
        let planned = DefaultPlanner.plan(&plan, &cx).unwrap();
        Runnable::new(planned, Arc::clone(&self.resolver) as _)
            .with_worker(|w| w.with_execution_reads(Arc::new(ExecutionReads::default())))
    }

    fn rows(&self, plan: &Runnable) -> usize {
        run(plan)
            .unwrap()
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum()
    }
}

/// The sorted values of the plan's first output column (an integer column).
fn first_column(plan: &Runnable) -> Vec<i64> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, Int64Type};
    let mut out = Vec::new();
    for b in run(plan).unwrap() {
        let c = b.column(0);
        if let Some(a) = c.as_primitive_opt::<Int64Type>() {
            out.extend(a.iter().flatten());
        } else {
            out.extend(
                c.as_primitive::<Int32Type>()
                    .iter()
                    .flatten()
                    .map(i64::from),
            );
        }
    }
    out.sort_unstable();
    out
}

/// Whether the plan text has a `Filter` operator line (not the scan's `PushedFilters`).
fn has_filter(plan: &Runnable) -> bool {
    plan.to_string().lines().any(|l| {
        l.trim_start_matches(['+', '-', ' ', ':'])
            .starts_with("Filter")
    })
}

fn decided(plan: &Runnable) -> bool {
    fn any(n: &PhysNode) -> bool {
        matches!(n, PhysNode::Scan(s) if s.decided.is_some()) || n.children().into_iter().any(any)
    }
    any(plan.root())
}

#[test]
fn a_claimed_conjunct_is_trusted_and_never_applied() {
    let f = fixture();
    let plan = f.plan("SELECT b FROM t WHERE a > 5 AND b > 0", true);
    assert!(decided(&plan), "{plan}");
    assert!(plan.to_string().contains("PushedFilters:"), "{plan}");
    // Everything it claimed is left out of the plan above the scan.
    assert!(!has_filter(&plan), "{plan}");
    assert_eq!(f.rows(&plan), 10);
    assert_eq!(f.log.decides.lock().unwrap().len(), 1);
}

#[test]
fn the_residual_keeps_what_the_source_did_not_claim() {
    let f = fixture();
    let plan = f.plan("SELECT b FROM t WHERE a > 5 AND (a = 1 OR b = 20)", true);
    assert!(has_filter(&plan), "{plan}");
    // `a > 5` is claimed and not applied; the disjunction is the residual: a = 1 or b = 20.
    assert_eq!(f.rows(&plan), 2);
    assert_eq!(first_column(&plan), [10, 20]);
    let decides = f.log.decides.lock().unwrap();
    assert!(
        decides[0]
            .iter()
            .any(|s| matches!(s, SourceConjunct::Translated)),
        "{decides:?}"
    );
}

#[test]
fn without_a_filter_the_source_is_not_asked() {
    let f = fixture();
    let plan = f.plan("SELECT b FROM t", true);
    assert!(!decided(&plan));
    assert_eq!(f.rows(&plan), 10);
    assert!(f.log.decides.lock().unwrap().is_empty());
}

#[test]
fn a_plan_made_without_decisions_keeps_its_whole_filter() {
    let f = fixture();
    let plan = f.plan("SELECT b FROM t WHERE a > 5", false);
    assert!(has_filter(&plan), "{plan}");
    assert!(f.log.decides.lock().unwrap().is_empty());
}

#[test]
fn undecided_deciding_scan_does_not_run() {
    let f = fixture();
    let plan = f.plan("SELECT b FROM t WHERE a > 5", false);
    let err = run(&plan).unwrap_err();
    assert!(err.to_string().contains("without its decision"), "{err}");
    assert_eq!(*f.log.reads.lock().unwrap(), 0);
}

#[test]
fn a_deciding_scan_with_no_filter_runs_without_decisions() {
    let f = fixture();
    let plan = f.plan("SELECT b FROM t", false);
    assert!(!decided(&plan), "{plan}");
    assert_eq!(f.rows(&plan), 10);
}

#[test]
fn a_plan_run_twice_reads_twice_and_tasks_share_one_read() {
    let f = fixture();
    let plan = f.plan("SELECT b FROM t WHERE a > 5", true);
    f.rows(&plan);
    assert_eq!(*f.log.reads.lock().unwrap(), 1);
    // The same context shares its execution's read.
    f.rows(&plan);
    assert_eq!(*f.log.reads.lock().unwrap(), 1);
    // A fresh execution reads again.
    let again = plan.with_worker(|w| w.with_execution_reads(Arc::new(ExecutionReads::default())));
    f.rows(&again);
    assert_eq!(*f.log.reads.lock().unwrap(), 2);
}

#[test]
fn a_decided_scan_outside_an_execution_is_internal() {
    let f = fixture();
    let plan = f.plan("SELECT b FROM t WHERE a > 5", true);
    let bare = Runnable::new(
        (*plan).clone(),
        Arc::clone(&f.resolver) as Arc<dyn TableProviderResolver>,
    );
    let err = run(&bare).unwrap_err();
    assert!(err.to_string().contains("outside an execution"), "{err}");
}

#[test]
fn a_split_like_keeps_spark_length_guard_above_the_scan() {
    // `s LIKE 's%1'` is StartsWith + EndsWith plus `length(s) >= 2`: a source that takes both leaves
    // must not lose the guard (`s1` overlaps; `s` alone would match both leaves).
    let f = fixture();
    let plan = f.plan("SELECT b FROM t WHERE s LIKE 's%1'", true);
    assert!(has_filter(&plan), "the LIKE stays above the scan: {plan}");
    let decides = f.log.decides.lock().unwrap();
    assert_eq!(decides[0].len(), 3, "{decides:?}");
    assert!(matches!(decides[0][2], SourceConjunct::Untranslated));
    drop(decides);
    // Both leaves were taken: `PushedFilters` lists the conjunct once.
    let text = plan.to_string();
    let pushed = text
        .split("PushedFilters: [")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
        .unwrap_or_default();
    assert_eq!(pushed.to_lowercase().matches("like").count(), 1, "{text}");
    // The fixture's source trusts both leaves and filters nothing: the rows that come back are the
    // residual's answer. `s%1` over s0..s9 is exactly `s1`.
    assert_eq!(f.rows(&plan), 1);
}

#[test]
fn under_a_barrier_the_source_is_not_asked_and_the_policy_filter_stays() {
    // Control: without the barrier the source claims `a > 5` and the plan keeps no filter.
    let f = fixture();
    let plain = f.plan("SELECT b FROM t WHERE a > 5", true);
    assert!(decided(&plain) && !has_filter(&plain), "{plain}");
    // Under one the filter is the policy's: the source's code is not offered it, the scan reads
    // undecided with nothing pushed, and the engine applies the whole filter.
    let f = fixture();
    let plan = f.plan_guarded("SELECT b FROM t WHERE a > 5");
    assert!(!decided(&plan), "{plan}");
    assert!(has_filter(&plan), "{plan}");
    assert!(!plan.to_string().contains("PushedFilters:"), "{plan}");
    // `a` is 0..9 with 3 and 7 NULL: 6, 8 and 9 pass.
    assert_eq!(first_column(&plan), [60, 80, 90]);
    assert!(f.log.decides.lock().unwrap().is_empty());
}
