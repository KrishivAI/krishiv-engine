// SPDX-License-Identifier: Apache-2.0
//! `PartialAggregate` below a join, merged above it (design P2-T167 unit 1): a
//! hand-built partial-then-final plan returns what the single aggregate does,
//! each join match being a weight on the states.

use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Decimal128Array, Int32Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::planner::{DefaultPlanner, PhysicalPlanner as _, PlanningContext};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{FileFormat as _, FormatOptions};
use krishiv_plan::{
    AttrId, BoundColumn, Expr, ExprNode, Grouping, LogicalPlan, Plan, PlanNode, ResolvedExpr,
    ResolvedFunction, partial_state_columns,
};
use proptest::prelude::*;

use super::tpch::{Order, Tpch, parquet_files, rows};
use super::{Runnable, run};

fn resolved(r: ResolvedExpr) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(r)))
}

/// The aggregate call under an output expression.
fn call_of(e: &Expr) -> &ResolvedFunction {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => call_of(expr),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => f,
        other => panic!("not an aggregate call: {other:?}"),
    }
}

/// `f`'s raw argument type: what its column or cast holds.
fn raw_type(f: &ResolvedFunction) -> krishiv_types::DataType {
    match f.args.first().map(|a| &a.node) {
        Some(ExprNode::Resolved(ResolvedExpr::Column(c))) => c.data_type.clone(),
        Some(ExprNode::Cast { to, .. }) => to.clone(),
        _ => krishiv_types::DataType::Long,
    }
}

/// `e` with its call's arguments replaced by `arg`.
fn with_state(e: &Arc<Expr>, arg: Arc<Expr>) -> Arc<Expr> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => resolved(ResolvedExpr::Alias {
            expr: with_state(expr, arg),
            output: output.clone(),
        }),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let mut f = f.clone();
            f.args = vec![arg];
            resolved(ResolvedExpr::Function(f))
        }
        other => panic!("not an aggregate call: {other:?}"),
    }
}

/// An aggregate over `input` split around its join: the calls run in a partial
/// aggregate over `f`'s side, keyed by `f.k`, and merge above.
fn split(
    input: &Arc<LogicalPlan>,
    keys: &[Arc<Expr>],
    outputs: &[Arc<Expr>],
    states_of: &dyn Fn(Vec<BoundColumn>) -> Vec<BoundColumn>,
) -> Arc<LogicalPlan> {
    let mut next = 1_u64 << 40;
    let mut calls = Vec::new();
    let mut states = Vec::new();
    let mut merged = Vec::new();
    for e in outputs {
        if matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Column(_))) {
            merged.push(Arc::clone(e));
            continue;
        }
        let f = call_of(e);
        let raw = raw_type(f);
        let columns = partial_state_columns(f, &raw, AttrId(next)).unwrap();
        next += columns.len() as u64;
        merged.push(with_state(
            e,
            resolved(ResolvedExpr::StateInput {
                columns: columns
                    .iter()
                    .map(|c| resolved(ResolvedExpr::Column(c.clone())))
                    .collect(),
                raw_type: raw,
            }),
        ));
        calls.push(resolved(ResolvedExpr::Function(f.clone())));
        states.extend(columns);
    }
    let states = states_of(states);
    let k_of = |side: &Arc<LogicalPlan>| {
        side.resolved_output()
            .and_then(|o| o.iter().find(|c| c.name == "k").cloned())
    };
    let partial = |side: &Arc<LogicalPlan>| {
        let k = k_of(side).unwrap();
        Arc::new(LogicalPlan::new(PlanNode::PartialAggregate {
            input: Arc::clone(side),
            keys: vec![resolved(ResolvedExpr::Column(k))],
            aggregates: calls.clone(),
            states: states.clone(),
        }))
    };
    // Projections above the join only pass the raw columns the calls read.
    let mut join = input;
    while let PlanNode::Project { input, .. } = &join.node {
        join = input;
    }
    let PlanNode::Join {
        left,
        right,
        kind,
        condition,
    } = &join.node
    else {
        panic!("not a join: {join:?}");
    };
    let in_left = k_of(left).is_some();
    let below = Arc::new(LogicalPlan::new(PlanNode::Join {
        left: if in_left {
            partial(left)
        } else {
            Arc::clone(left)
        },
        right: if in_left {
            Arc::clone(right)
        } else {
            partial(right)
        },
        kind: *kind,
        condition: condition.clone(),
    }));
    Arc::new(LogicalPlan::new(PlanNode::Aggregate {
        input: below,
        grouping: Grouping::GroupBy(keys.to_vec()),
        aggregates: merged,
    }))
}

/// `plan` with its aggregate split.
fn eager(plan: &Plan, states_of: &dyn Fn(Vec<BoundColumn>) -> Vec<BoundColumn>) -> Plan {
    let top = Arc::clone(plan.root())
        .transform_down(&mut |p: Arc<LogicalPlan>| {
            Ok::<_, Infallible>(match &p.node {
                PlanNode::Aggregate {
                    input,
                    grouping: Grouping::GroupBy(keys),
                    aggregates,
                } => split(input, keys, aggregates, states_of),
                _ => p,
            })
        })
        .unwrap();
    Plan::try_new(LogicalPlan::clone(&top)).unwrap()
}

fn physical(t: &Tpch, plan: &Plan) -> Result<Runnable, KrishivError> {
    let cx = PlanningContext::new(Arc::clone(&t.tables) as _, Arc::clone(&t.config));
    Ok(Runnable::new(
        DefaultPlanner.plan(plan, &cx)?,
        Arc::clone(&t.tables) as _,
    ))
}

fn write(dir: &Path, name: &str, batch: &RecordBatch) {
    let o = FormatOptions::defaults(ParquetFormat.options());
    let mut enc = ParquetFormat::encoder_with(&batch.schema(), &o, |p| p).unwrap();
    let mut bytes = Vec::new();
    enc.write(batch).unwrap();
    while let Some(b) = enc.take_bytes() {
        bytes.extend_from_slice(&b);
    }
    bytes.extend_from_slice(&enc.finish().unwrap().tail);
    std::fs::write(dir.join(name), bytes).unwrap();
}

/// `(key, int, decimal, string)` rows; `None` is NULL.
type Row = (i64, Option<i32>, Option<i128>, Option<u8>);

fn facts(rows: &[Row]) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("i", DataType::Int32, true),
        Field::new("d", DataType::Decimal128(10, 2), true),
        Field::new("s", DataType::Utf8View, true),
    ]);
    let d: Decimal128Array = rows.iter().map(|r| r.2).collect();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
            Arc::new(Int32Array::from_iter(rows.iter().map(|r| r.1))),
            Arc::new(d.with_precision_and_scale(10, 2).unwrap()),
            Arc::new(StringViewArray::from_iter(
                rows.iter().map(|r| r.3.map(|v| format!("v{v}"))),
            )),
        ],
    )
    .unwrap()
}

fn weights(keys: &[i64]) -> RecordBatch {
    let schema = Schema::new(vec![Field::new("wk", DataType::Int64, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int64Array::from_iter_values(keys.iter().copied()))],
    )
    .unwrap()
}

/// `f` on a join's null-supplying side: an unmatched `w` row gets NULL states.
const OUTER: &str = "SELECT w.wk, count(f.i), sum(f.i), avg(f.i), sum(f.d), avg(f.d), \
                     min(f.i), max(f.d), min(f.s), max(f.s) \
                     FROM w LEFT JOIN f ON w.wk = f.k GROUP BY w.wk";

const QUERY: &str = "SELECT f.k, count(f.i), count(*), sum(f.i), avg(f.i), sum(f.d), avg(f.d), \
                     min(f.i), max(f.d), min(f.s), max(f.s) \
                     FROM f JOIN w ON f.k = w.wk GROUP BY f.k";

fn tables(dir: &Path, rows: &[Row], keys: &[i64], rt: &tokio::runtime::Runtime) -> Tpch {
    write(dir, "f.parquet", &facts(rows));
    write(dir, "w.parquet", &weights(keys));
    let sources = [
        ("f", vec!["f.parquet".to_owned()]),
        ("w", vec!["w.parquet".to_owned()]),
    ];
    parquet_files(dir, &sources, true, rt)
}

fn sorted(batches: &[RecordBatch]) -> Vec<String> {
    let mut r = rows(batches);
    r.sort_unstable();
    r
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]
    /// Whatever the rows and however many times the join matches a key, the
    /// split plan returns the single aggregate's rows.
    #[test]
    fn partial_then_final_equals_the_single_aggregate(
        rows in prop::collection::vec(
            (
                0i64..12,
                prop::option::of(-1000i32..1000),
                prop::option::of(-100_000i128..100_000),
                prop::option::of(0u8..9),
            ),
            0..400,
        ),
        keys in prop::collection::vec(0i64..16, 0..60),
    ) {
        let rt = runtime();
        let dir = tempfile::tempdir().unwrap();
        let t = tables(dir.path(), &rows, &keys, &rt);
        for q in [QUERY, OUTER] {
            let plan = t.optimized(q, Order::Statistics).unwrap();
            let single = sorted(&run(&physical(&t, &plan).unwrap()).unwrap());
            let split = physical(&t, &eager(&plan, &|s| s)).unwrap();
            prop_assert!(split.to_string().contains("PartialHashAggregate"), "{}", split);
            prop_assert_eq!(sorted(&run(&split).unwrap()), single);
        }
    }
}

fn sample(rt: &tokio::runtime::Runtime, dir: &Path) -> Tpch {
    tables(
        dir,
        &[(1, Some(1), Some(100), Some(1)), (2, None, None, None)],
        &[1, 1, 2],
        rt,
    )
}

/// One group of known values: a key matched twice counts its rows twice, and
/// a group of NULLs stays NULL.
#[test]
fn join_matches_weigh_the_states() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(
        dir.path(),
        &[
            (1, Some(2), Some(100), Some(1)),
            (1, Some(4), Some(300), Some(2)),
            (2, None, None, None),
        ],
        &[1, 1, 2],
        &rt,
    );
    let q = "SELECT f.k, count(f.i), count(*), sum(f.i), avg(f.i), sum(f.d) \
             FROM f JOIN w ON f.k = w.wk GROUP BY f.k ORDER BY f.k";
    let plan = t.optimized(q, Order::Statistics).unwrap();
    let got = sorted(&run(&physical(&t, &eager(&plan, &|s| s)).unwrap()).unwrap());
    let single = sorted(&run(&physical(&t, &plan).unwrap()).unwrap());
    assert_eq!(got, single);
    assert_eq!(got.len(), 2, "{got:?}");
    assert!(got[0].starts_with("1|4|4|12|3"), "{got:?}");
    assert!(got[1].starts_with("2|0|1|NULL|NULL"), "{got:?}");
}

/// An unmatched outer row's NULL states are empty states: `count` is 0 and
/// the others NULL, not whatever the NULL slots hold.
#[test]
fn null_states_of_an_outer_join_are_empty() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(
        dir.path(),
        &[
            (1, Some(2), Some(100), Some(1)),
            (1, Some(4), Some(300), Some(2)),
        ],
        &[1, 3],
        &rt,
    );
    let q = "SELECT w.wk, count(f.i), sum(f.i), avg(f.i), sum(f.d), avg(f.d), min(f.s) \
             FROM w LEFT JOIN f ON w.wk = f.k GROUP BY w.wk";
    let plan = t.optimized(q, Order::Statistics).unwrap();
    let got = sorted(&run(&physical(&t, &eager(&plan, &|s| s)).unwrap()).unwrap());
    assert_eq!(got, sorted(&run(&physical(&t, &plan).unwrap()).unwrap()));
    assert!(
        got[1].starts_with("3|0|NULL|NULL|NULL|NULL|NULL"),
        "{got:?}"
    );
}

/// A change to a partial aggregate's keys and calls.
type Edit<'a> = dyn Fn(&mut Vec<Arc<Expr>>, &mut Vec<Arc<Expr>>) + 'a;

/// `plan`'s partial aggregate with `edit` applied to its keys and calls.
fn with_partial(plan: &Plan, edit: &Edit<'_>) -> Plan {
    let top = Arc::clone(plan.root())
        .transform_up(&mut |p: Arc<LogicalPlan>| {
            Ok::<_, Infallible>(match &p.node {
                PlanNode::PartialAggregate {
                    input,
                    keys,
                    aggregates,
                    states,
                } => {
                    let (mut keys, mut aggregates) = (keys.clone(), aggregates.clone());
                    edit(&mut keys, &mut aggregates);
                    Arc::new(LogicalPlan::new(PlanNode::PartialAggregate {
                        input: Arc::clone(input),
                        keys,
                        aggregates,
                        states: states.clone(),
                    }))
                }
                _ => p,
            })
        })
        .unwrap();
    Plan::try_new(LogicalPlan::clone(&top)).unwrap()
}

/// Only the optimizer builds the node, so a malformed one is an internal
/// error, naming what is wrong.
#[test]
fn a_malformed_partial_aggregate_is_refused() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = sample(&rt, dir.path());
    let plan = t.optimized(QUERY, Order::Statistics).unwrap();
    let split = eager(&plan, &|s| s);
    let literal = || {
        Arc::new(Expr::new(ExprNode::Literal(
            krishiv_plan::Literal::Integer(1),
        )))
    };
    let filtered = |e: &Arc<Expr>| {
        let mut f = call_of(e).clone();
        f.filter = Some(Arc::new(Expr::new(ExprNode::Literal(
            krishiv_plan::Literal::Boolean(true),
        ))));
        resolved(ResolvedExpr::Function(f))
    };
    let distinct = |e: &Arc<Expr>| {
        let mut f = call_of(e).clone();
        f.distinct = true;
        resolved(ResolvedExpr::Function(f))
    };
    let cases: [(&Edit<'_>, &str); 4] = [
        (&|k, _| k[0] = literal(), "key that is not a column"),
        (&|_, a| a[0] = literal(), "is not a partial aggregate call"),
        (&|_, a| a[0] = filtered(&a[0]), "cannot run as a partial"),
        (&|_, a| a[0] = distinct(&a[0]), "cannot run as a partial"),
    ];
    for (edit, want) in cases {
        let e = physical(&t, &with_partial(&split, edit)).err().unwrap();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        assert!(e.to_string().contains(want), "{e}");
    }
}

/// The rule runs before the optimizer's last passes, so they must keep the
/// split plan's rows: `Prune` narrows what the partial aggregate reads, and
/// leaves its state columns.
#[test]
fn the_optimizer_keeps_a_split_plan() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(
        dir.path(),
        &[
            (1, Some(2), Some(100), Some(1)),
            (1, Some(4), Some(300), Some(2)),
            (2, None, None, None),
        ],
        &[1, 1, 2, 3],
        &rt,
    );
    for q in [QUERY, OUTER] {
        let plan = t.optimized(q, Order::Statistics).unwrap();
        let again = krishiv_optimizer::optimize(&eager(&plan, &|s| s)).unwrap();
        let runnable = physical(&t, &again).unwrap();
        assert!(
            runnable.to_string().contains("PartialHashAggregate"),
            "{runnable}"
        );
        assert_eq!(
            sorted(&run(&runnable).unwrap()),
            sorted(&run(&physical(&t, &plan).unwrap()).unwrap())
        );
    }
}

/// `e` with a `FILTER` on its call.
fn with_filter(e: &Arc<Expr>) -> Arc<Expr> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => resolved(ResolvedExpr::Alias {
            expr: with_filter(expr),
            output: output.clone(),
        }),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            let mut f = f.clone();
            f.filter = Some(Arc::new(Expr::new(ExprNode::Literal(
                krishiv_plan::Literal::Boolean(true),
            ))));
            resolved(ResolvedExpr::Function(f))
        }
        other => panic!("not an aggregate call: {other:?}"),
    }
}

/// States are already filtered: a merge with a filter is a malformed plan.
#[test]
fn a_filter_on_a_merge_is_refused() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = sample(&rt, dir.path());
    let q = "SELECT f.k, sum(f.i) FROM f JOIN w ON f.k = w.wk GROUP BY f.k";
    let plan = t.optimized(q, Order::Statistics).unwrap();
    let split = eager(&plan, &|s| s);
    let filtered = Arc::clone(split.root())
        .transform_down(&mut |p: Arc<LogicalPlan>| {
            Ok::<_, Infallible>(match &p.node {
                PlanNode::Aggregate {
                    input,
                    grouping,
                    aggregates,
                } => {
                    let merged = aggregates
                        .iter()
                        .map(|e| {
                            if matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Column(_))) {
                                return Arc::clone(e);
                            }
                            with_filter(e)
                        })
                        .collect();
                    Arc::new(LogicalPlan::new(PlanNode::Aggregate {
                        input: Arc::clone(input),
                        grouping: grouping.clone(),
                        aggregates: merged,
                    }))
                }
                _ => p,
            })
        })
        .unwrap();
    let filtered = Plan::try_new(LogicalPlan::clone(&filtered)).unwrap();
    let e = physical(&t, &filtered).err().unwrap();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.to_string().contains("a filter on a call merging"), "{e}");
}

#[test]
fn the_plan_prints_the_split() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = sample(&rt, dir.path());
    let plan = t.optimized(QUERY, Order::Statistics).unwrap();
    let split = eager(&plan, &|s| s);
    let text = split.root().to_string();
    assert!(text.contains("PartialAggregate ["), "{text}");
    assert!(text.contains("state("), "{text}");
    let printed = physical(&t, &split).unwrap().to_string();
    assert!(printed.contains("PartialHashAggregate keys="), "{printed}");
}

/// The optimizer types the states without the executor: a plan where the two
/// disagree is refused, not run.
#[test]
fn states_the_executor_does_not_produce_are_refused() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = sample(&rt, dir.path());
    let plan = t.optimized(QUERY, Order::Statistics).unwrap();
    let wrong_type = eager(&plan, &|mut s| {
        s[0].data_type = krishiv_types::DataType::String;
        s
    });
    let missing = eager(&plan, &|mut s| {
        s.pop();
        s
    });
    for bad in [wrong_type, missing] {
        let e = physical(&t, &bad).err().unwrap();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        assert!(e.to_string().contains("partial states"), "{e}");
    }
}

/// An aggregate merging columns that are not its call's states is refused.
#[test]
fn a_merge_of_the_wrong_columns_is_refused() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = sample(&rt, dir.path());
    let q = "SELECT f.k, avg(f.i) FROM f JOIN w ON f.k = w.wk GROUP BY f.k";
    let plan = t.optimized(q, Order::Statistics).unwrap();
    let split = eager(&plan, &|s| s);
    // Reads the `sum` column twice instead of `sum` and `count`.
    let wrong = Arc::clone(split.root())
        .transform_down(&mut |p: Arc<LogicalPlan>| {
            Ok::<_, Infallible>(match &p.node {
                PlanNode::Aggregate {
                    input,
                    grouping,
                    aggregates,
                } => {
                    let merged = aggregates
                        .iter()
                        .map(|e| {
                            if matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Column(_))) {
                                return Arc::clone(e);
                            }
                            let f = call_of(e);
                            let ExprNode::Resolved(ResolvedExpr::StateInput { columns, raw_type }) =
                                &f.args[0].node
                            else {
                                panic!("not merging: {f:?}");
                            };
                            with_state(
                                e,
                                resolved(ResolvedExpr::StateInput {
                                    columns: vec![Arc::clone(&columns[0]), Arc::clone(&columns[0])],
                                    raw_type: raw_type.clone(),
                                }),
                            )
                        })
                        .collect();
                    Arc::new(LogicalPlan::new(PlanNode::Aggregate {
                        input: Arc::clone(input),
                        grouping: grouping.clone(),
                        aggregates: merged,
                    }))
                }
                _ => p,
            })
        })
        .unwrap();
    let wrong = Plan::try_new(LogicalPlan::clone(&wrong)).unwrap();
    let e = physical(&t, &wrong).err().unwrap();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.to_string().contains("where its states are"), "{e}");
}
