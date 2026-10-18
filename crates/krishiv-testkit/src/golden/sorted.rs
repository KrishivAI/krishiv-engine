// SPDX-License-Identifier: Apache-2.0
//! `SQLQueryTestHelper.isSemanticallySorted` (Spark 4.2.0) over Krishiv's resolved logical plan.
//!
//! `SQLQueryTestSuite` sorts a result's rows before comparing them unless the query's analysed
//! plan is "semantically sorted", so the verdict decides whether an engine's row order counts.
//! Spark's rule, in order:
//!
//! 1. a `Join`, `Aggregate`, `Generate`, `Sample` or `Distinct` is not sorted, whatever is below;
//! 2. a `DESCRIBE` command is sorted;
//! 3. a global `Sort`, with any `Project` and `Filter` nodes above it (`PhysicalOperation`), is
//!    sorted;
//! 4. otherwise the plan is sorted when any child is.
//!
//! "Child" is a child relation: plans inside subquery expressions are not children, so an
//! `ORDER BY` in an `IN`, `EXISTS` or scalar subquery does not count, while one in a derived table
//! does (unless a join or aggregate sits above it). The definitions of a `WITH` are children of
//! the `WithCte` node (Spark's `WithCTE`), so an `ORDER BY` in a CTE counts even when the main
//! query is unsorted.
//!
//! Plan nodes the IR does not have yet (`Distinct`, `Generate`, `Sample`, `Describe*`, set
//! operations, windows) are not matched here; a plan with such a node falls to rule 4, so each
//! one must be added to this function when it enters the IR (a `DISTINCT` lowered to an
//! `Aggregate` is rule 1). Windows are not a rule of Spark's: a window's own `ORDER BY` is not a
//! `Sort` node. The expected verdicts in `tests/fixtures/spark-4.2.0/semantically-sorted.json`
//! come from Spark running the original function on analysed plans.

use std::collections::HashSet;

use krishiv_plan::{LogicalPlan, PlanNode};

/// Whether the row order of the result of `plan` is significant, as `isSemanticallySorted`
/// decides on the analysed plan.
#[must_use]
pub fn is_semantically_sorted(plan: &LogicalPlan) -> bool {
    // An explicit stack instead of recursion (plans may be deep), and a visited set because
    // subtrees are shared: a node's verdict does not depend on the path to it.
    let mut stack = vec![plan];
    let mut seen: HashSet<*const LogicalPlan> = HashSet::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(node)) {
            continue;
        }
        match &node.node {
            PlanNode::Join { .. } | PlanNode::Aggregate { .. } => {}
            _ if sorts_below_projects(node) => return true,
            _ => node.for_each_child(&mut |child| stack.push(child)),
        }
    }
    false
}

/// `PhysicalOperation(_, _, Sort(_, true, _, _))`: `Project` and `Filter` nodes over a global
/// sort. (Spark stops collecting them at expressions it cannot collapse; the sort is then found
/// as a child of the node, so the verdict is the same.)
fn sorts_below_projects(plan: &LogicalPlan) -> bool {
    let mut node = plan;
    loop {
        match &node.node {
            PlanNode::Project { input, .. } | PlanNode::Filter { input, .. } => node = input,
            PlanNode::Sort { global, .. } => return *global,
            _ => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv_plan::{
        Cte, Expr, JoinCondition, JoinKind, Literal, ReadSource, Scan, SortOrder, SubqueryKind,
    };
    use serde_json::Value;

    use super::*;

    type R = Result<(), Box<dyn std::error::Error>>;
    type Plan = LogicalPlan;

    fn scan(name: &str) -> Plan {
        let source = ReadSource::Table {
            name: vec![name.to_owned()],
            options: std::collections::BTreeMap::new(),
        };
        Plan::new(PlanNode::Scan(Scan {
            source,
            output: Vec::new(),
            version: None,
        }))
        .alias(name)
    }

    fn col() -> Expr {
        Expr::column(["a"])
    }

    /// `SELECT a, b ...` (the star, expanded).
    fn project(p: Plan) -> Plan {
        p.project(vec![col(), Expr::column(["b"])])
    }

    fn order(p: Plan) -> Plan {
        p.sort(vec![SortOrder::asc(col())])
    }

    fn sort_by(input: Plan) -> Plan {
        Plan::new(PlanNode::Sort {
            input: Arc::new(input),
            order: vec![SortOrder::asc(col())],
            global: false,
        })
    }

    fn join(l: Plan, r: Plan) -> Plan {
        l.join(r, JoinKind::Inner, JoinCondition::None)
    }

    fn group(p: Plan) -> Plan {
        p.aggregate(vec![col()], vec![col()])
    }

    fn limit(p: Plan) -> Plan {
        p.limit(Expr::literal(Literal::Integer(1)))
    }

    fn filter(p: Plan) -> Plan {
        p.filter(Expr::literal(Literal::Boolean(true)))
    }

    fn filter_in(p: Plan, sub: Plan) -> Plan {
        p.filter(Expr::subquery(SubqueryKind::In(vec![Arc::new(col())]), sub))
    }

    fn with(main: Plan, defs: Vec<Plan>) -> Plan {
        let ctes = defs
            .into_iter()
            .enumerate()
            .map(|(i, plan)| Cte {
                name: format!("c{i}"),
                plan: Arc::new(plan),
            })
            .collect();
        Plan::new(PlanNode::WithCte {
            input: Arc::new(main),
            ctes,
        })
    }

    fn t1() -> Plan {
        scan("t1")
    }

    fn t2() -> Plan {
        scan("t2")
    }

    /// `(select * from t1 order by a) x`.
    fn sorted_x() -> Plan {
        order(project(t1())).alias("x")
    }

    /// The analysed plans of the queries of the fixture, built by hand.
    fn plans() -> Vec<(&'static str, Plan)> {
        let c = || scan("c");
        let scalar = |sub: Plan| Expr::subquery(SubqueryKind::Scalar, sub);
        vec![
            ("select * from t1", project(t1())),
            ("select * from t1 order by a, b", order(project(t1()))),
            (
                "select * from t1 join t2 on t1.a = t2.a",
                project(join(t1(), t2())),
            ),
            (
                "select * from t1 join t2 on t1.a = t2.a order by t1.a",
                order(project(join(t1(), t2()))),
            ),
            ("select a, max(b) from t1 group by a", group(t1())),
            (
                "select a, max(b) from t1 group by a order by a",
                order(group(t1())),
            ),
            ("select distinct a from t1", group(project(t1()))),
            (
                "select distinct a from t1 order by a",
                order(group(project(t1()))),
            ),
            ("select a from t1 order by a", order(project(t1()))),
            (
                "select * from t1 where a > 1 order by a",
                order(project(filter(t1()))),
            ),
            (
                "select a + 1 from (select * from t1 order by a) x",
                project(sorted_x()),
            ),
            (
                "select * from (select * from t1 order by a) x where a > 1",
                project(filter(sorted_x())),
            ),
            (
                "select * from (select * from t1 order by a) x limit 1",
                limit(project(sorted_x())),
            ),
            (
                "select * from t1 order by a limit 1",
                limit(order(project(t1()))),
            ),
            ("select * from t1 limit 1", limit(project(t1()))),
            ("select * from t1 sort by a", sort_by(project(t1()))),
            (
                "select * from (select * from t1 sort by a) x",
                project(sort_by(project(t1())).alias("x")),
            ),
            (
                "select * from (select * from t1 order by a) x join t2 on x.a = t2.a",
                project(join(sorted_x(), t2())),
            ),
            (
                "select * from t1 join (select * from t2 order by a) y on t1.a = y.a",
                project(join(t1(), order(project(t2())).alias("y"))),
            ),
            (
                "select * from (select * from t1 order by a) x join t2 on x.a = t2.a order by x.a",
                order(project(join(sorted_x(), t2()))),
            ),
            (
                "select count(*) from (select * from t1 order by a) x",
                group(sorted_x()),
            ),
            (
                "select a, count(*) from (select * from t1 order by a) x group by a",
                group(sorted_x()),
            ),
            (
                "select * from (select a, count(*) c from t1 group by a) x order by a",
                order(project(group(t1()).alias("x"))),
            ),
            (
                "select * from (select a, count(*) c from t1 group by a) x",
                project(group(t1()).alias("x")),
            ),
            (
                "select * from (select * from t1 order by a) x order by b",
                order(project(sorted_x())),
            ),
            (
                "select * from (select * from t1 order by a) x where a in (select a from t2)",
                project(filter_in(sorted_x(), project(t2()))),
            ),
            (
                "select * from t1 where a in (select a from t2 order by a)",
                project(filter_in(t1(), order(project(t2())))),
            ),
            (
                "select * from t1 where exists (select 1 from t2 order by a)",
                project(t1().filter(Expr::subquery(SubqueryKind::Exists, order(project(t2()))))),
            ),
            (
                "select a, (select b from t2 order by a limit 1) from t1",
                t1().project(vec![col(), scalar(limit(order(project(t2()))))]),
            ),
            (
                "select * from t1 where a = (select max(a) from t2)",
                project(t1().filter(scalar(group(t2())))),
            ),
            (
                "select * from t1 where a in (select a from t2) order by a",
                order(project(filter_in(t1(), project(t2())))),
            ),
            (
                "select * from t1 where a in (select a from (select * from t2 order by a) y)",
                project(filter_in(t1(), project(order(project(t2())).alias("y")))),
            ),
            (
                "select * from (select * from t1 where a in (select a from t2 order by a)) x",
                project(project(filter_in(t1(), order(project(t2())))).alias("x")),
            ),
            (
                "with c as (select * from t1 order by a) select * from c",
                with(project(c()), vec![order(project(t1()))]),
            ),
            (
                "with c as (select * from t1) select * from c order by a",
                with(order(project(c())), vec![project(t1())]),
            ),
            (
                "with c as (select * from t1) select * from c",
                with(project(c()), vec![project(t1())]),
            ),
            (
                "with c as (select * from t1 order by a) select * from t2",
                with(project(t2()), vec![order(project(t1()))]),
            ),
            (
                "with c as (select * from t1 order by a) select * from c join t2 on c.a = t2.a",
                with(project(join(c(), t2())), vec![order(project(t1()))]),
            ),
            (
                "with c as (select a from t1 group by a), d as (select * from t2 order by a) select * from c",
                with(project(c()), vec![group(t1()), order(project(t2()))]),
            ),
            (
                "select * from (with c as (select * from t1 order by a) select * from c) x",
                project(with(project(c()), vec![order(project(t1()))]).alias("x")),
            ),
        ]
    }

    /// Every query of the fixture, with Spark's verdict on its analysed plan, against the same
    /// query's hand-built plan; the planted subquery cases are among them.
    #[test]
    fn matches_spark_on_every_fixture_query() -> R {
        let path = crate::tests_dir().join("fixtures/spark-4.2.0/semantically-sorted.json");
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let cases = doc["cases"].as_array().ok_or("cases")?;
        let plans = plans();
        assert_eq!(cases.len(), plans.len());
        let (mut sorted, mut unsorted) = (0, 0);
        for case in cases {
            let sql = case["sql"].as_str().ok_or("sql")?;
            let want = case["sorted"].as_bool().ok_or("sorted")?;
            let (_, plan) = plans.iter().find(|(q, _)| *q == sql).ok_or(sql)?;
            assert_eq!(is_semantically_sorted(plan), want, "{sql}");
            if want {
                sorted += 1;
            } else {
                unsorted += 1;
            }
        }
        assert!(
            sorted > 10 && unsorted > 10,
            "{sorted} sorted, {unsorted} not"
        );
        Ok(())
    }

    #[test]
    fn a_non_global_sort_is_not_a_global_sort_but_its_children_still_count() {
        let below = order(project(t1()));
        assert!(is_semantically_sorted(&sort_by(below)));
        assert!(!is_semantically_sorted(&sort_by(project(t1()))));
    }

    /// A CTE referencing the one before it, thirty times: 2^30 paths, linear nodes.
    #[test]
    fn shared_subtrees_are_visited_once() {
        let mut plan = Arc::new(project(t1()));
        for _ in 0..30 {
            plan = Arc::new(Plan::new(PlanNode::WithCte {
                input: Arc::clone(&plan),
                ctes: vec![Cte {
                    name: "c".to_owned(),
                    plan: Arc::clone(&plan),
                }],
            }));
        }
        assert!(!is_semantically_sorted(&plan));
        let sorted = Arc::new(order(project(t1())));
        let shared = Plan::new(PlanNode::WithCte {
            input: Arc::clone(&sorted),
            ctes: vec![Cte {
                name: "c".to_owned(),
                plan: sorted,
            }],
        });
        assert!(is_semantically_sorted(&shared));
    }
}
