// SPDX-License-Identifier: Apache-2.0
//! Join ordering from TPC-H SF10 statistics: plan snapshots, no cross
//! products, greedy ordering of large join trees, and the same rows as
//! the unoptimised plans.

use std::collections::HashMap;

use krishiv_optimizer::{ColumnStats, Statistics, TableStats, optimize, optimize_with};
use krishiv_plan::{JoinCondition, Literal, LogicalPlan, PlanNode, ReadSource, Scan};
use serde_json::Value;

use crate::support::{self, data, eval};

/// Statistics by table name, from `tests/fixtures/tpch-sf10-stats.json`.
#[derive(Debug)]
struct Fixture(HashMap<String, TableStats>);

impl Statistics for Fixture {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        self.0.get(name.last()?).cloned()
    }
}

fn bound(v: &Value) -> Option<Literal> {
    if let Some(d) = v.get("date") {
        let p: Vec<i64> = d.as_str()?.split('-').map(|x| x.parse().unwrap()).collect();
        return Some(Literal::Date(eval::days(p[0], p[1], p[2])));
    }
    Some(Literal::Double(v.get("number")?.as_f64()?))
}

fn fixture() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tpch-sf10-stats.json"
    );
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let tables = v["tables"].as_object().unwrap().iter().map(|(name, t)| {
        let columns = t["columns"].as_object().unwrap().iter().map(|(c, s)| {
            let stats = ColumnStats {
                distinct: s["distinct"].as_u64(),
                nulls: Some(0),
                min: bound(&s["min"]),
                max: bound(&s["max"]),
            };
            (c.clone(), stats)
        });
        let stats = TableStats {
            rows: t["rows"].as_u64().unwrap(),
            columns: columns.collect(),
        };
        (name.clone(), stats)
    });
    Fixture(tables.collect())
}

/// Panics on a join without a condition anywhere in `root`.
fn no_cross_product(root: &LogicalPlan, what: &str) {
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        p.for_each_child(&mut |c| stack.push(c));
        p.for_each_expr(&mut |e| {
            if let krishiv_plan::ExprNode::Subquery { plan, .. } = &e.node {
                no_cross_product(plan, what);
            }
        });
        if let PlanNode::Join {
            condition: JoinCondition::None,
            ..
        } = &p.node
        {
            panic!("{what} plans a cross product:\n{root}");
        }
    }
}

#[test]
fn tpch_join_orders() {
    let ctx = support::ctx();
    let stats = fixture();
    let db = data::tpch(7, 150);
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let optimized = optimize_with(&plan, &stats).unwrap();
        let what = format!("q{q:02}");
        no_cross_product(optimized.root(), &what);
        assert_eq!(
            optimized.root().resolved_output(),
            plan.root().resolved_output()
        );
        let again = optimize_with(&optimized, &stats).unwrap();
        assert!(
            again.structurally_eq(&optimized),
            "{what} is a fixed point:\n{}",
            again.root()
        );
        assert_eq!(
            eval::rows(&db, optimized.root()),
            eval::rows(&db, plan.root()),
            "{what}"
        );
        insta::assert_snapshot!(what, optimized.root().to_string());
    }
}

#[test]
fn large_trees_are_ordered_greedily() {
    // A 12-way chain written backwards, with a filter on one end.
    let mut sql = "SELECT t1.b FROM t t12".to_owned();
    for i in (1..12).rev() {
        sql.push_str(&format!(" JOIN t t{i} ON t{i}.a = t{}.c", i + 1));
    }
    sql.push_str(" WHERE t12.b = 'x'");
    let plan = support::resolve(&sql, &support::ctx());
    let optimized = optimize(&plan).unwrap();
    no_cross_product(optimized.root(), "the chain");
    assert!(optimize(&optimized).unwrap().structurally_eq(&optimized));
    for seed in 1..=4 {
        let db = data::small(seed);
        assert_eq!(
            eval::rows(&db, optimized.root()),
            eval::rows(&db, plan.root())
        );
    }
    insta::assert_snapshot!(optimized.root().to_string());
}

#[test]
fn cross_products_the_query_has_stay_last() {
    let sql = "SELECT t.a, u.d, x.b FROM t, u, t x WHERE t.a = x.a AND u.d = 'y' AND x.c > t.c";
    let plan = support::resolve(sql, &support::ctx());
    let optimized = optimize(&plan).unwrap();
    for seed in 1..=4 {
        let db = data::small(seed);
        assert_eq!(
            eval::rows(&db, optimized.root()),
            eval::rows(&db, plan.root())
        );
    }
    insta::assert_snapshot!(optimized.root().to_string());
}

#[test]
fn estimates_read_every_shape() {
    let ctx = support::ctx();
    let stats = fixture();
    let cases = [
        "SELECT o_orderkey FROM (SELECT o_orderkey, o_orderdate FROM orders ORDER BY o_orderdate LIMIT 10) o JOIN lineitem ON l_orderkey = o_orderkey JOIN part ON p_partkey = l_partkey",
        "SELECT n_name FROM nation, supplier, (SELECT c_nationkey FROM customer GROUP BY c_nationkey) c WHERE 5 > n_nationkey AND n_nationkey = s_nationkey AND c.c_nationkey = n_nationkey",
        "SELECT r_name FROM region, nation, supplier WHERE TRUE AND r_regionkey = n_regionkey AND CAST(s_nationkey AS BIGINT) = n_nationkey AND s_acctbal > 100.00",
    ];
    for (i, sql) in cases.iter().enumerate() {
        let plan = support::resolve(sql, &ctx);
        let optimized = optimize_with(&plan, &stats).unwrap();
        no_cross_product(optimized.root(), sql);
        insta::assert_snapshot!(format!("shape_{i}"), optimized.root().to_string(), sql);
    }
}

#[test]
fn positional_roots_keep_their_columns() {
    let sql =
        "SELECT * FROM t JOIN u ON t.a = u.a JOIN t x ON x.c = u.a WHERE t.b = 'y' AND u.d = 'z'";
    let resolved = support::resolve(sql, &support::ctx());
    let PlanNode::Project { input, .. } = &resolved.root().node else {
        panic!("{}", resolved.root());
    };
    let plan = krishiv_plan::Plan::try_new(std::sync::Arc::clone(input)).unwrap();
    let optimized = optimize(&plan).unwrap();
    assert_eq!(
        optimized.root().resolved_output(),
        plan.root().resolved_output()
    );
    insta::assert_snapshot!(optimized.root().to_string());
}

/// Optimised and unoptimised `sql` return the same rows on small data.
fn same_rows(sql: &str) -> krishiv_plan::Plan {
    let plan = support::resolve(sql, &support::ctx());
    let optimized = optimize(&plan).unwrap();
    assert_eq!(
        optimized.root().resolved_output(),
        plan.root().resolved_output()
    );
    for seed in 1..=4 {
        let db = data::small(seed);
        assert_eq!(
            eval::rows(&db, optimized.root()),
            eval::rows(&db, plan.root()),
            "{sql}"
        );
    }
    optimized
}

#[test]
fn conjuncts_over_three_inputs_join_them_all() {
    same_rows("SELECT t.a, u.d, x.b FROM t, u, t x WHERE t.a + u.a = x.a");
    same_rows("SELECT t.a, u.d, x.b FROM t, u, t x WHERE t.a = u.a OR u.d = x.b");
    same_rows("SELECT t.a, u.d, x.b, y.d FROM t, u, t x, u y WHERE t.a = u.a AND u.a + x.a = y.a");
}

#[test]
fn cte_definitions_keep_their_columns() {
    same_rows(
        "SELECT s.*, y.c FROM (WITH c AS (SELECT * FROM t JOIN u ON t.a = u.a WHERE t.b = 'x') \
         SELECT c1.* FROM c c1 JOIN c c2 ON c1.d = c2.d) s JOIN t y ON s.b = y.b",
    );
}

#[test]
fn greedy_ordering_scales() {
    let mut sql = "SELECT t1.b FROM t t1".to_owned();
    for i in 2..=60 {
        sql.push_str(&format!(" JOIN t t{i} ON t{i}.a = t{}.c", i - 1));
    }
    let plan = support::resolve(&sql, &support::ctx());
    let started = std::time::Instant::now();
    let optimized = optimize(&plan).unwrap();
    assert!(started.elapsed().as_secs() < 10, "{:?}", started.elapsed());
    no_cross_product(optimized.root(), "the chain");
}

#[test]
fn subquery_conditions_keep_their_join() {
    let optimized = same_rows(
        "SELECT t.a, u.d FROM t JOIN u ON t.a = (SELECT max(x.a) FROM t x WHERE x.b = u.d) \
         JOIN t y ON y.c = t.c",
    );
    no_cross_product(optimized.root(), "a subquery join condition");
}
