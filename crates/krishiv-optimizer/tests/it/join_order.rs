// SPDX-License-Identifier: Apache-2.0
//! Join ordering from TPC-H SF10 statistics: plan snapshots, no cross
//! products, greedy ordering of large join trees, and the same rows as
//! the unoptimised plans.

use std::collections::HashMap;

use krishiv_optimizer::{
    ColumnStats, Options, StatSource, Statistics, TableStats, optimize, optimize_with,
    optimize_with_options,
};
use krishiv_plan::{
    BoundColumn, Builtin, Expr, ExprNode, FunctionId, JoinCondition, Literal, LogicalPlan,
    PlanNode, ReadSource, ResolvedExpr, Scan,
};
use serde_json::Value;

use crate::support::{self, data, eval};

/// Statistics by table name, from `tests/fixtures/tpch-sf10-stats.json`.
#[derive(Debug)]
pub(crate) struct Fixture(HashMap<String, TableStats>);

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

pub(crate) fn fixture() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tpch-sf10-stats.json"
    );
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let tables = v["tables"].as_object().unwrap().iter().map(|(name, t)| {
        let columns = t["columns"].as_object().unwrap().iter().map(|(c, s)| {
            let stats = ColumnStats {
                source: StatSource::default(),
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
pub(crate) fn no_cross_product(root: &LogicalPlan, what: &str) {
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

/// TPC-H statistics at scale factor `sf` as the benchmark runner reads
/// them from Parquet footers (`krishiv-exec` `planner/file_table.rs`): row
/// counts and bounds, no distinct counts; an integral column's distinct
/// count is its range, at most its rows (`StatSource::IntegralRange`), and
/// a decimal or string column has none. Rows and key ranges scale from the
/// SF10 fixture; `nation` and `region` keep theirs.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "TPC-H's integral bounds are small non-negative integers"
)]
fn footers(sf: u64) -> Fixture {
    let Fixture(mut tables) = fixture();
    let scaled = |n: &str| !matches!(n, "nation" | "region");
    // A key's range is its table's rows (keys run from 1).
    let keys: Vec<f64> = tables
        .iter()
        .filter(|(n, _)| scaled(n))
        .map(|(_, t)| t.rows as f64)
        .collect();
    let integral = |c: &str| c.ends_with("key") || matches!(c, "p_size" | "ps_availqty");
    for (name, t) in &mut tables {
        if scaled(name) {
            t.rows = t.rows * sf / 10;
        }
        for (col, c) in &mut t.columns {
            c.distinct = None;
            if let Some(Literal::Double(m)) = &mut c.max
                && keys.contains(m)
            {
                *m = *m * sf as f64 / 10.0;
            }
            if let (true, Some(Literal::Double(lo)), Some(Literal::Double(hi))) =
                (integral(col), &c.min, &c.max)
            {
                c.source = StatSource::IntegralRange;
                c.distinct = Some(((hi - lo) as u64 + 1).min(t.rows));
            }
        }
    }
    Fixture(tables)
}

/// The `Join Inner` lines of `text` whose condition reads `needle`.
fn joins_on<'a>(text: &'a str, needle: &str) -> Vec<&'a str> {
    text.lines()
        .filter(|l| l.contains("Join Inner") && l.contains(needle))
        .collect()
}

/// TPC-H q02 at SF1000 statistics (`research/bench/sf1000-2026-11-05.md`
/// section 5). `p_partkey`, `ps_partkey` and the decorrelated aggregate's
/// `ps_partkey` are one equivalence class, so partsupp and the aggregate
/// share a partkey equality even though the query writes it only through
/// part. The join on the minimum cost must carry it: on `ps_supplycost`
/// alone (about 100K values for 800M rows) that join is close to many to
/// many and filled a 40 GB spill cap. No join repeats a key of one class.
#[test]
fn q02_keeps_the_composite_key_at_sf1000() {
    let ctx = support::ctx();
    let plan = support::resolve(&support::tpch_sql(2), &ctx);
    let optimized = optimize_with(&plan, &footers(1000)).unwrap();
    let text = optimized.root().to_string();
    let cost = joins_on(&text, "min(ps_supplycost)");
    assert_eq!(cost.len(), 1, "{text}");
    assert!(cost[0].contains("ps_partkey"), "{text}");
    for l in text.lines().filter(|l| l.contains("Join Inner")) {
        let keys = l.split(" and ").filter(|c| c.contains("partkey")).count();
        assert!(keys <= 1, "a join repeats the partkey class: {l}\n{text}");
    }
    // The minimum groups only the filtered parts' partsupp rows (its
    // input is reduced by a semi join with them), not all 200M parts'.
    assert!(text.contains("Join LeftSemi"), "{text}");
    let db = data::tpch(7, 150);
    assert_eq!(
        eval::rows(&db, optimized.root()),
        eval::rows(&db, plan.root())
    );
    insta::assert_snapshot!(text);
}

/// Statistics propagation steers estimates only: with it off, every TPC-H
/// query returns the rows it returns with it on, and both the rows of the
/// unoptimised plan, whatever order the joins take.
#[test]
fn results_match_with_statistics_propagation_on_and_off() {
    let ctx = support::ctx();
    let stats = fixture();
    let db = data::tpch(7, 150);
    let off = support::options(|o| o.stats_propagation = false);
    let mut ordered_differently = 0;
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let on = optimize_with(&plan, &stats).unwrap();
        let no = optimize_with_options(&plan, &stats, &off).unwrap();
        let want = eval::rows(&db, plan.root());
        assert_eq!(eval::rows(&db, on.root()), want, "q{q:02} on");
        assert_eq!(eval::rows(&db, no.root()), want, "q{q:02} off");
        ordered_differently += usize::from(!on.structurally_eq(&no));
    }
    // The switch changes plans (else the test compares a plan with itself).
    assert!(ordered_differently > 0);
}

/// Design P2-T167's acceptance: Q13 runs its count below the join; the queries
/// where the rewrite does not pay stay as they were, with the gate that said so.
#[test]
fn eager_aggregation_fires_where_the_design_says() {
    let ctx = support::ctx();
    let stats = fixture();
    let decisions = |q: usize| -> (bool, Vec<String>) {
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let (optimized, steps) =
            krishiv_optimizer::optimize_traced_with_options(&plan, &stats, &Options::default())
                .unwrap();
        (
            optimized.root().to_string().contains("PartialAggregate"),
            steps
                .into_iter()
                .filter(|s| s.rule == "EagerAggregate")
                .map(|s| s.note)
                .collect(),
        )
    };
    let (fired, notes) = decisions(13);
    assert!(fired, "{notes:?}");
    assert!(
        notes[0].starts_with("pushed by [o_custkey]: rows "),
        "{notes:?}"
    );
    for q in [3, 10, 17, 18] {
        let (fired, notes) = decisions(q);
        assert!(!fired, "q{q}: {notes:?}");
    }
}

/// Switched off, nothing moves and nothing is said.
#[test]
fn eager_aggregation_can_be_switched_off() {
    let plan = support::resolve(&support::tpch_sql(13), &support::ctx());
    let off = support::options(|o| o.eager_aggregation = false);
    let optimized = optimize_with_options(&plan, &fixture(), &off).unwrap();
    assert!(!optimized.root().to_string().contains("PartialAggregate"));

    let library = krishiv_udf_sdk::UdfLibrary::new();
    let mut config = krishiv_common::config::Config::new();
    let on = krishiv_optimizer::optimize_with_udfs(&plan, &fixture(), &library, &config).unwrap();
    assert!(on.root().to_string().contains("PartialAggregate"));
    config
        .set("krishiv.optimizer.eagerAggregation", "false")
        .unwrap();
    let no = krishiv_optimizer::optimize_with_udfs(&plan, &fixture(), &library, &config).unwrap();
    assert!(!no.root().to_string().contains("PartialAggregate"));
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

/// A join condition holding a subquery stays on its join. (A correlated
/// scalar subquery in a join condition is rejected at analysis, as Spark
/// rejects it, so the subquery here is uncorrelated.)
#[test]
fn subquery_conditions_keep_their_join() {
    let optimized = same_rows(
        "SELECT t.a, u.d FROM t JOIN u ON t.a = (SELECT max(x.a) FROM t x WHERE x.b = 'x') \
         AND t.b = u.d JOIN t y ON y.c = t.c",
    );
    no_cross_product(optimized.root(), "a subquery join condition");
}

/// Whether `e` is, or has as a conjunct, an equality of two columns.
fn has_column_equality(e: &Expr) -> bool {
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
        return false;
    };
    match f.id {
        FunctionId::Builtin(Builtin::And) => f.args.iter().any(|a| has_column_equality(a)),
        FunctionId::Builtin(Builtin::Eq) => f
            .args
            .iter()
            .all(|a| matches!(a.node, ExprNode::Resolved(ResolvedExpr::Column(_)))),
        _ => false,
    }
}

/// Panics on a join without a column equality in its condition (a
/// nested-loop join) anywhere in `root`.
fn every_join_has_an_equality(root: &LogicalPlan, what: &str) {
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        p.for_each_child(&mut |c| stack.push(c));
        if let PlanNode::Join { condition, .. } = &p.node {
            let equi = matches!(condition, JoinCondition::On(c) if has_column_equality(c));
            assert!(equi, "{what} plans a nested-loop join:\n{root}");
        }
    }
}

/// TPC-DS SF1 statistics as the Parquet footers give them (`rows`, and per
/// column `(type, min, max, nulls, distinct)`): no distinct counts, so an
/// integral column's range bounds its distinct values (as the engine
/// derives them) and every other column counts as all distinct.
type Column = (&'static str, &'static str, f64, f64, u64, Option<u64>);
const TPCDS_SF1: &[(&str, u64, &[Column])] = &[
    (
        "web_sales",
        719_384,
        &[
            ("ws_item_sk", "bigint", 1.0, 18_000.0, 0, Some(18_000)),
            ("ws_order_number", "bigint", 1.0, 60_000.0, 0, Some(60_000)),
            ("ws_quantity", "bigint", 1.0, 100.0, 173, Some(100)),
            ("ws_sales_price", "decimal(7,2)", 0.0, 299.16, 170, None),
            (
                "ws_net_profit",
                "decimal(7,2)",
                -9_938.0,
                18_864.56,
                0,
                None,
            ),
        ],
    ),
    (
        "web_returns",
        71_654,
        &[
            ("wr_item_sk", "bigint", 1.0, 18_000.0, 0, Some(18_000)),
            ("wr_order_number", "bigint", 1.0, 59_999.0, 0, Some(59_999)),
            (
                "wr_refunded_addr_sk",
                "bigint",
                1.0,
                50_000.0,
                3_110,
                Some(50_000),
            ),
            ("wr_fee", "decimal(7,2)", 0.5, 100.0, 3_139, None),
        ],
    ),
    (
        "customer_address",
        50_000,
        &[
            ("ca_address_sk", "bigint", 1.0, 50_000.0, 0, Some(50_000)),
            ("ca_state", "string", f64::NAN, f64::NAN, 1_550, None),
            ("ca_country", "string", f64::NAN, f64::NAN, 1_525, None),
        ],
    ),
    // Footer statistics as Parquet gives them (TPC-DS SF1): row counts and
    // bounds; distinct counts only from integral bounds (`max - min + 1`,
    // at most the non-null rows), none for strings.
    (
        "catalog_sales",
        1_441_548,
        &[
            (
                "cs_sold_date_sk",
                "bigint",
                2_450_815.0,
                2_452_648.0,
                7_191,
                Some(1_834),
            ),
            (
                "cs_bill_cdemo_sk",
                "bigint",
                5.0,
                1_920_765.0,
                7_183,
                Some(1_434_365),
            ),
        ],
    ),
    (
        "date_dim",
        73_049,
        &[
            (
                "d_date_sk",
                "bigint",
                2_415_022.0,
                2_488_070.0,
                0,
                Some(73_049),
            ),
            ("d_year", "bigint", 1_900.0, 2_100.0, 0, Some(201)),
        ],
    ),
    (
        "customer_demographics",
        1_920_800,
        &[
            ("cd_demo_sk", "bigint", 1.0, 1_920_800.0, 0, Some(1_920_800)),
            ("cd_gender", "string", f64::NAN, f64::NAN, 0, None),
            ("cd_marital_status", "string", f64::NAN, f64::NAN, 0, None),
            ("cd_education_status", "string", f64::NAN, f64::NAN, 0, None),
        ],
    ),
];

#[derive(Debug)]
struct TpcdsFooters;

impl Statistics for TpcdsFooters {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let table = name.last()?;
        let (_, rows, columns) = TPCDS_SF1.iter().find(|t| t.0 == table)?;
        let columns = columns.iter().map(|&(c, _, min, max, nulls, distinct)| {
            let number = |v: f64| (!v.is_nan()).then_some(Literal::Double(v));
            let stats = ColumnStats {
                source: StatSource::default(),
                distinct,
                nulls: Some(nulls),
                min: number(min),
                max: number(max),
            };
            (c.to_owned(), stats)
        });
        Some(TableStats {
            rows: *rows,
            columns: columns.collect(),
        })
    }
}

/// The TPC-H catalog plus the TPC-DS tables of [`TPCDS_SF1`].
fn tpcds_ctx() -> krishiv_plan::analyzer::AnalysisContext {
    let mut cat = support::catalog();
    for (table, _, columns) in TPCDS_SF1 {
        let columns = columns
            .iter()
            .map(|(c, ty, ..)| krishiv_plan::analyzer::TableColumn {
                name: (*c).to_owned(),
                data_type: krishiv_types::parse_data_type(ty).unwrap(),
                nullable: true,
                char_varchar: None,
            })
            .collect();
        cat = cat.with_table(&["default"], table, columns, None);
    }
    support::ctx_over(cat)
}

/// TPC-DS q85, reduced to the three inputs of its failure: a disjunction
/// relates `web_sales` and `customer_address`, which otherwise meet only
/// through `web_returns`. Filter pushdown adds what the disjunction
/// implies on each input, so the disjunction's selectivity is conditional
/// on those filters (at SF1, 148M of the 245M pairs that pass them match).
/// Counted unconditionally it squared them, estimated the nested-loop join
/// of the two as tiny and planned it first: the hash join above it then
/// built 148M rows and the query ran out of memory. Joined through
/// `web_returns` instead, the disjunction is a residual of an equi-join.
#[test]
fn implied_filters_are_not_counted_twice() {
    let sql = "SELECT count(*), sum(ws_quantity), sum(wr_fee) \
               FROM web_sales, web_returns, customer_address \
               WHERE ws_item_sk = wr_item_sk AND ws_order_number = wr_order_number \
               AND ca_address_sk = wr_refunded_addr_sk \
               AND ws_sales_price BETWEEN 50.00 AND 200.00 \
               AND ((ca_country = 'United States' AND ca_state IN ('IN', 'OH', 'NJ') \
                     AND ws_net_profit BETWEEN 100 AND 200) \
                 OR (ca_country = 'United States' AND ca_state IN ('WI', 'CT', 'KY') \
                     AND ws_net_profit BETWEEN 150 AND 300) \
                 OR (ca_country = 'United States' AND ca_state IN ('LA', 'IA', 'AR') \
                     AND ws_net_profit BETWEEN 50 AND 250))";
    let plan = support::resolve(sql, &tpcds_ctx());
    let optimized = optimize_with(&plan, &TpcdsFooters).unwrap();
    every_join_has_an_equality(optimized.root(), "TPC-DS q85");
    insta::assert_snapshot!(optimized.root().to_string());
}

/// The same query with the disjunction written in `ON`: its implied
/// filters are added below the join too (once: optimizing again changes
/// nothing), so the estimate that divides them out holds for it as well.
#[test]
fn implied_filters_of_on_conditions() {
    let sql = "SELECT count(*), sum(ws_quantity), sum(wr_fee) \
               FROM web_sales JOIN customer_address \
               ON ((ca_country = 'United States' AND ca_state IN ('IN', 'OH', 'NJ') \
                     AND ws_net_profit BETWEEN 100 AND 200) \
                 OR (ca_country = 'United States' AND ca_state IN ('WI', 'CT', 'KY') \
                     AND ws_net_profit BETWEEN 150 AND 300)) \
               JOIN web_returns ON ws_item_sk = wr_item_sk AND ws_order_number = wr_order_number \
               AND ca_address_sk = wr_refunded_addr_sk";
    let plan = support::resolve(sql, &tpcds_ctx());
    let optimized = optimize_with(&plan, &TpcdsFooters).unwrap();
    every_join_has_an_equality(optimized.root(), "an ON disjunction");
    assert!(
        optimize_with(&optimized, &TpcdsFooters)
            .unwrap()
            .structurally_eq(&optimized)
    );
    insta::assert_snapshot!(optimized.root().to_string());
}

#[test]
fn implied_filters_of_on_conditions_keep_the_rows() {
    same_rows(
        "SELECT t.a, u.d FROM t JOIN u ON (t.b = 'x' AND u.d = 'y') OR (t.b = 'z' AND u.a > 2)",
    );
    same_rows(
        "SELECT t.a, u.d FROM t LEFT JOIN u ON (t.b = 'x' AND u.d = 'y') OR (t.b = 'z' AND u.a > 2)",
    );
}

/// The fixture's statistics without distinct counts for the columns
/// `keep` does not name, as Parquet footers give them (footers carry no
/// distinct counts).
fn without_distinct(keep: &[&str]) -> Fixture {
    let mut f = fixture();
    for t in f.0.values_mut() {
        for (name, c) in &mut t.columns {
            if !keep.contains(&name.as_str()) {
                c.distinct = None;
            }
        }
    }
    f
}

/// `col = literal` and `col IN (...)` read the column's distinct count
/// when one is known; without one the selectivity is the default, not
/// `1 / rows` (which would estimate a single matching row for
/// `l_returnflag = 'R'` and make 60M rows look like the smallest input).
#[test]
fn literal_equalities_without_a_distinct_count_use_the_default() {
    let ctx = support::ctx();
    let rows = |sql: &str, stats: &Fixture| {
        krishiv_optimizer::estimated_rows(support::resolve(sql, &ctx).root(), stats)
    };
    let known = fixture();
    let footers = without_distinct(&[]);
    let lineitem = 59_986_052.0;
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b;
    let eq = "SELECT l_orderkey FROM lineitem WHERE l_returnflag = 'R'";
    assert!(
        close(rows(eq, &known), lineitem / 3.0),
        "{}",
        rows(eq, &known)
    );
    assert!(
        close(rows(eq, &footers), lineitem * 0.2),
        "{}",
        rows(eq, &footers)
    );
    // The literal on the left reads the same.
    let flipped = "SELECT l_orderkey FROM lineitem WHERE 'R' = l_returnflag";
    assert!(close(rows(flipped, &footers), lineitem * 0.2));
    let within = "SELECT l_orderkey FROM lineitem WHERE l_shipmode IN ('MAIL', 'SHIP')";
    assert!(close(rows(within, &known), lineitem * 2.0 / 7.0));
    assert!(close(rows(within, &footers), lineitem * 0.2));
    // A distinct count from statistics still counts without the others.
    let only = without_distinct(&["l_returnflag"]);
    assert!(close(rows(eq, &only), lineitem / 3.0));
    // `c <=> NULL` is `c IS NULL`: the column's null fraction (none here).
    // Optimized first: folding makes the coerced NULL a literal.
    let folded = |sql: &str, stats: &Fixture| {
        let plan = optimize_with(&support::resolve(sql, &ctx), stats).unwrap();
        krishiv_optimizer::estimated_rows(plan.root(), stats)
    };
    let is_null = folded(
        "SELECT l_orderkey FROM lineitem WHERE l_returnflag IS NULL",
        &known,
    );
    for sql in [
        "SELECT l_orderkey FROM lineitem WHERE l_returnflag <=> NULL",
        "SELECT l_orderkey FROM lineitem WHERE NULL <=> l_returnflag",
    ] {
        let (k, f) = (folded(sql, &known), folded(sql, &footers));
        assert!(
            close(k, is_null) && close(f, is_null),
            "{sql}: {k} {f} vs {is_null}"
        );
    }
    // An equality between two columns of one input keeps `1 / max(ndv)`; a
    // count that is only the default guess reads as the rows (T166), so only
    // `col = literal` takes the default selectivity.
    let cols = "SELECT l_orderkey FROM lineitem WHERE l_commitdate = l_receiptdate";
    // The fixture has no distinct counts: each column reads as the rows.
    assert!(close(rows(cols, &footers), 1.0), "{}", rows(cols, &footers));
}

/// TPC-DS q26's shape: a fact table joined to a dimension filtered to one
/// year and to one filtered by three string equalities. A join with a
/// filtered input counts the distinct keys the filter keeps (at most its
/// rows), so the year of dates joins about a fifth of the fact rows (the
/// fact holds five years), not 1/200 of them as the whole 73,049-date
/// domain would say. The demographics join, the more selective one, then
/// comes first and the fact table stays the larger (probe) input of every
/// join, so each dimension's build can filter the fact scan.
#[test]
fn filtered_dimensions_count_the_keys_they_keep() {
    let sql = "SELECT count(*) FROM catalog_sales, customer_demographics, date_dim \
               WHERE cs_sold_date_sk = d_date_sk AND cs_bill_cdemo_sk = cd_demo_sk \
               AND cd_gender = 'M' AND cd_marital_status = 'S' \
               AND cd_education_status = 'College' AND d_year = 2000";
    let plan = support::resolve(sql, &tpcds_ctx());
    let optimized = optimize_with(&plan, &TpcdsFooters).unwrap();
    let text = optimized.root().to_string();
    let line = |needle: &str| {
        text.lines()
            .position(|l| l.contains("Join Inner") && l.contains(needle))
            .unwrap_or_else(|| panic!("no join on {needle}:\n{text}"))
    };
    // The date join is above (printed before) the demographics join.
    assert!(line("d_date_sk") < line("cd_demo_sk"), "{text}");
    insta::assert_snapshot!(text);
}

/// TPC-DS q13's shape: one `or` over three inputs whose disjuncts all hold
/// the same key equalities. The equalities are join conditions of their
/// own, so the inputs meet in equi-joins; as one condition over all three
/// inputs they could only meet in a cross product (q13 crossed a filtered
/// customer_demographics with the fact side).
#[test]
fn equalities_every_disjunct_holds_join_their_inputs() {
    let sql = "SELECT count(*) FROM lineitem, orders, customer \
               WHERE (l_orderkey = o_orderkey AND o_custkey = c_custkey \
                      AND l_quantity > 10 AND c_acctbal > 0) \
                  OR (l_orderkey = o_orderkey AND o_custkey = c_custkey \
                      AND l_quantity < 5 AND o_totalprice > 100)";
    let plan = support::resolve(sql, &support::ctx());
    let optimized = optimize_with(&plan, &fixture()).unwrap();
    no_cross_product(optimized.root(), sql);
    insta::assert_snapshot!(optimized.root().to_string());
    // The same rows, and a disjunct of only shared equalities leaves just
    // the equalities.
    same_rows(
        "SELECT t.a, u.d, x.b FROM t, u, t x \
         WHERE (t.a = u.a AND u.a = x.a AND t.b = 'p') OR (t.a = u.a AND u.a = x.a AND x.c > 1)",
    );
    let only = same_rows(
        "SELECT t.a, u.d, x.b FROM t, u, t x \
         WHERE (t.a = u.a AND u.a = x.a) OR (t.a = u.a AND u.a = x.a AND x.c > 1)",
    );
    assert!(!only.root().to_string().contains(" or "), "{}", only.root());
    // Equalities only some disjuncts hold stay inside the `or`.
    same_rows("SELECT t.a, u.d FROM t, u WHERE (t.a = u.a AND t.b = 'p') OR (t.c = u.a)");
}

/// TPC-DS q85's shape: an `or` over three inputs that other conditions
/// already join. Its shared equality is not needed to connect them, so it
/// stays inside the `or` instead of becoming an extra join key (q85 would
/// otherwise join its two demographics instances on two string columns).
#[test]
fn shared_equalities_stay_in_the_or_when_the_inputs_are_joined() {
    let sql = "SELECT count(*) FROM lineitem, orders, customer \
               WHERE l_orderkey = o_orderkey AND o_custkey = c_custkey \
               AND ((o_orderpriority = c_mktsegment AND l_quantity > 10) \
                 OR (o_orderpriority = c_mktsegment AND l_quantity < 5))";
    let plan = support::resolve(sql, &support::ctx());
    let optimized = optimize_with(&plan, &fixture()).unwrap();
    let text = optimized.root().to_string();
    assert_eq!(text.matches("(o_orderpriority#").count(), 2, "{text}");
    insta::assert_snapshot!(text);
}

/// A grouping-set column passed through `Expand` keeps whether its
/// distinct count comes from statistics, so `col = literal` above a
/// `ROLLUP` reads the count when the statistics have one and the default
/// otherwise.
#[test]
fn expanded_columns_keep_whether_their_distinct_count_is_known() {
    let ctx = support::ctx();
    let rows = |sql: &str, stats: &Fixture| {
        krishiv_optimizer::estimated_rows(support::resolve(sql, &ctx).root(), stats)
    };
    let all = "SELECT * FROM (SELECT l_returnflag, l_orderkey FROM lineitem \
               GROUP BY ROLLUP (l_returnflag, l_orderkey))";
    let eq = "SELECT * FROM (SELECT l_returnflag, l_orderkey FROM lineitem \
              GROUP BY ROLLUP (l_returnflag, l_orderkey)) WHERE l_returnflag = 'R'";
    let (known, footers) = (fixture(), without_distinct(&[]));
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b;
    // Three return flags in the statistics: a third of the rows.
    let k = rows(eq, &known) / rows(all, &known);
    assert!(close(k, 1.0 / 3.0), "{k}");
    // No distinct count: the default selectivity.
    let f = rows(eq, &footers) / rows(all, &footers);
    assert!(close(f, krishiv_optimizer::DEFAULT_SELECTIVITY), "{f}");
}

/// Whether `p` or anything under it is a union.
fn holds_union(p: &LogicalPlan) -> bool {
    if matches!(p.node, krishiv_plan::PlanNode::Union { .. }) {
        return true;
    }
    let mut found = false;
    p.for_each_child(&mut |c| found |= holds_union(c));
    found
}

/// Panics when a join builds (its right input) on a union.
fn no_union_build(p: &LogicalPlan, what: &str) {
    if let krishiv_plan::PlanNode::Join { right, .. } = &p.node {
        assert!(
            !holds_union(right),
            "{what}: a join builds on the union\n{p}"
        );
    }
    p.for_each_child(&mut |c| no_union_build(c, what));
}

/// TPC-DS q05's shape: sales and returns of one channel as a union, joined
/// to a filtered date dimension and a second dimension. The union's columns
/// carry their inputs' statistics (a key's distinct values, its range), so
/// the join on the date key is estimated as selective as it is and the
/// union stays on the probe side. Without them every union value counted
/// as distinct, each join with the union looked as small as the dimension,
/// and the 3M-row union side was built.
#[test]
fn a_union_keeps_its_column_statistics() {
    let sql = "SELECT count(*) \
               FROM (SELECT cs_sold_date_sk AS d, cs_bill_cdemo_sk AS c FROM catalog_sales \
                     UNION ALL SELECT cs_sold_date_sk, cs_bill_cdemo_sk FROM catalog_sales) s, \
                    date_dim, customer_demographics \
               WHERE s.d = d_date_sk AND d_year = 2000 AND s.c = cd_demo_sk AND cd_gender = 'F'";
    let plan = support::resolve(sql, &tpcds_ctx());
    let optimized = optimize_with(&plan, &TpcdsFooters).unwrap();
    no_union_build(optimized.root(), "q05 shape");
    insta::assert_snapshot!(optimized.root().to_string());
}

/// A composite key whose columns are also equal to a third input's (q09:
/// lineitem and partsupp on part and supplier keys, part and supplier on
/// each) is planned the same however its equalities are written, and the
/// plan is a fixed point: the key's selectivity sits on the same conjunct
/// whichever the query wrote first.
#[test]
fn composite_keys_over_a_class_do_not_depend_on_the_written_order() {
    let ctx = support::ctx();
    let written = support::tpch_sql(9);
    let swapped = written.replace(
        "AND ps_suppkey = l_suppkey\n        AND ps_partkey = l_partkey",
        "AND ps_partkey = l_partkey\n        AND ps_suppkey = l_suppkey",
    );
    assert_ne!(written, swapped);
    for stats in [fixture(), footers(1000)] {
        let plans: Vec<_> = [&written, &swapped]
            .into_iter()
            .map(|sql| {
                let plan = support::resolve(sql, &ctx);
                let optimized = optimize_with(&plan, &stats).unwrap();
                let again = optimize_with(&optimized, &stats).unwrap();
                assert!(
                    again.structurally_eq(&optimized),
                    "q09 is a fixed point:\n{}\n{}",
                    optimized.root(),
                    again.root()
                );
                // Conjuncts keep the order the query wrote them in.
                let text = optimized.root().to_string();
                let sorted = text.lines().map(|l| match l.split_once("Join Inner, ") {
                    Some((head, cond)) => {
                        let cond = cond.replace(['(', ')'], "");
                        let mut parts: Vec<&str> = cond.split(" and ").collect();
                        parts.sort_unstable();
                        format!("{head}Join Inner, {}", parts.join(" and "))
                    }
                    None => l.to_owned(),
                });
                sorted.collect::<Vec<_>>().join("\n")
            })
            .collect();
        assert_eq!(plans[0], plans[1]);
    }
}

/// The columns of each column equality in `e`'s conjuncts.
fn equalities(e: &Expr) -> Vec<[&BoundColumn; 2]> {
    fn col(a: &Expr) -> Option<&BoundColumn> {
        match &a.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c),
            _ => None,
        }
    }
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
        return Vec::new();
    };
    match (&f.id, f.args.as_slice()) {
        (FunctionId::Builtin(Builtin::And), args) => {
            args.iter().flat_map(|a| equalities(a)).collect()
        }
        (FunctionId::Builtin(Builtin::Eq), [a, b]) => match (col(a), col(b)) {
            (Some(a), Some(b)) => vec![[a, b]],
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// Panics on a join holding two equalities over one column (a key the
/// joins below already imply).
fn no_repeated_key(root: &LogicalPlan, what: &str) {
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        p.for_each_child(&mut |c| stack.push(c));
        if let PlanNode::Join {
            condition: JoinCondition::On(c),
            ..
        } = &p.node
        {
            let mut ids: Vec<_> = equalities(c).iter().flatten().map(|c| c.id).collect();
            let n = ids.len();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), n, "{what} repeats a join key:\n{root}");
        }
    }
}

/// Inputs the query relates only through a third one join on the implied
/// equality, each join holds one equality of the class, and nulls (which
/// `=` never matches) keep the rows the written plan keeps.
#[test]
fn implied_equalities_join_inputs_related_through_a_third() {
    let sql = "SELECT t1.b, t2.b, t3.b FROM t t1 JOIN t t2 ON t1.a = t2.a \
               JOIN t t3 ON t3.a = t1.a JOIN u ON u.a = t2.a";
    let optimized = same_rows(sql);
    no_cross_product(optimized.root(), sql);
    no_repeated_key(optimized.root(), sql);
    assert!(optimize(&optimized).unwrap().structurally_eq(&optimized));
    insta::assert_snapshot!(optimized.root().to_string());
}

/// Two columns of one input in a class: the input is filtered on their
/// equality, and the join takes one of them.
#[test]
fn a_class_with_two_columns_of_one_input_filters_that_input() {
    let sql = "SELECT t1.b, s.b FROM t t1 JOIN (SELECT a AS x, a AS y, b FROM t) s \
               ON t1.a = s.x AND t1.a = s.y";
    let optimized = same_rows(sql);
    no_repeated_key(optimized.root(), sql);
    assert!(optimize(&optimized).unwrap().structurally_eq(&optimized));
    let text = optimized.root().to_string();
    assert_eq!(text.matches(" == ").count(), 2, "{text}");
    insta::assert_snapshot!(text);
}

/// Nested columns form no class: two columns of one input equal to a third
/// stay two join keys, as written, instead of a key and a filter of that
/// input (the executor's `=` cannot compare a struct holding an array, which
/// its join keys can).
#[test]
fn nested_equalities_stay_as_written() {
    let ty = krishiv_types::parse_data_type("struct<x: array<int>>").unwrap();
    let col = |name: &str| krishiv_plan::analyzer::TableColumn {
        name: name.to_owned(),
        data_type: ty.clone(),
        nullable: true,
        char_varchar: None,
    };
    let cat = support::catalog()
        .with_table(&["default"], "n1", vec![col("a"), col("b")], None)
        .with_table(&["default"], "n2", vec![col("c")], None);
    let sql = "SELECT 1 FROM n1 t JOIN n2 u ON t.a = u.c AND t.b = u.c";
    let plan = support::resolve(sql, &support::ctx_over(cat));
    let optimized = optimize(&plan).unwrap();
    let text = optimized.root().to_string();
    assert!(!text.contains("Filter"), "{text}");
    assert_eq!(text.matches(" == ").count(), 2, "{text}");
}

/// A written equality the others imply is left out of the plan, and the
/// rows stay those of the written plan.
#[test]
fn a_written_equality_the_others_imply_is_left_out() {
    let sql = "SELECT t1.b FROM t t1 JOIN t t2 ON t1.a = t2.a JOIN t t3 \
               ON t2.a = t3.a AND t1.a = t3.a";
    let optimized = same_rows(sql);
    no_repeated_key(optimized.root(), sql);
    let joins = optimized.root().to_string().matches("Join Inner").count();
    let eqs = optimized.root().to_string().matches(" == ").count();
    assert_eq!((joins, eqs), (2, 2), "{}", optimized.root());
}

/// One class over many inputs (every input joined on the same column)
/// derives an equality per pair of inputs; planning stays fast, every join
/// holds one of them and the rows are the written plan's.
#[test]
fn a_class_over_many_inputs_plans_quickly() {
    let mut sql = "SELECT t1.b FROM t t1".to_owned();
    for i in 2..=30 {
        sql.push_str(&format!(" JOIN t t{i} ON t{i}.a = t{}.a", i - 1));
    }
    let plan = support::resolve(&sql, &support::ctx());
    let started = std::time::Instant::now();
    let optimized = optimize(&plan).unwrap();
    assert!(started.elapsed().as_secs() < 10, "{:?}", started.elapsed());
    no_cross_product(optimized.root(), "the class");
    no_repeated_key(optimized.root(), "the class");
    let eqs = optimized.root().to_string().matches(" == ").count();
    assert_eq!(eqs, 29, "{}", optimized.root());
}

/// The right side of the semi join restricting the aggregate in `text`
/// (the lines under its `Shift`), or `None` without one.
fn restriction_domain(text: &str) -> Option<String> {
    let start = text.lines().position(|l| l.contains("Join LeftSemi"))?;
    let lines: Vec<&str> = text.lines().skip(start + 1).collect();
    let shift = lines.iter().position(|l| l.contains("Shift +"))?;
    let indent = lines[shift].find("Shift").unwrap_or(0);
    let mut out = vec![lines[shift]];
    out.extend(lines[shift + 1..].iter().take_while(|l| {
        l.find(|c: char| c.is_alphanumeric())
            .is_some_and(|i| i > indent)
    }));
    Some(out.join("\n"))
}

/// An aggregate joined to `partsupp ⋈ filtered part` on partsupp's key is
/// restricted by the filtered parts alone: the domain follows the inner
/// join's `ps_partkey = p_partkey` into part, which scans far fewer rows
/// than partsupp joined to it (without that step the domain is the join,
/// reading all of partsupp again).
#[test]
fn a_restriction_domain_follows_an_inner_join_equality() {
    let sql = "SELECT x.ps_partkey, g.m FROM \
               (SELECT ps_partkey, ps_suppkey FROM partsupp JOIN part ON ps_partkey = p_partkey \
                WHERE p_size = 15 AND p_type LIKE '%BRASS') x \
               LEFT JOIN (SELECT l_partkey AS k, sum(l_quantity) AS m FROM lineitem GROUP BY l_partkey) g \
               ON g.k = x.ps_partkey";
    let ctx = support::ctx();
    let plan = support::resolve(sql, &ctx);
    let optimized = optimize_with(&plan, &fixture()).unwrap();
    let text = optimized.root().to_string();
    let domain = restriction_domain(&text).unwrap_or_else(|| panic!("not restricted:\n{text}"));
    assert!(domain.contains("p_partkey"), "{text}");
    assert!(
        !domain.contains("partsupp"),
        "the domain reads partsupp:\n{text}"
    );
    assert!(
        optimize_with(&optimized, &fixture())
            .unwrap()
            .structurally_eq(&optimized)
    );
    let db = data::tpch(7, 150);
    assert_eq!(
        eval::rows(&db, optimized.root()),
        eval::rows(&db, plan.root())
    );
}

/// Through an outer join the equality does not hold of every row (the
/// preserved side keeps its unmatched keys), so the domain never moves
/// across it.
#[test]
fn a_restriction_domain_does_not_follow_an_outer_join_equality() {
    let sql = "SELECT x.ps_partkey, g.m FROM \
               (SELECT ps_partkey, ps_suppkey FROM partsupp LEFT JOIN \
                (SELECT p_partkey FROM part WHERE p_size = 15 AND p_type LIKE '%BRASS') p \
                ON ps_partkey = p_partkey) x \
               LEFT JOIN (SELECT l_partkey AS k, sum(l_quantity) AS m FROM lineitem GROUP BY l_partkey) g \
               ON g.k = x.ps_partkey";
    let ctx = support::ctx();
    let plan = support::resolve(sql, &ctx);
    let optimized = optimize_with(&plan, &fixture()).unwrap();
    let text = optimized.root().to_string();
    if let Some(domain) = restriction_domain(&text) {
        assert!(
            !domain.contains("p_partkey"),
            "the domain crossed an outer join:\n{text}"
        );
    }
    let db = data::tpch(7, 150);
    assert_eq!(
        eval::rows(&db, optimized.root()),
        eval::rows(&db, plan.root())
    );
}

/// The scans of `text`, in plan order (a join's build side prints after its
/// probe side).
fn scans(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|l| l.split("Scan ").nth(1))
        .map(|s| s.split(' ').next().unwrap_or(""))
        .collect()
}

/// E5a (`research/perf-parity/05-q10-join-order-range-estimate.md` section 3,
/// variant v6): a one-day `o_orderdate` range keeps ~1/2,400 of orders at SF10.
/// Estimated per bound against the unnarrowed orders it was about a quarter,
/// and customer (1.5M rows) became the build side; estimated together the
/// filtered orders is the build, whichever way the bounds are written.
#[test]
fn a_two_sided_range_is_estimated_between_its_bounds() {
    let ctx = support::ctx();
    let sql = |a: &str, b: &str| {
        format!(
            "SELECT c_name, o_totalprice FROM customer JOIN orders ON c_custkey = o_custkey \
             WHERE {a} AND {b}"
        )
    };
    let (lo, hi) = (
        "o_orderdate >= DATE '1995-03-01'",
        "o_orderdate < DATE '1995-03-02'",
    );
    let mut plans = Vec::new();
    for q in [sql(lo, hi), sql(hi, lo)] {
        let optimized = optimize_with(&support::resolve(&q, &ctx), &fixture()).unwrap();
        let text = optimized.root().to_string();
        let order = scans(&text);
        assert_eq!(
            order,
            [
                "spark_catalog.default.customer",
                "spark_catalog.default.orders"
            ],
            "orders is the build side: {text}"
        );
        plans.push(order.join(","));
    }
    assert_eq!(plans[0], plans[1]);
}

/// The estimate of an input's conjuncts does not depend on their written order,
/// even where one changes what the next keeps (a null-rejecting conjunct before
/// a range; an `IN` list before a bound on the same column).
#[test]
fn conjunct_order_does_not_change_the_join_order() {
    let ctx = support::ctx();
    for (a, b) in [
        ("o_totalprice IS NOT NULL", "o_totalprice > 3000"),
        ("o_orderkey IN (1, 2, 3)", "o_orderkey > 2"),
        (
            "o_orderdate >= DATE '1995-03-01'",
            "o_orderdate < DATE '1995-03-02'",
        ),
    ] {
        let plan = |x: &str, y: &str| {
            let q = format!(
                "SELECT c_name FROM customer JOIN orders ON c_custkey = o_custkey \
                 JOIN nation ON c_nationkey = n_nationkey WHERE {x} AND {y}"
            );
            let optimized = optimize_with(&support::resolve(&q, &ctx), &fixture()).unwrap();
            scans(&optimized.root().to_string())
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        };
        assert_eq!(plan(a, b), plan(b, a), "{a} AND {b}");
    }
}
