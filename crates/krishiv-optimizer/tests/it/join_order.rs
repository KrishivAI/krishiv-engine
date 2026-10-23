// SPDX-License-Identifier: Apache-2.0
//! Join ordering from TPC-H SF10 statistics: plan snapshots, no cross
//! products, greedy ordering of large join trees, and the same rows as
//! the unoptimised plans.

use std::collections::HashMap;

use krishiv_optimizer::{ColumnStats, Statistics, TableStats, optimize, optimize_with};
use krishiv_plan::{
    Builtin, Expr, ExprNode, FunctionId, JoinCondition, Literal, LogicalPlan, PlanNode, ReadSource,
    ResolvedExpr, Scan,
};
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
    // An equality between two columns of one input keeps `1 / ndv`.
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
    let cols = "SELECT l_orderkey FROM lineitem WHERE l_commitdate = l_receiptdate";
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
