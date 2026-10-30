// SPDX-License-Identifier: Apache-2.0
//! Statistics propagation (`PlanStats`): what each operator leaves known
//! about its columns. One test per rule, over a table whose statistics are
//! chosen so the expected numbers can be worked by hand.
//!
//! `t(a, b, s, x, d)` has 1,000 rows. `a` is an int in `0..=99` with 100
//! distinct values and 10% nulls; `b` an int in `1..=10`; `x` a bigint in
//! `0..=499`. `u(k, v)` has 200 rows, `k` in `50..=149` with 40 distinct
//! values; `w(wk, wv)` has 20 rows, `wk` in `50..=149`; `g(ga)` has no
//! statistics.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_optimizer::{
    AttrStats, ColumnStats, Options, PlanStats, StatSource, Statistics, TableStats, estimated_rows,
    optimize_with, optimize_with_options,
};
use krishiv_plan::analyzer::{AnalysisContext, MemoryCatalog, TableColumn};
use krishiv_plan::{AttrId, ExprNode, Literal, LogicalPlan, Plan, PlanNode, ReadSource, Scan};
use krishiv_types::{DataType, DecimalType};

use crate::support;

fn column(name: &str, t: DataType) -> TableColumn {
    TableColumn {
        name: name.to_owned(),
        data_type: t,
        nullable: true,
    }
}

fn ctx() -> AnalysisContext {
    let t = vec![
        column("a", DataType::Integer),
        column("b", DataType::Integer),
        column("s", DataType::String),
        column("x", DataType::Long),
        column("d", DataType::Date),
        column("n", DataType::Decimal(DecimalType::try_new(10, 2).unwrap())),
    ];
    let catalog = MemoryCatalog::new("spark_catalog")
        .with_table(&["default"], "t", t, None)
        .with_table(
            &["default"],
            "u",
            vec![
                column("k", DataType::Integer),
                column("v", DataType::Integer),
            ],
            None,
        )
        .with_table(
            &["default"],
            "w",
            vec![
                column("wk", DataType::Integer),
                column("wv", DataType::Integer),
            ],
            None,
        )
        .with_table(
            &["default"],
            "g",
            vec![column("ga", DataType::Integer)],
            None,
        )
        .with_table(
            &["default"],
            "g2",
            vec![column("gb", DataType::Integer)],
            None,
        );
    support::ctx_over(catalog)
}

fn col(distinct: u64, nulls: u64, min: i32, max: i32) -> ColumnStats {
    ColumnStats {
        distinct: Some(distinct),
        nulls: Some(nulls),
        min: Some(Literal::Integer(min)),
        max: Some(Literal::Integer(max)),
        ..ColumnStats::default()
    }
}

/// Statistics by table name.
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

fn fixture() -> Fixture {
    let t = TableStats {
        rows: 1000,
        columns: HashMap::from([
            ("a".to_owned(), col(100, 100, 0, 99)),
            ("b".to_owned(), col(10, 0, 1, 10)),
            (
                "s".to_owned(),
                ColumnStats {
                    distinct: Some(50),
                    nulls: Some(0),
                    ..ColumnStats::default()
                },
            ),
            (
                "x".to_owned(),
                ColumnStats {
                    distinct: Some(500),
                    nulls: Some(0),
                    min: Some(Literal::Long(0)),
                    max: Some(Literal::Long(499)),
                    ..ColumnStats::default()
                },
            ),
            (
                "d".to_owned(),
                ColumnStats {
                    distinct: Some(1000),
                    nulls: Some(0),
                    min: Some(Literal::Date(18_000)),
                    max: Some(Literal::Date(19_000)),
                    ..ColumnStats::default()
                },
            ),
        ]),
    };
    let u = TableStats {
        rows: 200,
        columns: HashMap::from([
            (
                "k".to_owned(),
                ColumnStats {
                    source: StatSource::IntegralRange,
                    ..col(40, 20, 50, 149)
                },
            ),
            ("v".to_owned(), col(20, 0, 0, 19)),
        ]),
    };
    let w = TableStats {
        rows: 20,
        columns: HashMap::from([
            ("wk".to_owned(), col(20, 0, 50, 149)),
            ("wv".to_owned(), col(20, 0, 0, 19)),
        ]),
    };
    Fixture(HashMap::from([
        ("t".to_owned(), t),
        ("u".to_owned(), u),
        ("w".to_owned(), w),
    ]))
}

fn resolve(sql: &str) -> Plan {
    support::resolve(sql, &ctx())
}

/// The first relation of `plan` (parents first) that `is` accepts.
fn find(plan: &Arc<LogicalPlan>, is: impl Fn(&LogicalPlan) -> bool) -> Arc<LogicalPlan> {
    let mut stack = vec![Arc::clone(plan)];
    while let Some(p) = stack.pop() {
        if is(&p) {
            return p;
        }
        let mut children = Vec::new();
        p.for_each_child(&mut |c| children.push(Arc::clone(c)));
        stack.extend(children.into_iter().rev());
    }
    panic!("no such relation in\n{plan}");
}

fn attr(node: &LogicalPlan, name: &str) -> AttrId {
    node.resolved_output()
        .unwrap()
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no column {name} in\n{node}"))
        .id
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-6 * want.abs().max(1.0),
        "{what}: {got} vs {want}"
    );
}

/// The estimate of a relation of `sql`'s plan, and the relation.
struct Estimated {
    stats: PlanStats,
    node: Arc<LogicalPlan>,
}

impl Estimated {
    fn of(sql: &str, is: impl Fn(&LogicalPlan) -> bool) -> Self {
        let plan = resolve(sql);
        let stats = PlanStats::compute(plan.root(), &fixture());
        let node = find(plan.root(), is);
        Self { stats, node }
    }

    fn rows(&self) -> f64 {
        self.stats.get(&self.node).unwrap().rows
    }

    fn col(&self, name: &str) -> AttrStats {
        let id = attr(&self.node, name);
        self.stats.get(&self.node).unwrap().cols[&id].clone()
    }
}

fn filter(sql: &str) -> Estimated {
    Estimated::of(sql, |p| matches!(p.node, PlanNode::Filter { .. }))
}

fn join(sql: &str) -> Estimated {
    Estimated::of(sql, |p| matches!(p.node, PlanNode::Join { .. }))
}

#[test]
fn a_filter_narrows_the_range_and_a_later_predicate_reads_it() {
    let e = filter("SELECT a, b FROM t WHERE a > 49");
    close(e.rows(), 1000.0 * 50.0 / 99.0, "rows");
    let a = e.col("a");
    assert_eq!((a.min, a.max), (Some(49.0), Some(99.0)));
    // `a < 60` reads `[49, 99]`: 11 of its 50 values, not 60 of 99.
    let e = filter("SELECT a FROM t WHERE a > 49 AND a < 60");
    close(e.rows(), 1000.0 * 50.0 / 99.0 * 11.0 / 50.0, "conjunction");
    let a = e.col("a");
    assert_eq!((a.min, a.max), (Some(49.0), Some(60.0)));
    // Across two filters, through the projection between them.
    let e = Estimated::of(
        "SELECT * FROM (SELECT a FROM t WHERE a > 49) q WHERE a < 60",
        |p| matches!(&p.node, PlanNode::Filter { input, .. } if !matches!(input.node, PlanNode::Scan(_))),
    );
    close(
        e.rows(),
        1000.0 * 50.0 / 99.0 * 11.0 / 50.0,
        "stacked filters",
    );
    let a = e.col("a");
    assert_eq!((a.min, a.max), (Some(49.0), Some(60.0)));
    // A range predicate with no literal bound narrows nothing it cannot.
    let e = filter("SELECT a FROM t WHERE a > b");
    let a = e.col("a");
    assert_eq!((a.min, a.max), (Some(0.0), Some(99.0)));
    assert_eq!(a.null_frac, 0.0, "a comparison rejects NULL");
}

#[test]
fn an_equality_sets_the_range_to_the_value() {
    let e = filter("SELECT a FROM t WHERE a = 7");
    close(e.rows(), 10.0, "rows");
    let a = e.col("a");
    assert_eq!((a.min, a.max, a.null_frac), (Some(7.0), Some(7.0), 0.0));
    close(a.ndv, 1.0, "ndv");
}

#[test]
fn an_equality_outside_the_range_estimates_no_rows_and_removes_nothing() {
    let e = filter("SELECT a FROM t WHERE a = 500");
    assert_eq!(e.rows(), 0.0);
    let project = Estimated::of("SELECT a FROM t WHERE a = 500", |p| {
        matches!(p.node, PlanNode::Project { .. })
    });
    assert_eq!(project.rows(), 0.0, "the zero reaches the nodes above");
    // Within the range is an ordinary equality.
    close(filter("SELECT a FROM t WHERE a = 99").rows(), 10.0, "edge");
    close(filter("SELECT a FROM t WHERE a = 0").rows(), 10.0, "edge");

    // Nothing is pruned: the plan keeps its filter and its scan, with
    // propagation on or off, and costing still reads one row.
    let plan = resolve("SELECT a FROM t WHERE a = 500");
    let stats = fixture();
    let on = optimize_with(&plan, &stats).unwrap();
    let off = optimize_with_options(
        &plan,
        &stats,
        &Options {
            stats_propagation: false,
            ..Options::default()
        },
    )
    .unwrap();
    let text = on.root().to_string();
    assert!(text.contains("Filter") && text.contains("Scan"), "{text}");
    assert!(text.contains("== 500"), "{text}");
    assert!(on.structurally_eq(&off), "{text}\n{}", off.root());
    assert_eq!(estimated_rows(on.root(), &stats), 1.0);
    // A string equality has no range to be outside of.
    assert!(filter("SELECT s FROM t WHERE s = 'zz'").rows() > 0.0);
}

#[test]
fn null_rejecting_predicates_clear_the_null_fraction() {
    let null_frac = |pred: &str| {
        filter(&format!("SELECT a FROM t WHERE {pred}"))
            .col("a")
            .null_frac
    };
    for rejecting in [
        "a > 5",
        "a <= 5",
        "a = 5",
        "a <=> 5",
        "a BETWEEN 1 AND 5",
        "a IN (1, 2)",
        "a IS NOT NULL",
        "a = b",
        "a < b",
    ] {
        assert_eq!(null_frac(rejecting), 0.0, "{rejecting}");
    }
    // What does not reject NULL leaves the fraction, and so does an
    // unknown predicate or a negation (their rows may be null).
    for keeping in [
        "a IS NULL",
        "a <=> b",
        "NOT (a > 5)",
        "a % 2 = 0",
        "b = 3",
        "a < 5 OR b = 3",
    ] {
        close(null_frac(keeping), 0.1, keeping);
    }
    // A disjunction rejects NULL only if every branch does.
    assert_eq!(null_frac("a < 5 OR a > 90"), 0.0);
}

#[test]
fn a_disjunction_takes_the_union_of_its_branches_ranges() {
    let e = filter("SELECT a FROM t WHERE a BETWEEN 10 AND 20 OR a BETWEEN 50 AND 60");
    let (a, sel) = (e.col("a"), 10.0 / 99.0);
    assert_eq!((a.min, a.max), (Some(10.0), Some(60.0)));
    close(e.rows(), 1000.0 * (2.0 * sel - sel * sel), "rows");
    // The branches' distinct values add (each keeps `sel` of 100).
    close(a.ndv, 200.0 * sel, "ndv");
    // A branch that says nothing of `a` leaves its whole range.
    let e = filter("SELECT a FROM t WHERE a < 10 OR b = 3");
    let a = e.col("a");
    assert_eq!((a.min, a.max), (Some(0.0), Some(99.0)));
}

#[test]
fn negation_and_unknown_predicates_leave_the_ranges() {
    let e = filter("SELECT a FROM t WHERE NOT (a > 50)");
    let a = e.col("a");
    assert_eq!((a.min, a.max), (Some(0.0), Some(99.0)));
    close(e.rows(), 1000.0 * 50.0 / 99.0, "rows");
    let e = filter("SELECT a FROM t WHERE a % 2 = 0");
    let a = e.col("a");
    assert_eq!((a.min, a.max), (Some(0.0), Some(99.0)));
    close(e.rows(), 200.0, "the default selectivity");
    // Narrowing is not undone by a negation elsewhere in the conjunction.
    let e = filter("SELECT a FROM t WHERE a > 49 AND NOT (b = 3)");
    let (a, b) = (e.col("a"), e.col("b"));
    assert_eq!(a.min, Some(49.0));
    assert_eq!((b.min, b.max), (Some(1.0), Some(10.0)));
}

#[test]
fn distinct_counts_follow_the_filter_rules() {
    let ndv = |pred: &str, column: &str| {
        let e = filter(&format!("SELECT a, b, s, x FROM t WHERE {pred}"));
        (e.rows(), e.col(column).ndv)
    };
    // `col = lit`: one.
    assert_eq!(ndv("a = 7", "a").1, 1.0);
    // `IN (k literals)`: `min(k, ndv)`.
    assert_eq!(ndv("a IN (1, 2, 3)", "a").1, 3.0);
    assert_eq!(
        ndv("b IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)", "b").1,
        10.0
    );
    // Duplicates count once; an element that is not a literal can hold any
    // value of its column: capped by the 20 surviving rows, not by the
    // list length.
    assert_eq!(ndv("a IN (1, 1, 1)", "a").1, 1.0);
    assert_eq!(ndv("a IN (b, 2)", "a").1, 20.0);
    // Duplicates count once; an element that is not a literal can hold any
    // value of its column: capped by the 20 surviving rows, not by the
    // list length.
    assert_eq!(ndv("a IN (1, 1, 1)", "a").1, 1.0);
    assert_eq!(ndv("a IN (b, 2)", "a").1, 20.0);
    // `IS NOT NULL` keeps it.
    assert_eq!(ndv("a IS NOT NULL", "a").1, 100.0);
    // A range predicate scales it by its selectivity, at least 1.
    close(ndv("a > 49", "a").1, 100.0 * 50.0 / 99.0, "range");
    close(ndv("a > 98", "a").1, 100.0 / 99.0, "narrow range");
    assert_eq!(ndv("a > 99", "a").1, 1.0, "never below 1");
    // Every other column scales by the fraction of rows kept (at least 1,
    // at most the rows).
    let (rows, x) = ndv("a = 7", "x");
    close(rows, 10.0, "rows");
    close(x, 5.0, "x scales by 1/100");
    assert_eq!(
        ndv("a = 7", "s").1,
        1.0,
        "50 values scaled by 1/100: at least 1"
    );
    let (rows, x) = ndv("a > 98", "x");
    assert!(x <= rows, "{x} {rows}");
    // Other predicates keep `min(ndv, rows_out)` for the column they read.
    assert_eq!(ndv("a % 2 = 0", "a").1, 100.0);
    assert_eq!(ndv("x % 2 = 0", "x").1, 200.0);
}

#[test]
fn predicates_the_estimator_cannot_read_use_the_defaults() {
    let rows = |pred: &str| filter(&format!("SELECT a FROM t WHERE {pred}")).rows();
    // An expression, a pattern and a range over an expression are the
    // defaults (a third for each side of a range).
    for (pred, want) in [
        ("a + 1 BETWEEN 1 AND 2", 1000.0 / 9.0),
        ("a + 1 IN (1, 2)", 200.0),
        ("a + 1 IS NULL", 200.0),
        ("a + 1 IS NOT NULL", 800.0),
        ("s LIKE 'x%'", 200.0),
        // A list of two over 100 distinct values.
        ("a IN (b, 2)", 20.0),
        ("true", 1000.0),
        // A constant false is empty.
        ("false", 0.0),
    ] {
        close(rows(pred), want, pred);
    }
}

#[test]
fn an_inner_join_narrows_its_keys_and_scales_the_other_columns() {
    let e = join("SELECT * FROM t JOIN u ON t.a = u.k");
    // `1 / max(100, 40)` of the 200,000 pairs.
    close(e.rows(), 2000.0, "rows");
    for key in ["a", "k"] {
        let c = e.col(key);
        close(c.ndv, 40.0, "key ndv is the smaller side's");
        assert_eq!(
            (c.min, c.max),
            (Some(50.0), Some(99.0)),
            "{key}: ranges intersect"
        );
        assert_eq!(c.null_frac, 0.0, "{key}: an inner join's keys are not null");
    }
    // 40 of `t.a`'s 100 values find a match: the other columns of `t`
    // keep that fraction of their distinct values; `u` keeps all of its.
    close(e.col("b").ndv, 4.0, "b");
    close(e.col("x").ndv, 200.0, "x");
    close(e.col("v").ndv, 20.0, "v");
    assert_eq!(e.col("v").max, Some(19.0), "other ranges stay");
}

#[test]
fn outer_joins_give_the_null_supplying_side_the_unmatched_fraction_as_nulls() {
    // 200 of `t`'s 1,000 rows match.
    let e = join("SELECT * FROM t LEFT JOIN w ON t.a = w.wk");
    close(e.rows(), 1000.0, "rows");
    for c in ["wk", "wv"] {
        close(e.col(c).null_frac, 0.8, c);
    }
    // The preserved side keeps its statistics, keys included.
    let a = e.col("a");
    assert_eq!((a.min, a.max), (Some(0.0), Some(99.0)));
    close(a.null_frac, 0.1, "a");
    close(a.ndv, 100.0, "a ndv");
    // The supplied key's values all come from matches.
    let wk = e.col("wk");
    assert_eq!((wk.min, wk.max), (Some(50.0), Some(99.0)));
    // A right join supplies the left side; here every right row matches
    // many left rows, so nothing is unmatched.
    let e = join("SELECT * FROM t RIGHT JOIN w ON t.a = w.wk");
    close(e.rows(), 200.0, "right rows");
    assert_eq!(e.col("wv").null_frac, 0.0);
    // A full join supplies both sides.
    let e = join("SELECT * FROM t FULL JOIN w ON t.a = w.wk");
    close(e.rows(), 1000.0, "full rows");
    close(e.col("wv").null_frac, 0.8, "full: w");
    close(e.col("b").null_frac, 0.0, "full: t (no unmatched w rows)");
}

#[test]
fn semi_and_anti_joins_keep_the_left_side() {
    let e = join("SELECT * FROM t LEFT SEMI JOIN w ON t.a = w.wk");
    close(e.rows(), 200.0, "semi rows");
    let a = e.col("a");
    assert_eq!((a.min, a.max, a.null_frac), (Some(50.0), Some(99.0), 0.0));
    let stats = e.stats.get(&e.node).unwrap();
    assert_eq!(stats.cols.len(), 6, "only the left columns");
    let e = join("SELECT * FROM t LEFT ANTI JOIN w ON t.a = w.wk");
    close(e.rows(), 800.0, "anti rows");
    // The rows that find no match can be anywhere in the range.
    let a = e.col("a");
    assert_eq!((a.min, a.max), (Some(0.0), Some(99.0)));
    close(a.null_frac, 0.1, "a");
}

#[test]
fn an_aggregate_keeps_its_keys_and_bounds_min_max_and_count() {
    let agg = |sql: &str| Estimated::of(sql, |p| matches!(p.node, PlanNode::Aggregate { .. }));
    let e =
        agg("SELECT b, count(*) AS c, min(a) AS lo, max(a) AS hi, sum(x) AS sx FROM t GROUP BY b");
    close(e.rows(), 10.0, "rows are the key's distinct values");
    let b = e.col("b");
    assert_eq!((b.min, b.max, b.ndv), (Some(1.0), Some(10.0), 10.0));
    assert_eq!(b.source, StatSource::Provided);
    for (name, lo, hi) in [("lo", 0.0, 99.0), ("hi", 0.0, 99.0)] {
        let c = e.col(name);
        assert_eq!((c.min, c.max), (Some(lo), Some(hi)), "{name}");
        assert!(!c.ndv_known(), "{name}: no distinct count");
    }
    let c = e.col("c");
    assert_eq!(
        (c.min, c.max),
        (Some(0.0), Some(1000.0)),
        "count: 0 ..= input rows"
    );
    let sx = e.col("sx");
    assert_eq!((sx.min, sx.max), (None, None), "sum: nothing known");
    // Keys multiply, at most the input's rows.
    close(
        agg("SELECT a, b FROM t GROUP BY a, b").rows(),
        1000.0,
        "100 * 10",
    );
    close(
        agg("SELECT a, x FROM t GROUP BY a, x").rows(),
        1000.0,
        "clamped to rows",
    );
    close(
        agg("SELECT count(*) FROM t").rows(),
        1.0,
        "global aggregate",
    );
}

#[test]
fn projections_keep_what_a_column_or_a_safe_cast_keeps() {
    let e = Estimated::of(
        "SELECT a AS pa, CAST(a AS BIGINT) AS big, CAST(a AS DOUBLE) AS dbl, \
         CAST(a AS DECIMAL(12, 2)) AS dec, CAST(a AS STRING) AS str, \
         CAST(x AS INT) AS narrow, CAST(x AS DOUBLE) AS lossy, a + 1 AS plus, \
         CAST(a AS DECIMAL(9, 2)) AS tight, CAST(d AS TIMESTAMP) AS ts FROM t",
        |p| matches!(p.node, PlanNode::Project { .. }),
    );
    let bounds = |name: &str| {
        let c = e.col(name);
        (c.ndv, c.null_frac, c.min, c.max)
    };
    let kept = (100.0, 0.1, Some(0.0), Some(99.0));
    // A column, and the casts that are injective, never raise and keep
    // order (int to bigint, to double, into a decimal with room): all of
    // distinct count, null fraction and bounds.
    for name in ["pa", "big", "dbl", "dec"] {
        assert_eq!(bounds(name), kept, "{name}");
    }
    // Injective and non-raising but not order-preserving: no bounds.
    assert_eq!(bounds("str"), (100.0, 0.1, None, None));
    // Date to timestamp keeps order; the bounds are microseconds, widened
    // by a day for the time zone.
    let ts = e.col("ts");
    let day = 86_400_000_000.0;
    assert_eq!(ts.min, Some(17_999.0 * day));
    assert_eq!(ts.max, Some(19_001.0 * day));
    assert_eq!(ts.ndv, 1000.0);
    // Not injective (`bigint` to `double`), may raise (`bigint` to `int`),
    // may overflow (`decimal(9, 2)` cannot hold an int) or an expression:
    // none of it carried.
    for name in ["narrow", "lossy", "tight", "plus"] {
        let c = e.col(name);
        assert!(!c.ndv_known(), "{name}");
        assert_eq!((c.null_frac, c.min, c.max), (0.0, None, None), "{name}");
    }
}

#[test]
fn a_union_adds_rows_takes_the_most_distinct_counts_and_widens_ranges() {
    let e = Estimated::of("SELECT a FROM t UNION ALL SELECT k FROM u", |p| {
        matches!(p.node, PlanNode::Union { .. })
    });
    close(e.rows(), 1200.0, "rows");
    let a = e.col("a");
    // The more of 100 and 40 distinct values (branches share a key
    // domain); the bounds of both; nulls (100 + 20) / 1200.
    close(a.ndv, 100.0, "ndv");
    assert_eq!((a.min, a.max), (Some(0.0), Some(149.0)));
    close(a.null_frac, 120.0 / 1200.0, "nulls");
    // The weakest source: `u.k`'s bounded range.
    assert_eq!(a.source, StatSource::IntegralRange);
    let e = Estimated::of("SELECT b FROM t UNION ALL SELECT v FROM u", |p| {
        matches!(p.node, PlanNode::Union { .. })
    });
    close(e.col("b").ndv, 20.0, "max(10, 20)");
}

#[test]
fn an_unknown_distinct_count_reads_as_the_rows_in_a_join_selectivity() {
    // Neither side has one: each reads as its rows, `1 / max(rows)` (T166
    // note, section 2); only `col = literal` takes the default selectivity.
    let e = join("SELECT * FROM g JOIN g2 ON g.ga = g2.gb");
    close(e.rows(), 1e6, "no counts");
    // One side has one: the other reads as its rows, which is larger.
    let e = join("SELECT * FROM t JOIN g ON t.a = g.ga");
    close(e.rows(), 1000.0, "one count");
    // The weakest source wins on the key: a guess on either side.
    for key in ["a", "ga"] {
        assert!(!e.col(key).ndv_known(), "{key}");
    }
    // A count bounded by an integral range is a count.
    let e = join("SELECT * FROM u JOIN w ON u.k = w.wk");
    close(e.rows(), 200.0 * 20.0 / 40.0, "range counts");
    for key in ["k", "wk"] {
        assert_eq!(e.col(key).source, StatSource::IntegralRange, "{key}");
    }
    // A table without statistics has default rows and guessed columns.
    let e = Estimated::of("SELECT ga FROM g", |p| matches!(p.node, PlanNode::Scan(_)));
    close(e.rows(), 1e6, "default rows");
    assert_eq!(e.col("ga").source, StatSource::Guess);
    assert!(!e.stats.has_statistics(&e.node));
}

#[test]
fn only_plans_over_tables_with_statistics_are_measured() {
    let plan = resolve("SELECT * FROM t JOIN u ON t.a = u.k");
    let stats = PlanStats::compute(plan.root(), &fixture());
    assert!(stats.has_statistics(plan.root()));
    let plan = resolve("SELECT * FROM t JOIN g ON t.a = g.ga");
    let stats = PlanStats::compute(plan.root(), &fixture());
    assert!(!stats.has_statistics(plan.root()));
    let t = find(
        plan.root(),
        |p| matches!(&p.node, PlanNode::Scan(s) if s.output.iter().any(|c| c.name == "a")),
    );
    assert!(stats.has_statistics(&t));
    // A relation of another plan is not held.
    let other = resolve("SELECT * FROM t");
    assert!(!stats.has_statistics(other.root()));
    assert!(stats.get(other.root()).is_none());
    assert_eq!(stats.rows(other.root()), krishiv_optimizer::DEFAULT_ROWS);
    // A `WITH` reference counts as default-sized.
    let plan = resolve("WITH c AS (SELECT a FROM t) SELECT * FROM c");
    let stats = PlanStats::compute(plan.root(), &fixture());
    let reference = find(plan.root(), |p| matches!(p.node, PlanNode::CteRef(_)));
    assert!(!stats.has_statistics(&reference));
}

/// Every relation is estimated once, shared subtrees and subquery plans
/// included, and asking again estimates nothing more.
#[test]
fn each_relation_is_estimated_once() {
    fn relations(plan: &Arc<LogicalPlan>) -> usize {
        let mut seen = HashSet::new();
        let mut stack = vec![Arc::clone(plan)];
        while let Some(p) = stack.pop() {
            if !seen.insert(Arc::as_ptr(&p)) {
                continue;
            }
            p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
            let mut exprs: Vec<Arc<krishiv_plan::Expr>> = Vec::new();
            p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
            while let Some(e) = exprs.pop() {
                if let ExprNode::Subquery { plan, .. } = &e.node {
                    stack.push(Arc::clone(plan));
                }
                e.for_each_child(&mut |c| exprs.push(Arc::clone(c)));
            }
        }
        seen.len()
    }
    for sql in [
        "SELECT a FROM t JOIN u ON t.a = u.k WHERE b > 2",
        "SELECT a FROM t WHERE a > (SELECT max(k) FROM u WHERE v > 3)",
        "WITH c AS (SELECT a FROM t) SELECT * FROM c c1 JOIN c c2 ON c1.a = c2.a",
        "SELECT a FROM t UNION ALL SELECT a FROM t",
    ] {
        let plan = resolve(sql);
        let stats = PlanStats::compute(plan.root(), &fixture());
        assert_eq!(stats.computed(), relations(plan.root()), "{sql}");
        assert_eq!(stats.len(), stats.computed(), "{sql}");
        assert!(!stats.is_empty());
        for _ in 0..3 {
            let _ = stats.rows(plan.root());
            let _ = stats.has_statistics(plan.root());
        }
        assert_eq!(stats.computed(), relations(plan.root()), "{sql}: lookups");
    }
    // The subquery's plan has its own estimate.
    let plan = resolve("SELECT a FROM t WHERE a > (SELECT max(k) FROM u WHERE v > 3)");
    let stats = PlanStats::compute(plan.root(), &fixture());
    let mut found = false;
    let filter = find(plan.root(), |p| matches!(p.node, PlanNode::Filter { .. }));
    filter.for_each_expr(&mut |e| {
        let mut stack = vec![Arc::clone(e)];
        while let Some(e) = stack.pop() {
            if let ExprNode::Subquery { plan, .. } = &e.node {
                found = true;
                assert!(stats.get(plan).is_some());
            }
            e.for_each_child(&mut |c| stack.push(Arc::clone(c)));
        }
    });
    assert!(found, "the query has a subquery");
}

/// `EXPLAIN COST` text: the size, the row count, and Spark's formats.
#[test]
fn statistics_print_as_spark_prints_them() {
    let plan = resolve("SELECT a, s FROM t");
    let stats = PlanStats::compute(plan.root(), &fixture());
    // 8 bytes per row plus 4 (int) and 20 (string); 1,000 rows.
    assert_eq!(
        stats.describe(plan.root()).unwrap(),
        "Statistics(sizeInBytes=31.3 KiB, rowCount=1.00E+3)"
    );
    let plan = resolve("SELECT a FROM t WHERE a = 500");
    let stats = PlanStats::compute(plan.root(), &fixture());
    assert_eq!(
        stats.describe(plan.root()).unwrap(),
        "Statistics(sizeInBytes=1.0 B, rowCount=0)"
    );
    let plan = resolve("SELECT count(*) AS c FROM t");
    let stats = PlanStats::compute(plan.root(), &fixture());
    assert_eq!(
        stats.describe(plan.root()).unwrap(),
        "Statistics(sizeInBytes=16.0 B, rowCount=1)"
    );
    assert!(stats.describe(resolve("SELECT 1").root()).is_none());
}

mod ranges {
    //! Propagation only ever widens or keeps: the narrowed range holds
    //! every value of the rows that survive, on generated data.

    use proptest::prelude::*;

    use super::*;

    #[derive(Debug, Clone)]
    enum P {
        Cmp(&'static str, i32),
        Between(i32, i32),
        In(Vec<i32>),
        NotNull,
        IsNull,
        And(Box<P>, Box<P>),
        Or(Box<P>, Box<P>),
        Not(Box<P>),
    }

    impl P {
        fn sql(&self) -> String {
            match self {
                Self::Cmp(op, v) => format!("a {op} {v}"),
                Self::Between(lo, hi) => format!("a BETWEEN {lo} AND {hi}"),
                Self::In(vs) => {
                    let list: Vec<String> = vs.iter().map(ToString::to_string).collect();
                    format!("a IN ({})", list.join(", "))
                }
                Self::NotNull => "a IS NOT NULL".to_owned(),
                Self::IsNull => "a IS NULL".to_owned(),
                Self::And(x, y) => format!("(({}) AND ({}))", x.sql(), y.sql()),
                Self::Or(x, y) => format!("(({}) OR ({}))", x.sql(), y.sql()),
                Self::Not(x) => format!("(NOT ({}))", x.sql()),
            }
        }

        /// SQL three-valued logic.
        fn eval(&self, a: Option<i32>) -> Option<bool> {
            match self {
                Self::Cmp(op, v) => a.map(|a| match *op {
                    "=" => a == *v,
                    "<" => a < *v,
                    "<=" => a <= *v,
                    ">" => a > *v,
                    _ => a >= *v,
                }),
                Self::Between(lo, hi) => a.map(|a| *lo <= a && a <= *hi),
                Self::In(vs) => a.map(|a| vs.contains(&a)),
                Self::NotNull => Some(a.is_some()),
                Self::IsNull => Some(a.is_none()),
                Self::And(x, y) => match (x.eval(a), y.eval(a)) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
                Self::Or(x, y) => match (x.eval(a), y.eval(a)) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                },
                Self::Not(x) => x.eval(a).map(|v| !v),
            }
        }
    }

    fn predicate() -> impl Strategy<Value = P> {
        let value = 0..30;
        let leaf = prop_oneof![
            (
                prop_oneof![Just("="), Just("<"), Just("<="), Just(">"), Just(">=")],
                value.clone()
            )
                .prop_map(|(op, v)| P::Cmp(op, v)),
            (value.clone(), value.clone()).prop_map(|(a, b)| P::Between(a, b)),
            prop::collection::vec(value, 1..5).prop_map(P::In),
            Just(P::NotNull),
            Just(P::IsNull),
        ];
        leaf.prop_recursive(3, 12, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(x, y)| P::And(Box::new(x), Box::new(y))),
                (inner.clone(), inner.clone()).prop_map(|(x, y)| P::Or(Box::new(x), Box::new(y))),
                inner.prop_map(|x| P::Not(Box::new(x))),
            ]
        })
    }

    /// The statistics a column of `values` would report.
    fn table_stats(values: &[Option<i32>]) -> TableStats {
        let present: Vec<i32> = values.iter().flatten().copied().collect();
        let distinct = present.iter().collect::<HashSet<_>>().len();
        let stats = ColumnStats {
            distinct: Some(distinct as u64),
            nulls: Some((values.len() - present.len()) as u64),
            min: present.iter().min().map(|v| Literal::Integer(*v)),
            max: present.iter().max().map(|v| Literal::Integer(*v)),
            ..ColumnStats::default()
        };
        TableStats {
            rows: values.len() as u64,
            columns: HashMap::from([("a".to_owned(), stats)]),
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn the_narrowed_range_holds_every_surviving_value(
            values in prop::collection::vec(prop::option::weighted(0.8, 0..30), 1..60),
            p in predicate(),
        ) {
            let ctx = support::ctx_over(
                MemoryCatalog::new("spark_catalog").with_table(
                    &["default"],
                    "r",
                    vec![column("a", DataType::Integer)],
                    None,
                ),
            );
            let plan = support::resolve(&format!("SELECT a FROM r WHERE {}", p.sql()), &ctx);
            let stats = Fixture(HashMap::from([("r".to_owned(), table_stats(&values))]));
            let planned = PlanStats::compute(plan.root(), &stats);
            let filter = find(plan.root(), |n| matches!(n.node, PlanNode::Filter { .. }));
            let est = planned.get(&filter).unwrap();
            let a = &est.cols[&attr(&filter, "a")];
            let survivors: Vec<Option<i32>> =
                values.iter().copied().filter(|v| p.eval(*v) == Some(true)).collect();
            let present: Vec<f64> = survivors.iter().flatten().map(|v| f64::from(*v)).collect();
            if let (Some(lo), Some(hi)) = (
                present.iter().copied().reduce(f64::min),
                present.iter().copied().reduce(f64::max),
            ) {
                prop_assert!(a.min.is_none_or(|m| m <= lo), "{} min {:?} > {lo}", p.sql(), a.min);
                prop_assert!(a.max.is_none_or(|m| m >= hi), "{} max {:?} < {hi}", p.sql(), a.max);
            }
            if a.null_frac == 0.0 {
                // A cleared fraction is a claim about the survivors.
                prop_assert!(survivors.iter().all(Option::is_some), "{}: nulls survive", p.sql());
            }
            prop_assert!(a.ndv >= 1.0 && a.ndv.is_finite());
            prop_assert!(est.rows >= 0.0 && est.rows <= values.len() as f64 + 1e-9);
        }
    }
}
