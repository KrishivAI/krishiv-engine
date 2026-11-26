// SPDX-License-Identifier: Apache-2.0
//! Grouping sets aggregate their input before the expansion (`eager` module `expand`,
//! `krishiv.optimizer.preAggregateExpand`): where it applies and where it refuses, that the output
//! keeps its attributes, that the rows never change and that the plan is a fixed point.

use krishiv_common::config::Config;
use krishiv_optimizer::{
    ColumnStats, Statistics, TableStats, optimize_traced_with_udfs, optimize_with_udfs,
};
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};
use krishiv_plan::{Literal, Plan, ReadSource, Scan};
use krishiv_udf_sdk::UdfLibrary;

use crate::support::eval::{self, Db, V};
use crate::support::{self, Rng};

/// `h(c int, k double, s string, v bigint, w int)`.
const COLUMNS: &[(&str, &str)] = &[
    ("c", "int"),
    ("k", "double"),
    ("s", "string"),
    ("v", "bigint"),
    ("w", "int"),
];

fn catalog() -> MemoryCatalog {
    let columns = COLUMNS
        .iter()
        .map(|(c, t)| TableColumn {
            name: (*c).to_owned(),
            data_type: krishiv_types::parse_data_type(t).unwrap(),
            nullable: true,
            char_varchar: None,
        })
        .collect();
    MemoryCatalog::new("spark_catalog").with_table(&["default"], "h", columns, None)
}

/// A million rows of `h`, `c` and `w` within `[-1000, 1000]`.
#[derive(Debug)]
struct Stats;

impl Statistics for Stats {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        if name.last()?.as_str() != "h" {
            return None;
        }
        let columns = scan.output.iter().map(|c| {
            let (min, max) = match c.name.as_str() {
                "c" | "w" => (Some(Literal::Integer(-1000)), Some(Literal::Integer(1000))),
                _ => (None, None),
            };
            let stats = ColumnStats {
                distinct: Some(100),
                nulls: Some(0),
                min,
                max,
                ..ColumnStats::default()
            };
            (c.name.clone(), stats)
        });
        Some(TableStats {
            rows: 1_000_000,
            columns: columns.collect(),
        })
    }
}

fn config(on: bool) -> Config {
    let mut c = Config::new();
    c.set(
        "krishiv.optimizer.preAggregateExpand",
        if on { "true" } else { "false" },
    )
    .unwrap();
    c
}

/// Rows with NULL keys and arguments, and duplicate groups.
fn db(seed: u64) -> Db {
    let mut r = Rng(seed | 1);
    let mut db = Db::default();
    let mut rows = Vec::new();
    for i in 0..80 {
        let c = match r.range(0, 4) {
            0 => V::Null,
            n => V::int(n - 2),
        };
        let w = match r.range(0, 5) {
            0 => V::Null,
            n => V::int(n * 7 - 20),
        };
        rows.push(vec![
            c,
            V::int(r.range(-1, 1)),
            V::Str(r.pick(&["a", "b", "c"]).into()),
            V::int(i % 9),
            w,
        ]);
    }
    db.tables.insert(
        "h".to_owned(),
        (COLUMNS.iter().map(|c| c.0.to_owned()).collect(), rows),
    );
    db
}

/// What optimizing one query gives.
struct Run {
    on: Plan,
    notes: Vec<String>,
}

impl Run {
    fn fired(&self) -> bool {
        self.on.root().to_string().contains("PartialAggregate")
    }
}

/// Optimizes `sql` with the rule on, checking that the output keeps its attributes, that the plan
/// is a fixed point, that the rows equal the unoptimized plan's and the rule-off plan's over small
/// data, and that the plan with the rule off has no partial aggregate.
fn run(sql: &str) -> Run {
    let ctx = support::ctx_over(catalog());
    let plan = support::resolve(sql, &ctx);
    let udfs = UdfLibrary::new();
    let (on, steps) = optimize_traced_with_udfs(&plan, &Stats, &udfs, &config(true)).unwrap();
    assert_eq!(
        on.root().resolved_output(),
        plan.root().resolved_output(),
        "{sql}: the output keeps its attributes\n{}",
        on.root()
    );
    let again = optimize_with_udfs(&on, &Stats, &udfs, &config(true)).unwrap();
    assert!(
        again.structurally_eq(&on),
        "{sql}: a fixed point\n{}\nagain:\n{}",
        on.root(),
        again.root()
    );
    let off = optimize_with_udfs(&plan, &Stats, &udfs, &config(false)).unwrap();
    assert!(
        !off.root().to_string().contains("PartialAggregate"),
        "{sql}: switched off\n{}",
        off.root()
    );
    for seed in [3, 11, 29] {
        let db = db(seed);
        let want = eval::rows(&db, plan.root());
        assert_eq!(eval::rows(&db, on.root()), want, "{sql} (seed {seed})");
        assert_eq!(eval::rows(&db, off.root()), want, "{sql} off (seed {seed})");
    }
    let notes = steps
        .into_iter()
        .filter(|s| s.rule == "EagerAggregate" && !s.note.is_empty())
        .map(|s| s.note)
        .collect();
    Run { on, notes }
}

fn assert_fires(sql: &str) -> Run {
    let r = run(sql);
    assert!(r.fired(), "{sql}: {:?}\n{}", r.notes, r.on.root());
    assert!(
        r.notes
            .iter()
            .any(|n| n.starts_with("pre-aggregated below the Expand by [")),
        "{sql}: {:?}",
        r.notes
    );
    r
}

fn assert_refuses(sql: &str, why: &str) {
    let r = run(sql);
    assert!(!r.fired(), "{sql}: {:?}\n{}", r.notes, r.on.root());
    assert!(
        r.notes
            .iter()
            .any(|n| n.contains("below the Expand: ") && n.contains(why)),
        "{sql}: wanted {why:?} in {:?}",
        r.notes
    );
}

/// A `ROLLUP` of the calls eager aggregation moves: the partial aggregate groups by both columns.
#[test]
fn a_rollup_aggregates_its_input_by_the_finest_set() {
    let r = assert_fires(
        "SELECT c, s, sum(w), count(*), count(w), min(s), max(v), avg(w) FROM h \
         GROUP BY ROLLUP(c, s)",
    );
    let text = r.on.root().to_string();
    let partial = text
        .lines()
        .find(|l| l.contains("PartialAggregate"))
        .unwrap();
    assert!(partial.contains("PartialAggregate [c#0, s#2], "), "{text}");
    assert!(text.contains("Expand"), "{text}");
}

/// `CUBE`, `grouping` and `grouping_id`, `HAVING` and `ORDER BY` read the copies and the id.
#[test]
fn a_cube_keeps_its_grouping_ids() {
    assert_fires(
        "SELECT c, s, grouping_id() AS g, grouping(c) AS gc, count(v) AS n FROM h \
         GROUP BY CUBE(c, s) HAVING count(v) > 1 ORDER BY g, c, s",
    );
}

/// Equal sets stay separate rows (the expansion numbers its sets); a grand total is one of them.
#[test]
fn repeated_sets_and_the_grand_total_keep_their_rows() {
    assert_fires(
        "SELECT c, sum(w) AS t FROM h GROUP BY GROUPING SETS ((c), (c), ()) ORDER BY c, t",
    );
}

/// A column that is both a grouping column and an argument.
#[test]
fn an_argument_that_is_also_a_key() {
    assert_fires("SELECT c, sum(c), max(c) FROM h GROUP BY ROLLUP(c)");
}

/// A grouping expression: the projection below the expansion computes it, and the partial
/// aggregate groups by its value.
#[test]
fn a_grouping_expression_groups_by_its_value() {
    let r = assert_fires("SELECT c % 2 AS p, s, count(*) FROM h GROUP BY ROLLUP(c % 2, s)");
    let text = r.on.root().to_string();
    assert!(
        text.contains("PartialAggregate [s#2, (c % 2)#8], "),
        "{text}"
    );
}

/// q67's argument: exact decimal arithmetic under `coalesce` raises on no row, and the partial
/// reads the rows the aggregate above reads, so it moves below the expansion.
#[test]
fn an_exact_decimal_argument_under_coalesce_moves() {
    assert_fires(
        "SELECT c, s, sum(coalesce(CAST(c AS decimal(12,2)) * w, 0)) FROM h GROUP BY ROLLUP(c, s)",
    );
}

/// A product whose type Spark adjusts for precision loss (`decimal(38,10) * decimal(38,10)`) is
/// not exact decimal arithmetic: it can round or overflow, so under `coalesce` too it stays above
/// the expansion. `min`, so the refusal is the argument's and not a `sum` buffer's precision.
#[test]
fn an_adjusted_decimal_product_under_coalesce_stays() {
    assert_refuses(
        "SELECT c, min(coalesce(CAST(c AS decimal(38,10)) * CAST(w AS decimal(38,10)), 0)) \
         FROM h GROUP BY ROLLUP(c)",
        "can raise",
    );
}

/// Integer `*` raises on overflow under ANSI: it stays above the expansion.
#[test]
fn an_integer_product_stays() {
    assert_refuses(
        "SELECT c, sum(coalesce(c * w, 0)) FROM h GROUP BY ROLLUP(c)",
        "can raise",
    );
}

/// Floating-point sums depend on the order of their rows.
#[test]
fn a_floating_point_sum_stays() {
    assert_refuses(
        "SELECT c, sum(k) FROM h GROUP BY ROLLUP(c)",
        "can depend on the order",
    );
}

/// `-0.0` and `0.0` would merge in the partial aggregate.
#[test]
fn a_floating_point_key_stays() {
    assert_refuses(
        "SELECT k, count(*) FROM h GROUP BY ROLLUP(k)",
        "floating point or collated key",
    );
}

/// Only empty grouping sets: no column of the input is read above the expansion, and a keyless
/// partial aggregate would yield a row over empty input (Spark goldens `postgreSQL/groupingsets`
/// #16 and #44).
#[test]
fn only_empty_grouping_sets_stay() {
    assert_refuses(
        "SELECT sum(w), count(*) FROM h GROUP BY GROUPING SETS ((), (), ())",
        "no key",
    );
}

/// A distinct aggregate exports no states.
#[test]
fn a_distinct_aggregate_stays() {
    let r = run("SELECT c, count(DISTINCT v) FROM h GROUP BY ROLLUP(c)");
    assert!(!r.fired(), "{:?}\n{}", r.notes, r.on.root());
}

/// One plain `GROUP BY` has no expansion: nothing to do.
#[test]
fn a_plain_group_by_is_left_alone() {
    let r = run("SELECT c, sum(w) FROM h GROUP BY c");
    assert!(!r.fired(), "{:?}\n{}", r.notes, r.on.root());
}

/// The rewrite of several distinct calls is an expansion too: the regular call merges below it,
/// keyed by the distinct arguments (Opus review of b58a8e4e7, finding 2).
#[test]
fn the_multi_distinct_rewrite_aggregates_its_input() {
    let r = assert_fires("SELECT count(DISTINCT s), count(DISTINCT c), max(w) FROM h");
    let text = r.on.root().to_string();
    assert!(
        text.contains("PartialAggregate [c#0, s#2], [max(w#4)]"),
        "{text}"
    );
}

/// With a key, the rewrite aggregates its regular calls in an aggregate of their own (with a
/// projection of the grouping id), which the rule does not match; the rows stay the same.
#[test]
fn the_multi_distinct_rewrite_with_a_key_stays() {
    let r =
        run("SELECT c, count(DISTINCT s), count(DISTINCT w), count(*), max(v) FROM h GROUP BY c");
    assert!(!r.fired(), "{:?}\n{}", r.notes, r.on.root());
}

/// A `FILTER` on a regular call beside several distinct calls stays: the filtered call is not
/// one accumulator per group.
#[test]
fn a_filtered_regular_call_beside_distinct_calls_stays() {
    assert_refuses(
        "SELECT count(DISTINCT s), count(DISTINCT c), max(w) FILTER (WHERE c > 0) FROM h",
        "exports no states",
    );
}

/// A key the projection below the expansion computes nondeterministically is computed once per
/// input row, below the partial aggregate, which groups by its value: as many evaluations as
/// without the rule. (The reference interpreter has no `rand`, so only the plan is checked.)
#[test]
fn a_nondeterministic_key_is_computed_once_below_the_partial_aggregate() {
    let sql = "SELECT c, r, count(*) FROM (SELECT c, CAST(rand(7) * 10 AS INT) AS r FROM h) \
               GROUP BY ROLLUP(c, r)";
    let ctx = support::ctx_over(catalog());
    let plan = support::resolve(sql, &ctx);
    let udfs = UdfLibrary::new();
    let on = optimize_with_udfs(&plan, &Stats, &udfs, &config(true)).unwrap();
    assert_eq!(
        on.root().resolved_output(),
        plan.root().resolved_output(),
        "the output keeps its attributes\n{}",
        on.root()
    );
    let text = on.root().to_string();
    assert_eq!(text.matches("rand(7)").count(), 1, "{text}");
    let line = text
        .lines()
        .find(|l| l.contains("PartialAggregate ["))
        .expect(&text);
    let keys = &line[line.find('[').unwrap()..line.find(']').unwrap()];
    assert!(keys.contains("c#") && keys.contains("r#"), "{text}");
    let partial = text.find(line).unwrap();
    assert!(text[partial..].contains("rand(7)"), "{text}");
    let again = optimize_with_udfs(&on, &Stats, &udfs, &config(true)).unwrap();
    assert!(again.structurally_eq(&on), "{text}");
    let off = optimize_with_udfs(&plan, &Stats, &udfs, &config(false)).unwrap();
    assert!(
        !off.root().to_string().contains("PartialAggregate"),
        "switched off\n{}",
        off.root()
    );
}

/// An uncorrelated scalar subquery the key reads becomes a column of the input, so the rule
/// groups by it below the expansion with the same rows.
#[test]
fn a_scalar_subquery_key_below_the_expansion_is_a_column() {
    assert_fires(
        "SELECT c, x, count(*) FROM (SELECT c, (SELECT max(w) FROM h) AS x FROM h) \
         GROUP BY ROLLUP(c, x)",
    );
}

/// Grouping sets with no aggregate call have no states to merge below the expansion.
#[test]
fn grouping_sets_without_a_call_stay() {
    assert_refuses(
        "SELECT c, s FROM h GROUP BY ROLLUP(c, s)",
        "no aggregate call",
    );
}

/// A scalar subquery a projection above a join computes, read by the aggregate's key: the
/// subquery would move below the merge with the projection, so the call stays above the join.
#[test]
fn a_subquery_between_the_aggregate_and_the_join_keeps_the_call_above() {
    let r = run(
        "SELECT x, sum(w) FROM (SELECT a.w, (SELECT max(v) FROM h) + a.c AS x \
         FROM h a JOIN h b ON a.c = b.c) GROUP BY x",
    );
    assert!(!r.fired(), "{:?}\n{}", r.notes, r.on.root());
    assert!(
        r.notes
            .iter()
            .any(|n| n
                .contains("a projection between the aggregate and its new input holds a subquery")),
        "{:?}\n{}",
        r.notes,
        r.on.root()
    );
}
