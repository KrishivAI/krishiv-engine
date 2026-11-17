// SPDX-License-Identifier: Apache-2.0
//! `DerivedKeysAbove` (P3-A part 2c): a group key computed from other group keys is computed above
//! the aggregate (ClickBench q36). Where it applies and where it refuses, that the outputs keep
//! their attributes, that the rows never change and that the plan is a fixed point.

use krishiv_common::config::Config;
use krishiv_optimizer::{
    ColumnStats, Statistics, TableStats, optimize_traced_with_udfs, optimize_with_udfs,
};
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};
use krishiv_plan::{Literal, Plan, ReadSource, Scan};
use krishiv_udf_sdk::UdfLibrary;

use crate::support::eval::{self, Db, V};
use crate::support::{self, Rng};

/// `h(c int, k double, s string, sl string collate UTF8_LCASE, v bigint, w int)`: ClickBench's `hits`
/// in small.
const COLUMNS: &[(&str, &str)] = &[
    ("c", "int"),
    ("k", "double"),
    ("s", "string"),
    ("sl", "string collate UTF8_LCASE"),
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
    let dim = [("dk", "int"), ("x", "int")]
        .iter()
        .map(|(c, t)| TableColumn {
            name: (*c).to_owned(),
            data_type: krishiv_types::parse_data_type(t).unwrap(),
            nullable: true,
            char_varchar: None,
        })
        .collect();
    MemoryCatalog::new("spark_catalog")
        .with_table(&["default"], "h", columns, None)
        .with_table(&["default"], "d", dim, None)
}

/// A million rows of `h`; the range of `c` and `w` (`None`: no minimum and maximum known).
#[derive(Debug, Clone, Copy)]
struct Stats(Option<(i32, i32)>);

/// The footer range of ClickBench's `ClientIP`, well inside `int`.
const FOOTER: Stats = Stats(Some((-1_000_000_000, 1_000_000_000)));

impl Statistics for Stats {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let rows = match name.last()?.as_str() {
            "h" => 1_000_000,
            "d" => 1_000,
            _ => return None,
        };
        let columns = scan.output.iter().map(|c| {
            let (min, max) = match (self.0, c.name.as_str()) {
                (Some((lo, hi)), "c" | "w" | "dk" | "x") => {
                    (Some(Literal::Integer(lo)), Some(Literal::Integer(hi)))
                }
                _ => (None, None),
            };
            let stats = ColumnStats {
                distinct: Some(1_000),
                nulls: Some(0),
                min,
                max,
                ..ColumnStats::default()
            };
            (c.name.clone(), stats)
        });
        Some(TableStats {
            rows,
            columns: columns.collect(),
        })
    }
}

/// The session configuration: the rule on (its default) and ANSI as given.
fn config(ansi: bool) -> Config {
    config_with(ansi, &[])
}

/// [`config`] with more keys set.
fn config_with(ansi: bool, more: &[(&str, &str)]) -> Config {
    let mut c = Config::new();
    for (k, v) in more {
        c.set(k, v).unwrap();
    }
    c.set("krishiv.optimizer.derivedKeysAbove", "true").unwrap();
    c.set(
        "krishiv.sql.ansi.enabled",
        if ansi { "true" } else { "false" },
    )
    .unwrap();
    c
}

/// Rows with NULL keys, duplicates and a key at each end of `int`. The keys at the ends of `int`
/// outrun the `FOOTER` range the tests declare under ANSI, so the interpreter comparison does not
/// check error order (it wraps); the executor differential (`krishiv-exec` `derived_keys`) does.
fn db(seed: u64) -> Db {
    let mut r = Rng(seed | 1);
    let mut db = Db::default();
    let mut rows = Vec::new();
    let keys = [
        V::Null,
        V::int(0),
        V::int(1),
        V::int(7),
        V::int(-3),
        V::int(i64::from(i32::MAX)),
        V::int(i64::from(i32::MIN)),
    ];
    for i in 0..60 {
        let pick = |r: &mut Rng| keys[usize::try_from(r.range(0, 6)).unwrap()].clone();
        rows.push(vec![
            pick(&mut r),
            V::int(r.range(0, 3)),
            V::Str(r.pick(&["a", "b", "c"]).into()),
            V::Str(r.pick(&["k", "K", "m"]).into()),
            V::int(i),
            pick(&mut r),
        ]);
    }
    db.tables.insert(
        "h".to_owned(),
        (COLUMNS.iter().map(|c| c.0.to_owned()).collect(), rows),
    );
    db
}

/// What running one query gives.
struct Run {
    /// The plan as resolved.
    plan: Plan,
    /// The optimized plan with the rule on.
    on: Plan,
    /// The notes of the rule.
    notes: Vec<String>,
}

impl Run {
    fn text(&self) -> String {
        self.on.root().to_string()
    }

    /// Whether the rule rewrote anything: its output differs from the rule switched off.
    fn moved(&self) -> bool {
        self.notes.iter().any(|n| n.starts_with("moved"))
    }
}

/// Optimizes `sql` with the rule on, checking that the output keeps its attributes, that the plan
/// is a fixed point and (`rows`) that the rows equal the unoptimized plan's and the rule-off plan's
/// over small data.
fn run_over(sql: &str, stats: Stats, ansi: bool, rows: bool) -> Run {
    run_conf(sql, stats, ansi, rows, &[])
}

/// [`run_over`] with more session keys set.
fn run_conf(sql: &str, stats: Stats, ansi: bool, rows: bool, more: &[(&str, &str)]) -> Run {
    let ctx = support::ctx_over(catalog());
    let plan = support::resolve(sql, &ctx);
    let config = config_with(ansi, more);
    let udfs = UdfLibrary::new();
    let (on, steps) = optimize_traced_with_udfs(&plan, &stats, &udfs, &config).unwrap();
    assert_eq!(
        on.root().resolved_output(),
        plan.root().resolved_output(),
        "{sql}: the output keeps its attributes\n{}",
        on.root()
    );
    let again = optimize_with_udfs(&on, &stats, &udfs, &config).unwrap();
    assert!(
        again.structurally_eq(&on),
        "{sql}: a fixed point\n{}\nagain:\n{}",
        on.root(),
        again.root()
    );
    let mut off_config = config;
    off_config
        .set("krishiv.optimizer.derivedKeysAbove", "false")
        .unwrap();
    let off = optimize_with_udfs(&plan, &stats, &udfs, &off_config).unwrap();
    if rows {
        for seed in [3, 11, 29] {
            let db = db(seed);
            let want = eval::rows(&db, plan.root());
            assert_eq!(eval::rows(&db, on.root()), want, "{sql} (seed {seed})");
            assert_eq!(eval::rows(&db, off.root()), want, "{sql} off (seed {seed})");
        }
    }
    let notes: Vec<String> = steps
        .into_iter()
        .filter(|s| s.rule == "DerivedKeysAbove" && !s.note.is_empty())
        .map(|s| s.note)
        .collect();
    let run = Run { plan, on, notes };
    if !run.moved() {
        assert!(
            off.structurally_eq(&run.on),
            "{sql}: no key moved but the plan changed\n{}\noff:\n{}",
            run.on.root(),
            off.root()
        );
    }
    run
}

fn run(sql: &str) -> Run {
    run_over(sql, FOOTER, true, true)
}

/// The grouping line of the aggregate in `text`.
fn aggregate_line(text: &str) -> &str {
    text.lines().find(|l| l.contains("Aggregate")).unwrap()
}

const Q36: &str = "SELECT c, c - 1, c - 2, c - 3, count(*) AS n FROM h \
                   GROUP BY c, c - 1, c - 2, c - 3 ORDER BY n DESC LIMIT 10";

/// ClickBench q36: four group keys become one, the other three computed above.
#[test]
fn q36_groups_by_the_column_alone() {
    let r = run(Q36);
    assert!(r.moved(), "{:?}\n{}", r.notes, r.text());
    let line = aggregate_line(&r.text()).to_owned();
    assert!(line.contains("Aggregate [c#0], "), "{line}");
    insta::assert_snapshot!(
        "q36",
        format!("{}\n{}\n{}", r.plan.root(), r.text(), r.notes.join("\n"))
    );
}

/// A derived key a derived table defines (the key is a bare column of the project below).
#[test]
fn a_derived_column_the_project_below_defines_moves() {
    let r = run(
        "SELECT c, d, e, count(*) AS n FROM (SELECT c, c - 1 AS d, c + 5 AS e, v FROM h) \
                 GROUP BY c, d, e",
    );
    assert!(r.moved(), "{:?}\n{}", r.notes, r.text());
    insta::assert_snapshot!(
        "derived_column",
        format!("{}\n{}\n{}", r.plan.root(), r.text(), r.notes.join("\n"))
    );
}

/// An output that reads a moved key inside a larger expression, and a key an alias names.
#[test]
fn outputs_over_a_moved_key_are_recomputed() {
    let r = run(
        "SELECT (c - 1) * 2 AS x, c AS y, c - 1 AS z, count(*) AS n, sum(v) AS sv FROM h \
                 GROUP BY c, c - 1",
    );
    assert!(r.moved(), "{:?}\n{}", r.notes, r.text());
    // The output order is the query's: the calls sit between the keys.
    let names: Vec<_> =
        r.on.root()
            .resolved_output()
            .unwrap()
            .into_iter()
            .map(|c| c.name)
            .collect();
    assert_eq!(names, ["x", "y", "z", "n", "sv"]);
}

/// A key nothing outputs moves too, the one that stays is output for the project above.
#[test]
fn a_key_no_output_names_moves() {
    let r = run("SELECT count(*) AS n FROM h GROUP BY c, c - 1");
    assert!(r.moved(), "{:?}\n{}", r.notes, r.text());
}

/// A filter above the aggregate on the moved column reaches the aggregate's input.
#[test]
fn a_having_on_a_moved_key_still_filters() {
    let r = run("SELECT c, c - 1 AS d, count(*) AS n FROM h GROUP BY c, c - 1 HAVING c - 1 > 2");
    assert!(r.moved(), "{:?}\n{}", r.notes, r.text());
}

/// `sql` keeps its keys, the rule noting `why` (the reference interpreter lacks some of the
/// functions the refusals use, so the rows are compared only where `rows`).
fn refused_rows(sql: &str, why: &str, rows: bool) {
    let r = run_over(sql, FOOTER, true, rows);
    assert!(!r.moved(), "{}", r.text());
    assert!(
        r.notes.iter().any(|n| n.contains(why)),
        "wanted {why:?} in {:?}",
        r.notes
    );
}

fn refused(sql: &str, why: &str) {
    refused_rows(sql, why, true);
}

#[test]
fn a_floating_point_key_refuses() {
    refused(
        "SELECT k, k + 1, count(*) AS n FROM h GROUP BY k, k + 1",
        "whose grouping is not identity",
    );
}

#[test]
fn a_non_binary_collation_refuses() {
    // The collation-key rule runs first: a collated key becomes `__collation_key(sl, ..)`, a
    // registry function, so the key never reaches the rule as a bare collated column (the type
    // gate is a second guard) and is refused as not a built-in operator, the note naming it.
    let r = run_over(
        "SELECT sl, CAST(sl AS STRING COLLATE UTF8_BINARY) AS t, count(*) AS n FROM h \
         GROUP BY sl, CAST(sl AS STRING COLLATE UTF8_BINARY)",
        FOOTER,
        true,
        false,
    );
    assert!(!r.moved(), "{}", r.text());
    assert!(
        r.notes
            .iter()
            .any(|n| n.contains("__collation_key") && n.contains("not a built-in operator")),
        "{:?}",
        r.notes
    );
}

#[test]
fn a_registry_function_refuses() {
    refused_rows(
        "SELECT s, length(s), count(*) AS n FROM h GROUP BY s, length(s)",
        "not a built-in operator",
        false,
    );
}

#[test]
fn an_expression_that_may_raise_refuses() {
    refused_rows(
        "SELECT c, CAST(c AS TINYINT) AS t, count(*) AS n FROM h GROUP BY c, CAST(c AS TINYINT)",
        "may raise",
        false,
    );
}

#[test]
fn a_nondeterministic_call_refuses() {
    refused_rows(
        "SELECT c, c + CAST(rand() AS INT) AS r, count(*) AS n FROM h GROUP BY c, c + CAST(rand() AS INT)",
        "not deterministic",
        false,
    );
}

#[test]
fn a_key_that_is_not_a_group_key_refuses() {
    refused(
        "SELECT c, w + 1, count(*) AS n FROM h GROUP BY c, w + 1",
        "not a bare group key",
    );
}

#[test]
fn grouping_sets_and_rollup_keep_their_keys() {
    for sql in [
        "SELECT c, c - 1, count(*) AS n FROM h GROUP BY ROLLUP (c, c - 1)",
        "SELECT c, c - 1, count(*) AS n FROM h GROUP BY GROUPING SETS ((c, c - 1), (c))",
        "SELECT c, c - 1, count(*) AS n FROM h GROUP BY CUBE (c, c - 1)",
    ] {
        let r = run(sql);
        assert!(!r.moved(), "{sql}\n{}", r.text());
    }
}

/// Under ANSI, `c - 1` may overflow where the range is not proven; the rule needs the statistics.
#[test]
fn an_unproven_range_under_ansi_refuses() {
    let r = run_over(Q36, Stats(None), true, true);
    assert!(!r.moved(), "{}", r.text());
    assert!(
        r.notes.iter().any(|n| n.contains("no minimum and maximum")),
        "{:?}",
        r.notes
    );
}

/// A `count`'s range rests on the estimated input rows, so under ANSI it proves no `+` safe: an
/// estimate of 1 row says nothing of a count of 10^10 (`n + 9223372026854775807` would overflow).
/// Without ANSI `+` wraps and needs no proof.
#[test]
fn a_count_range_is_an_estimate_and_proves_nothing_under_ansi() {
    let sql = "SELECT n, n + 9223372026854775807 AS m FROM \
               (SELECT c, count(*) AS n FROM h GROUP BY c) GROUP BY n, n + 9223372026854775807";
    let ansi = run_over(sql, FOOTER, true, false);
    assert!(!ansi.moved(), "{}", ansi.text());
    assert!(
        ansi.notes
            .iter()
            .any(|n| n.contains("the range of n is an estimate")),
        "{:?}",
        ansi.notes
    );
    let wrapping = run_over(sql, FOOTER, false, false);
    assert!(
        wrapping.moved(),
        "{:?}\n{}",
        wrapping.notes,
        wrapping.text()
    );
}

/// A range that reaches the type's end is no proof; wrapping needs none.
#[test]
fn a_key_at_int_max_rewrites_only_without_ansi() {
    let at_max = Stats(Some((0, i32::MAX)));
    let ansi = run_over(Q36, at_max, true, true);
    assert!(!ansi.moved(), "{}", ansi.text());
    assert!(
        ansi.notes.iter().any(|n| n.contains("may overflow")),
        "{:?}",
        ansi.notes
    );
    let wrapping = run_over(Q36, at_max, false, true);
    assert!(
        wrapping.moved(),
        "{:?}\n{}",
        wrapping.notes,
        wrapping.text()
    );
}

/// An output that holds a call and reads a moved key outside it, and a column a call reads.
#[test]
fn an_output_mixing_a_call_and_a_moved_key_refuses() {
    refused(
        "SELECT c, count(*) + (c - 1) AS m FROM h GROUP BY c, c - 1",
        "beside an aggregate call",
    );
    refused(
        "SELECT c, d, sum(d) AS sd FROM (SELECT c, c - 1 AS d FROM h) GROUP BY c, d",
        "an aggregate call reads it",
    );
}

#[test]
fn switched_off_nothing_moves_and_the_default_is_on() {
    let ctx = support::ctx_over(catalog());
    let plan = support::resolve(Q36, &ctx);
    let udfs = UdfLibrary::new();
    // The key off.
    let mut c = config(true);
    c.set("krishiv.optimizer.derivedKeysAbove", "false")
        .unwrap();
    let off = optimize_with_udfs(&plan, &FOOTER, &udfs, &c).unwrap();
    assert!(
        aggregate_line(&off.root().to_string()).contains("c#0 - 1"),
        "{}",
        off.root()
    );
    // The default (no key set) is the rule on.
    let on = optimize_with_udfs(&plan, &FOOTER, &udfs, &config(true)).unwrap();
    let default = optimize_with_udfs(&plan, &FOOTER, &udfs, &Config::new()).unwrap();
    assert!(on.structurally_eq(&default), "{}", default.root());
    assert!(!off.structurally_eq(&default), "{}", default.root());
}

/// Both switches that read an aggregate's shape on, over a join whose derived key is a dimension
/// column: the rule runs first, so the group-key code and eager aggregation see the narrowed
/// aggregate and a second optimization finds nothing more.
#[test]
fn a_join_with_the_group_key_code_and_eager_aggregation_on_is_a_fixed_point() {
    let on = [
        ("krishiv.optimizer.groupKeyCode", "true"),
        ("krishiv.optimizer.eagerAggregation", "true"),
    ];
    for sql in [
        "SELECT d.x, d.x - 1 AS y, sum(h.v) AS sv FROM h JOIN d ON h.w = d.dk GROUP BY d.x, d.x - 1",
        "SELECT d.x, d.x - 1 AS y, count(*) AS n FROM h JOIN d ON h.w = d.dk GROUP BY d.x, d.x - 1",
    ] {
        let r = run_conf(sql, FOOTER, true, false, &on);
        assert!(r.moved(), "{sql}: {:?}\n{}", r.notes, r.text());
    }
}

/// The derived-key aggregate below a three-way join: narrowing its keys changes its estimate (a
/// non-column key counts as the input's rows), so the joins above it are ordered again and a second
/// optimization finds the same plan, with the group-key code and eager aggregation on.
#[test]
fn an_aggregate_below_a_three_way_join_is_a_fixed_point() {
    let on = [
        ("krishiv.optimizer.groupKeyCode", "true"),
        ("krishiv.optimizer.eagerAggregation", "true"),
    ];
    let sql = "SELECT a.c, a.d, a.n, d.x, g.v FROM \
               (SELECT c, c - 1 AS d, count(*) AS n FROM h GROUP BY c, c - 1) a \
               JOIN h AS g ON a.c = g.c JOIN d ON a.c = d.dk";
    let r = run_conf(sql, FOOTER, true, false, &on);
    assert!(r.moved(), "{:?}\n{}", r.notes, r.text());
}
