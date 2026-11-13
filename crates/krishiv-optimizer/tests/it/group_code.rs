// SPDX-License-Identifier: Apache-2.0
//! `GroupKeyCode` (design P2b D2): an aggregate's group keys from one
//! dimension travel as one integer code. First the fence it relies on: a
//! `Shift` the optimizer did not make stays, filters cross it, and join
//! ordering does not look through it.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_optimizer::{
    ColumnStats, Options, Statistics, TableStats, optimize_traced_with_options, optimize_with,
    optimize_with_options,
};
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};
use krishiv_plan::{LogicalPlan, Plan, PlanNode, ReadSource, Scan};

use crate::support::eval::{Db, V};
use crate::support::{self, AllDistinct, Rng, data, eval};

/// `plan` with every scan of `table` under a `Shift` of offset 0.
fn shift_scans(plan: &Plan, table: &str) -> Plan {
    let root = Arc::clone(plan.root())
        .transform_up(&mut |p: Arc<LogicalPlan>| {
            let wrap = matches!(&p.node, PlanNode::Scan(s) if s.output.first().is_some_and(|_| {
                matches!(&s.source, krishiv_plan::ReadSource::Table { name, .. } if name.last().is_some_and(|n| n == table))
            }));
            Ok::<_, std::convert::Infallible>(if wrap {
                Arc::new(LogicalPlan::new(PlanNode::Shift {
                    input: p,
                    offset: 0,
                }))
            } else {
                p
            })
        })
        .unwrap();
    Plan::try_new(root).unwrap()
}

/// The fence D2 builds `D_c` with (design §2.3 "Fence"): a `Shift` of offset
/// 0 the optimizer did not make is kept through every batch (Tidy
/// included), a filter on its columns moves below it, and an optimized plan
/// with it is a fixed point.
#[test]
fn a_shift_the_optimizer_did_not_make_is_kept_and_filters_cross_it() {
    let ctx = support::ctx();
    let sql = "SELECT t.a, u.d FROM t JOIN u ON t.a = u.a WHERE u.d = 'x'";
    let plan = shift_scans(&support::resolve(sql, &ctx), "u");
    let optimized = optimize_with(&plan, &AllDistinct).unwrap();
    let text = optimized.root().to_string();
    let shift = text
        .lines()
        .position(|l| l.contains("Shift"))
        .unwrap_or_else(|| panic!("the Shift is gone:\n{text}"));
    let filter = text
        .lines()
        .position(|l| l.contains("Filter") && l.contains("d#"))
        .unwrap_or_else(|| panic!("no filter on d:\n{text}"));
    assert!(filter > shift, "the filter stays above the Shift:\n{text}");
    let again = optimize_with(&optimized, &AllDistinct).unwrap();
    assert!(again.structurally_eq(&optimized), "{}", again.root());
    let db = data::small(3);
    assert_eq!(
        eval::rows(&db, optimized.root()),
        eval::rows(&db, plan.root())
    );
}

/// Join ordering treats a `Shift` as one input: the join under it stays
/// under it.
#[test]
fn join_ordering_does_not_look_through_a_shift() {
    let ctx = support::ctx();
    let sql = "SELECT t.a, x.d FROM t JOIN (SELECT u.a, u.d FROM u JOIN t AS v ON u.a = v.a) AS x ON t.a = x.a";
    let resolved = support::resolve(sql, &ctx);
    // Wrap the derived table's join in a Shift of offset 0.
    let root = Arc::clone(resolved.root())
        .transform_up(&mut |p: Arc<LogicalPlan>| {
            let wrap = matches!(&p.node, PlanNode::SubqueryAlias { alias, .. } if alias == "x");
            Ok::<_, std::convert::Infallible>(if wrap {
                Arc::new(LogicalPlan::new(PlanNode::Shift {
                    input: p,
                    offset: 0,
                }))
            } else {
                p
            })
        })
        .unwrap();
    let plan = Plan::try_new(root).unwrap();
    let optimized = optimize_with(&plan, &AllDistinct).unwrap();
    let text = optimized.root().to_string();
    let shift = text
        .lines()
        .position(|l| l.contains("Shift"))
        .unwrap_or_else(|| panic!("the Shift is gone:\n{text}"));
    let joins: Vec<usize> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("Join Inner"))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(joins.len(), 2, "{text}");
    assert!(joins[0] < shift && shift < joins[1], "{text}");
    let again = optimize_with(&optimized, &AllDistinct).unwrap();
    assert!(again.structurally_eq(&optimized), "{}", again.root());
    let db = data::small(5);
    assert_eq!(
        eval::rows(&db, optimized.root()),
        eval::rows(&db, plan.root())
    );
}

/// `f` a fact, `d` and `e` dimensions:
/// `f(fk int, ek2 int, x int, y string, k2 int)`,
/// `d(dk int, g1 string, g2 string, g3 int, gd double, gl string collate UTF8_LCASE, dyear int)`,
/// `e(ek int, h1 string, h2 string)`.
const TABLES: &[(&str, &[(&str, &str)])] = &[
    (
        "f",
        &[
            ("fk", "int"),
            ("ek2", "int"),
            ("x", "int"),
            ("y", "string"),
            ("k2", "int"),
        ],
    ),
    (
        "d",
        &[
            ("dk", "int"),
            ("g1", "string"),
            ("g2", "string"),
            ("g3", "int"),
            ("gd", "double"),
            ("gl", "string collate UTF8_LCASE"),
            ("dyear", "int"),
        ],
    ),
    ("e", &[("ek", "int"), ("h1", "string"), ("h2", "string")]),
];

fn catalog() -> MemoryCatalog {
    let mut cat = MemoryCatalog::new("spark_catalog");
    for (name, cols) in TABLES {
        let columns = cols
            .iter()
            .map(|(c, t)| TableColumn {
                name: (*c).to_owned(),
                data_type: krishiv_types::parse_data_type(t).unwrap(),
                nullable: true,
                char_varchar: None,
            })
            .collect();
        cat = cat.with_table(&["default"], name, columns, None);
    }
    cat
}

/// Row counts and distinct counts by table; a column not listed has as many
/// distinct values as rows, `None` an unknown count.
#[derive(Debug, Clone)]
struct Stats(HashMap<&'static str, (u64, HashMap<&'static str, Option<u64>>)>);

impl Stats {
    /// `f` 1M rows (1,000 values of `fk` and of `ek2`), `d` 1,000 (10 values
    /// of `g1`, `g2`), `e` 1,000 (10 of `h1`, `h2`).
    fn new() -> Self {
        let mut s = Self(HashMap::new());
        s.0.insert("f", (1_000_000, HashMap::new()));
        s.0.insert("d", (1_000, HashMap::new()));
        s.0.insert("e", (1_000, HashMap::new()));
        for (t, c) in [("d", "g1"), ("d", "g2"), ("e", "h1"), ("e", "h2")] {
            s.set(t, c, Some(10));
        }
        // A fact's foreign keys hold at most the dimension's keys.
        s.set("f", "fk", Some(1_000)).set("f", "ek2", Some(1_000));
        s.set("d", "dyear", Some(5));
        s.set("f", "k2", Some(10));
        s
    }

    fn set(&mut self, table: &'static str, col: &'static str, ndv: Option<u64>) -> &mut Self {
        self.0.get_mut(table).unwrap().1.insert(col, ndv);
        self
    }

    fn rows(&mut self, table: &'static str, rows: u64) -> &mut Self {
        self.0.get_mut(table).unwrap().0 = rows;
        self
    }
}

impl Statistics for Stats {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        let (rows, ndv) = self.0.get(name.last()?.as_str())?;
        let columns = scan.output.iter().map(|c| {
            let distinct = ndv.get(c.name.as_str()).copied().unwrap_or(Some(*rows));
            let stats = ColumnStats {
                distinct,
                nulls: Some(0),
                ..ColumnStats::default()
            };
            (c.name.clone(), stats)
        });
        Some(TableStats {
            rows: *rows,
            columns: columns.collect(),
        })
    }
}

/// Small data: `d` repeats join keys with different group keys (the review's
/// counter-example to a code from the join key) and holds null keys; some
/// `f` rows match nothing.
fn db(seed: u64) -> Db {
    let mut r = Rng(seed | 1);
    let mut db = Db::default();
    let maybe = |r: &mut Rng, v: V| if r.range(0, 6) == 0 { V::Null } else { v };
    let word = |r: &mut Rng, w: &[&str]| V::Str(r.pick(w).into());
    let mut d = Vec::new();
    for i in 0..14 {
        let dk = V::int(i % 9);
        let g1 = word(&mut r, &["a", "b", "c"]);
        let g1 = maybe(&mut r, g1);
        let g2 = word(&mut r, &["p", "q"]);
        let g2 = maybe(&mut r, g2);
        let gl = word(&mut r, &["k", "K", "m"]);
        d.push(vec![
            dk,
            g1,
            g2,
            V::int(r.range(0, 2)),
            V::int(r.range(0, 2)),
            gl,
            V::int(r.range(2000, 2002)),
        ]);
    }
    let mut f = Vec::new();
    for _ in 0..40 {
        let y = word(&mut r, &["a", "b", "z"]);
        let d = V::int(r.range(0, 10));
        f.push(vec![
            maybe(&mut r, d),
            V::int(r.range(0, 7)),
            V::int(r.range(1, 9)),
            y,
            V::int(r.range(0, 3)),
        ]);
    }
    let mut e = Vec::new();
    for i in 0..8 {
        let h1 = word(&mut r, &["u", "v"]);
        let h2 = word(&mut r, &["s", "t", "w"]);
        e.push(vec![V::int(i), maybe(&mut r, h1), h2]);
    }
    for (name, rows) in [("d", d), ("f", f), ("e", e)] {
        let cols = TABLES.iter().find(|t| t.0 == name).unwrap().1;
        db.tables.insert(
            name.to_owned(),
            (cols.iter().map(|c| c.0.to_owned()).collect(), rows),
        );
    }
    db
}

/// What `GroupKeyCode` said and the plan, with the rule on; checks that the
/// output is kept, the plan is a fixed point, and that it returns the rows of
/// the unoptimised plan and of the plan optimised with the rule off.
fn run(sql: &str, stats: &Stats) -> (String, Vec<String>) {
    run_checked(sql, stats, true)
}

/// [`run`], comparing rows only when `rows` (the reference interpreter groups
/// a collated string by its value, not by its collation key).
fn run_checked(sql: &str, stats: &Stats, rows: bool) -> (String, Vec<String>) {
    let ctx = support::ctx_over(catalog());
    let plan = support::resolve(sql, &ctx);
    let (optimized, steps) =
        optimize_traced_with_options(&plan, stats, &Options::default()).unwrap();
    assert_eq!(
        optimized.root().resolved_output(),
        plan.root().resolved_output(),
        "{sql}: the output keeps its attributes"
    );
    let again = optimize_with(&optimized, stats).unwrap();
    assert!(
        again.structurally_eq(&optimized),
        "{sql}: a fixed point\n{}\nagain:\n{}",
        optimized.root(),
        again.root()
    );
    let off = support::options(|o| o.group_key_code = false);
    let without = optimize_with_options(&plan, stats, &off).unwrap();
    if !rows {
        // Unchecked rows only where the rule leaves the plan alone.
        assert!(
            without.structurally_eq(&optimized),
            "{sql}: the rule changed a plan whose rows are not compared\n{}",
            optimized.root()
        );
    }
    for seed in [3, 11, 29].into_iter().filter(|_| rows) {
        let db = db(seed);
        let want = eval::rows(&db, plan.root());
        assert_eq!(
            eval::rows(&db, optimized.root()),
            want,
            "{sql} (seed {seed})"
        );
        assert_eq!(
            eval::rows(&db, without.root()),
            want,
            "{sql} off (seed {seed})"
        );
    }
    let notes = steps
        .into_iter()
        .filter(|s| s.rule == "GroupKeyCode" && !s.note.is_empty())
        .map(|s| s.note)
        .collect();
    (optimized.root().to_string(), notes)
}

fn coded(text: &str) -> bool {
    text.contains("krishiv_group_code")
}

/// Dimension keys that nothing reads on the way are coded: the aggregate
/// groups by the code alone, and the keys come back from the dictionary.
#[test]
fn single_dimension_keys_travel_as_one_code() {
    let (text, notes) = run(
        "SELECT g1, g2, count(*) AS n, sum(x) AS s FROM f JOIN d ON fk = dk GROUP BY g1, g2",
        &Stats::new(),
    );
    assert!(coded(&text), "{notes:?}\n{text}");
    assert!(notes[0].starts_with("coded [g1, g2]"), "{notes:?}");
    insta::assert_snapshot!(text);
}

/// A key read by a cross-table condition, or by an aggregate call, stays
/// carried, and the aggregate still groups by the code; the other key is
/// dropped.
#[test]
fn keys_read_on_the_way_stay_carried() {
    for (name, sql) in [
        (
            "condition",
            "SELECT g1, g2, count(*) AS n FROM f JOIN d ON fk = dk AND g2 <> y GROUP BY g1, g2",
        ),
        (
            "call",
            "SELECT g1, g2, count(DISTINCT g2) AS n FROM f JOIN d ON fk = dk GROUP BY g1, g2",
        ),
    ] {
        let (text, notes) = run(sql, &Stats::new());
        assert!(coded(&text), "{notes:?}\n{text}");
        assert!(notes[0].contains("20 key bytes dropped"), "{notes:?}");
        insta::assert_snapshot!(format!("keys_read_on_the_way_{name}"), text);
    }
}

/// An output over keys moves above the decode join.
#[test]
fn outputs_over_keys_move_to_the_top() {
    let (text, notes) = run(
        "SELECT upper(g1) AS u, g2, k2, sum(x) AS s FROM f JOIN d ON fk = dk GROUP BY g1, g2, k2",
        &Stats::new(),
    );
    assert!(coded(&text), "{notes:?}\n{text}");
    insta::assert_snapshot!(text);
}

/// Refusals, each with its reason in `EXPLAIN VERBOSE`; every plan returns the
/// rows of the unoptimised one.
#[test]
fn refusals_say_why() {
    let base = Stats::new();
    let cases: Vec<(&str, Stats, &str)> = vec![
        (
            "SELECT g1, gd, count(*) FROM f JOIN d ON fk = dk GROUP BY g1, gd",
            base.clone(),
            "gd is of type double",
        ),
        (
            "SELECT dk, g1, g2, count(*) FROM f JOIN d ON fk = dk GROUP BY dk, g1, g2",
            base.clone(),
            "dk is a group key and its join's key",
        ),
        (
            "SELECT g1, g2, count(*) FROM f LEFT JOIN (SELECT * FROM e JOIN d ON ek = dk) z ON f.fk = z.ek GROUP BY g1, g2",
            base.clone(),
            "null-padded side",
        ),
        (
            "SELECT g1, g2, count(*) FROM f FULL JOIN d ON fk = dk GROUP BY g1, g2",
            base.clone(),
            "null-padded side",
        ),
        (
            "SELECT g1, g2, count(*) FROM f JOIN d ON fk = dk GROUP BY g1, g2",
            {
                let mut s = base.clone();
                s.set("d", "g1", None);
                s
            },
            "the distinct count of g1 is a guess",
        ),
        (
            "SELECT g1, g2, count(*) FROM f JOIN d ON fk = dk GROUP BY g1, g2",
            {
                let mut s = base.clone();
                s.set("d", "g1", Some(1_000))
                    .set("d", "g2", Some(1_000))
                    .rows("f", 3_000);
                s
            },
            "too many groups",
        ),
        (
            "SELECT g1, g2, count(*) FROM f JOIN d ON fk = dk GROUP BY g1, g2",
            {
                let mut s = base.clone();
                s.rows("d", 600_000);
                s
            },
            "the dimension is too large",
        ),
        (
            "SELECT g1, g2, count(*) FROM f JOIN (SELECT * FROM d LIMIT 100) z ON fk = dk GROUP BY g1, g2",
            base.clone(),
            "do not come from a scan with filters",
        ),
        (
            "SELECT g3, count(*) FROM f JOIN d ON fk = dk GROUP BY g3",
            base.clone(),
            "4 bytes wide",
        ),
        (
            "SELECT g1, g2, h1, h2, count(*) FROM f JOIN d ON fk = dk JOIN e ON ek2 = ek GROUP BY g1, g2, h1, h2",
            base,
            "2 dimensions qualify",
        ),
    ];
    for (sql, stats, why) in cases {
        let (text, notes) = run(sql, &stats);
        assert!(!coded(&text), "{sql}\n{text}");
        assert!(
            notes.iter().any(|n| n.contains(why)),
            "{sql}: want {why:?} in {notes:?}"
        );
    }
}

/// Grouping sets and collated keys are not plain column keys at this point:
/// nothing is coded. (Rows are compared for the rollup only: the reference
/// interpreter has no collation-aware grouping.)
#[test]
fn grouping_sets_and_collated_keys_are_left_alone() {
    for (sql, rows) in [
        (
            "SELECT g1, g2, count(*) FROM f JOIN d ON fk = dk GROUP BY ROLLUP(g1, g2)",
            true,
        ),
        (
            "SELECT gl, g1, count(*) FROM f JOIN d ON fk = dk GROUP BY gl, g1",
            false,
        ),
    ] {
        let (text, notes) = run_checked(sql, &Stats::new(), rows);
        assert!(!coded(&text), "{sql}: {notes:?}\n{text}");
        if !rows {
            assert!(
                notes
                    .iter()
                    .any(|n| n.contains("gl is a collated string, grouped by its collation key")),
                "{sql}: {notes:?}"
            );
        }
    }
}

/// A filter of `D` that can raise under ANSI (a string cast to `int`) is not
/// read twice: refused. (Rows not compared: the reference interpreter does
/// not raise.)
#[test]
fn a_dimension_filter_that_can_raise_is_refused() {
    let sql = "SELECT g1, g2, count(*) FROM f JOIN (SELECT * FROM d WHERE CAST(g1 AS INT) > 0) z \
               ON fk = dk GROUP BY g1, g2";
    let (text, notes) = run_checked(sql, &Stats::new(), false);
    assert!(!coded(&text), "{notes:?}\n{text}");
    assert!(
        notes.iter().any(|n| n.contains("a filter that can raise")),
        "{notes:?}"
    );
}

/// `D` on the preserved side of an outer join above its own join: coded, with
/// the data's null keys (a real null group) and without them.
#[test]
fn an_outer_join_above_the_dimension_keeps_the_code() {
    for sql in [
        "SELECT g1, g2, count(*) AS n, sum(x) AS s, count(h1) AS h \
         FROM f JOIN d ON fk = dk LEFT JOIN e ON ek2 = ek GROUP BY g1, g2",
        "SELECT g1, g2, count(*) AS n, sum(x) AS s, count(h1) AS h \
         FROM f JOIN d ON fk = dk LEFT JOIN e ON ek2 = ek \
         WHERE g1 IS NOT NULL AND g2 IS NOT NULL GROUP BY g1, g2",
    ] {
        let (text, notes) = run(sql, &Stats::new());
        assert!(coded(&text), "{sql}: {notes:?}\n{text}");
    }
}

/// Mixed keys from a second dimension (design section 2.2 2, a q03-like
/// report): `e` is refused (too large), `d` is coded, and the aggregate
/// groups by `(code, h1)`.
#[test]
fn a_second_dimension_key_stays_beside_the_code() {
    let mut stats = Stats::new();
    stats.rows("e", 600_000);
    let (text, notes) = run(
        "SELECT g1, g2, h1, count(*) AS n FROM f JOIN d ON fk = dk JOIN e ON ek2 = ek \
         GROUP BY g1, g2, h1",
        &stats,
    );
    assert!(coded(&text), "{notes:?}\n{text}");
    assert!(
        text.lines()
            .any(|l| l.contains("Aggregate [krishiv_group_code") && l.contains("h1#")),
        "{text}"
    );
}

/// With a session configuration `ShareCte` turns both reads of `dict` into
/// `CteRef`s of one `WITH` definition; a second optimization sees the code
/// through them ("already a code") and returns the plan unchanged.
#[test]
fn shared_dictionaries_stay_a_fixed_point() {
    let ctx = support::ctx_over(catalog());
    let mut config = krishiv_common::config::Config::new();
    // Off by default until its timed A/B passes.
    config
        .set("krishiv.optimizer.groupKeyCode", "true")
        .unwrap();
    let udfs = krishiv_udf_sdk::UdfLibrary::new();
    let mut mixed = Stats::new();
    mixed.rows("e", 600_000);
    for (sql, stats) in [
        (
            "SELECT g1, g2, count(*) AS n, sum(x) AS s FROM f JOIN d ON fk = dk GROUP BY g1, g2",
            Stats::new(),
        ),
        (
            "SELECT g1, g2, h1, count(*) AS n FROM f JOIN d ON fk = dk JOIN e ON ek2 = ek \
             GROUP BY g1, g2, h1",
            mixed,
        ),
    ] {
        let plan = support::resolve(sql, &ctx);
        let (once, _) =
            krishiv_optimizer::optimize_traced_with_udfs(&plan, &stats, &udfs, &config).unwrap();
        let text = once.root().to_string();
        assert!(coded(&text) && text.contains("CteRef"), "{sql}\n{text}");
        let (twice, steps) =
            krishiv_optimizer::optimize_traced_with_udfs(&once, &stats, &udfs, &config).unwrap();
        assert!(
            twice.structurally_eq(&once),
            "{sql}\n{text}\nagain:\n{}",
            twice.root()
        );
        assert!(
            steps
                .iter()
                .any(|s| s.rule == "GroupKeyCode" && s.note.contains("already a code")),
            "{sql}"
        );
        for seed in [3, 11] {
            let db = db(seed);
            assert_eq!(
                eval::rows(&db, once.root()),
                eval::rows(&db, plan.root()),
                "{sql} (seed {seed})"
            );
        }
    }
}

/// TPC-H q21 groups by `s_name` behind `EXISTS` / `NOT EXISTS`, which keep
/// a small part of their input: the gate reads the semi and anti joins as
/// estimated (only a null-aware `NOT IN` is read through), and refuses.
#[test]
fn tpch_q21_is_refused() {
    let ctx = support::ctx();
    let stats = crate::join_order::fixture();
    let plan = support::resolve(&support::tpch_sql(21), &ctx);
    let (optimized, steps) =
        optimize_traced_with_options(&plan, &stats, &Options::default()).unwrap();
    let notes: Vec<String> = steps
        .into_iter()
        .filter(|s| s.rule == "GroupKeyCode" && !s.note.is_empty())
        .map(|s| s.note)
        .collect();
    assert!(!coded(&optimized.root().to_string()), "{notes:?}");
    assert!(notes.iter().any(|n| n.contains("[s_name]")), "{notes:?}");
}

/// TPC-H q16 at SF10 statistics (design section 4: NDV ratio ~4.3, rows ~4):
/// `p_brand, p_type, p_size` travel as one code through the anti join.
#[test]
fn tpch_q16_codes_its_part_keys() {
    let ctx = support::ctx();
    let stats = crate::join_order::fixture();
    let plan = support::resolve(&support::tpch_sql(16), &ctx);
    let (optimized, steps) =
        optimize_traced_with_options(&plan, &stats, &Options::default()).unwrap();
    let notes: Vec<String> = steps
        .into_iter()
        .filter(|s| s.rule == "GroupKeyCode" && !s.note.is_empty())
        .map(|s| s.note)
        .collect();
    let text = optimized.root().to_string();
    assert!(coded(&text), "{notes:?}\n{text}");
    let again = optimize_with(&optimized, &stats).unwrap();
    assert!(again.structurally_eq(&optimized), "{}", again.root());
    let db = data::tpch(7, 150);
    assert_eq!(
        eval::rows(&db, optimized.root()),
        eval::rows(&db, plan.root())
    );
    insta::assert_snapshot!(text);
}
