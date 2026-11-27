// SPDX-License-Identifier: Apache-2.0
//! `PayloadAboveAggregate` (design P2b A5): an aggregate that groups by a
//! dimension's key and payload groups by the other side's key instead, and the
//! dimension joins above it. Where it fires (q10 with `nation` absorbed, other
//! grouping keys, a key the plan proves), every gate that refuses it, that it
//! is a fixed point, and that the rows never change, except with a false key
//! fact, the only way it fires over a repeated key.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_optimizer::{
    ColumnStats, Statistics, TableStats, optimize_traced_with_udfs, optimize_with_udfs,
};
use krishiv_plan::analyzer::{MemoryCatalog, TableColumn};
use krishiv_plan::{Literal, LogicalPlan, Plan, PlanNode, ReadSource, Scan};
use krishiv_udf_sdk::UdfLibrary;

use crate::join_order::fixture;
use crate::support::eval::{Db, V};
use crate::support::{self, Rng, data, eval};

mod requests;

const RULE: &str = "PayloadAboveAggregate";

/// Statistics with key facts: `inner`'s tables, and per table name the column
/// sets declared unique.
#[derive(Debug)]
struct Keyed<S> {
    inner: S,
    keys: HashMap<&'static str, Vec<Vec<String>>>,
}

impl<S> Keyed<S> {
    fn new(inner: S, keys: &[(&'static str, &[&str])]) -> Self {
        let mut map: HashMap<&'static str, Vec<Vec<String>>> = HashMap::new();
        for (t, cols) in keys {
            map.entry(t)
                .or_default()
                .push(cols.iter().map(|c| (*c).to_owned()).collect());
        }
        Self { inner, keys: map }
    }
}

fn table_name(scan: &Scan) -> Option<&str> {
    let ReadSource::Table { name, .. } = &scan.source else {
        return None;
    };
    name.last().map(String::as_str)
}

impl<S: Statistics> Statistics for Keyed<S> {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        self.inner.table(scan)
    }

    fn unique_keys(&self, scan: &Scan) -> Vec<Vec<String>> {
        table_name(scan)
            .and_then(|t| self.keys.get(t))
            .cloned()
            .unwrap_or_default()
    }
}

fn config(on: bool) -> Config {
    config_with(on, &[])
}

/// The rule on or off, and the `extra` settings.
fn config_with(on: bool, extra: &[(&str, &str)]) -> Config {
    let mut c = Config::new();
    c.set(
        "krishiv.optimizer.payloadAboveAggregate",
        if on { "true" } else { "false" },
    )
    .unwrap();
    for (k, v) in extra {
        c.set(k, v).unwrap();
    }
    c
}

/// [`optimized`] with eager aggregation off: where no fact proves a key,
/// eager aggregation takes the aggregate first and the rule then sees its
/// partial aggregate, so tests of the rule's own refusals switch it off.
fn optimized_alone(plan: &Plan, stats: &dyn Statistics, on: bool) -> (Plan, Vec<String>) {
    optimized_with(
        plan,
        stats,
        &config_with(on, &[("krishiv.optimizer.eagerAggregation", "false")]),
    )
}

/// The plan optimized with the rule on or off, and the rule's notes.
fn optimized(plan: &Plan, stats: &dyn Statistics, on: bool) -> (Plan, Vec<String>) {
    optimized_with(plan, stats, &config(on))
}

fn optimized_with(plan: &Plan, stats: &dyn Statistics, config: &Config) -> (Plan, Vec<String>) {
    let (out, steps) = optimize_traced_with_udfs(plan, stats, &UdfLibrary::new(), config).unwrap();
    let notes = steps
        .into_iter()
        .filter(|s| s.rule == RULE && !s.note.is_empty())
        .map(|s| s.note)
        .collect();
    (out, notes)
}

fn moved(notes: &[String]) -> Vec<&String> {
    notes.iter().filter(|n| n.starts_with("moved")).collect()
}

/// Optimizes `plan` with the rule on; checks it keeps the output, is a fixed
/// point, and (`dbs`) returns the unoptimised plan's rows. Returns the plan,
/// its text and the notes.
fn check(plan: &Plan, stats: &dyn Statistics, dbs: &[Db]) -> (Plan, String, Vec<String>) {
    check_in(plan, stats, dbs, &config(true))
}

/// [`check`] under `config`.
fn check_in(
    plan: &Plan,
    stats: &dyn Statistics,
    dbs: &[Db],
    config: &Config,
) -> (Plan, String, Vec<String>) {
    let (out, notes) = optimized_with(plan, stats, config);
    let text = out.root().to_string();
    assert_eq!(
        out.root().resolved_output(),
        plan.root().resolved_output(),
        "{text}"
    );
    let again = optimize_with_udfs(&out, stats, &UdfLibrary::new(), config).unwrap();
    assert!(
        again.structurally_eq(&out),
        "not a fixed point:\n{text}\nagain:\n{}",
        again.root()
    );
    for db in dbs {
        assert_eq!(
            eval::rows(db, out.root()),
            eval::rows(db, plan.root()),
            "{text}"
        );
    }
    (out, text, notes)
}

// ---------------------------------------------------------------- TPC-H

/// TPC-H q10 with the keys `customer.c_custkey` and `nation.n_nationkey`
/// declared: the aggregate groups by `o_custkey` over `lineitem ⋈ orders'`,
/// and `customer ⋈ nation` (nation absorbed from above the customer join)
/// joins above it.
#[test]
fn q10_groups_by_the_order_s_customer_and_joins_customer_and_nation_above() {
    let stats = Keyed::new(
        fixture(),
        &[("customer", &["c_custkey"]), ("nation", &["n_nationkey"])],
    );
    let plan = support::resolve(&support::tpch_sql(10), &support::ctx());
    let (_, text, notes) = check(&plan, &stats, &[data::tpch(7, 150)]);
    assert_eq!(moved(&notes).len(), 1, "{notes:?}\n{text}");
    let note = moved(&notes)[0];
    assert!(note.contains("o_custkey = c_custkey"), "{note}");
    assert!(note.contains("absorbed [nation]"), "{note}");
    assert!(note.contains("statistics"), "{note}");
    insta::assert_snapshot!("payload_above_q10", text);
    // The aggregate groups by the order's customer only.
    let agg = text.lines().find(|l| l.contains("Aggregate [")).unwrap();
    assert!(agg.contains("Aggregate [o_custkey#9], "), "{agg}");
}

/// Without key facts no TPC-H plan changes, with the switch on (its default)
/// or off.
#[test]
fn tpch_plans_do_not_change_off_or_without_facts() {
    let stats = Keyed::new(fixture(), &[]);
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &support::ctx());
        let default =
            optimize_with_udfs(&plan, &stats, &UdfLibrary::new(), &Config::new()).unwrap();
        let (off, notes_off) = optimized(&plan, &stats, false);
        assert!(off.structurally_eq(&default), "q{q:02}");
        assert!(notes_off.is_empty(), "q{q:02}: {notes_off:?}");
        let (on, notes) = optimized(&plan, &stats, true);
        assert!(moved(&notes).is_empty(), "q{q:02}: {notes:?}");
        assert!(on.structurally_eq(&off), "q{q:02}:\n{}", on.root());
    }
}

// ---------------------------------------------------------------- small tables

const TABLES: &[(&str, &[(&str, &str)])] = &[
    (
        "f",
        &[
            ("fk", "int"),
            ("fv", "decimal(10,2)"),
            ("fi", "int"),
            ("fl", "bigint"),
            ("fd", "double"),
            ("fs", "string"),
        ],
    ),
    (
        "d",
        &[
            ("dk", "int"),
            ("dname", "string"),
            ("dp", "string"),
            ("dek", "int"),
            ("dd", "double"),
        ],
    ),
    ("e", &[("ek", "int"), ("ename", "string")]),
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

/// A column's name, distinct count and range.
type ColumnSpec = (&'static str, u64, Option<(f64, f64)>);

/// `f` 1M rows, `d` 1,000, `e` 10; `fk` holds `d`'s 1,000 keys, `dp` 10
/// values, `dek` `e`'s 10 keys. `fl` has no range.
#[derive(Debug)]
struct Small;

impl Statistics for Small {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let (rows, cols): (u64, &[ColumnSpec]) = match table_name(scan)? {
            "f" => (
                1_000_000,
                &[
                    ("fk", 1_000, Some((0.0, 999.0))),
                    ("fv", 100_000, Some((-1_000.0, 1_000.0))),
                    ("fi", 100, Some((0.0, 100.0))),
                    ("fl", 1_000_000, None),
                    ("fd", 1_000, Some((0.0, 999.0))),
                    ("fs", 3, None),
                ],
            ),
            "d" => (
                1_000,
                &[
                    ("dk", 1_000, Some((0.0, 999.0))),
                    ("dname", 1_000, None),
                    ("dp", 10, None),
                    ("dek", 10, Some((0.0, 9.0))),
                    ("dd", 1_000, Some((0.0, 999.0))),
                ],
            ),
            "e" => (10, &[("ek", 10, Some((0.0, 9.0))), ("ename", 10, None)]),
            _ => return None,
        };
        let columns = cols
            .iter()
            .map(|(c, ndv, range)| {
                let stats = ColumnStats {
                    distinct: Some(*ndv),
                    nulls: Some(0),
                    min: range.map(|r| Literal::Double(r.0)),
                    max: range.map(|r| Literal::Double(r.1)),
                    ..ColumnStats::default()
                };
                ((*c).to_owned(), stats)
            })
            .collect();
        Some(TableStats { rows, columns })
    }
}

/// `d` unique on `dk`, `e` on `ek`.
fn facts() -> Keyed<Small> {
    Keyed::new(Small, &[("d", &["dk"]), ("e", &["ek"])])
}

fn no_facts() -> Keyed<Small> {
    Keyed::new(Small, &[])
}

/// Small data. `d`'s keys are unique among its non-NULL keys (two rows have a
/// NULL key) unless `dup`: then key 1 repeats with the same payload and key 2
/// with another. `f` holds NULL keys and keys `d` lacks.
fn db(seed: u64, dup: bool) -> Db {
    let mut r = Rng(seed | 1);
    let mut db = Db::default();
    let word = |r: &mut Rng, w: &[&str]| V::Str(r.pick(w).into());
    let mut d = Vec::new();
    for i in 0..8 {
        let name = word(&mut r, &["ann", "bob", "cy"]);
        let p = word(&mut r, &["a", "b"]);
        let ek = if r.range(0, 5) == 0 {
            V::Null
        } else {
            V::int(r.range(0, 4))
        };
        d.push(vec![
            V::int(i),
            name,
            p,
            ek,
            V::Num(i128::from(i) * eval::SCALE),
        ]);
    }
    for _ in 0..2 {
        let name = word(&mut r, &["ann", "dee"]);
        d.push(vec![V::Null, name, V::Str("a".into()), V::int(1), V::Null]);
    }
    if dup {
        let same = d[1].clone();
        d.push(same);
        let mut other = d[2].clone();
        other[1] = V::Str("zed".into());
        d.push(other);
    }
    let mut f = Vec::new();
    for _ in 0..40 {
        let k = if r.range(0, 6) == 0 {
            V::Null
        } else {
            V::int(r.range(0, 10))
        };
        let fd = V::Num(i128::from(r.range(0, 9)) * eval::SCALE);
        f.push(vec![
            k,
            V::cents(r.range(-5_000, 5_000)),
            V::int(r.range(0, 100)),
            V::int(r.range(0, 1_000)),
            fd,
            word(&mut r, &["x", "y", "z"]),
        ]);
    }
    let e = (0..4)
        .map(|i| vec![V::int(i), V::Str(format!("e{i}").into())])
        .collect();
    for (name, rows) in [("d", d), ("f", f), ("e", e)] {
        let cols = TABLES.iter().find(|t| t.0 == name).unwrap().1;
        db.tables.insert(
            name.to_owned(),
            (cols.iter().map(|c| c.0.to_owned()).collect(), rows),
        );
    }
    db
}

fn dbs(dup: bool) -> Vec<Db> {
    [3, 11, 29].into_iter().map(|s| db(s, dup)).collect()
}

fn resolve(sql: &str) -> Plan {
    support::resolve(sql, &support::ctx_over(catalog()))
}

/// The rule fires on `sql` with `stats`, is exact on data with unique keys
/// (NULL keys on both sides included) and a fixed point; returns the text
/// and the move note.
fn fires(sql: &str, stats: &dyn Statistics) -> (String, String) {
    let plan = resolve(sql);
    let (_, text, notes) = check(&plan, stats, &dbs(false));
    let m = moved(&notes);
    assert_eq!(m.len(), 1, "{sql}\n{notes:?}\n{text}");
    (text, m[0].clone())
}

/// The rule refuses `sql` with `stats` and notes a refusal containing `why`;
/// the plan is the one optimized with the rule off.
fn refused(sql: &str, stats: &dyn Statistics, why: &str) {
    let plan = resolve(sql);
    let (on, notes) = optimized_alone(&plan, stats, true);
    let (off, _) = optimized_alone(&plan, stats, false);
    assert!(moved(&notes).is_empty(), "{sql}\n{notes:?}\n{}", on.root());
    assert!(
        on.structurally_eq(&off),
        "{sql}\n{}\noff:\n{}",
        on.root(),
        off.root()
    );
    assert!(
        notes.iter().any(|n| n.contains(why)),
        "{sql}: no note with {why:?}: {notes:?}"
    );
}

const PLAIN: &str = "SELECT dk, dname, dp, sum(fv), count(*) FROM f, d WHERE fk = dk \
                     GROUP BY dk, dname, dp";

#[test]
fn a_dimension_key_proven_by_a_fact_moves_its_payload_above() {
    let (text, note) = fires(PLAIN, &facts());
    assert!(note.contains("fk = dk"), "{note}");
    assert!(note.contains("statistics"), "{note}");
    insta::assert_snapshot!("payload_above_plain", text);
}

#[test]
fn an_inner_join_above_reading_only_the_dimension_is_absorbed() {
    let sql = "SELECT dk, dname, ename, sum(fv) FROM f, d, e WHERE fk = dk AND dek = ek \
               GROUP BY dk, dname, ename";
    let (text, note) = fires(sql, &facts());
    // Absorbed from above the join, or proven where ordering put it below.
    assert!(note.contains("key [ek]"), "{note}");
    // `e` joins with `d`, above the aggregate, which groups by `fk` only.
    let agg = text.lines().find(|l| l.contains("Aggregate [")).unwrap();
    assert!(agg.contains("Aggregate [fk#0], "), "{text}");
}

#[test]
fn other_grouping_keys_of_the_fact_side_stay_in_the_aggregate() {
    let sql = "SELECT dk, dname, fs, sum(fv) FROM f, d WHERE fk = dk GROUP BY dk, dname, fs";
    let (text, _) = fires(sql, &facts());
    let agg = text.lines().find(|l| l.contains("Aggregate [")).unwrap();
    assert!(agg.contains("fs#") && agg.contains("fk#"), "{agg}");
    assert!(!agg.contains("dname#"), "{agg}");
}

#[test]
fn a_dimension_grouped_by_its_key_in_the_plan_needs_no_fact() {
    let sql = "SELECT t.k, t.n, sum(fv) FROM f, \
               (SELECT dk AS k, max(dname) AS n FROM d GROUP BY dk) t \
               WHERE fk = t.k GROUP BY t.k, t.n";
    let (_, note) = fires(sql, &no_facts());
    assert!(note.contains("the plan"), "{note}");
}

#[test]
fn without_a_fact_the_rule_refuses() {
    refused(PLAIN, &no_facts(), "unique");
}

#[test]
fn the_switch_off_leaves_the_plan_and_is_on_by_default() {
    let plan = resolve(PLAIN);
    let (off, notes) = optimized(&plan, &facts(), false);
    assert!(notes.is_empty(), "{notes:?}");
    let (on, _) = optimized(&plan, &facts(), true);
    assert!(!on.structurally_eq(&off));
    let default = optimize_with_udfs(&plan, &facts(), &UdfLibrary::new(), &Config::new()).unwrap();
    assert!(default.structurally_eq(&on), "{}", default.root());
}

#[test]
fn a_call_reading_the_payload_refuses() {
    refused(
        "SELECT dk, dname, sum(dek) FROM f, d WHERE fk = dk GROUP BY dk, dname",
        &facts(),
        "reads the dimension",
    );
    refused(
        "SELECT dk, dname, sum(fv) FILTER (WHERE dp = 'a') FROM f, d WHERE fk = dk \
         GROUP BY dk, dname",
        &facts(),
        "reads the dimension",
    );
}

#[test]
fn an_argument_that_may_raise_refuses() {
    refused(
        "SELECT dk, dname, sum(fi + 1) FROM f, d WHERE fk = dk GROUP BY dk, dname",
        &facts(),
        "may raise",
    );
    // An unmatched row's string could fail the cast where Spark never casts it.
    refused(
        "SELECT dk, dname, sum(cast(fs AS int)) FROM f, d WHERE fk = dk GROUP BY dk, dname",
        &facts(),
        "may raise",
    );
}

#[test]
fn an_unprovable_sum_bound_refuses() {
    refused(
        "SELECT dk, dname, sum(fl) FROM f, d WHERE fk = dk GROUP BY dk, dname",
        &facts(),
        "overflow",
    );
}

#[test]
fn a_floating_point_key_refuses() {
    refused(
        "SELECT dd, dname, sum(fv) FROM f, d WHERE fd = dd GROUP BY dd, dname",
        &Keyed::new(Small, &[("d", &["dd"])]),
        "key",
    );
}

/// A floating-point `sum` or `avg` would round differently once the join below
/// it is gone (summation order), so it never moves.
#[test]
fn a_floating_point_sum_refuses() {
    for call in ["sum(fd)", "avg(fd)"] {
        refused(
            &format!("SELECT dk, dname, {call} FROM f, d WHERE fk = dk GROUP BY dk, dname"),
            &facts(),
            "not known to accumulate",
        );
    }
}

#[test]
fn a_collated_key_never_moves() {
    let sql = "SELECT dk, dname, sum(fv) FROM f, d \
               WHERE fs COLLATE UTF8_LCASE = dname COLLATE UTF8_LCASE GROUP BY dk, dname";
    let plan = resolve(sql);
    let (on, notes) = optimized(&plan, &Keyed::new(Small, &[("d", &["dname"])]), true);
    assert!(moved(&notes).is_empty(), "{notes:?}\n{}", on.root());
}

#[test]
fn a_dimension_key_not_grouped_refuses() {
    refused(
        "SELECT dname, dp, sum(fv) FROM f, d WHERE fk = dk GROUP BY dname, dp",
        &facts(),
        "not all grouped",
    );
}

#[test]
fn grouping_sets_refuse() {
    refused(
        "SELECT dk, dname, sum(fv) FROM f, d WHERE fk = dk GROUP BY ROLLUP(dk, dname)",
        &facts(),
        "grouping sets",
    );
}

#[test]
fn a_selective_dimension_refuses_on_retention() {
    refused(
        "SELECT dk, dname, sum(fv) FROM f, d WHERE fk = dk AND dp = 'a' GROUP BY dk, dname",
        &facts(),
        "retention",
    );
}

/// The aggregate's input read by a second aggregate: rebuilding it would copy it.
#[test]
fn a_path_read_by_another_parent_refuses() {
    let plan = resolve(PLAIN);
    let root = Arc::clone(plan.root());
    // Find the aggregate and read its input twice.
    let mut agg = None;
    let mut stack = vec![Arc::clone(&root)];
    while let Some(p) = stack.pop() {
        if matches!(p.node, PlanNode::Aggregate { .. }) {
            agg = Some(p);
            break;
        }
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    let agg = agg.unwrap();
    let twin = Arc::new(LogicalPlan::clone(&agg));
    let union = Arc::new(LogicalPlan::new(PlanNode::Union {
        inputs: vec![agg, twin],
        by_name: false,
        allow_missing_columns: false,
    }));
    let plan = Plan::try_new(union).unwrap();
    let (on, notes) = optimized(&plan, &facts(), true);
    let (off, _) = optimized(&plan, &facts(), false);
    assert!(moved(&notes).is_empty(), "{notes:?}\n{}", on.root());
    assert!(on.structurally_eq(&off), "{}", on.root());
}

// ---------------------------------------------------------------- differential

/// A repeated key, with the same or another payload: without a fact the rule
/// cannot fire, and the rows are kept.
#[test]
fn a_repeated_key_without_a_fact_keeps_the_rows() {
    let plan = resolve(PLAIN);
    let (out, notes) = optimized(&plan, &no_facts(), true);
    assert!(moved(&notes).is_empty(), "{notes:?}");
    for db in dbs(true) {
        assert_eq!(eval::rows(&db, out.root()), eval::rows(&db, plan.root()));
    }
}

/// A false fact is the only way the rule fires over a repeated key, and then
/// the rows change: the fact is the whole proof of FD2.
#[test]
fn only_a_false_fact_fires_over_a_repeated_key_and_it_changes_the_rows() {
    let plan = resolve(PLAIN);
    let (out, notes) = optimized(&plan, &facts(), true);
    assert_eq!(moved(&notes).len(), 1, "{notes:?}");
    let differs = dbs(true)
        .iter()
        .any(|db| eval::rows(db, out.root()) != eval::rows(db, plan.root()));
    assert!(differs, "a false fact went unnoticed:\n{}", out.root());
}

/// NULL keys on both sides, keys the dimension lacks, and an absorbed join.
#[test]
fn null_and_unmatched_keys_keep_the_rows() {
    for sql in [
        PLAIN,
        "SELECT dk, dname, ename, sum(fv), max(fs) FROM f, d, e WHERE fk = dk AND dek = ek \
         GROUP BY dk, dname, ename",
        "SELECT dk, dname, fs, count(fi), avg(fv) FROM f, d WHERE fk = dk \
         GROUP BY dk, dname, fs",
    ] {
        fires(sql, &facts());
    }
}

// ---------------------------------------------------------------- sweeps with keys

/// One query of a sweep under `config`: where the rule moves nothing the plan
/// is the rule-off plan; where it moves, [`check_in`] (output, fixed point,
/// rows over `dbs`). Returns the moves. Where the rule-off plan is itself not
/// a fixed point under a session configuration, only output and rows.
fn sweep_one(plan: &Plan, stats: &dyn Statistics, dbs: &[Db], config: &Config) -> usize {
    let (on, notes) = optimized_with(plan, stats, config);
    let moves = moved(&notes).len();
    let mut off_config = config.clone();
    off_config
        .set("krishiv.optimizer.payloadAboveAggregate", "false")
        .unwrap();
    let (off, _) = optimized_with(plan, stats, &off_config);
    if moves == 0 {
        assert!(on.structurally_eq(&off), "{}", on.root());
        return moves;
    }
    let off_again = optimize_with_udfs(&off, stats, &UdfLibrary::new(), &off_config).unwrap();
    if off_again.structurally_eq(&off) {
        check_in(plan, stats, dbs, config);
    } else {
        // Not a fixed point without the rule either: under a session
        // configuration `ShareCte`'s `CteRef`s gain inferred `isnotnull`
        // filters on a second pass (TPC-DS q23, before this rule). Output and
        // rows here, and the full check with `ShareCte` off, where the
        // rule-off plan is a fixed point.
        assert_eq!(on.root().resolved_output(), plan.root().resolved_output());
        for db in dbs {
            assert_eq!(eval::rows(db, on.root()), eval::rows(db, plan.root()));
        }
        let mut unshared = config.clone();
        unshared
            .set("krishiv.sql.optimizer.shareCteInstances", "false")
            .unwrap();
        check_in(plan, stats, dbs, &unshared);
    }
    moves
}

/// The rule on, `SubqueryAsWindow` off (the reference interpreter runs no
/// window functions).
fn sweep_config() -> Config {
    config_with(true, &[("krishiv.optimizer.subqueryAsWindow", "false")])
}

/// Every TPC-H primary key.
const TPCH_KEYS: &[(&str, &[&str])] = &[
    ("customer", &["c_custkey"]),
    ("lineitem", &["l_orderkey", "l_linenumber"]),
    ("nation", &["n_nationkey"]),
    ("orders", &["o_orderkey"]),
    ("part", &["p_partkey"]),
    ("partsupp", &["ps_partkey", "ps_suppkey"]),
    ("region", &["r_regionkey"]),
    ("supplier", &["s_suppkey"]),
];

/// With every primary key declared, each TPC-H query keeps its output, is a
/// fixed point and returns the unoptimised plan's rows; the queries the rule
/// changes are listed, so a change to the list is a reviewed decision.
#[test]
fn tpch_with_every_key_declared_keeps_rows_and_is_a_fixed_point() {
    let stats = Keyed::new(fixture(), TPCH_KEYS);
    let db = data::tpch(7, 150);
    let mut changed = Vec::new();
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &support::ctx());
        let moves = sweep_one(&plan, &stats, std::slice::from_ref(&db), &sweep_config());
        if moves > 0 {
            changed.push((q, moves));
        }
    }
    assert!(changed.iter().any(|(q, _)| *q == 10), "{changed:?}");
    insta::assert_debug_snapshot!("payload_above_tpch_changed", changed);
}

/// TPC-H q18 with its keys: the aggregate groups by the customer's and the
/// order's columns; the rule keeps the order's (`K_X`) and `o_custkey` and
/// joins `customer` above it.
#[test]
fn q18_with_its_keys_keeps_rows() {
    let stats = Keyed::new(fixture(), TPCH_KEYS);
    let plan = support::resolve(&support::tpch_sql(18), &support::ctx());
    let (_, text, notes) = check_in(
        &plan,
        &stats,
        &[data::tpch(7, 150), data::tpch(11, 150)],
        &sweep_config(),
    );
    assert_eq!(moved(&notes).len(), 1, "{notes:?}\n{text}");
    insta::assert_snapshot!("payload_above_q18", text);
}

/// The surrogate keys of the TPC-DS dimensions.
const TPCDS_KEYS: &[(&str, &[&str])] = &[
    ("call_center", &["cc_call_center_sk"]),
    ("catalog_page", &["cp_catalog_page_sk"]),
    ("customer", &["c_customer_sk"]),
    ("customer_address", &["ca_address_sk"]),
    ("customer_demographics", &["cd_demo_sk"]),
    ("date_dim", &["d_date_sk"]),
    ("household_demographics", &["hd_demo_sk"]),
    ("income_band", &["ib_income_band_sk"]),
    ("item", &["i_item_sk"]),
    ("promotion", &["p_promo_sk"]),
    ("reason", &["r_reason_sk"]),
    ("ship_mode", &["sm_ship_mode_sk"]),
    ("store", &["s_store_sk"]),
    ("time_dim", &["t_time_sk"]),
    ("warehouse", &["w_warehouse_sk"]),
    ("web_page", &["wp_web_page_sk"]),
    ("web_site", &["web_site_sk"]),
];

/// Over the 99 TPC-DS queries with SF1 row counts (the optimizer tests have
/// no TPC-DS footer fixture) and the dimensions' keys declared: each keeps
/// its output and is a fixed point; the queries the rule changes are listed.
/// **No row comparison:** the reference interpreter has no TPC-DS data, so
/// the moves here (q23, q64) are checked for shape only; their rows are the
/// TPC-H sweep's and the executor differential's argument, plus review.
#[test]
fn tpcds_with_keys_declared_is_a_fixed_point() {
    let stats = Keyed::new(crate::late_payload::TpcdsRows, TPCDS_KEYS);
    let ctx = support::ctx_over(crate::tpcds::catalog());
    let mut changed = Vec::new();
    for q in 1..=99 {
        let Some(plan) = support::try_resolve(&crate::tpcds::sql(q), &ctx) else {
            continue;
        };
        let moves = sweep_one(&plan, &stats, &[], &sweep_config());
        if moves > 0 {
            changed.push((q, moves));
        }
    }
    insta::assert_debug_snapshot!("payload_above_tpcds_changed", changed);
}

/// With the group-key code on too, q10 and a two-dimension shape are fixed
/// points (review m5: the code may fire on the new aggregate).
#[test]
fn with_the_group_key_code_on_too_the_plan_is_a_fixed_point() {
    let both = config_with(
        true,
        &[
            ("krishiv.optimizer.groupKeyCode", "true"),
            ("krishiv.optimizer.subqueryAsWindow", "false"),
        ],
    );
    let stats = Keyed::new(fixture(), TPCH_KEYS);
    for q in [10, 18, 3, 5] {
        let plan = support::resolve(&support::tpch_sql(q), &support::ctx());
        check_in(&plan, &stats, &[data::tpch(7, 150)], &both);
    }
    let plan = resolve(
        "SELECT dk, dname, dp, ename, sum(fv) FROM f, d, e WHERE fk = dk AND fi = ek \
         GROUP BY dk, dname, dp, ename",
    );
    check_in(&plan, &facts(), &dbs(false), &both);
}

// ---------------------------------------------------------------- accepted shapes

/// Shapes the rule accepts, each exact on data with NULL and unmatched keys.
#[test]
fn accepted_shapes_keep_the_rows() {
    let multi = Keyed::new(Small, &[("d", &["dk", "dek"]), ("e", &["ek"])]);
    let cases: &[(&str, &dyn Statistics)] = &[
        // A DISTINCT call sees the same rows a group.
        (
            "SELECT dk, dname, count(DISTINCT fs) FROM f, d WHERE fk = dk GROUP BY dk, dname",
            &facts(),
        ),
        // A two-column key.
        (
            "SELECT dk, dek, dname, sum(fv) FROM f, d WHERE fk = dk AND fi = dek \
             GROUP BY dk, dek, dname",
            &multi,
        ),
        // A kept inner join of the fact side.
        (
            "SELECT dk, dname, sum(fv), max(ename) FROM f, d, e WHERE fk = dk AND fi = ek \
             GROUP BY dk, dname",
            &facts(),
        ),
        // Kept semi and anti joins.
        (
            "SELECT dk, dname, sum(fv) FROM f, d WHERE fk = dk \
             AND EXISTS (SELECT 1 FROM e WHERE ek = fi) GROUP BY dk, dname",
            &facts(),
        ),
        (
            "SELECT dk, dname, sum(fv) FROM f, d WHERE fk = dk \
             AND NOT EXISTS (SELECT 1 FROM e WHERE ek = fi) GROUP BY dk, dname",
            &facts(),
        ),
        // A kept left outer join of the fact side.
        (
            "SELECT dk, dname, sum(fv), max(ename) FROM f JOIN d ON fk = dk \
             LEFT JOIN e ON fi = ek GROUP BY dk, dname",
            &facts(),
        ),
        // The proof through a renaming projection.
        (
            "SELECT t.k, t.n, sum(fv) FROM f, (SELECT dk AS k, dname AS n FROM d) t \
             WHERE fk = t.k GROUP BY t.k, t.n",
            &facts(),
        ),
        // The proof through a `Shift` (a second instance of `d`).
        (
            "SELECT b.dk, b.dname, sum(fv) FROM f JOIN d a ON fk = a.dk \
             JOIN d b ON fk = b.dk GROUP BY b.dk, b.dname",
            &facts(),
        ),
    ];
    for (sql, stats) in cases {
        fires(sql, *stats);
    }
}

/// A left outer join above the dimension's join that reads the dimension
/// stays: the dimension cannot move above it.
#[test]
fn an_outer_join_reading_the_dimension_refuses() {
    refused(
        "SELECT dk, dname, ename, sum(fv) FROM f JOIN d ON fk = dk \
         LEFT JOIN e ON dek = ek GROUP BY dk, dname, ename",
        &facts(),
        "join on the path reads the dimension",
    );
}

/// An inner join above the dimension's that the dimension cannot absorb: `e`
/// has no key fact.
#[test]
fn a_join_on_a_key_not_proven_unique_refuses() {
    refused(
        "SELECT dk, dname, ename, sum(fv) FROM f, d, e WHERE fk = dk AND dek = ek \
         GROUP BY dk, dname, ename",
        &Keyed::new(Small, &[("d", &["dk"])]),
        "unique",
    );
}
