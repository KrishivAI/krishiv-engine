// SPDX-License-Identifier: Apache-2.0
//! Build-side projection (S4'): where an expression above an inner join is
//! computed on the build input, the plan returns what it returns with the
//! rule off, under ANSI on and off; the gates keep the other expressions
//! above the join.

use super::partial_aggregate::{Row, physical, runtime, sorted, tables};
use super::run;
use super::tpch::{Order, TABLES, Tpch, parquet_files_with, sql};

const OFF: (&str, &str) = ("krishiv.exec.join.buildProjection", "false");

/// 240 rows over keys 0 to 11, with NULLs in every other column.
fn many() -> Vec<Row> {
    (0..240_i64)
        .map(|n| {
            (
                n % 12,
                (n % 7 != 0).then(|| i32::try_from(n).unwrap()),
                (n % 5 != 0).then(|| i128::from(n) * 100),
                (n % 3 != 0).then(|| u8::try_from(n % 9).unwrap()),
            )
        })
        .collect()
}

/// Every key once, and two keys `f` does not hold.
const KEYS: [i64; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

/// Joins written with `f` on the right, so `f` is the build input.
const FIRES: [&str; 4] = [
    // `decimal(10,2) + cast(i AS decimal(10,0))` is `decimal(13,2)`: 8 bytes
    // (a `Decimal64`) for 12.
    "SELECT w.wk, f.d + f.i AS di FROM w JOIN f ON w.wk = f.k",
    // An unaliased item keeps its name.
    "SELECT w.wk, f.d + f.i FROM w JOIN f ON w.wk = f.k",
    // Part of a mixed expression; the probe column stays above.
    "SELECT (f.d + f.i) * w.wk FROM w JOIN f ON w.wk = f.k",
    // An aggregate's argument.
    "SELECT w.wk, sum(f.d + f.i), count(*) FROM w JOIN f ON w.wk = f.k GROUP BY w.wk",
];

/// Each stays above the join, for the reason in its comment.
const STAYS: [&str; 4] = [
    // `d` is still read above: the sum would free only `i` (4 bytes for 8).
    "SELECT w.wk, f.d + f.i, f.d FROM w JOIN f ON w.wk = f.k",
    // A `decimal(21,2)` product (16 bytes) frees only 12.
    "SELECT w.wk, f.d * f.i FROM w JOIN f ON w.wk = f.k",
    // A string result is variable-width.
    "SELECT w.wk, concat(f.s, f.s) FROM w JOIN f ON w.wk = f.k",
    // An aggregate's grouping key never moves.
    "SELECT f.d + f.i, count(*) FROM w JOIN f ON w.wk = f.k GROUP BY f.d + f.i",
];

/// Whether a join of the plan's text tree computes something on its build
/// side (the join line's `computes on build:` detail).
fn projects_below_join(text: &str) -> bool {
    text.lines()
        .any(|l| l.contains("Join") && l.contains("computes on build:"))
}

fn answer(t: &Tpch, q: &str) -> (Vec<String>, String) {
    let plan = t.optimized(q, Order::Written).unwrap();
    let runnable = physical(t, &plan).unwrap();
    let text = runnable.to_string();
    (sorted(&run(&runnable).unwrap()), text)
}

#[test]
fn moved_expressions_return_what_the_join_returns() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    tables(dir.path(), &many(), &KEYS, &rt);
    let sources = [
        ("f", vec!["f.parquet".to_owned()]),
        ("w", vec!["w.parquet".to_owned()]),
    ];
    for ansi in [true, false] {
        let on = parquet_files_with(dir.path(), &sources, ansi, &[], &rt);
        let off = parquet_files_with(dir.path(), &sources, ansi, &[OFF], &rt);
        for q in FIRES {
            let (rows_on, plan_on) = answer(&on, q);
            let (rows_off, plan_off) = answer(&off, q);
            assert_eq!(rows_on, rows_off, "{q} (ansi {ansi})");
            assert!(!rows_on.is_empty(), "{q}");
            assert!(
                projects_below_join(&plan_on),
                "{q} (ansi {ansi}):\n{plan_on}"
            );
            assert!(!projects_below_join(&plan_off), "{q}:\n{plan_off}");
        }
        for q in STAYS {
            let (rows_on, plan_on) = answer(&on, q);
            let (rows_off, _) = answer(&off, q);
            assert_eq!(rows_on, rows_off, "{q} (ansi {ansi})");
            assert!(
                !projects_below_join(&plan_on),
                "{q} (ansi {ansi}):\n{plan_on}"
            );
        }
    }
}

#[test]
fn an_unaliased_moved_item_keeps_its_column_name() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = tables(dir.path(), &many(), &KEYS, &rt);
    let q = FIRES[1];
    let plan = t.optimized(q, Order::Written).unwrap();
    let runnable = physical(&t, &plan).unwrap();
    let batches = run(&runnable).unwrap();
    let names: Vec<String> = batches[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let off = parquet_files_with(
        dir.path(),
        &[
            ("f", vec!["f.parquet".to_owned()]),
            ("w", vec!["w.parquet".to_owned()]),
        ],
        true,
        &[OFF],
        &rt,
    );
    let plan_off = off.optimized(q, Order::Written).unwrap();
    let off_batches = run(&physical(&off, &plan_off).unwrap()).unwrap();
    let names_off: Vec<String> = off_batches[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, names_off);
}

/// TPC-H q09 at SF 0.01: the orders join computes `amount` on its build
/// side (four decimals for one) and returns what it returns without the
/// rule. All 22 queries against Spark's answers with the rule on (its
/// default) are `tpch::tpch_sf0_01_matches_spark`.
#[test]
fn tpch_q09_computes_amount_on_the_orders_join_build() {
    let rt = runtime();
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let sources: Vec<(&str, Vec<String>)> = TABLES
        .iter()
        .map(|t| (*t, vec![format!("{t}.parquet")]))
        .collect();
    let on = parquet_files_with(&dir, &sources, true, &[], &rt);
    let off = parquet_files_with(&dir, &sources, true, &[OFF], &rt);
    let q = sql(9);
    let run_with = |t: &Tpch| {
        let plan = t.optimized(&q, Order::Statistics).unwrap();
        let runnable = physical(t, &plan).unwrap();
        (sorted(&run(&runnable).unwrap()), runnable.to_string())
    };
    let (rows_on, plan_on) = run_with(&on);
    let (rows_off, plan_off) = run_with(&off);
    assert_eq!(rows_on, rows_off);
    assert!(
        plan_on
            .lines()
            .any(|l| l.contains("Join") && l.contains("computes on build: amount#")),
        "{plan_on}"
    );
    assert!(!plan_off.contains("computes on build"), "{plan_off}");
}
