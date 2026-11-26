// SPDX-License-Identifier: Apache-2.0
//! Bounds from a hash join's comparison residual (design P3-0.3.1 D2,
//! `krishiv.runtimeFilter.residualBounds`) never change results, and drop
//! probe rows no build row can match: randomized joins over Parquet files
//! with `<`, `<=`, `>`, `>=` conjuncts on integers, decimals and floats
//! (signed zeros, NaN, NULLs), operands either way round, inner and semi
//! joins (bounded), and outer joins whose probe side is preserved (not).

use proptest::prelude::*;

use super::runtime_filters::{Counters, configured, run, sorted, table, write};
use super::tpch::parquet_files;

const OFF: &[(&str, &str)] = &[("krishiv.runtimeFilter.enabled", "false")];
const NO_BOUNDS: &[(&str, &str)] = &[("krishiv.runtimeFilter.residualBounds", "false")];
const BOUNDS: &[(&str, &str)] = &[
    // No adaptive switch-off: every planned bound runs on every row.
    ("krishiv.runtimeFilter.adaptive.maxPassPercent", "100"),
];

/// Joins with a comparison residual; `big` is the probe side.
const JOINS: [&str; 10] = [
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k32 = s.k32 AND b.v < s.v",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k32 = s.k32 AND s.v >= b.v",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k64 = s.k64 AND b.v > s.v",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k32 = s.k32 AND b.d102 <= s.d102",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k32 = s.k32 AND b.x < s.x",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k32 = s.k32 AND b.x >= s.x",
    "SELECT b.v, s.v FROM big b JOIN small s ON b.k32 = s.k32 AND b.f < s.x",
    "SELECT b.v FROM big b LEFT SEMI JOIN small s ON b.k32 = s.k32 AND b.v <= s.v",
    "SELECT b.v, s.v FROM big b RIGHT JOIN small s ON b.k32 = s.k32 AND b.v < s.v",
    "SELECT b.v, s.v FROM big b LEFT JOIN small s ON b.k32 = s.k32 AND b.v < s.v",
];

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]
    #[test]
    fn residual_bounds_change_no_result(
        seed in 1u64..u64::MAX,
        domain in prop_oneof![Just(4i32), Just(40)],
        nulls in 0u32..30,
        small_rows in prop_oneof![Just(0usize), Just(5), Just(40)],
    ) {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut rng = seed;
        write(dir.path(), "big.parquet", &table(&mut rng, 1500, domain, nulls));
        let small = table(&mut rng, small_rows, domain, nulls);
        // The build stores only rows with a key: with none, no bound has a
        // value and nothing more is pruned.
        let keyed = small.column(0).null_count() < small.num_rows();
        write(dir.path(), "small.parquet", &small);
        let sources = [
            ("big", vec!["big.parquet".to_owned()]),
            ("small", vec!["small.parquet".to_owned()]),
        ];
        let mut t = parquet_files(dir.path(), &sources, true, &rt);
        let (mut with, mut without) = (Counters::default(), Counters::default());
        for q in JOINS {
            configured(&mut t, OFF);
            let off = sorted(&run(&t.plan_sql(q).unwrap()).0);
            for (mode, total) in [(NO_BOUNDS, &mut without), (BOUNDS, &mut with)] {
                configured(&mut t, mode);
                let plan = t.plan_sql(q).unwrap();
                let (got, c) = run(&plan);
                total.rows_dropped += c.rows_dropped;
                total.units_pruned += c.units_pruned;
                prop_assert_eq!(&sorted(&got), &off, "{}:\n{}", q, plan);
            }
        }
        // `small`'s `v` holds row numbers below its row count, `big`'s up
        // to 1,499 in file order: an upper bound on `b.v` prunes most of
        // `big`'s row groups by their statistics.
        if small_rows == 5 && keyed {
            prop_assert!(with.units_pruned > without.units_pruned, "{:?} {:?}", with, without);
        }
    }
}

/// Which conjuncts plan a bound: the side follows the operator and the
/// operands' order; a float's lower bound, a left join, a residual that
/// may raise and the setting turned off plan none.
#[test]
fn bounds_are_planned_where_sound() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut rng = 7;
    write(dir.path(), "big.parquet", &table(&mut rng, 1500, 40, 5));
    write(dir.path(), "small.parquet", &table(&mut rng, 40, 40, 5));
    let sources = [
        ("big", vec!["big.parquet".to_owned()]),
        ("small", vec!["small.parquet".to_owned()]),
    ];
    let mut t = parquet_files(dir.path(), &sources, true, &rt);
    configured(&mut t, &[]);
    let explain = |t: &super::tpch::Tpch, q: &str| t.plan_sql(q).unwrap().to_string();
    // The bounds `EXPLAIN` lists on the join: `RuntimeFilter n: <probe>
    // <= max(<build>) -> <scan column>`.
    let bound = |text: &str, agg: &str| -> Vec<String> {
        text.split("RuntimeFilter ")
            .skip(1)
            .filter(|f| f.contains(&format!(" {agg}(")))
            .map(|f| f.split(", ").next().unwrap_or(f).trim().to_owned())
            .collect()
    };
    let upper = explain(&t, JOINS[0]);
    let lines = bound(&upper, "max");
    assert_eq!(lines.len(), 1, "{upper}");
    assert!(
        lines[0].contains("<= max(") && lines[0].ends_with("big.v"),
        "{upper}"
    );
    // `s.v >= b.v` is `b.v <= s.v`.
    assert_eq!(bound(&explain(&t, JOINS[1]), "max").len(), 1);
    let lower = explain(&t, JOINS[2]);
    assert!(
        bound(&lower, "min").iter().all(|l| l.contains(">= min(")),
        "{lower}"
    );
    assert_eq!(bound(&lower, "min").len(), 1, "{lower}");
    assert_eq!(bound(&explain(&t, JOINS[3]), "max").len(), 1);
    assert_eq!(bound(&explain(&t, JOINS[4]), "max").len(), 1);
    // A NaN probe value is larger than every value: no lower float bound.
    assert!(bound(&explain(&t, JOINS[5]), "min").is_empty());
    // The left join keeps every probe row.
    assert!(bound(&explain(&t, JOINS[9]), "max").is_empty());
    // A conjunct that may raise (ANSI division) keeps every row its pairs
    // are evaluated on.
    let raises = "SELECT b.v, s.v FROM big b JOIN small s \
                  ON b.k32 = s.k32 AND b.v < s.v AND b.v DIV s.k32 > 0";
    assert!(bound(&explain(&t, raises), "max").is_empty());
    // So does a probe key that may raise: it is computed above the scan, on
    // every row the scan keeps (review of #1029).
    let key_raises = "SELECT b.v, s.v FROM big b JOIN small s \
                      ON b.k32 DIV b.v = s.k32 AND b.v < s.v";
    assert!(bound(&explain(&t, key_raises), "max").is_empty());
    configured(&mut t, NO_BOUNDS);
    assert!(bound(&explain(&t, JOINS[0]), "max").is_empty());
}

/// A `TIMESTAMP_NTZ` probe column against a `TIMESTAMP` build column: Spark
/// compares `CAST(ntz AS TIMESTAMP)`, which shifts by the session zone's
/// offset, so neither an equality key's filter nor a residual's bound may
/// test the raw NTZ values (review of D2: `widens_to`).
#[test]
fn ntz_against_timestamp_keeps_every_matching_row() {
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, TimestampMicrosecondArray};
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use std::sync::Arc;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let base: i64 = 1_700_000_000_000_000;
    let hour: i64 = 3_600_000_000;
    let batch = |rows: i32, t: &dyn Fn(i32) -> i64, zone: Option<&str>| {
        let ts = (0..rows)
            .map(|i| Some(t(i)))
            .collect::<TimestampMicrosecondArray>();
        let ts = match zone {
            Some(z) => ts.with_timezone(z),
            None => ts,
        };
        let columns: Vec<ArrayRef> = vec![
            Arc::new((0..rows).map(|i| i % 8).collect::<Int32Array>()),
            Arc::new(ts),
        ];
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new(
                "t",
                DataType::Timestamp(TimeUnit::Microsecond, zone.map(Into::into)),
                false,
            ),
        ]));
        RecordBatch::try_new(schema, columns).unwrap()
    };
    // In Asia/Kolkata (UTC+5:30) the probe's wall clock `base + 1h` is the
    // instant `base - 4h30` (< every build value), `base + 5h30` is `base`
    // (= every build value), `base + 10h` is past both.
    let probe = |i: i32| {
        base + match i % 3 {
            0 => hour,
            1 => 11 * hour / 2,
            _ => 10 * hour,
        }
    };
    write(dir.path(), "big.parquet", &batch(1500, &probe, None));
    write(
        dir.path(),
        "small.parquet",
        &batch(8, &|_| base, Some("UTC")),
    );
    let sources = [
        ("big", vec!["big.parquet".to_owned()]),
        ("small", vec!["small.parquet".to_owned()]),
    ];
    let mut t = parquet_files(dir.path(), &sources, true, &rt);
    let zone = ("spark.sql.session.timeZone", "Asia/Kolkata");
    for q in [
        "SELECT b.k, b.t FROM big b JOIN small s ON b.k = s.k AND b.t < s.t",
        "SELECT b.k, b.t, s.k FROM big b JOIN small s ON b.t = s.t",
    ] {
        configured(&mut t, &[OFF[0], zone]);
        let off = sorted(&run(&t.plan_sql(q).unwrap()).0);
        assert!(!off.is_empty(), "{q}: no matching rows to lose");
        configured(&mut t, &[BOUNDS[0], zone]);
        let plan = t.plan_sql(q).unwrap();
        assert!(!plan.to_string().contains(" max("), "{q}:\n{plan}");
        assert_eq!(sorted(&run(&plan).0), off, "{q}:\n{plan}");
    }
}

/// A decimal bound keeps one end: with equal keys, a probe value of `1.00`
/// below the build's `10.00` matches `b.d <= s.d`, which a closed
/// `[10.00, 10.00]` range would drop; `11.00` is past the bound (review of
/// #1029: the random tables tie `d102` to the key).
#[test]
fn a_decimal_bound_keeps_values_below_the_build_maximum() {
    use arrow_array::{ArrayRef, Decimal128Array, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let batch = |values: &[(i32, i32, i128)]| {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(values.iter().map(|v| v.0).collect::<Int32Array>()),
            Arc::new(values.iter().map(|v| v.1).collect::<Int32Array>()),
            Arc::new(
                values
                    .iter()
                    .map(|v| v.2)
                    .collect::<Decimal128Array>()
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
        ];
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("k", DataType::Int32, false),
            Field::new("d", DataType::Decimal128(10, 2), false),
        ]));
        RecordBatch::try_new(schema, columns).unwrap()
    };
    write(
        dir.path(),
        "big.parquet",
        &batch(&[(1, 7, 100), (2, 7, 1_100), (3, 8, 100)]),
    );
    write(dir.path(), "small.parquet", &batch(&[(9, 7, 1_000)]));
    let sources = [
        ("big", vec!["big.parquet".to_owned()]),
        ("small", vec!["small.parquet".to_owned()]),
    ];
    let mut t = parquet_files(dir.path(), &sources, true, &rt);
    let q = "SELECT b.id, s.id FROM big b JOIN small s ON b.k = s.k AND b.d <= s.d";
    configured(&mut t, BOUNDS);
    let plan = t.plan_sql(q).unwrap();
    let text = plan.to_string();
    assert!(text.contains("<= max("), "{text}");
    let got = sorted(&run(&plan).0);
    configured(&mut t, OFF);
    assert_eq!(got, sorted(&run(&t.plan_sql(q).unwrap()).0));
    assert_eq!(got, vec!["1|9".to_owned()]);
}
