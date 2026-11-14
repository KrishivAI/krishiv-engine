// SPDX-License-Identifier: Apache-2.0
//! `GroupKeyCode` (design P2b D2) through the physical planner and the
//! executor: with statistics under which the rule fires, the plan with the
//! rule on returns the rows of the plan with it off (and Spark's, for TPC-H),
//! with the dictionary read twice (no session configuration) and shared
//! (`ShareCte`, a session configuration), and a null dimension key survives
//! the null-safe encoding join.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_optimizer::{ColumnStats, Options, StatSource, Statistics, TableStats};
use krishiv_plan::analyzer::Analyzer as _;
use krishiv_plan::analyzer::DefaultAnalyzer;
use krishiv_plan::analyzer::testing::block_on;
use krishiv_plan::{Literal, Plan, ReadSource, Scan, SourceId};
use krishiv_udf_sdk::UdfLibrary;
use serde_json::Value;

use super::partial_aggregate::{physical, runtime, sorted, write};
use super::run;
use super::tpch::{Tpch, answers, expected_rows, parquet_files, parquet_tables, rows, sql};

/// Statistics by table name.
#[derive(Debug)]
struct ByName(HashMap<String, TableStats>);

impl Statistics for ByName {
    fn table(&self, scan: &Scan) -> Option<TableStats> {
        let ReadSource::Table { name, .. } = &scan.source else {
            return None;
        };
        self.0.get(name.last()?).cloned()
    }
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days(y: i64, m: i64, d: i64) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    i32::try_from(era * 146_097 + doe - 719_468).unwrap()
}

fn bound(v: &Value) -> Option<Literal> {
    if let Some(d) = v.get("date") {
        let p: Vec<i64> = d.as_str()?.split('-').map(|x| x.parse().unwrap()).collect();
        return Some(Literal::Date(days(p[0], p[1], p[2])));
    }
    Some(Literal::Double(v.get("number")?.as_f64()?))
}

/// TPC-H SF10 statistics (the optimizer's fixture, where the rule fires on
/// q05, q10 and q16) over the SF 0.01 files: statistics choose the plan,
/// never its rows.
fn sf10() -> ByName {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../krishiv-optimizer/tests/fixtures/tpch-sf10-stats.json");
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
    ByName(tables.collect())
}

fn analyze(t: &Tpch, text: &str) -> Plan {
    let plan =
        krishiv_sql::parse_and_lower(text, &krishiv_sql::ParserOptions::default(), SourceId(0))
            .unwrap();
    block_on(DefaultAnalyzer.analyze(&plan, &t.ctx))
        .unwrap()
        .plan
}

/// How the plan is optimized: by `Options` alone (no session configuration,
/// so no `ShareCte`: `dict` is computed per read) or under a session
/// configuration (`ShareCte` on: one `WITH` definition).
#[derive(Clone, Copy, Debug)]
enum Session {
    None,
    Config,
}

fn optimized(t: &Tpch, text: &str, stats: &dyn Statistics, on: bool, s: Session) -> Plan {
    let plan = analyze(t, text);
    match s {
        Session::None => {
            let mut options = Options::default();
            options.group_key_code = on;
            krishiv_optimizer::optimize_with_options(&plan, stats, &options).unwrap()
        }
        Session::Config => {
            let mut config = Config::new();
            config
                .set(
                    "krishiv.optimizer.groupKeyCode",
                    if on { "true" } else { "false" },
                )
                .unwrap();
            krishiv_optimizer::optimize_with_udfs(&plan, stats, &UdfLibrary::new(), &config)
                .unwrap()
        }
    }
}

fn coded(plan: &Plan) -> bool {
    plan.root().to_string().contains("krishiv_group_code")
}

/// TPC-H q05, q10 and q16 at SF 0.01 under SF10 statistics: the rule fires,
/// and the plan returns Spark's rows and the rows of the plan with it off.
#[test]
fn tpch_coded_queries_return_spark_rows() {
    let rt = runtime();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let t = parquet_tables(&dir, &rt);
    let answers = answers(&dir.join("spark-answers.json"));
    let stats = sf10();
    for q in [5, 10, 16] {
        let want = expected_rows(&answers, &format!("q{q:02}"));
        for s in [Session::None, Session::Config] {
            let on = optimized(&t, &sql(q), &stats, true, s);
            assert!(coded(&on), "q{q:02} {s:?}: not coded\n{}", on.root());
            let off = optimized(&t, &sql(q), &stats, false, s);
            assert!(!coded(&off), "q{q:02} {s:?}");
            let got = rows(&run(&physical(&t, &on).unwrap()).unwrap());
            assert_eq!(got, want, "q{q:02} {s:?} on\n{}", on.root());
            let got = rows(&run(&physical(&t, &off).unwrap()).unwrap());
            assert_eq!(got, want, "q{q:02} {s:?} off");
        }
    }
}

/// `f(fk, x)` 400 rows; `d(dk, g1, g2)` 20 rows with duplicate `dk` values
/// of different keys, and null `g1` / `g2` (whole null tuples too).
fn null_tables(dir: &Path, rt: &tokio::runtime::Runtime) -> Tpch {
    let f = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("fk", DataType::Int64, true),
            Field::new("x", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int64Array::from_iter(
                (0..400_i64).map(|n| (n % 13 != 0).then_some(n % 11)),
            )),
            Arc::new(Int32Array::from_iter_values((0..400_i32).map(|n| n % 17))),
        ],
    )
    .unwrap();
    let word = |n: i64, w: &[&str]| w[usize::try_from(n).unwrap() % w.len()].to_owned();
    let d = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("dk", DataType::Int64, true),
            Field::new("g1", DataType::Utf8View, true),
            Field::new("g2", DataType::Utf8View, true),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values((0..20_i64).map(|n| n % 9))),
            Arc::new(StringViewArray::from_iter(
                (0..20_i64).map(|n| (n % 4 != 0).then(|| word(n, &["a", "b", "c"]))),
            )),
            Arc::new(StringViewArray::from_iter((0..20_i64).map(|n| {
                (n % 4 != 0 && n % 5 != 0).then(|| word(n, &["p", "q"]))
            }))),
        ],
    )
    .unwrap();
    write(dir, "f.parquet", &f);
    write(dir, "d.parquet", &d);
    let sources = [
        ("f", vec!["f.parquet".to_owned()]),
        ("d", vec!["d.parquet".to_owned()]),
    ];
    parquet_files(dir, &sources, true, rt)
}

/// Statistics under which `d`'s keys are coded: `f` 1M rows over 1,000
/// keys, `d` 1,000 rows with 10 values of `g1` and of `g2`.
fn null_stats() -> ByName {
    let col = |n: u64| ColumnStats {
        distinct: Some(n),
        nulls: Some(0),
        ..ColumnStats::default()
    };
    ByName(HashMap::from([
        (
            "f".to_owned(),
            TableStats {
                rows: 1_000_000,
                columns: HashMap::from([("fk".to_owned(), col(1_000)), ("x".to_owned(), col(17))]),
            },
        ),
        (
            "d".to_owned(),
            TableStats {
                rows: 1_000,
                columns: HashMap::from([
                    ("dk".to_owned(), col(1_000)),
                    ("g1".to_owned(), col(10)),
                    ("g2".to_owned(), col(10)),
                ]),
            },
        ),
    ]))
}

/// A null key tuple is one `dict` row, joined by `<=>`: the null groups
/// survive the encoding join, with and without a shared dictionary, and
/// under an outer join above `D`.
#[test]
fn null_dimension_keys_keep_their_groups() {
    let rt = runtime();
    let dir = tempfile::tempdir().unwrap();
    let t = null_tables(dir.path(), &rt);
    let stats = null_stats();
    for q in [
        "SELECT g1, g2, count(*), sum(x) FROM f JOIN d ON fk = dk GROUP BY g1, g2",
        "SELECT upper(g1), g2, count(DISTINCT g2), sum(x) FROM f JOIN d ON fk = dk \
         GROUP BY g1, g2",
        "SELECT g1, g2, count(*), count(f2.x) FROM f JOIN d ON f.fk = dk \
         LEFT JOIN f AS f2 ON f.x = f2.fk GROUP BY g1, g2",
    ] {
        for s in [Session::None, Session::Config] {
            let on = optimized(&t, q, &stats, true, s);
            assert!(coded(&on), "{q} {s:?}: not coded\n{}", on.root());
            let off = optimized(&t, q, &stats, false, s);
            let got = sorted(&run(&physical(&t, &on).unwrap()).unwrap());
            let want = sorted(&run(&physical(&t, &off).unwrap()).unwrap());
            assert_eq!(got, want, "{q} {s:?}\n{}", on.root());
            assert!(
                want.iter().any(|r| r.starts_with("NULL|NULL|")),
                "{q}: the data has a null group: {want:?}"
            );
        }
    }
}

/// The encoding join (`D` joined to `dict` by `<=>` on the keys, the join
/// under the `D_c` fence, `Shift +0`, printed `(ids +0)`) publishes no runtime
/// filter onto `D`: `dict` holds every key `D` has, so the filter would drop
/// nothing and cost a probe per row (design section 2.3).
#[test]
fn the_encoding_join_publishes_no_runtime_filter() {
    let rt = runtime();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let t = parquet_tables(&dir, &rt);
    let on = optimized(&t, &sql(16), &sf10(), true, Session::None);
    let text = physical(&t, &on).unwrap().to_string();
    let lines: Vec<&str> = text.lines().collect();
    let encoding: Vec<&str> = lines
        .windows(2)
        .filter(|w| w[0].contains("(ids +0)") && w[1].contains("HashJoin"))
        .map(|w| w[1])
        .collect();
    assert!(!encoding.is_empty(), "{text}");
    assert!(
        encoding.iter().all(|l| !l.contains("RuntimeFilter")),
        "{text}"
    );
}
