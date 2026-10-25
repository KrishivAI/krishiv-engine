// SPDX-License-Identifier: Apache-2.0
//! A scan never drops a row that a filter conjunct that may raise still
//! has to see. Spark's Parquet reader drops no row on its own (only whole
//! row groups and pages), its filter evaluates the conjuncts in written
//! order, and its runtime filter is the filter's last conjunct. So, as
//! Spark 4.2 does, with `spark.sql.ansi.enabled=true` and a row
//! `(k = 0, s = 'abc')`:
//!
//! - `WHERE CAST(s AS INT) > 0 AND k = 1` raises `CAST_INVALID_INPUT`,
//! - `WHERE k = 1 AND CAST(s AS INT) > 0` returns the matching rows,
//! - a join whose runtime filter rejects that row still raises, even with
//!   the row alone in its own file,
//! - with two bad rows the error is the first one's.
//!
//! Late materialisation is forced on and the matching rows are rare (the
//! scan would drop the bad row before the filter if it could), and a
//! differential test compares the scan with its row-level stages against
//! the scan without them, over generated predicates: same rows or the
//! same error.

use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{FileFormat as _, FormatOptions};
use krishiv_memory::{MemoryConfig, TreePool};
use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode, ResolvedExpr};
use proptest::prelude::*;

use super::Runnable;
use super::tpch::{Order, Tpch, parquet_files_with, parquet_tables, rows, sql};

/// Rows of the main table: `k = 1` on one row in 2,000 (0.05%, so late
/// materialisation filters), `k = 0` elsewhere.
const N: usize = 20_000;

/// Late materialisation forced on.
const LATE: (&str, &str) = ("krishiv.io.lateMaterialization", "on");

/// Runtime filters forced: an exact set for every build, never switched
/// off.
const RUNTIME: [(&str, &str); 3] = [
    ("krishiv.runtimeFilter.bloom.minApplicationRows", "0"),
    ("krishiv.runtimeFilter.exactSet.maxValues", "1000000"),
    ("krishiv.runtimeFilter.adaptive.maxPassPercent", "100"),
];

/// `(k, s)` rows: `bad` lists `(row, k, s)` overrides.
fn table(n: usize, bad: &[(usize, Option<i32>, Option<&str>)]) -> RecordBatch {
    let mut k: Vec<Option<i32>> = (0..n).map(|i| Some(i32::from(i % 2000 == 5))).collect();
    let mut s: Vec<Option<String>> = (0..n)
        .map(|i| Some(if i % 2000 == 5 { "1" } else { "7" }.to_owned()))
        .collect();
    for (i, kv, sv) in bad {
        k[*i] = *kv;
        s[*i] = sv.map(str::to_owned);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("s", DataType::Utf8View, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(k)) as ArrayRef,
            Arc::new(StringViewArray::from(s)),
        ],
    )
    .unwrap()
}

/// The dimension of the join tests: `dk = 1` has `x = 1`, others `x = 0`.
fn dim() -> RecordBatch {
    let dk: Vec<i32> = std::iter::once(1).chain(100..300).collect();
    let x: Vec<i32> = dk.iter().map(|k| i32::from(*k == 1)).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("dk", DataType::Int32, false),
        Field::new("x", DataType::Int32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(dk)) as ArrayRef,
            Arc::new(Int32Array::from(x)),
        ],
    )
    .unwrap()
}

fn write(dir: &Path, name: &str, batch: &RecordBatch) {
    let o = FormatOptions::defaults(ParquetFormat.options());
    let mut enc = ParquetFormat::encoder_with(&batch.schema(), &o, |p| {
        p.set_max_row_group_row_count(Some(5000))
    })
    .unwrap();
    let mut bytes = Vec::new();
    enc.write(batch).unwrap();
    while let Some(b) = enc.take_bytes() {
        bytes.extend_from_slice(&b);
    }
    bytes.extend_from_slice(&enc.finish().unwrap().tail);
    std::fs::write(dir.join(name), bytes).unwrap();
}

struct Fixture {
    dir: tempfile::TempDir,
    rt: tokio::runtime::Runtime,
}

impl Fixture {
    /// Tables `t` (a bad row `(0, 'abc')` in the middle), `two` (bad rows
    /// `'abc'` then `'xyz'`, both `k = 0`), `split` (`t`'s good rows, and
    /// the bad row alone in its own file) and `d` (the dimension).
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write(
            p,
            "t.parquet",
            &table(N, &[(N / 2 + 1, Some(0), Some("abc"))]),
        );
        write(
            p,
            "two.parquet",
            &table(
                N,
                &[(3001, Some(0), Some("abc")), (15001, Some(0), Some("xyz"))],
            ),
        );
        write(p, "good.parquet", &table(N, &[]));
        write(p, "bad.parquet", &table(1, &[(0, Some(0), Some("abc"))]));
        write(p, "d.parquet", &dim());
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        Self { dir, rt }
    }

    fn tables(&self, ansi: bool, settings: &[(&str, &str)]) -> Tpch {
        let one = |t: &str| vec![format!("{t}.parquet")];
        let sources = [
            ("t", one("t")),
            ("two", one("two")),
            (
                "split",
                vec!["bad.parquet".to_owned(), "good.parquet".to_owned()],
            ),
            ("d", one("d")),
        ];
        parquet_files_with(self.dir.path(), &sources, ansi, settings, &self.rt)
    }
}

/// The rows of `q`, or its error, and the rows its file scans returned.
fn run(t: &Tpch, q: &str) -> (Result<Vec<String>, KrishivError>, u64) {
    let plan: Runnable = t.plan_sql(q).unwrap_or_else(|e| panic!("{q}: {e}"));
    let root = TreePool::new(&MemoryConfig {
        limit: Some(1 << 30),
        ..MemoryConfig::default()
    });
    let (out, query) = super::run_in(&plan, &root);
    let scanned = query
        .metrics()
        .operators
        .values()
        .filter(|e| e.name == "FileScan")
        .map(|e| e.metrics.rows_out)
        .sum();
    let mut out = out.map(|b| rows(&b));
    if let Ok(r) = out.as_mut() {
        r.sort_unstable();
    }
    (out, scanned)
}

fn cast_error(r: &Result<Vec<String>, KrishivError>, value: &str, q: &str) {
    match r {
        Err(e) => {
            assert_eq!(e.error_class(), ErrorClass::CastInvalidInput, "{q}: {e}");
            assert!(e.to_string().contains(&format!("'{value}'")), "{q}: {e}");
        }
        Ok(rows) => panic!("{q}: no error, {rows:?}"),
    }
}

fn count(r: &Result<Vec<String>, KrishivError>, n: usize, q: &str) {
    match r {
        Ok(rows) => assert_eq!(rows, &[n.to_string()], "{q}"),
        Err(e) => panic!("{q}: {e}"),
    }
}

/// Spark 4.2: `CAST(s AS INT) > 0 AND k = 1` raises on the `k = 0` row;
/// `k = 1 AND CAST(s AS INT) > 0` does not, and the scan still drops the
/// rows `k = 1` rejects.
#[test]
fn a_raising_conjunct_sees_every_row_written_conjuncts_before_it_keep() {
    let f = Fixture::new();
    let t = f.tables(true, &[LATE]);
    let q = "SELECT count(*) FROM t WHERE CAST(s AS INT) > 0 AND k = 1";
    cast_error(&run(&t, q).0, "abc", q);
    let q = "SELECT count(*) FROM t WHERE k = 1 AND CAST(s AS INT) > 0";
    let (r, scanned) = run(&t, q);
    count(&r, 10, q);
    assert!(scanned < 100, "{q}: the scan returned {scanned} rows");
    // Inside a disjunction the cast still sees the `k = 0` row.
    let q = "SELECT count(*) FROM t WHERE k = 0 AND (k = 1 OR CAST(s AS INT) > 0)";
    cast_error(&run(&t, q).0, "abc", q);
}

/// With two bad rows, the error names the first in row order.
#[test]
fn the_error_is_the_first_bad_rows() {
    let f = Fixture::new();
    let t = f.tables(true, &[LATE]);
    let q = "SELECT count(*) FROM two WHERE CAST(s AS INT) > 0 AND k = 1";
    cast_error(&run(&t, q).0, "abc", q);
}

/// Conjuncts that cannot raise still drop rows in the scan (`s` is not
/// read by them, so late materialisation pays): with an ANSI-off cast or
/// `TRY_CAST` after them, or before a conjunct that may raise.
#[test]
fn non_raising_conjuncts_still_drop_rows_in_the_scan() {
    let f = Fixture::new();
    let t = f.tables(false, &[LATE]);
    for q in [
        "SELECT count(*) FROM t WHERE CAST(s AS INT) > 0 AND k = 1",
        "SELECT count(*) FROM t WHERE k IN (1, 2) AND TRY_CAST(s AS INT) > 0",
        // `length` is not known not to raise: `k = 1`, written before it,
        // still drops rows.
        "SELECT count(*) FROM t WHERE k = 1 AND length(s) > 0",
    ] {
        let (r, scanned) = run(&t, q);
        count(&r, 10, q);
        assert!(scanned < 100, "{q}: the scan returned {scanned} rows");
    }
}

/// Spark 4.2 puts its runtime filter last in the filter over the scan: a
/// bad row whose key the join's build side does not hold still raises,
/// also alone in a file the runtime filter would skip.
#[test]
fn a_runtime_filter_never_hides_a_row_from_a_raising_conjunct() {
    let f = Fixture::new();
    let mut settings = vec![LATE];
    settings.extend(RUNTIME);
    let t = f.tables(true, &settings);
    for fact in ["t", "split"] {
        let q = format!(
            "SELECT count(*) FROM {fact} JOIN d ON {fact}.k = d.dk \
             WHERE d.x = 1 AND CAST({fact}.s AS INT) > 0"
        );
        cast_error(&run(&t, &q).0, "abc", &q);
    }
    // A non-raising conjunct first: the row never reaches the cast.
    let q = "SELECT count(*) FROM t JOIN d ON t.k = d.dk \
             WHERE d.x = 1 AND t.k = 1 AND CAST(t.s AS INT) > 0";
    count(&run(&t, q).0, 10, q);
    // Without a raising conjunct the runtime filter still applies.
    let q = "SELECT count(*) FROM t JOIN d ON t.k = d.dk WHERE d.x = 1";
    let plan = t.plan_sql(q).unwrap();
    assert!(plan.to_string().contains("RuntimeFilter #"), "{plan}");
    let (r, scanned) = run(&t, q);
    count(&r, 10, q);
    assert!(scanned < 100, "{q}: the scan returned {scanned} rows");
}

/// Conjunct shapes for the differential test, raising (under ANSI) and
/// not. The table has rare `k = 1` rows, a `(0, 'abc')` row, a `k = 7`
/// row (`k * 1000000000` overflows), NULLs in `k` and `s`.
const SHAPES: [&str; 12] = [
    "k = 1",
    "k IN (1, 3)",
    "k IS NULL",
    "k IS NOT NULL",
    "s LIKE '1%'",
    "NOT (k = 0)",
    "(k = 1 OR s = 'zz')",
    "k BETWEEN 1 AND 2",
    "CAST(s AS INT) > 0",
    "k * 1000000000 > 0",
    "CAST(s AS INT) IS NULL",
    "TRY_CAST(s AS INT) > 0",
];

fn diff_table(dir: &Path) {
    let mut bad = vec![(9001, Some(0), Some("abc")), (12_345, Some(7), Some("8"))];
    bad.extend((0..N).step_by(997).map(|i| (i, None, Some("5"))));
    bad.extend((3..N).step_by(1499).map(|i| (i, Some(0), None)));
    bad.sort_by_key(|b| b.0);
    bad.dedup_by_key(|b| b.0);
    write(dir, "g.parquet", &table(N, &bad));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]
    /// The scan with its row-level stages (late materialisation forced on)
    /// returns what the scan without them returns under the same filter:
    /// the same rows or the same error, message included.
    #[test]
    fn row_level_stages_never_change_the_result(
        picks in proptest::collection::vec(0..SHAPES.len(), 1..=4),
        ansi in any::<bool>(),
    ) {
        let dir = tempfile::tempdir().unwrap();
        diff_table(dir.path());
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        let sources = [("g", vec!["g.parquet".to_owned()])];
        let late = parquet_files_with(dir.path(), &sources, ansi, &[LATE], &rt);
        let plain = parquet_files_with(
            dir.path(),
            &sources,
            ansi,
            &[("krishiv.io.lateMaterialization", "off")],
            &rt,
        );
        let predicate: Vec<&str> = picks.iter().map(|i| SHAPES[*i]).collect();
        let q = format!("SELECT k, s FROM g WHERE {}", predicate.join(" AND "));
        let (want, _) = run(&plain, &q);
        let (got, _) = run(&late, &q);
        match (&want, &got) {
            (Ok(w), Ok(g)) => prop_assert_eq!(w, g, "{}", q),
            (Err(w), Err(g)) => {
                prop_assert_eq!(w.error_class(), g.error_class(), "{}", q);
                prop_assert_eq!(w.to_string(), g.to_string(), "{}", q);
            }
            _ => prop_assert!(false, "{}: {:?} vs {:?}", q, want.map(|r| r.len()), got.map(|r| r.len())),
        }
    }
}

/// The filters directly over a scan in `p` (subquery plans included).
fn scan_filters<'a>(p: &'a Arc<LogicalPlan>, out: &mut Vec<&'a Expr>) {
    if let PlanNode::Filter { input, predicate } = &p.node {
        if matches!(input.node, PlanNode::Scan(_)) {
            out.push(predicate);
        }
        subquery_filters(predicate, out);
    }
    p.for_each_child(&mut |c| scan_filters(c, out));
}

fn subquery_filters<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match &e.node {
        ExprNode::Subquery { plan, .. } => scan_filters(plan, out),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            f.args.iter().for_each(|a| subquery_filters(a, out));
        }
        ExprNode::Cast { expr, .. } => subquery_filters(expr, out),
        _ => {}
    }
}

/// No TPC-H filter over a scan has a conjunct that may raise (ANSI on):
/// every scan keeps its row-level drops and runtime filters.
#[test]
fn no_tpch_scan_filter_may_raise() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let tpch = parquet_tables(&dir, &rt);
    let mut raising = Vec::new();
    let mut seen = 0;
    for q in 1..=22 {
        let plan = tpch.optimized(&sql(q), Order::Statistics).unwrap();
        let mut filters = Vec::new();
        scan_filters(plan.root(), &mut filters);
        seen += filters.len();
        for f in filters {
            for c in krishiv_plan::conjuncts(f) {
                if krishiv_plan::may_raise(c, true) {
                    raising.push(format!("q{q:02}: {c}"));
                }
            }
        }
    }
    assert!(seen >= 22, "{seen} filters over scans");
    // q22's `c_acctbal > (SELECT avg(...))` compares with a scalar
    // subquery (`SCALAR_SUBQUERY_TOO_MANY_ROWS`); that filter has a
    // subquery, so it is not pushed into the scan either way, and the
    // customer scan is the preserved side of an anti join (no runtime
    // filter targets it).
    assert_eq!(
        raising,
        ["q22: (CAST(c_acctbal#5 AS DECIMAL(19,6)) > scalar-subquery)"],
        "{raising:#?}"
    );
}
