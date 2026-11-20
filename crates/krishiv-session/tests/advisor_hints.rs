// SPDX-License-Identifier: Apache-2.0
//! The advisor hints (P3-T073) on planted and clean queries: the profile `EXPLAIN ANALYZE` prints and the
//! `system.queries.hints` row of the same run carry the same hints.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_exec::profile::{Hint, HintCode, TextOptions};
use krishiv_session::Session;

fn base(builder: krishiv_session::SessionBuilder) -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let s = builder
        // The session creates the spill root itself (mode 0700).
        .config(
            "krishiv.spill.dir",
            dir.path().join("spill").to_str().unwrap(),
        )
        .build()
        .unwrap();
    (s, dir)
}

fn register(s: &Session, name: &str, cols: &[(&str, Vec<i64>)]) {
    let schema = Arc::new(Schema::new(
        cols.iter()
            .map(|(n, _)| Field::new(*n, DataType::Int64, false))
            .collect::<Vec<_>>(),
    ));
    let batches = (0..cols[0].1.len().div_ceil(65_536))
        .map(|b| {
            let range = b * 65_536..((b + 1) * 65_536).min(cols[0].1.len());
            RecordBatch::try_new(
                Arc::clone(&schema),
                cols.iter()
                    .map(|(_, v)| Arc::new(Int64Array::from(v[range.clone()].to_vec())) as _)
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    s.register_batches(name, schema, batches).unwrap();
}

/// As [`register`], in batches of 8,192 rows.
fn register_chunked(s: &Session, name: &str, cols: &[(&str, Vec<i64>)]) {
    let schema = Arc::new(Schema::new(
        cols.iter()
            .map(|(n, _)| Field::new(*n, DataType::Int64, false))
            .collect::<Vec<_>>(),
    ));
    let batches = (0..cols[0].1.len().div_ceil(8192))
        .map(|b| {
            let range = b * 8192..((b + 1) * 8192).min(cols[0].1.len());
            RecordBatch::try_new(
                Arc::clone(&schema),
                cols.iter()
                    .map(|(_, v)| Arc::new(Int64Array::from(v[range.clone()].to_vec())) as _)
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    s.register_batches(name, schema, batches).unwrap();
}

/// `part(p_partkey, p_name)`: 100,000 parts of which every hundredth is a `forest` one.
fn register_part(s: &Session) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int64, false),
        Field::new("p_name", DataType::Utf8, false),
    ]));
    let keys: Vec<i64> = (0..100_000).collect();
    let names: Vec<String> = keys
        .iter()
        .map(|k| {
            if k % 100 == 0 {
                format!("forest {k}")
            } else {
                format!("other {k}")
            }
        })
        .collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap();
    s.register_batches("part", schema, vec![batch]).unwrap();
}

/// `lineitem(l_partkey, l_qty)`: 3 million rows over the 100,000 parts.
fn register_lineitem(s: &Session) {
    let n = 3_000_000_i64;
    register(
        s,
        "lineitem",
        &[
            ("l_partkey", (0..n).map(|i| i % 100_000).collect()),
            ("l_qty", (0..n).map(|i| i % 50).collect()),
        ],
    );
}

fn run(s: &Session, sql: &str) -> (Vec<Hint>, String) {
    let df = block_on(s.sql(sql)).unwrap();
    let (_, profile) = block_on(df.collect_with_profile()).unwrap();
    (
        profile.hints.clone(),
        profile.to_text(TextOptions::default()),
    )
}

fn codes(h: &[Hint]) -> Vec<&'static str> {
    h.iter().map(|h| h.code.as_str()).collect()
}

const FOREST_JOIN: &str = "SELECT count(*) FROM lineitem JOIN part ON l_partkey + 0 = p_partkey \
                           WHERE p_name LIKE 'forest%'";
const PLAIN_JOIN: &str = "SELECT count(*) FROM lineitem JOIN part ON l_partkey = p_partkey \
                          WHERE p_name LIKE 'forest%'";

#[test]
fn hint_runtime_filter_raised() {
    let (s, _d) = base(Session::builder());
    register_lineitem(&s);
    register_part(&s);
    // A computed key reaches no scan.
    let (hints, text) = run(&s, FOREST_JOIN);
    let h: Vec<_> = hints
        .iter()
        .filter(|h| h.code == HintCode::RuntimeFilterOpportunity)
        .collect();
    assert_eq!(h.len(), 1, "{hints:#?}\n{text}");
    assert!(h[0].message.contains("is computed"), "{}", h[0].message);
    assert!(text.contains("== Hints =="), "{text}");
    // The plain key with filters off.
    let (off, _d2) = base(Session::builder().config("krishiv.runtimeFilter.enabled", "false"));
    register_lineitem(&off);
    register_part(&off);
    let (hints, text) = run(&off, PLAIN_JOIN);
    let h: Vec<_> = hints
        .iter()
        .filter(|h| h.code == HintCode::RuntimeFilterOpportunity)
        .collect();
    assert_eq!(h.len(), 1, "{hints:#?}\n{text}");
    assert!(
        h[0].message
            .contains("krishiv.runtimeFilter.enabled is false"),
        "{}",
        h[0].message
    );
}

#[test]
fn no_hint_on_clean_queries() {
    let (s, _d) = base(Session::builder());
    register_lineitem(&s);
    register_part(&s);
    register(
        &s,
        "small",
        &[("k", (0..1000).collect()), ("v", (0..1000).collect())],
    );
    for sql in [
        // A filtered scan.
        "SELECT l_partkey FROM lineitem WHERE l_partkey = 7",
        // A selective join whose runtime filter is published.
        PLAIN_JOIN,
        // A small aggregate.
        "SELECT k % 10, sum(v) FROM small GROUP BY k % 10",
    ] {
        let (hints, text) = run(&s, sql);
        assert!(hints.is_empty(), "{sql}\n{hints:#?}\n{text}");
        assert!(!text.contains("== Hints =="), "{text}");
    }
}

#[test]
fn hint_missing_stats_raised() {
    let (s, _d) = base(Session::builder());
    let dir = tempfile::tempdir().unwrap();
    let rows: String = (0..3000).map(|i| format!("{i},{}\n", i % 7)).collect();
    std::fs::write(dir.path().join("f.csv"), format!("k,v\n{rows}")).unwrap();
    block_on(s.sql(&format!(
        "CREATE TABLE sales_csv (k INT, v INT) USING csv LOCATION '{}' OPTIONS (header 'true')",
        dir.path().display()
    )))
    .unwrap();
    register(
        &s,
        "dim",
        &[("k", (0..200_000).collect()), ("w", (0..200_000).collect())],
    );
    let sql = "SELECT count(*) FROM dim JOIN sales_csv ON dim.k = sales_csv.k";
    let (hints, text) = run(&s, sql);
    let h: Vec<_> = hints
        .iter()
        .filter(|h| h.code == HintCode::MissingStatistics)
        .collect();
    assert_eq!(h.len(), 1, "{hints:#?}\n{text}");
    assert!(h[0].message.contains("sales_csv"), "{}", h[0].message);
}

#[test]
fn hint_spill_raised() {
    // The join and aggregate of `tests/spill.rs` under a 6 MiB limit spill; the byte floor is lowered to the
    // size of this planted case. The run is a plain one: the hint is read from its `system.queries` row.
    let (s, _d) = base(
        Session::builder()
            .config("krishiv.memory.limit", "6m")
            .config("krishiv.advisor.spill.minBytes", "64k"),
    );
    let (fact, keys) = (300_000_i64, 60_000_i64);
    let rows = |f: &dyn Fn(i64) -> i64, n: i64| (0..n).map(f).collect::<Vec<i64>>();
    register_chunked(
        &s,
        "f",
        &[
            ("k", rows(&|i| i % keys, fact)),
            ("v", rows(&|i| i % 1000, fact)),
        ],
    );
    register_chunked(&s, "d", &[("k", rows(&|i| i, keys))]);
    let sql = "SELECT f.k, sum(f.v) AS s, count(*) AS n FROM f JOIN d ON f.k = d.k \
               GROUP BY f.k ORDER BY s DESC, f.k";
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.collect()).unwrap();
    let row = row_hints(&s, "ORDER BY s DESC").expect("a finished query has a hints list");
    assert!(row.iter().any(|h| h.0 == "SPILL"), "{row:#?}");
}

#[test]
fn hint_skew_raised() {
    // Four compute threads whatever the host has: the hot key (90% of the rows) is then 4 drivers' skew.
    let executor = krishiv_exec::TokioExecutor::new(&krishiv_exec::ExecutorConfig {
        compute_threads: 4,
        io_threads: 1,
        blocking_threads: 4,
        io_blocking_threads: 2,
    })
    .unwrap();
    let (s, _d) = base(Session::builder().executor(executor));
    let n = 4_000_000_i64;
    register(
        &s,
        "skewed",
        &[
            (
                "k",
                (0..n).map(|i| if i % 10 == 0 { i } else { 0 }).collect(),
            ),
            ("v", (0..n).collect()),
        ],
    );
    let (hints, text) = run(
        &s,
        "SELECT k, sum(v) OVER (PARTITION BY k ORDER BY v ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
         FROM skewed",
    );
    assert!(codes(&hints).contains(&"SKEW"), "{hints:#?}\n{text}");
}

#[test]
fn advisor_disabled_raises_nothing() {
    let (s, _d) = base(Session::builder().config("krishiv.advisor.enabled", "false"));
    register_lineitem(&s);
    register_part(&s);
    let (hints, text) = run(&s, FOREST_JOIN);
    assert!(hints.is_empty(), "{hints:#?}");
    assert!(!text.contains("== Hints =="));
}

/// The hints of the newest `system.queries` row that has the text `needle`.
/// `(code, node id, operator id, message)` of one `system.queries.hints` element.
type HintCells = (String, Option<i32>, Option<i32>, String);

fn row_hints(s: &Session, needle: &str) -> Option<Vec<HintCells>> {
    let df = block_on(s.sql("SELECT query_text, hints FROM system.queries ORDER BY submitted_at"))
        .unwrap();
    let batches = block_on(df.collect()).unwrap();
    let mut found = None;
    for b in &batches {
        let texts = b.column(0).as_string_view_opt().map(|a| {
            (0..a.len())
                .map(|i| a.is_valid(i).then(|| a.value(i).to_owned()))
                .collect::<Vec<_>>()
        });
        let texts = texts.unwrap_or_else(|| {
            let a = b.column(0).as_string::<i32>();
            (0..a.len())
                .map(|i| a.is_valid(i).then(|| a.value(i).to_owned()))
                .collect()
        });
        let list = b.column(1).as_list::<i32>();
        for (i, t) in texts.iter().enumerate() {
            if t.as_deref().is_some_and(|t| t.contains(needle)) {
                found = Some(if list.is_null(i) {
                    None
                } else {
                    let v = list.value(i);
                    let st = v.as_struct();
                    let text = |c: usize| {
                        let a = st.column(c);
                        (0..st.len())
                            .map(|j| {
                                a.as_string_view_opt().map_or_else(
                                    || a.as_string::<i32>().value(j).to_owned(),
                                    |a| a.value(j).to_owned(),
                                )
                            })
                            .collect::<Vec<_>>()
                    };
                    let (codes, msgs) = (text(0), text(3));
                    let nodes = st.column(1).as_primitive::<Int32Type>();
                    let ops = st.column(2).as_primitive::<Int32Type>();
                    Some(
                        (0..st.len())
                            .map(|j| {
                                (
                                    codes[j].clone(),
                                    nodes.is_valid(j).then(|| nodes.value(j)),
                                    ops.is_valid(j).then(|| ops.value(j)),
                                    msgs[j].clone(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    )
                });
            }
        }
    }
    found.flatten()
}

#[test]
fn hints_agree_across_surfaces() {
    let (s, _d) = base(Session::builder());
    register_lineitem(&s);
    register_part(&s);
    let (hints, _) = run(&s, FOREST_JOIN);
    assert!(!hints.is_empty());
    let row = row_hints(&s, "l_partkey + 0").expect("the profiled run has a row with hints");
    let from_profile: Vec<_> = hints
        .iter()
        .map(|h| {
            (
                h.code.as_str().to_owned(),
                h.node.map(|n| i32::try_from(n).unwrap()),
                h.operator.map(|o| i32::try_from(o.0).unwrap()),
                h.message.clone(),
            )
        })
        .collect();
    // Same code, node, operator and message; the numbers a message quotes are the run's own, so compare the
    // run that produced the row: its codes and nodes match and its messages name the same operator.
    assert_eq!(
        row.iter().map(|r| (&r.0, r.1, r.2)).collect::<Vec<_>>(),
        from_profile
            .iter()
            .map(|r| (&r.0, r.1, r.2))
            .collect::<Vec<_>>()
    );
    assert_eq!(row, from_profile);
}

/// Every hint the 22 TPC-H queries raise over the SF1 files in `KRISHIV_TPCH_SF1` (the PR's review list):
/// `cargo nextest run -p krishiv-session --test advisor_hints --run-ignored only tpch_sf1_hint_review
/// --no-capture`. Prints one line per hint; asserts nothing, a reviewer reads the list.
#[test]
#[ignore = "needs TPC-H SF1 files in KRISHIV_TPCH_SF1; prints the review list"]
fn tpch_sf1_hint_review() {
    let dir = std::env::var("KRISHIV_TPCH_SF1")
        .expect("KRISHIV_TPCH_SF1 must name a directory with the TPC-H SF1 parquet files");
    let (s, _d) = base(Session::builder());
    for t in [
        "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
    ] {
        let df = s.read().parquet([format!("{dir}/{t}.parquet")]);
        block_on(df.create_or_replace_temp_view(t)).unwrap();
    }
    for q in 1..=22 {
        let path = format!(
            "{}/../../bench/queries/tpch/q{q:02}.sql",
            env!("CARGO_MANIFEST_DIR")
        );
        let sql = std::fs::read_to_string(path).unwrap();
        let df = block_on(s.sql(&sql)).unwrap();
        let started = std::time::Instant::now();
        let (_, profile) = block_on(df.collect_with_profile()).unwrap();
        println!(
            "q{q:02}: {} hints in {:?}",
            profile.hints.len(),
            started.elapsed()
        );
        for h in &profile.hints {
            println!("q{q:02}:   {}", h.line(&profile.plan));
        }
    }
}
