// SPDX-License-Identifier: Apache-2.0
//! Build columns with a string or binary column kept as Arrow columns
//! (perf F-H1, `krishiv.exec.join.columnPayload`): every join type returns
//! what the arrow-row payload returns, over `Utf8View`, `Utf8` and
//! `BinaryView` columns with short (inline) and long values and NULLs,
//! with a residual filter that reads build columns, in both key formats,
//! in memory and under a budget that spills; and the build table holds no
//! more than a tenth above the arrow-row payload for customer-like names.

use arrow_array::{BinaryViewArray, StringViewArray};
use krishiv_exec::FnPredicate;

use super::*;

/// Key, then the payload columns.
#[derive(Clone, Debug)]
struct Wide {
    k: Option<i64>,
    s: Option<String>,
    t: Option<String>,
    b: Option<Vec<u8>>,
    v: i64,
}

fn wide_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("s", DataType::Utf8View, true),
        Field::new("t", DataType::Utf8, true),
        Field::new("b", DataType::BinaryView, true),
        Field::new("v", DataType::Int64, false),
    ]))
}

fn wide_data(rows: &[Wide]) -> RecordBatch {
    RecordBatch::try_new(
        wide_schema(),
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.k).collect::<Vec<_>>(),
            )),
            Arc::new(StringViewArray::from_iter(
                rows.iter().map(|r| r.s.as_deref()),
            )),
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.t.as_deref()).collect::<Vec<_>>(),
            )),
            Arc::new(BinaryViewArray::from_iter(
                rows.iter().map(|r| r.b.as_deref()),
            )),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.v))),
        ],
    )
    .unwrap()
}

/// `n` rows with keys in `0..keys` (NULL one time in 17), strings of 0 to
/// 40 bytes (inline up to 12, in data buffers past it) and NULLs.
fn wide_rows(n: usize, keys: i64, salt: u64) -> Vec<Wide> {
    let mut x = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    (0..n)
        .map(|i| {
            let r = next();
            let text = |r: u64, tag: &str| {
                let len = (r % 41) as usize;
                (!r.is_multiple_of(11)).then(|| format!("{tag}{i:0>len$}"))
            };
            Wide {
                k: (r % 17 != 0).then(|| i64::try_from(r % keys.unsigned_abs()).unwrap()),
                s: text(next(), "s"),
                t: text(next(), "t"),
                b: text(next(), "b").map(String::into_bytes),
                v: i64::try_from(r % 20).unwrap(),
            }
        })
        .collect()
}

/// `probe.v + build.v` not a multiple of 3: a residual that reads a build
/// column (the probe's 5 columns, then the build's).
fn residual(b: &RecordBatch) -> Result<BooleanArray> {
    let p = b.column(4).as_primitive::<Int64Type>();
    let q = b.column(9).as_primitive::<Int64Type>();
    Ok(p.iter()
        .zip(q.iter())
        .map(|(p, q)| Some((p.unwrap_or(0) + q.unwrap_or(0)) % 3 != 0))
        .collect())
}

/// One run of a join of `probe` against `build`.
#[derive(Clone, Copy, Debug)]
struct Run {
    join_type: JoinType,
    filter: bool,
    row_layout: bool,
    column_payload: bool,
    limit: usize,
}

async fn run_wide(
    r: Run,
    probe: &[Wide],
    build: &[Wide],
) -> (Vec<String>, Arc<krishiv_exec::QueryContext>) {
    let root = root(r.limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: 1024,
        join_row_layout: r.row_layout,
        join_column_payload: r.column_payload,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let spec = JoinSpec {
        join_type: r.join_type,
        probe_schema: wide_schema(),
        build_schema: wide_schema(),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: (r.filter && r.join_type != JoinType::NullAwareAnti)
            .then(|| Arc::new(FnPredicate(residual)) as Arc<dyn Predicate>),
        single_match: false,
        spill: Some(scope.clone()),
    };
    let plan = join(
        spec,
        scan(&wide_schema(), batches(&wide_data(build), 300)),
        scan(&wide_schema(), batches(&wide_data(probe), 300)),
    );
    let out = run(&runner_for(3, r.limit), plan, &q).await.unwrap();
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0, "spill files are deleted");
    for b in &out {
        let declared: Vec<DataType> = b
            .data()
            .schema()
            .fields()
            .iter()
            .map(|f| f.data_type().clone())
            .collect();
        let got: Vec<DataType> = b
            .data()
            .columns()
            .iter()
            .map(|c| c.data_type().clone())
            .collect();
        assert_eq!(got, declared, "{r:?}: output types");
    }
    (render(&out), q)
}

/// Every join type, with and without a residual, in both key formats: the
/// column payload returns the arrow-row payload's rows.
#[tokio::test]
async fn every_join_returns_the_arrow_row_payloads_rows() {
    let build = wide_rows(600, 40, 1);
    let probe = wide_rows(500, 50, 2);
    for join_type in TYPES {
        for (filter, row_layout) in [(false, false), (true, false), (false, true), (true, true)] {
            let r = Run {
                join_type,
                filter,
                row_layout,
                column_payload: false,
                limit: 64 * MIB,
            };
            let (expect, _) = run_wide(r, &probe, &build).await;
            let on = Run {
                column_payload: true,
                ..r
            };
            let (got, _) = run_wide(on, &probe, &build).await;
            assert_eq!(got, expect, "{on:?}");
        }
    }
}

/// Under a budget that spills the build (grace partitions, unmatched build
/// rows of right and full outer joins read back from disk): the in-memory
/// arrow-row answer.
#[tokio::test]
async fn spilled_column_payloads_equal_the_in_memory_join() {
    let build = wide_rows(20_000, 6_000, 3);
    let probe = wide_rows(12_000, 8_000, 4);
    for join_type in [
        JoinType::Inner,
        JoinType::RightOuter,
        JoinType::FullOuter,
        JoinType::RightSemi,
        JoinType::RightAnti,
    ] {
        for row_layout in [false, true] {
            let r = Run {
                join_type,
                filter: join_type == JoinType::Inner,
                row_layout,
                column_payload: false,
                limit: 256 * MIB,
            };
            let (expect, _) = run_wide(r, &probe, &build).await;
            let on = Run {
                column_payload: true,
                limit: MIB,
                ..r
            };
            let (got, q) = run_wide(on, &probe, &build).await;
            assert_eq!(got, expect, "{on:?}");
            assert!(
                metric(&q, "HashJoinBuild", "spilled_partitions") > 0,
                "{on:?}: no spill"
            );
        }
    }
}

/// Customer-like names (TPC-DS `c_last_name`, `c_first_name`: 3 to 13
/// bytes): the build table with the column payload is at most a tenth
/// above the arrow-row payload's (the F-H1 kill line).
#[tokio::test]
async fn customer_names_cost_no_more_than_a_tenth_above_arrow_rows() {
    const LAST: [&str; 8] = [
        "Smith",
        "Johnson",
        "Brown",
        "Rodriguez",
        "Lee",
        "Vanderbilt",
        "Ng",
        "Williamson",
    ];
    const FIRST: [&str; 8] = [
        "Ann",
        "Christopher",
        "Maria",
        "Bob",
        "Elizabeth",
        "Kai",
        "Jonathan",
        "Eve",
    ];
    let n = 100_000;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("last", DataType::Utf8View, true),
        Field::new("first", DataType::Utf8View, true),
    ]));
    let data = |rows: std::ops::Range<i64>| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(rows.clone())),
                Arc::new(StringViewArray::from_iter(rows.clone().map(|i| {
                    (i % 50 != 0).then(|| LAST[usize::try_from(i.rem_euclid(8)).unwrap_or(0)])
                }))),
                Arc::new(StringViewArray::from_iter(rows.map(|i| {
                    (i % 47 != 0)
                        .then(|| FIRST[usize::try_from((i / 8).rem_euclid(8)).unwrap_or(0)])
                }))),
            ],
        )
        .unwrap()
    };
    let mut bytes = Vec::new();
    for column_payload in [false, true] {
        let root = root(512 * MIB);
        let config = ExecConfig {
            join_column_payload: column_payload,
            ..ExecConfig::default()
        };
        let q = query(&root, 1, config);
        let spec = JoinSpec {
            join_type: JoinType::Inner,
            probe_schema: Arc::clone(&schema),
            build_schema: Arc::clone(&schema),
            probe_keys: vec![0],
            build_keys: vec![0],
            null_safe: vec![false],
            filter: None,
            single_match: false,
            spill: None,
        };
        let plan = join(
            spec,
            scan(&schema, batches(&data(0..n), 4096)),
            scan(&schema, batches(&data(0..2000), 4096)),
        );
        let out = run(&runner_for(2, 512 * MIB), plan, &q).await.unwrap();
        assert_eq!(out.iter().map(Batch::num_rows).sum::<usize>(), 2000);
        bytes.push(metric(&q, "HashJoinBuild", "arena_bytes"));
        drop(out);
        assert_released(&root);
    }
    let (rows, columns) = (bytes[0], bytes[1]);
    eprintln!("arena bytes for {n} customer rows: arrow rows {rows}, columns {columns}");
    assert!(rows > 0 && columns > 0);
    assert!(
        columns * 10 <= rows * 11,
        "column payload {columns} B vs arrow rows {rows} B"
    );
}
