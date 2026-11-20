// SPDX-License-Identifier: Apache-2.0
//! Build columns with a variable-width column stay Arrow columns in the
//! row layout (`krishiv.exec.join.columnarPayload`, design
//! `perf1-join-columnar-string-payload`): every join type returns exactly
//! what the arrow-row payload returns (strings inline and past 12 bytes,
//! binaries, NULLs, plain `Utf8`), in memory and under a budget that
//! spills.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BinaryViewArray, Int32Array, Int64Array, RecordBatch, StringArray,
    StringViewArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_exec::{ExecConfig, JoinSpec, JoinType};
use krishiv_types::batch::Batch;
use proptest::prelude::*;

use super::util::*;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("s", DataType::Utf8View, true),
        Field::new("b", DataType::BinaryView, true),
        Field::new("u", DataType::Utf8, true),
        Field::new("i", DataType::Int32, false),
    ]))
}

/// One row: key, then payload values (`None` = NULL).
type Row = (
    Option<i64>,
    Option<String>,
    Option<Vec<u8>>,
    Option<String>,
    i32,
);

fn data(rows: &[Row]) -> RecordBatch {
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringViewArray::from(
            rows.iter().map(|r| r.1.as_deref()).collect::<Vec<_>>(),
        )),
        Arc::new(BinaryViewArray::from(
            rows.iter().map(|r| r.2.as_deref()).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.3.as_deref()).collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.4).collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema(), cols).unwrap()
}

/// Every cell exactly, with its column's type, rows sorted.
fn exact(batches: &[Batch]) -> Vec<String> {
    let cell = |a: &ArrayRef, i: usize| -> String {
        if a.is_null(i) {
            return "∅".to_owned();
        }
        match a.data_type() {
            DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
            DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
            DataType::Utf8View => format!("v:{}", a.as_string_view().value(i)),
            DataType::BinaryView => format!("x:{:?}", a.as_binary_view().value(i)),
            DataType::Utf8 => format!("s:{}", a.as_string::<i32>().value(i)),
            DataType::Boolean => a.as_boolean().value(i).to_string(),
            t => format!("<{t}>"),
        }
    };
    let mut out: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(move |i| {
                let cols = b.data().columns();
                cols.iter()
                    .map(|c| cell(c, i))
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect();
    out.sort();
    out
}

/// How the build keeps its payload.
#[derive(Debug, Clone, Copy)]
enum Form {
    /// Row layout off: arrow-row keys and payload.
    Arrow,
    /// Row-layout keys, arrow-row payload.
    Rows,
    /// Row-layout keys, Arrow payload columns.
    Columnar,
}

async fn run_join(
    join_type: JoinType,
    probe: &[Row],
    build: &[Row],
    form: Form,
    bits: u32,
    limit: usize,
    // The operators' batch size (the input's batches: at least 97 rows).
    batch: usize,
) -> (Vec<String>, u64) {
    let root = root(limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: batch,
        join_row_layout: !matches!(form, Form::Arrow),
        join_columnar_payload: matches!(form, Form::Columnar),
        join_radix_bits: bits,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let spec = JoinSpec {
        join_type,
        probe_schema: schema(),
        build_schema: schema(),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill: Some(scope.clone()),
    };
    let plan = join(
        spec,
        scan(&schema(), batches(&data(build), batch.max(97))),
        scan(&schema(), batches(&data(probe), batch.max(97))),
    );
    let out = run(&runner_for(2, limit), plan, &q).await.unwrap();
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0);
    (
        exact(&out),
        metric(&q, "HashJoinBuild", "spilled_partitions"),
    )
}

fn text() -> impl Strategy<Value = Option<String>> {
    prop_oneof![
        1 => Just(None),
        1 => Just(Some(String::new())),
        3 => "[a-c]{1,12}".prop_map(Some),
        3 => "[a-z]{13,40}".prop_map(Some),
    ]
}

fn row(keys: i64) -> impl Strategy<Value = Row> {
    (
        prop::option::weighted(0.9, 0..keys),
        text(),
        prop::option::of(prop::collection::vec(any::<u8>(), 0..30)),
        text(),
        any::<i32>(),
    )
}

const TYPES: [JoinType; 9] = [
    JoinType::Inner,
    JoinType::LeftOuter,
    JoinType::RightOuter,
    JoinType::FullOuter,
    JoinType::LeftSemi,
    JoinType::LeftAnti,
    JoinType::RightSemi,
    JoinType::RightAnti,
    JoinType::NullAwareAnti,
];

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]
    #[test]
    fn columnar_payload_equals_arrow_rows(
        t in 0usize..TYPES.len(),
        probe in prop::collection::vec(row(40), 0..300),
        build in prop::collection::vec(row(40), 0..300),
    ) {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let jt = TYPES[t];
        let (expect, _) = rt.block_on(run_join(jt, &probe, &build, Form::Arrow, 3, 64 * MIB, 64));
        for (form, bits) in [(Form::Rows, 3), (Form::Columnar, 3), (Form::Columnar, 6)] {
            let (got, _) = rt.block_on(run_join(jt, &probe, &build, form, bits, 64 * MIB, 64));
            prop_assert_eq!(&got, &expect, "{:?} {:?} {} bits", jt, form, bits);
        }
    }
}

/// A build several times the budget spills, reloads its payload columns and
/// returns the in-memory answer of the arrow-row path, for the joins that
/// read build columns from a reloaded partition. Batches of 2,048 rows: with
/// a handful of rows per spill frame, a dozen padded buffers per frame make
/// the partitions repartition (correct, but slow under a debug build).
#[tokio::test(start_paused = true)]
async fn spilled_columnar_payload_round_trips() {
    let rows: Vec<Row> = (0..60_000i64)
        .map(|i| {
            let s = match i % 4 {
                0 => None,
                1 => Some(format!("s{i}")),
                _ => Some(format!("a longer string past the inline limit {i}")),
            };
            let b = (i % 3 != 0).then(|| i.to_le_bytes().repeat(usize::try_from(i % 4).unwrap()));
            let u = (i % 5 != 0).then(|| format!("u{}", i % 977));
            (Some(i % 20_000), s, b, u, 7)
        })
        .collect();
    let probe: Vec<Row> = rows.iter().step_by(3).cloned().collect();
    for jt in [JoinType::Inner, JoinType::RightOuter, JoinType::FullOuter] {
        let (expect, mem) = run_join(jt, &probe, &rows, Form::Arrow, 3, 256 * MIB, 2048).await;
        assert_eq!(mem, 0);
        for bits in [3, 6] {
            let (got, spilled) =
                run_join(jt, &probe, &rows, Form::Columnar, bits, 3 * MIB, 2048).await;
            assert!(spilled > 0, "the build spills ({jt:?})");
            assert_eq!(got.len(), expect.len());
            assert!(
                got == expect,
                "spilled columnar payload differs ({jt:?}, {bits} bits)"
            );
        }
    }
}
