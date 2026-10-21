// SPDX-License-Identifier: Apache-2.0
//! Fixed-width build columns travel as payload rows in the row layout
//! (P2-T158): every join type returns exactly what the arrow-row path
//! returns, bit for bit (`-0.0`, NaN payloads, NULLs, decimals, dates,
//! booleans), in memory and under a budget that spills (fast == slow).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Decimal128Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int32Array,
    Int64Array, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_exec::{ExecConfig, JoinSpec, JoinType};
use krishiv_types::batch::Batch;
use proptest::prelude::*;

use super::util::*;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("f", DataType::Float64, true),
        Field::new("d", DataType::Decimal128(20, 3), true),
        Field::new("t", DataType::Date32, true),
        Field::new("b", DataType::Boolean, true),
        Field::new("i", DataType::Int32, false),
    ]))
}

/// One row: key, then payload values (`None` = NULL).
type Row = (
    Option<i64>,
    Option<f64>,
    Option<i128>,
    Option<i32>,
    Option<bool>,
    i32,
);

fn data(rows: &[Row]) -> RecordBatch {
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        )),
        Arc::new(
            Decimal128Array::from(rows.iter().map(|r| r.2).collect::<Vec<_>>())
                .with_precision_and_scale(20, 3)
                .unwrap(),
        ),
        Arc::new(Date32Array::from(
            rows.iter().map(|r| r.3).collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|r| r.4).collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.5).collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema(), cols).unwrap()
}

/// Every cell exactly (floats by their bits), rows sorted.
fn exact(batches: &[Batch]) -> Vec<String> {
    let cell = |a: &ArrayRef, i: usize| -> String {
        if a.is_null(i) {
            return "∅".to_owned();
        }
        match a.data_type() {
            DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
            DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
            DataType::Float64 => {
                format!("{:x}", a.as_primitive::<Float64Type>().value(i).to_bits())
            }
            DataType::Decimal128(..) => a.as_primitive::<Decimal128Type>().value(i).to_string(),
            DataType::Date32 => a.as_primitive::<Date32Type>().value(i).to_string(),
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

async fn run_join(
    join_type: JoinType,
    probe: &[Row],
    build: &[Row],
    row_layout: bool,
    limit: usize,
) -> (Vec<String>, u64) {
    let root = root(limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: 64,
        join_row_layout: row_layout,
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
        spill: Some(scope.clone()),
    };
    let plan = join(
        spec,
        scan(&schema(), batches(&data(build), 97)),
        scan(&schema(), batches(&data(probe), 97)),
    );
    let out = run(&runner_for(2, limit), plan, &q).await.unwrap();
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0);
    (
        exact(&out),
        metric(&q, "HashJoinBuild", "spilled_partitions"),
    )
}

fn row(keys: i64) -> impl Strategy<Value = Row> {
    let f = prop_oneof![
        1 => Just(None),
        1 => Just(Some(-0.0)),
        1 => Just(Some(f64::NAN)),
        1 => Just(Some(f64::from_bits(0x7ff8_0000_0000_0001))),
        4 => (-1e6f64..1e6).prop_map(Some),
    ];
    (
        prop::option::weighted(0.9, 0..keys),
        f,
        prop::option::of(-10_i128.pow(17)..10_i128.pow(17)),
        prop::option::of(-20_000i32..20_000),
        prop::option::of(any::<bool>()),
        any::<i32>(),
    )
}

const TYPES: [JoinType; 5] = [
    JoinType::Inner,
    JoinType::LeftOuter,
    JoinType::LeftSemi,
    JoinType::LeftAnti,
    JoinType::NullAwareAnti,
];

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]
    #[test]
    fn payload_rows_equal_arrow_rows(
        t in 0usize..5,
        probe in prop::collection::vec(row(40), 0..300),
        build in prop::collection::vec(row(40), 0..300),
    ) {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let jt = TYPES[t];
        let (expect, _) = rt.block_on(run_join(jt, &probe, &build, false, 64 * MIB));
        let (got, _) = rt.block_on(run_join(jt, &probe, &build, true, 64 * MIB));
        prop_assert_eq!(got, expect);
    }
}

/// A build several times the budget spills, reloads its payload rows and
/// returns the in-memory answer of the arrow-row path.
#[tokio::test(start_paused = true)]
async fn spilled_payload_rows_round_trip() {
    let rows: Vec<Row> = (0..60_000i64)
        .map(|i| {
            let f = match i % 5 {
                0 => Some(-0.0),
                1 => Some(f64::NAN),
                2 => None,
                _ => Some(i as f64 / 7.0),
            };
            let d = (i % 3 != 0).then_some(i128::from(i) * 1_000_003);
            (
                Some(i % 20_000),
                f,
                d,
                Some(i32::try_from(i % 9000).unwrap()),
                Some(i % 2 == 0),
                7,
            )
        })
        .collect();
    let probe: Vec<Row> = rows.iter().step_by(3).cloned().collect();
    let (expect, mem) = run_join(JoinType::Inner, &probe, &rows, false, 256 * MIB).await;
    assert_eq!(mem, 0);
    let (got, spilled) = run_join(JoinType::Inner, &probe, &rows, true, 2 * MIB).await;
    assert!(spilled > 0, "the build spills");
    assert_eq!(got.len(), expect.len());
    assert!(got == expect, "spilled payload rows differ");
}
