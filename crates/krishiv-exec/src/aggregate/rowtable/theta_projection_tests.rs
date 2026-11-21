// SPDX-License-Identifier: Apache-2.0
//! The row table's projection of the Theta union and intersection over a raw sketch column: the bytes of the
//! sketches are charged (for a `BinaryView` and a `Binary` column alike), and a partition that gets no row of the
//! batch is charged none of them.

use arrow_array::{BinaryArray, BinaryViewArray, Int64Array};
use arrow_schema::{DataType, Field, Schema};
use krishiv_functions::sketch::theta;

use super::*;
use crate::aggregate::{AggregateExpr, AggregateFunction, AggregateOutput, AggregateSpec};

fn core(function: AggregateFunction, sketches: DataType) -> Arc<AggCore> {
    let input = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Int64, true),
        Field::new("s", sketches, true),
    ]));
    let spec = AggregateSpec {
        input,
        group_by: vec![0],
        aggregates: vec![AggregateExpr::new(function, 1, DataType::Binary, "t")],
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        legacy_statistical: false,
        udafs: None,
    };
    Arc::new(AggCore::new(spec).unwrap())
}

fn big_sketch() -> Vec<u8> {
    let mut s = theta::new_sketch(16, "theta_sketch_agg").unwrap();
    for v in 0..300_000_i64 {
        theta::update_long(&mut s, v);
    }
    theta::to_bytes(&theta::compact(&mut s))
}

/// `rows` rows, all of group `groups[i % groups.len()]`, each one big sketch.
fn batch(core: &AggCore, view: bool, rows: usize, groups: &[i64]) -> (Batch, usize) {
    let bytes = big_sketch();
    let g: Vec<i64> = (0..rows).map(|i| groups[i % groups.len()]).collect();
    let sk: ArrayRef = if view {
        Arc::new(BinaryViewArray::from_iter_values(
            (0..rows).map(|_| bytes.clone()),
        ))
    } else {
        Arc::new(BinaryArray::from_iter_values(
            (0..rows).map(|_| bytes.clone()),
        ))
    };
    let cols: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(g)), sk];
    let b = Batch::new(RecordBatch::try_new(Arc::clone(&core.spec.input), cols).unwrap());
    (b, rows * bytes.len())
}

fn projection(function: AggregateFunction, view: bool, groups: &[i64]) -> (usize, usize) {
    let ty = if view {
        DataType::BinaryView
    } else {
        DataType::Binary
    };
    let core = core(function, ty);
    let seed = HashSeed::new(Some(3));
    let t = RowTable::new(&core, &seed, Mode::Grow, true, RADIX_BITS).unwrap();
    let (b, bytes) = batch(&core, view, 8, groups);
    let p = t.prepare(&b, false).unwrap();
    (t.projected(&p) - t.memory_size(), bytes)
}

#[test]
fn a_raw_sketch_column_is_charged_by_its_bytes_whatever_the_carrier() {
    for function in [
        AggregateFunction::ThetaIntersectionAgg,
        AggregateFunction::ThetaUnionAgg { lg_nom: 12 },
    ] {
        for view in [true, false] {
            let (growth, bytes) = projection(function.clone(), view, &[7]);
            // The views alone would be 8 * 16 bytes; the sketches are what is folded. A union's table is capped
            // by its lgNomEntries, an intersection's follows its input.
            let floor = if matches!(function, AggregateFunction::ThetaIntersectionAgg) {
                bytes
            } else {
                100_000
            };
            assert!(
                growth > floor,
                "{function:?} view={view}: {growth} <= {floor}"
            );
        }
    }
}

#[test]
fn a_partition_with_no_row_is_not_charged_the_column() {
    let function = AggregateFunction::ThetaIntersectionAgg;
    let (one, bytes) = projection(function, true, &[7]);
    // One group in one partition: the column is charged once (32 table bytes per byte, doubled by the transient; 16
    // partitions charged alike would be 16 times that).
    assert!(one < 70 * bytes + (8 << 20), "{one} for {bytes}");
    // ... and the sketch bytes are charged at all (a projection that stopped charging them would pass the cap alone).
    assert!(one >= 32 * bytes, "{one} for {bytes}");
}
