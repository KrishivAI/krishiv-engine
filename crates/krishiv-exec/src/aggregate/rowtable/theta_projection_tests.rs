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

/// One big sketch and fifteen small ones in sixteen groups spread over several radix partitions: each partition is
/// charged the sketches of its own rows (the fold's partition assignment), so the big sketch is charged once, not
/// once per partition the batch touches.
#[test]
fn a_sketch_is_charged_to_the_partition_that_reads_it() {
    let function = AggregateFunction::ThetaIntersectionAgg;
    let core = core(function, DataType::BinaryView);
    let seed = HashSeed::new(Some(3));
    let t = RowTable::new(&core, &seed, Mode::Grow, true, RADIX_BITS).unwrap();
    let big = big_sketch();
    let mut small = theta::new_sketch(12, "theta_sketch_agg").unwrap();
    theta::update_long(&mut small, 1);
    let small = theta::to_bytes(&theta::compact(&mut small));
    let rows = 16_usize;
    let sk: ArrayRef = Arc::new(BinaryViewArray::from_iter_values(
        (0..rows).map(|i| if i == 0 { big.clone() } else { small.clone() }),
    ));
    let g: Vec<i64> = (0..16).collect();
    let cols: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(g)), sk];
    let b = Batch::new(RecordBatch::try_new(Arc::clone(&core.spec.input), cols).unwrap());
    let p = t.prepare(&b, false).unwrap();
    let touched = p.counts.iter().filter(|&&c| c > 0).count();
    assert!(touched > 2, "the groups must spread: {touched} partitions");
    let growth = t.projected(&p) - t.memory_size();
    let bytes = big.len() + 15 * small.len();
    // As one partition holding the whole column (`a_partition_with_no_row_is_not_charged_the_column`): about 64
    // bytes per sketch byte, once. Charging the column to every touched partition would be `touched` times that.
    assert!(growth >= 32 * big.len(), "{growth} for {bytes}");
    assert!(
        growth < 70 * bytes + (8 << 20),
        "{growth} for {bytes} over {touched} partitions"
    );
}

/// A raw `ARRAY<BIGINT>` value is copied into one buffer before it is hashed: the projection holds the largest
/// array's bytes (`len * 8`), once, and nothing for a column without one.
#[test]
fn the_buffer_of_the_largest_array_is_projected() {
    use arrow_array::ListArray;
    use arrow_array::types::Int64Type;
    let item = Arc::new(Field::new("item", DataType::Int64, true));
    let input = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Int64, true),
        Field::new("a", DataType::List(Arc::clone(&item)), true),
    ]));
    let spec = AggregateSpec {
        input: Arc::clone(&input),
        group_by: vec![0],
        aggregates: vec![AggregateExpr::new(
            AggregateFunction::ThetaSketchAgg { lg_nom: 12 },
            1,
            DataType::Binary,
            "t",
        )],
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        legacy_statistical: false,
        udafs: None,
    };
    let core = Arc::new(AggCore::new(spec).unwrap());
    let seed = HashSeed::new(Some(3));
    let t = RowTable::new(&core, &seed, Mode::Grow, true, 0).unwrap();
    let growth = |arrays: Vec<Vec<Option<i64>>>| {
        let n = arrays.len();
        let list: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
            arrays.into_iter().map(Some),
        ));
        let g: ArrayRef = Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>()));
        let b = Batch::new(RecordBatch::try_new(Arc::clone(&input), vec![g, list]).unwrap());
        let p = t.prepare(&b, false).unwrap();
        t.projected(&p) + t.transient(&p) - t.memory_size()
    };
    let small = growth(vec![vec![Some(1); 4]; 2]);
    let big = growth(vec![vec![Some(1); 4], vec![Some(1); 1_000_000]]);
    assert!(big >= small + 7_900_000, "{big} against {small}");
}
