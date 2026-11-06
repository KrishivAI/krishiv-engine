// SPDX-License-Identifier: Apache-2.0
//! The batch fold of a struct `MIN`/`MAX` against feeding the same rows one
//! at a time (the row-at-a-time fold the batch path replaces).

use super::*;
use arrow_array::{Float64Array, Int32Array, StringArray, StructArray};
use arrow_schema::Field;
use proptest::prelude::*;

type Cell = Option<(Option<i32>, Option<&'static str>, f64)>;

fn column(rows: &[(u32, Cell)]) -> ArrayRef {
    let fields: arrow_schema::Fields = vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8, true),
        Field::new("c", DataType::Float64, true),
    ]
    .into();
    let a = Int32Array::from_iter(rows.iter().map(|(_, c)| c.and_then(|c| c.0)));
    let b = StringArray::from_iter(rows.iter().map(|(_, c)| c.and_then(|c| c.1)));
    let c = Float64Array::from_iter(rows.iter().map(|(_, c)| c.map(|c| c.2)));
    let nulls = NullBuffer::from_iter(rows.iter().map(|(_, c)| c.is_some()));
    Arc::new(StructArray::new(
        fields,
        vec![Arc::new(a) as ArrayRef, Arc::new(b), Arc::new(c)],
        Some(nulls),
    ))
}

fn struct_type() -> DataType {
    column(&[]).data_type().clone()
}

fn run(f: AggregateFunction, batches: &[Vec<(u32, Cell)>]) -> String {
    run_groups(f, batches, 5)
}

fn run_groups(f: AggregateFunction, batches: &[Vec<(u32, Cell)>], groups: u32) -> String {
    let t = struct_type();
    let mut acc = Accumulator::new(&AggregateExpr::new(f, 0, t, "a"), true).unwrap();
    acc.resize(groups as usize);
    for rows in batches {
        let groups: Vec<u32> = rows.iter().map(|r| r.0).collect();
        acc.update(&groups, &[column(rows)], &Weights::Const(1))
            .unwrap();
    }
    let ids: Vec<u32> = (0..groups).collect();
    format!("{:?}", acc.evaluate(&ids).unwrap())
}

fn cell() -> impl Strategy<Value = Cell> {
    let a = prop_oneof![Just(None), (0..3).prop_map(Some)];
    let b = prop_oneof![Just(None), Just(Some("x")), Just(Some("y")), Just(Some(""))];
    let c = prop_oneof![Just(f64::NAN), Just(-0.0), Just(0.0), Just(1.0), Just(-1.0)];
    prop_oneof![1 => Just(None), 4 => (a, b, c).prop_map(Some)]
}

/// Timing note (run with `--ignored --nocapture`): 200,000 rows over 100
/// groups folded in batches of 4,096 against one row per batch (the cost
/// profile of the row-at-a-time fold).
#[test]
#[ignore = "timing note, not a check"]
fn batch_fold_timing() {
    let rows: Vec<(u32, Cell)> = (0..200_000u32)
        .map(|i| {
            let v = i32::try_from(i.wrapping_mul(2_654_435_761) % 1000).unwrap_or(0);
            (i % 100, Some((Some(v), Some("x"), f64::from(v))))
        })
        .collect();
    let batched: Vec<Vec<(u32, Cell)>> = rows.chunks(4096).map(<[_]>::to_vec).collect();
    let single: Vec<Vec<(u32, Cell)>> = rows.iter().map(|r| vec![*r]).collect();
    let t = std::time::Instant::now();
    let a = run_groups(AggregateFunction::Max, &batched, 100);
    let batch_time = t.elapsed();
    let t = std::time::Instant::now();
    let b = run_groups(AggregateFunction::Max, &single, 100);
    let single_time = t.elapsed();
    assert_eq!(a, b);
    println!("TIMING batches of 4096: {batch_time:?}; one row per batch: {single_time:?}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn batch_fold_equals_row_at_a_time(
        rows in proptest::collection::vec((0u32..5, cell()), 0..48),
        split in 1usize..6,
    ) {
        for f in [AggregateFunction::Max, AggregateFunction::Min] {
            // Whole batches of `split` rows against one row per batch.
            let batched: Vec<Vec<(u32, Cell)>> =
                rows.chunks(split).map(<[_]>::to_vec).collect();
            let single: Vec<Vec<(u32, Cell)>> =
                rows.iter().map(|r| vec![*r]).collect();
            prop_assert_eq!(run(f.clone(), &batched), run(f, &single));
        }
    }
}
