// SPDX-License-Identifier: Apache-2.0
//! Spark's `NumericHistogram` against the Spark 4.3.0 golden of `group-by.sql`.

use super::*;

fn hist(nb: usize, values: impl IntoIterator<Item = f64>) -> NumericHistogram {
    let mut h = NumericHistogram::new(nb);
    for v in values {
        h.add(v);
    }
    h
}

fn fifty() -> Vec<f64> {
    (1..=50)
        .map(|i| if i == 36 { 3.0 } else { f64::from(i) })
        .collect()
}

#[expect(clippy::cast_possible_truncation, reason = "test centers are small")]
fn rounded(h: &NumericHistogram) -> Vec<(i64, f64)> {
    h.bins().iter().map(|b| (b.0 as i64, b.1)).collect()
}

#[test]
fn java_random_matches_the_jdk() {
    // `new Random(42).nextDouble()` is 0.7275636800328681.
    let mut r = JavaRandom {
        seed: (42 ^ JavaRandom::MULTIPLIER) & JavaRandom::MASK,
    };
    assert_eq!(r.next_double(), 0.727_563_680_032_868_1);
}

/// The golden's rows in `parts` contiguous slices (Spark's local relation
/// under `local[2]`), each aggregated, serialized and merged in order into a
/// fresh histogram, as Spark's partial and final phases do.
fn two_phase(nb: usize, parts: usize) -> NumericHistogram {
    let v = fifty();
    let mut fin = NumericHistogram::new(nb);
    for part in 0..parts {
        let (lo, hi) = (part * v.len() / parts, (part + 1) * v.len() / parts);
        let p = hist(nb, v[lo..hi].iter().copied());
        fin.merge(&NumericHistogram::deserialize(&p.serialize().unwrap()).unwrap());
    }
    fin
}

#[test]
fn bins_match_the_spark_golden() {
    // `group-by.sql` under Spark 4.3.0: the local relation has two partitions.
    assert_eq!(rounded(&two_phase(2, 2)), vec![(12, 26.0), (38, 24.0)]);
    assert_eq!(
        rounded(&two_phase(3, 2)),
        vec![(9, 20.0), (25, 11.0), (40, 19.0)]
    );
    assert_eq!(
        rounded(&two_phase(5, 2)),
        vec![(5, 11.0), (14, 8.0), (22, 7.0), (30, 10.0), (43, 14.0)]
    );
    assert_eq!(
        rounded(&two_phase(10, 2)),
        vec![
            (3, 6.0),
            (8, 6.0),
            (13, 4.0),
            (17, 3.0),
            (20, 4.0),
            (25, 6.0),
            (31, 7.0),
            (39, 5.0),
            (43, 4.0),
            (48, 5.0)
        ]
    );
}

#[test]
fn one_partition_differs_from_two() {
    // The result depends on the partitioning, as Spark's does.
    assert_eq!(rounded(&two_phase(2, 1)), vec![(17, 36.0), (43, 14.0)]);
}

#[test]
fn fewer_values_than_bins_stay_exact() {
    let h = hist(3, [1.0, 2.0, 3.0]);
    assert_eq!(h.bins(), &[(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]);
    let h = hist(5, [4.0, 4.0, 4.0]);
    assert_eq!(h.bins(), &[(4.0, 3.0)]);
}

#[test]
fn state_round_trips_and_merges_like_the_original() {
    let v = fifty();
    let (a, b) = v.split_at(20);
    let mut left = hist(5, a.iter().copied());
    let right = hist(5, b.iter().copied());
    let revived = NumericHistogram::deserialize(&right.serialize().unwrap()).unwrap();
    assert_eq!(revived.bins(), right.bins());
    left.merge(&revived);
    assert!(left.bins().len() <= 5);
    let total: f64 = left.bins().iter().map(|b| b.1).sum();
    assert_eq!(total, 50.0);
    // Merging into an empty histogram copies.
    let mut empty = NumericHistogram::new(5);
    empty.merge(&right);
    assert_eq!(empty.bins(), right.bins());
}

#[test]
fn bad_state_is_refused() {
    assert!(NumericHistogram::deserialize(&[0, 0, 0, 2, 0, 0, 0, 1]).is_err());
    assert!(NumericHistogram::deserialize(&[1, 2]).is_err());
    // nbins 0 or 1: `trim` would index bins[1].
    assert!(NumericHistogram::deserialize(&[0, 0, 0, 1, 0, 0, 0, 0]).is_err());
    assert!(NumericHistogram::deserialize(&[0, 0, 0, 0, 0, 0, 0, 0]).is_err());
}

#[test]
fn nan_and_infinities_match_spark() {
    // Spark 4.3.0, `local[1]`, `histogram_numeric(col, 3)` over `VALUES`:
    // NaN, 1, +Inf, -Inf, 2, NaN gave [{NaN,3},{Infinity,2},{NaN,1}] ...
    let h = hist(
        3,
        [
            f64::NAN,
            1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            2.0,
            f64::NAN,
        ],
    );
    let bins = h.bins();
    assert_eq!(bins.len(), 3);
    assert!(bins[0].0.is_nan() && bins[0].1 == 3.0);
    assert!(bins[1].0 == f64::INFINITY && bins[1].1 == 2.0);
    assert!(bins[2].0.is_nan() && bins[2].1 == 1.0);
    // ... and 1, +Inf, -Inf, 2, 5 gave [{-Infinity,1},{2.666666666666667,3},{Infinity,1}].
    let h = hist(3, [1.0, f64::INFINITY, f64::NEG_INFINITY, 2.0, 5.0]);
    assert_eq!(
        h.bins(),
        &[
            (f64::NEG_INFINITY, 1.0),
            (2.666_666_666_666_667, 3.0),
            (f64::INFINITY, 1.0)
        ]
    );
}

#[test]
fn decimal_centers_round_in_two_steps() {
    // 1.2449999 -> 4 significant digits 1.245 -> scale 2 half up 1.25.
    assert_eq!(decimal_unscaled_in(1.244_999_9, 4, 2), 125);
    assert_eq!(decimal_unscaled_in(2.0, 4, 2), 200);
    assert_eq!(decimal_unscaled_in(-2.5, 4, 1), -25);
}

/// The accumulator's partial and final phases (`Accumulator::update`, `state`,
/// `merge`) equal the direct two-partition merge of the Spark golden.
#[test]
fn the_accumulator_merges_partial_states_like_the_golden() {
    use crate::aggregate::{Accumulator, AggInput, AggregateExpr, AggregateFunction};
    use arrow_array::{Array, BinaryArray, Float64Array};
    let expr = AggregateExpr {
        function: AggregateFunction::HistogramNumeric { nb: 3 },
        args: vec![0],
        input: AggInput::Raw,
        arg_type: DataType::Float64,
        name: "h".to_owned(),
        mask: None,
        element: None,
    };
    let acc = || {
        let mut a = Accumulator::new(&expr, true).unwrap();
        a.resize(1);
        a
    };
    let v = fifty();
    let mut top = acc();
    for part in v.chunks(25) {
        let mut a = acc();
        let col = Arc::new(Float64Array::from(part.to_vec())) as ArrayRef;
        a.update(
            &vec![0; part.len()],
            &[col],
            &krishiv_types::batch::Weights::Const(1),
        )
        .unwrap();
        // The projection counts the serialized bins of a state batch, twice.
        let states = a.state(&[0]).unwrap();
        let held = top.projected(0, Some(&states[0]), None, false);
        let bins = states[0]
            .as_any()
            .downcast_ref::<BinaryArray>()
            .map(|b| (b.value(0).len() - 8) / 16)
            .unwrap();
        assert!(held >= top.size() + 2 * bins * 16, "{held} for {bins} bins");
        top.merge(&[0], &states).unwrap();
    }
    let merged = top.state(&[0]).unwrap();
    let merged = merged[0].as_any().downcast_ref::<BinaryArray>().unwrap();
    assert_eq!(
        merged.value(0),
        two_phase(3, 2).serialize().unwrap().as_slice()
    );
    // A group with no value has a NULL state, which a merge skips.
    let empty = acc();
    let none = empty.state(&[0]).unwrap();
    assert!(
        none[0]
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .is_null(0)
    );
}
