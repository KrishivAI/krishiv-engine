// SPDX-License-Identifier: Apache-2.0
//! `counter_diff` against the rows of Spark 4.3.0's `counter-diff.sql` golden (each case names its query).

use arrow_array::{
    Array, ArrayRef, Decimal128Array, Float32Array, Float64Array, Int32Array,
    TimestampMicrosecondArray,
};

use super::*;
use arrow_array::types::{Decimal128Type, Float32Type, Float64Type, Int32Type};

fn utc() -> ZoneRef {
    krishiv_kernels::datetime::resolve_zone("UTC").unwrap()
}

fn ints(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn run(counter: &ArrayRef) -> Vec<Option<i32>> {
    let out = CounterState::default().run(counter, None, &utc()).unwrap();
    out.as_primitive::<Int32Type>().iter().collect()
}

#[test]
fn a_reset_is_null_and_the_next_row_diffs_from_it() {
    // (100, 200, 400, 50, 100): NULL 100 200 NULL 50.
    let c = ints(&[Some(100), Some(200), Some(400), Some(50), Some(100)]);
    assert_eq!(run(&c), [None, Some(100), Some(200), None, Some(50)]);
}

#[test]
fn equal_values_diff_to_zero_and_a_single_row_is_null() {
    assert_eq!(
        run(&ints(&[Some(100), Some(100), Some(200)])),
        [None, Some(0), Some(100)]
    );
    assert_eq!(run(&ints(&[Some(50)])), [None]);
}

#[test]
fn a_null_counter_is_null_and_skipped() {
    assert_eq!(
        run(&ints(&[Some(100), None, Some(200)])),
        [None, None, Some(100)]
    );
    assert_eq!(run(&ints(&[None, None])), [None, None]);
}

#[test]
fn a_negative_counter_is_an_error_even_after_a_null() {
    for c in [ints(&[Some(-5)]), ints(&[Some(100), None, Some(-5)])] {
        let e = CounterState::default().run(&c, None, &utc()).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "COUNTER_DIFF_NEGATIVE_COUNTER_VALUE"
        );
        assert!(e.to_string().contains("-5"), "{e}");
    }
}

#[test]
fn the_negative_value_is_quoted_as_spark_casts_it() {
    let d: ArrayRef = Arc::new(
        Decimal128Array::from(vec![-5500_i128])
            .with_precision_and_scale(10, 3)
            .unwrap(),
    );
    let e = CounterState::default().run(&d, None, &utc()).unwrap_err();
    assert!(e.to_string().contains("-5.500"), "{e}");
    let f: ArrayRef = Arc::new(Float64Array::from(vec![f64::NEG_INFINITY]));
    let e = CounterState::default().run(&f, None, &utc()).unwrap_err();
    assert!(e.to_string().contains("-Infinity"), "{e}");
    let f: ArrayRef = Arc::new(Float64Array::from(vec![-5.0]));
    let e = CounterState::default().run(&f, None, &utc()).unwrap_err();
    assert!(e.to_string().contains("-5.0"), "{e}");
}

#[test]
fn floats_order_with_nan_above_everything() {
    let nan = f64::NAN;
    // 100, NaN, NaN, 200, 400, NaN, 50, 100: NULL NaN NaN NULL 200 NaN NULL 50.
    let c: ArrayRef = Arc::new(Float64Array::from(vec![
        100.0, nan, nan, 200.0, 400.0, nan, 50.0, 100.0,
    ]));
    let out = CounterState::default().run(&c, None, &utc()).unwrap();
    let out = out.as_primitive::<Float64Type>();
    let shown: Vec<Option<String>> = out.iter().map(|v| v.map(|v| v.to_string())).collect();
    let s = |v: &str| Some(v.to_owned());
    assert_eq!(
        shown,
        [
            None,
            s("NaN"),
            s("NaN"),
            None,
            s("200"),
            s("NaN"),
            None,
            s("50")
        ]
    );
    // Infinity: 100, inf, 200, inf, inf: NULL inf NULL inf NaN.
    let inf = f64::INFINITY;
    let c: ArrayRef = Arc::new(Float64Array::from(vec![100.0, inf, 200.0, inf, inf]));
    let out = CounterState::default().run(&c, None, &utc()).unwrap();
    let shown: Vec<Option<String>> = out
        .as_primitive::<Float64Type>()
        .iter()
        .map(|v| v.map(|v| v.to_string()))
        .collect();
    assert_eq!(shown, [None, s("inf"), None, s("inf"), s("NaN")]);
}

#[test]
fn a_float_diff_is_a_float_diff() {
    let c: ArrayRef = Arc::new(Float32Array::from(vec![1.5_f32, 3.5]));
    let out = CounterState::default().run(&c, None, &utc()).unwrap();
    assert_eq!(
        out.as_primitive::<Float32Type>().iter().collect::<Vec<_>>(),
        [None, Some(2.0)]
    );
}

#[test]
fn a_decimal_keeps_its_type_and_every_digit() {
    let c: ArrayRef = Arc::new(
        Decimal128Array::from(vec![
            12_345_678_901_234_567_890_123_456_789_012_123_456_i128,
            12_345_678_901_234_567_890_123_456_789_012_123_457_i128,
        ])
        .with_precision_and_scale(38, 6)
        .unwrap(),
    );
    let out = CounterState::default().run(&c, None, &utc()).unwrap();
    assert_eq!(out.data_type(), &arrow_schema::DataType::Decimal128(38, 6));
    assert_eq!(
        out.as_primitive::<Decimal128Type>()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some(1)]
    );
}

fn times(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(TimestampMicrosecondArray::from(v.to_vec()))
}

const MIN: i64 = 60_000_000;

fn with_start(counter: &[Option<i32>], start: &[Option<i64>]) -> Result<Vec<Option<i32>>> {
    let out = CounterState::default().run(&ints(counter), Some(&times(start)), &utc())?;
    Ok(out.as_primitive::<Int32Type>().iter().collect())
}

#[test]
fn an_advanced_start_time_is_a_reset() {
    let got = with_start(
        &[Some(100), Some(200), Some(400)],
        &[Some(0), Some(0), Some(151 * 1_000_000)],
    )
    .unwrap();
    assert_eq!(got, [None, Some(100), None]);
}

#[test]
fn a_null_start_time_skips_the_check_on_its_row_and_the_next() {
    let got = with_start(
        &[Some(100), Some(200), Some(300)],
        &[Some(0), None, Some(MIN)],
    )
    .unwrap();
    assert_eq!(got, [None, Some(100), Some(100)]);
}

#[test]
fn a_decreased_start_time_is_an_error_naming_both() {
    let e = with_start(&[Some(100), Some(200)], &[Some(5 * MIN), Some(MIN)]).unwrap_err();
    assert_eq!(e.error_class().name(), "COUNTER_DIFF_START_TIME_DECREASED");
    let text = e.to_string();
    assert!(
        text.contains("1970-01-01 00:05:00") && text.contains("1970-01-01 00:01:00"),
        "{text}"
    );
}

#[test]
fn a_null_counter_row_absorbs_its_start_time_until_the_next_row() {
    // The decrease on the NULL row is absorbed ...
    let got = with_start(&[Some(100), None], &[Some(5 * MIN), Some(MIN)]).unwrap();
    assert_eq!(got, [None, None]);
    // ... but the next counter row still sees it.
    assert!(
        with_start(
            &[Some(100), None, Some(300)],
            &[Some(5 * MIN), Some(MIN), Some(MIN)]
        )
        .is_err()
    );
    // An increase on the NULL row is a reset on the next counter row.
    let got = with_start(
        &[Some(100), None, Some(300)],
        &[Some(5 * MIN), Some(6 * MIN), Some(7 * MIN)],
    )
    .unwrap();
    assert_eq!(got, [None, None, None]);
}

#[test]
fn the_state_carries_over_from_one_call_to_the_next() {
    // A partition read in two batches (the on-disk path) gives what one batch gives.
    let mut state = CounterState::default();
    let a = state
        .run(&ints(&[Some(100), Some(200)]), None, &utc())
        .unwrap();
    let b = state
        .run(&ints(&[Some(150), Some(400)]), None, &utc())
        .unwrap();
    assert_eq!(
        a.as_primitive::<Int32Type>().iter().collect::<Vec<_>>(),
        [None, Some(100)]
    );
    assert_eq!(
        b.as_primitive::<Int32Type>().iter().collect::<Vec<_>>(),
        [None, Some(250)]
    );
}

#[test]
fn every_integral_width_and_a_decimal64_keep_their_type() {
    use arrow_array::types::{Decimal64Type, Int8Type, Int16Type, Int64Type};
    use arrow_array::{Decimal64Array, Int8Array, Int16Array, Int64Array};
    let c: ArrayRef = Arc::new(Int8Array::from(vec![Some(10_i8), None, Some(15)]));
    let out = CounterState::default().run(&c, None, &utc()).unwrap();
    assert_eq!(
        out.as_primitive::<Int8Type>().iter().collect::<Vec<_>>(),
        [None, None, Some(5)]
    );
    let c: ArrayRef = Arc::new(Int16Array::from(vec![300_i16, 100, 250]));
    let out = CounterState::default().run(&c, None, &utc()).unwrap();
    assert_eq!(
        out.as_primitive::<Int16Type>().iter().collect::<Vec<_>>(),
        [None, None, Some(150)]
    );
    let c: ArrayRef = Arc::new(Int64Array::from(vec![1_i64 << 40, (1_i64 << 40) + 7]));
    let out = CounterState::default().run(&c, None, &utc()).unwrap();
    assert_eq!(
        out.as_primitive::<Int64Type>().iter().collect::<Vec<_>>(),
        [None, Some(7)]
    );
    let c: ArrayRef = Arc::new(
        Decimal64Array::from(vec![1_000_i64, 1_250])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    );
    let out = CounterState::default().run(&c, None, &utc()).unwrap();
    assert_eq!(out.data_type(), &arrow_schema::DataType::Decimal64(10, 2));
    assert_eq!(
        out.as_primitive::<Decimal64Type>()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some(250)]
    );
}

#[test]
fn a_float32_follows_the_same_nan_order_and_refuses_a_negative() {
    let nan = f32::NAN;
    let c: ArrayRef = Arc::new(Float32Array::from(vec![100.0_f32, nan, nan, 200.0]));
    let out = CounterState::default().run(&c, None, &utc()).unwrap();
    let shown: Vec<Option<String>> = out
        .as_primitive::<Float32Type>()
        .iter()
        .map(|v| v.map(|v| v.to_string()))
        .collect();
    assert_eq!(
        shown,
        [None, Some("NaN".to_owned()), Some("NaN".to_owned()), None]
    );
    let c: ArrayRef = Arc::new(Float32Array::from(vec![-1.5_f32]));
    let e = CounterState::default().run(&c, None, &utc()).unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "COUNTER_DIFF_NEGATIVE_COUNTER_VALUE"
    );
}

#[test]
fn a_type_the_function_does_not_take_is_an_internal_error() {
    let c: ArrayRef = Arc::new(arrow_array::StringArray::from(vec!["a"]));
    assert!(CounterState::default().run(&c, None, &utc()).is_err());
    let d: ArrayRef = Arc::new(arrow_array::Date32Array::from(vec![1]));
    assert!(
        CounterState::default()
            .run(&ints(&[Some(1)]), Some(&d), &utc())
            .is_err()
    );
}
