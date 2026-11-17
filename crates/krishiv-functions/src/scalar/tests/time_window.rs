// SPDX-License-Identifier: Apache-2.0
//! `__time_windows` and `__window_time`: the windows of Spark's `window(..)` (checked against the
//! `time-window.sql` golden of Spark 4.3.0 and a real Spark run; the session golden run covers the SQL forms).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::TimestampMicrosecondType;
use arrow_array::{Array, ArrayRef, Int64Array};

use super::{BindArg, BindOptions, bind_scalar, col, lit, ltz, ntz};
use crate::scalar::Arg;

const MIN: i64 = 60_000_000;

fn long(v: i64) -> ArrayRef {
    Arc::new(Int64Array::from(vec![v]))
}

/// The `(start, end)` pairs of each row of `ts` under `(window, slide, start)`.
fn windows(ts: &ArrayRef, (w, s, t): (i64, i64, i64)) -> Vec<Option<Vec<(i64, i64)>>> {
    let args = vec![
        BindArg::from(ts.data_type().clone()),
        BindArg::Literal(long(w)),
        BindArg::Literal(long(s)),
        BindArg::Literal(long(t)),
    ];
    let f = bind_scalar("__time_windows", &args, &BindOptions::new()).unwrap();
    let call: Vec<Arg> = vec![col(ts), lit(&long(w)), lit(&long(s)), lit(&long(t))];
    let out = f.invoke(&call, ts.len()).unwrap();
    let list = out.as_list::<i32>();
    (0..list.len())
        .map(|i| {
            list.is_valid(i).then(|| {
                let v = list.value(i);
                let st = v.as_struct();
                let (a, b) = (
                    st.column(0).as_primitive::<TimestampMicrosecondType>(),
                    st.column(1).as_primitive::<TimestampMicrosecondType>(),
                );
                (0..v.len()).map(|j| (a.value(j), b.value(j))).collect()
            })
        })
        .collect()
}

#[test]
fn a_tumbling_window_is_the_aligned_bucket_and_a_null_time_has_none() {
    let out = windows(
        &ltz(&[Some(4 * MIN + MIN / 2), Some(6 * MIN), None, Some(5 * MIN)]),
        (5 * MIN, 5 * MIN, 0),
    );
    assert_eq!(
        out,
        [
            Some(vec![(0, 5 * MIN)]),
            Some(vec![(5 * MIN, 10 * MIN)]),
            None,
            Some(vec![(5 * MIN, 10 * MIN)]),
        ]
    );
}

#[test]
fn a_time_before_the_epoch_floors() {
    // 1969-12-31 23:59:59.5 is in [-5 min, 0).
    let out = windows(&ntz(&[Some(-500_000)]), (5 * MIN, 5 * MIN, 0));
    assert_eq!(out, [Some(vec![(-5 * MIN, 0)])]);
}

#[test]
fn a_sliding_window_lists_every_window_holding_the_time() {
    // `window(ts, '10 minutes', '5 minutes')` at 00:00:00 and 00:04:30 (the golden's `sliding_window_basic`).
    let out = windows(
        &ltz(&[Some(0), Some(4 * MIN + MIN / 2)]),
        (10 * MIN, 5 * MIN, 0),
    );
    let sorted = |mut v: Vec<(i64, i64)>| {
        v.sort_unstable();
        v
    };
    assert_eq!(
        out.into_iter()
            .map(|r| sorted(r.unwrap()))
            .collect::<Vec<_>>(),
        [
            vec![(-5 * MIN, 5 * MIN), (0, 10 * MIN)],
            vec![(-5 * MIN, 5 * MIN), (0, 10 * MIN)],
        ]
    );
}

#[test]
fn a_window_that_does_not_divide_evenly_filters_the_windows_without_the_time() {
    // `window(ts, '7 minutes', '3 minutes')` at 00:00:00: three candidates, the ones holding 0.
    let mut out = windows(&ltz(&[Some(0)]), (7 * MIN, 3 * MIN, 0))
        .pop()
        .unwrap()
        .unwrap();
    out.sort_unstable();
    assert_eq!(out, [(-6 * MIN, MIN), (-3 * MIN, 4 * MIN), (0, 7 * MIN)]);
}

#[test]
fn a_start_time_shifts_the_windows_and_may_be_negative() {
    // `window(ts, '10 seconds', '10 seconds', '-5 seconds')` at 19:39:30 (the golden's `negative_start_time`).
    let s = 1_000_000;
    let out = windows(&ltz(&[Some(30 * s)]), (10 * s, 10 * s, -5 * s));
    assert_eq!(out, [Some(vec![(25 * s, 35 * s)])]);
    let out = windows(&ltz(&[Some(30 * s)]), (10 * s, 10 * s, 5 * s));
    assert_eq!(out, [Some(vec![(25 * s, 35 * s)])]);
}

#[test]
fn the_result_keeps_the_time_columns_type() {
    let ts = ntz(&[Some(0)]);
    let args = vec![
        BindArg::from(ts.data_type().clone()),
        BindArg::Literal(long(MIN)),
        BindArg::Literal(long(MIN)),
        BindArg::Literal(long(0)),
    ];
    let f = bind_scalar("__time_windows", &args, &BindOptions::new()).unwrap();
    let arrow_schema::DataType::List(field) = f.return_type() else {
        panic!("not a list");
    };
    let arrow_schema::DataType::Struct(fields) = field.data_type() else {
        panic!("not a struct");
    };
    assert!(
        fields
            .iter()
            .all(|f| f.data_type() == ts.data_type() && f.is_nullable())
    );
}

#[test]
fn window_time_is_the_last_microsecond_of_the_window() {
    let ts = ltz(&[Some(0), None]);
    let args = vec![
        BindArg::from(ts.data_type().clone()),
        BindArg::Literal(long(5 * MIN)),
        BindArg::Literal(long(5 * MIN)),
        BindArg::Literal(long(0)),
    ];
    let make = bind_scalar("__time_windows", &args, &BindOptions::new()).unwrap();
    let call: Vec<Arg> = vec![
        col(&ts),
        lit(&long(5 * MIN)),
        lit(&long(5 * MIN)),
        lit(&long(0)),
    ];
    let list = make.invoke(&call, 2).unwrap();
    // The first row's only window, as a struct column.
    let first = list.as_list::<i32>().value(0);
    let window: ArrayRef = Arc::new(first.as_struct().clone());
    let f = bind_scalar(
        "__window_time",
        &[BindArg::from(window.data_type().clone())],
        &BindOptions::new(),
    )
    .unwrap();
    let out = f.invoke(&[col(&window)], 1).unwrap();
    assert_eq!(
        out.as_primitive::<TimestampMicrosecondType>().value(0),
        5 * MIN - 1
    );
}
