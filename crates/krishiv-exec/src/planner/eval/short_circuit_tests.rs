// SPDX-License-Identifier: Apache-2.0
//! `AND`, `OR` and `BETWEEN` give the same answers whether their last
//! operand runs on every row (it never raises) or only on the rows the
//! first one leaves open (it may raise, so Spark's short circuit shows).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::config::Config;
use krishiv_plan::Literal;

use super::logic::and_or;
use super::{Column, EvalEnv, Lit, PhysExpr, Value};
use crate::error::Result;

/// An expression that counts the rows it is evaluated over.
#[derive(Debug)]
struct Counted {
    inner: Arc<dyn PhysExpr>,
    rows: AtomicUsize,
}

impl Counted {
    fn new(inner: Arc<dyn PhysExpr>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            rows: AtomicUsize::new(0),
        })
    }

    fn rows(&self) -> usize {
        self.rows.load(Ordering::Relaxed)
    }
}

impl PhysExpr for Counted {
    fn data_type(&self) -> &ArrowType {
        self.inner.data_type()
    }

    fn nullable(&self) -> bool {
        self.inner.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        self.rows.fetch_add(batch.num_rows(), Ordering::Relaxed);
        self.inner.eval(batch, partition)
    }
}

fn column(index: usize, data_type: ArrowType) -> Arc<dyn PhysExpr> {
    Arc::new(Column {
        index,
        data_type,
        nullable: true,
    })
}

/// Every pair of `true`, `false` and NULL, as columns `l` and `r`.
fn pairs() -> RecordBatch {
    let side = |first: bool| -> ArrayRef {
        let values = [Some(true), Some(false), None];
        let all: Vec<Option<bool>> = (0..9)
            .map(|i| values[if first { i / 3 } else { i % 3 }])
            .collect();
        Arc::new(BooleanArray::from(all))
    };
    let schema = Schema::new(vec![
        Field::new("l", ArrowType::Boolean, true),
        Field::new("r", ArrowType::Boolean, true),
    ]);
    RecordBatch::try_new(Arc::new(schema), vec![side(true), side(false)]).unwrap()
}

fn answers(e: &dyn PhysExpr, batch: &RecordBatch) -> Vec<Option<bool>> {
    let v = e
        .eval(batch, 0)
        .unwrap()
        .into_array(batch.num_rows())
        .unwrap();
    v.as_boolean().iter().collect()
}

/// Kleene logic on both paths; a right side that may raise sees only the
/// six rows the left side leaves open, one that never raises sees all nine.
#[test]
fn and_or_answer_alike_on_every_row_or_on_the_open_rows() {
    let batch = pairs();
    let (t, f, n) = (Some(true), Some(false), None);
    let and = [t, f, n, f, f, f, n, f, n];
    let or = [t, t, t, t, f, n, t, n, n];
    for (is_and, expected) in [(true, and), (false, or)] {
        for (never_raises, seen) in [(true, 9), (false, 6)] {
            let r = Counted::new(column(1, ArrowType::Boolean));
            let e = and_or(
                is_and,
                column(0, ArrowType::Boolean),
                Arc::clone(&r) as Arc<dyn PhysExpr>,
                never_raises,
            );
            assert_eq!(answers(e.as_ref(), &batch), expected, "and={is_and}");
            assert_eq!(r.rows(), seen, "and={is_and} never_raises={never_raises}");
        }
    }
}

/// A left side that decides every row leaves the right side unevaluated,
/// even when it never raises.
#[test]
fn a_left_side_that_decides_every_row_skips_the_right_side() {
    let batch = pairs().slice(3, 3); // `l` is false on every row
    let r = Counted::new(column(1, ArrowType::Boolean));
    let e = and_or(
        true,
        column(0, ArrowType::Boolean),
        Arc::clone(&r) as Arc<dyn PhysExpr>,
        true,
    );
    assert_eq!(answers(e.as_ref(), &batch), [Some(false); 3]);
    assert_eq!(r.rows(), 0);
}

/// `v BETWEEN 1 AND hi` runs `hi` only where `v` is not NULL and `v >= 1`
/// is not false, unless `hi` never raises; the answers are the same.
#[test]
fn between_runs_a_raising_upper_bound_only_on_the_open_rows() {
    let env = EvalEnv::new(&Config::new(), Arc::from("test"));
    let v: ArrayRef = Arc::new(Int64Array::from(vec![
        Some(0),
        Some(5),
        Some(20),
        None,
        Some(7),
    ]));
    let hi: ArrayRef = Arc::new(Int64Array::from(vec![
        Some(10),
        Some(10),
        Some(10),
        Some(10),
        None,
    ]));
    let schema = Schema::new(vec![
        Field::new("v", ArrowType::Int64, true),
        Field::new("hi", ArrowType::Int64, true),
    ]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![v, hi]).unwrap();
    let expected = [Some(false), Some(true), Some(false), None, None];
    for (never_raises, seen) in [(true, 5), (false, 3)] {
        let hi = Counted::new(column(1, ArrowType::Int64));
        let lo: Arc<dyn PhysExpr> = Arc::new(Lit::new(&Literal::Long(1)).unwrap());
        let e = super::between::between(
            column(0, ArrowType::Int64),
            lo,
            Arc::clone(&hi) as Arc<dyn PhysExpr>,
            [true, never_raises],
            &env,
        )
        .unwrap();
        assert_eq!(answers(e.as_ref(), &batch), expected);
        assert_eq!(hi.rows(), seen, "never_raises={never_raises}");
    }
}

/// Scalar operands give one scalar answer; a `v >= lo` false for the
/// literal decides without the upper bound.
#[test]
fn between_of_literals_is_one_value() {
    let env = EvalEnv::new(&Config::new(), Arc::from("test"));
    let lit = |x: i64| -> Arc<dyn PhysExpr> { Arc::new(Lit::new(&Literal::Long(x)).unwrap()) };
    let batch = pairs();
    for (v, expected) in [(0, false), (3, true), (9, false)] {
        let e = super::between::between(lit(v), lit(1), lit(5), [true; 2], &env).unwrap();
        let out = e.eval(&batch, 0).unwrap();
        assert!(out.is_scalar());
        assert_eq!(out.array().as_boolean().value(0), expected, "v={v}");
    }
}

/// `lo` runs only where `v` is not NULL unless it never raises; a NULL
/// literal `v` answers NULL without running either bound.
#[test]
fn between_runs_no_bound_where_v_is_null() {
    let env = EvalEnv::new(&Config::new(), Arc::from("test"));
    let v: ArrayRef = Arc::new(Int64Array::from(vec![Some(3), None, Some(0), None]));
    let schema = Schema::new(vec![Field::new("v", ArrowType::Int64, true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![v]).unwrap();
    let five: Arc<dyn PhysExpr> = Arc::new(Lit::new(&Literal::Long(5)).unwrap());
    for (never_raises, seen) in [(true, 4), (false, 2)] {
        let lo = Counted::new(column(0, ArrowType::Int64));
        let e = super::between::between(
            column(0, ArrowType::Int64),
            Arc::clone(&lo) as Arc<dyn PhysExpr>,
            Arc::clone(&five),
            [never_raises, true],
            &env,
        )
        .unwrap();
        // `v BETWEEN v AND 5`: true for 3 and 0, NULL for the NULLs.
        assert_eq!(
            answers(e.as_ref(), &batch),
            [Some(true), None, Some(true), None]
        );
        assert_eq!(lo.rows(), seen, "never_raises={never_raises}");
    }
    let null: Arc<dyn PhysExpr> = Arc::new(Lit::new(&Literal::Null(None)).unwrap());
    let (lo, hi) = (
        Counted::new(column(0, ArrowType::Int64)),
        Counted::new(column(0, ArrowType::Int64)),
    );
    let e = super::between::between(
        null,
        Arc::clone(&lo) as Arc<dyn PhysExpr>,
        Arc::clone(&hi) as Arc<dyn PhysExpr>,
        [false; 2],
        &env,
    );
    // An untyped NULL meets the bounds' type at analysis; here the kernels
    // see `Null` beside `Int64`, which `compare` accepts.
    let e = e.unwrap();
    assert_eq!(answers(e.as_ref(), &batch), [None; 4]);
    assert_eq!((lo.rows(), hi.rows()), (0, 0));
}
