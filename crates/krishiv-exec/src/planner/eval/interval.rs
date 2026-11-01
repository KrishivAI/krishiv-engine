// SPDX-License-Identifier: Apache-2.0
//! Interval operators: the rewrites of Spark's `BinaryArithmeticWithDatetimeResolver`
//! and the arithmetic of intervals themselves.

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::types::{DurationMicrosecondType, Int32Type, Int64Type, IntervalYearMonthType};
use arrow_array::{Array, ArrayRef, RecordBatch, new_null_array};
use arrow_schema::{DataType as ArrowType, IntervalUnit, TimeUnit};
use krishiv_common::error::KrishivError;
use krishiv_kernels::datetime::{
    ZoneRef, date_add_months, timestamp_add_day_time, timestamp_add_months,
};
use krishiv_kernels::interval::{
    AddOp, ScaleOp, add_day_time, add_year_month, extract_days, negate_day_time, negate_year_month,
    scale_day_time, scale_year_month, subtract_dates, subtract_timestamps,
};
use krishiv_plan::Builtin;

use super::logic::prim;
use super::{EvalEnv, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::internal;

/// Whether `t` carries an ANSI interval.
pub(super) fn is_ansi(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::Interval(IntervalUnit::YearMonth) | ArrowType::Duration(TimeUnit::Microsecond)
    )
}

/// A two-operand interval operator.
#[derive(Debug)]
struct Binary {
    op: Builtin,
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
    out: ArrowType,
    nullable: bool,
    /// The session zone, when an operand is a `TIMESTAMP`.
    zone: Option<ZoneRef>,
    kernels: krishiv_kernels::KernelCtx,
}

/// `op(l, r)`; `out` is the result's Arrow type.
pub(super) fn binary(
    op: Builtin,
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
    out: ArrowType,
    nullable: bool,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let zone = match l.data_type() {
        ArrowType::Timestamp(_, Some(_)) => Some(env.zone_rules()?),
        _ => None,
    };
    Ok(Arc::new(Binary {
        op,
        l,
        r,
        out,
        nullable,
        zone,
        kernels: env.kernels,
    }))
}

/// The months of a year-month value as plain integers, as the date kernels take them.
fn months(v: &Value) -> Value {
    let a: ArrayRef = Arc::new(
        v.array()
            .as_primitive::<IntervalYearMonthType>()
            .reinterpret_cast::<Int32Type>(),
    );
    match v {
        Value::Array(_) => Value::Array(a),
        Value::Scalar(_) => Value::Scalar(a),
    }
}

/// The microseconds of a day-time value as plain integers.
fn micros(v: &Value) -> Value {
    let a: ArrayRef = Arc::new(
        v.array()
            .as_primitive::<DurationMicrosecondType>()
            .reinterpret_cast::<Int64Type>(),
    );
    match v {
        Value::Array(_) => Value::Array(a),
        Value::Scalar(_) => Value::Scalar(a),
    }
}

impl Binary {
    fn apply(&self, l: &Value, r: &Value) -> Result<ArrayRef> {
        let zone = self
            .zone
            .as_ref()
            .map(|z| z as &dyn krishiv_kernels::datetime::TimeZoneRules);
        let op = |b| {
            if b == Builtin::Add {
                AddOp::Add
            } else {
                AddOp::Subtract
            }
        };
        let scale_op = |b| {
            if b == Builtin::Multiply {
                ScaleOp::Multiply
            } else {
                ScaleOp::Divide
            }
        };
        Ok(match self.op {
            Builtin::DateAddYmInterval => {
                let r = months(r);
                Arc::new(date_add_months(
                    prim(l),
                    prim(&r),
                    false,
                    false,
                    &self.kernels,
                )?)
            }
            Builtin::TimestampAddYmInterval => {
                let r = months(r);
                Arc::new(
                    timestamp_add_months(prim(l), prim(&r), false, zone, false)?
                        .with_data_type(self.out.clone()),
                )
            }
            Builtin::TimestampAddInterval => {
                let r = micros(r);
                Arc::new(
                    timestamp_add_day_time(prim(l), prim(&r), false, zone, false)?
                        .with_data_type(self.out.clone()),
                )
            }
            Builtin::SubtractDates => Arc::new(subtract_dates(prim(l), prim(r))?),
            Builtin::SubtractTimestamps => Arc::new(subtract_timestamps(prim(l), prim(r), zone)?),
            b @ (Builtin::Add | Builtin::Subtract) => match self.out {
                ArrowType::Interval(_) => Arc::new(add_year_month(prim(l), prim(r), op(b), false)?),
                _ => Arc::new(add_day_time(prim(l), prim(r), op(b), false)?),
            },
            b @ (Builtin::Multiply | Builtin::Divide) => match self.out {
                ArrowType::Interval(_) => Arc::new(scale_year_month(
                    l.array().as_primitive(),
                    r.array().as_ref(),
                    scale_op(b),
                    false,
                )?),
                _ => Arc::new(scale_day_time(
                    l.array().as_primitive(),
                    r.array().as_ref(),
                    scale_op(b),
                    false,
                )?),
            },
            other => {
                return Err(internal(&format!(
                    "{} is not an interval operator",
                    other.name()
                )));
            }
        })
    }
}

impl PhysExpr for Binary {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn is_constant(&self) -> bool {
        self.l.is_constant() && self.r.is_constant()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let (mut l, r) = (
            self.l.eval(batch, partition)?,
            self.r.eval(batch, partition)?,
        );
        let scalar = l.is_scalar() && r.is_scalar();
        if scalar {
            l = Value::Array(Arc::clone(l.array()));
        }
        let out = self.apply(&l, &r)?;
        Ok(if scalar {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}

/// A one-operand interval operator.
#[derive(Debug)]
struct Unary {
    op: Builtin,
    e: Arc<dyn PhysExpr>,
    out: ArrowType,
}

/// `op(e)`: unary `-`, or `ExtractANSIIntervalDays`.
pub(super) fn unary(op: Builtin, e: Arc<dyn PhysExpr>) -> Arc<dyn PhysExpr> {
    let out = if op == Builtin::ExtractAnsiIntervalDays {
        ArrowType::Int32
    } else {
        e.data_type().clone()
    };
    Arc::new(Unary { op, e, out })
}

impl PhysExpr for Unary {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.e.nullable()
    }

    fn is_constant(&self) -> bool {
        self.e.is_constant()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.e.eval(batch, partition)?;
        let a = v.array();
        let out: ArrayRef = match (self.op, a.data_type()) {
            (Builtin::ExtractAnsiIntervalDays, ArrowType::Duration(_)) => {
                Arc::new(extract_days(a.as_primitive()))
            }
            (_, ArrowType::Interval(_)) => Arc::new(negate_year_month(a.as_primitive())?),
            (_, ArrowType::Duration(_)) => Arc::new(negate_day_time(a.as_primitive())?),
            (_, t) => return Err(internal(&format!("an interval operator on {t}"))),
        };
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}

/// Spark's `TryEval`: the child's values, NULL in a row where it fails. The
/// batch runs whole first; only a failure pays for going row by row.
#[derive(Debug)]
struct TryEval {
    e: Arc<dyn PhysExpr>,
}

pub(super) fn try_eval(e: Arc<dyn PhysExpr>) -> Arc<dyn PhysExpr> {
    Arc::new(TryEval { e })
}

impl PhysExpr for TryEval {
    fn data_type(&self) -> &ArrowType {
        self.e.data_type()
    }

    fn nullable(&self) -> bool {
        true
    }

    fn is_constant(&self) -> bool {
        self.e.is_constant()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        match self.e.eval(batch, partition) {
            Ok(v) => return Ok(v),
            Err(e) if !is_row_failure(&e) => return Err(e),
            Err(_) => {}
        }
        let null = || new_null_array(self.e.data_type(), 1);
        if self.e.is_constant() {
            return Ok(Value::Scalar(null()));
        }
        let mut rows: Vec<ArrayRef> = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            rows.push(match self.e.eval(&batch.slice(row, 1), partition) {
                Ok(v) => Arc::clone(v.array()),
                Err(e) if !is_row_failure(&e) => return Err(e),
                Err(_) => null(),
            });
        }
        let parts: Vec<&dyn Array> = rows.iter().map(AsRef::as_ref).collect();
        let out =
            arrow_select::concat::concat(&parts).map_err(|e| internal(&format!("TryEval: {e}")))?;
        Ok(Value::Array(out))
    }
}

/// Whether `e` is what a row's value raised, which `TryEval` turns into NULL,
/// and not a failure of the query: out of memory, cancelled, a bug, or
/// something Krishiv has not built.
fn is_row_failure(e: &KrishivError) -> bool {
    let class = e.error_class().name();
    !(class.starts_with("KRISHIV_")
        || class.starts_with("INTERNAL_ERROR")
        || class.starts_with("UNABLE_TO_ACQUIRE_MEMORY"))
}

#[cfg(test)]
mod tests {
    use arrow_array::Int32Array;
    use arrow_schema::{Field, Schema};
    use krishiv_common::error::{IntervalDividedByZero, KrishivUnsupportedFeature};

    use super::*;

    #[derive(Debug)]
    struct Fails(fn() -> KrishivError);

    impl PhysExpr for Fails {
        fn data_type(&self) -> &ArrowType {
            &ArrowType::Int32
        }

        fn nullable(&self) -> bool {
            false
        }

        fn is_constant(&self) -> bool {
            false
        }

        fn eval(&self, _: &RecordBatch, _: u32) -> Result<Value> {
            Err((self.0)())
        }
    }

    fn run(fail: fn() -> KrishivError) -> Result<Value> {
        let schema = Schema::new(vec![Field::new("a", ArrowType::Int32, false)]);
        let column: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let batch = RecordBatch::try_new(Arc::new(schema), vec![column])
            .map_err(|e| internal(&e.to_string()))?;
        try_eval(Arc::new(Fails(fail))).eval(&batch, 0)
    }

    #[test]
    fn try_eval_gives_null_for_a_row_and_raises_what_is_no_row_failure()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let nulls = run(|| IntervalDividedByZero {}.into())?;
        assert_eq!(nulls.array().null_count(), 2);
        let e = run(|| {
            KrishivUnsupportedFeature {
                feature: "x".to_owned(),
            }
            .into()
        })
        .err()
        .ok_or("raised nothing")?;
        assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
        let e = run(|| internal("a bug")).err().ok_or("raised nothing")?;
        assert_eq!(e.error_class().name(), "INTERNAL_ERROR");
        Ok(())
    }
}
