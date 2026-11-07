// SPDX-License-Identifier: Apache-2.0
//! `v BETWEEN lo AND hi`, Spark's `And(GreaterThanOrEqual(v, lo),
//! LessThanOrEqual(v, hi))` over one evaluation of `v`.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType as ArrowType;
use krishiv_kernels::KernelCtx;
use krishiv_kernels::compare::CompareKind;
use krishiv_plan::Builtin;

use super::cond::string_target;
use super::logic::{cmp, compare, kleene_bits};
use super::{EvalEnv, PhysExpr, Value};
use crate::error::Result;

/// `v BETWEEN lo AND hi` with `v` evaluated once: `v >= lo AND v <= hi`
/// under Kleene logic. Operands of one Spark `STRING`/`BINARY` in
/// different encodings meet in the canonical carrier; every other
/// mismatch fails here exactly as the two comparisons do. `never_raises`
/// says, for `lo` and then `hi`, that it raises on no row and reads no
/// state, so it may run on rows Spark does not evaluate it on.
pub(super) fn between(
    v: Arc<dyn PhysExpr>,
    lo: Arc<dyn PhysExpr>,
    hi: Arc<dyn PhysExpr>,
    never_raises: [bool; 2],
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let (mut v, mut lo, mut hi) = (v, lo, hi);
    if let Some(target) = string_target([v.data_type(), lo.data_type(), hi.data_type()]) {
        for e in [&mut v, &mut lo, &mut hi] {
            if e.data_type() != &target && *e.data_type() != ArrowType::Null {
                *e = super::cast::coerce_to(Arc::clone(e), &target, env)?;
            }
        }
    }
    compare(Builtin::GtEq, Arc::clone(&v), Arc::clone(&lo), env)?;
    compare(Builtin::LtEq, Arc::clone(&v), Arc::clone(&hi), env)?;
    Ok(Arc::new(Between {
        v,
        lo,
        hi,
        never_raises,
        kernels: env.kernels,
    }))
}

/// `BETWEEN` over one evaluation of `v`: both bounds reuse the evaluated
/// canonical `Value` instead of converting it once per bound. As in
/// Spark, a bound runs only on the rows that reach it: neither where `v`
/// is NULL (a comparison with a NULL left side does not evaluate its right
/// side), and `hi` not where `v >= lo` is false (the `AND`).
#[derive(Debug)]
struct Between {
    v: Arc<dyn PhysExpr>,
    lo: Arc<dyn PhysExpr>,
    hi: Arc<dyn PhysExpr>,
    /// `lo` and `hi` raise on no row and read no state.
    never_raises: [bool; 2],
    kernels: KernelCtx,
}

impl Between {
    /// Bound `e` over the rows `reached` (every row when `None`), NULL
    /// elsewhere; a bound that never raises runs over every row.
    fn bound(
        e: &dyn PhysExpr,
        never_raises: bool,
        reached: Option<&BooleanBuffer>,
        batch: &RecordBatch,
        partition: u32,
    ) -> Result<Value> {
        match reached {
            Some(r) if !never_raises && r.count_set_bits() < r.len() => {
                Ok(Value::Array(super::eval_where(e, batch, r, partition)?))
            }
            _ => e.eval(batch, partition),
        }
    }
}

impl PhysExpr for Between {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.v.nullable() || self.lo.nullable() || self.hi.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.v.eval(batch, partition)?;
        let rows = batch.num_rows();
        let v_valid = match &v {
            // A NULL `v` answers NULL without either bound.
            Value::Scalar(s) if s.logical_null_count() > 0 => {
                return Ok(Value::Scalar(Arc::new(BooleanArray::new_null(1))));
            }
            Value::Scalar(_) => None,
            Value::Array(a) => a.logical_nulls().map(|n| n.inner().clone()),
        };
        let lo = Self::bound(
            self.lo.as_ref(),
            self.never_raises[0],
            v_valid.as_ref(),
            batch,
            partition,
        )?;
        let ge = cmp(CompareKind::GreaterEq, &v, &lo, &self.kernels)?;
        // `hi` runs where `v >= lo` is true or NULL with `v` not NULL. `ge`
        // has one row only for a scalar `v` and `lo`, whose one row is open.
        let (g, g_ok) = kleene_bits(&ge, ge.len());
        let mut open = !&(&g_ok & &!&g);
        if let Some(valid) = &v_valid {
            open = &open & valid;
        }
        let hi = if open.count_set_bits() == 0 {
            None
        } else {
            let hi = Self::bound(
                self.hi.as_ref(),
                self.never_raises[1],
                Some(&open),
                batch,
                partition,
            )?;
            Some(hi)
        };
        let all_scalar =
            v.is_scalar() && lo.is_scalar() && hi.as_ref().is_none_or(Value::is_scalar);
        let n = if all_scalar { 1 } else { rows };
        let (g, g_ok) = kleene_bits(&ge, n);
        let (l, l_ok) = match &hi {
            Some(hi) => kleene_bits(&cmp(CompareKind::LessEq, &v, hi, &self.kernels)?, n),
            // Not evaluated: no row is decided by it.
            None => (BooleanBuffer::new_unset(n), BooleanBuffer::new_unset(n)),
        };
        // Kleene `AND`: a valid `false` on either side wins over NULL.
        let valid = &(&(&g_ok & &!&g) | &(&l_ok & &!&l)) | &(&g_ok & &l_ok);
        let out: ArrayRef = Arc::new(BooleanArray::new(&g & &l, Some(NullBuffer::new(valid))));
        Ok(if all_scalar {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}
