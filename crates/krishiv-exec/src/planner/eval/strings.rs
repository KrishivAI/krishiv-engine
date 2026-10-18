// SPDX-License-Identifier: Apache-2.0
//! String operators (`LIKE`, `||`) and the `Utf8View` ↔ `Utf8` boundary of
//! the string kernels.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, RecordBatch, StringArray,
    StringViewArray,
};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType as ArrowType;
use krishiv_kernels::string::{self, LikePattern, StrOperand};

use super::logic::str_value;
use super::{PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};

/// `a` with views converted to the kernels' offset arrays (`Utf8View` to
/// `Utf8`, `BinaryView` to `Binary`); other arrays as they are.
pub(super) fn from_view(a: &ArrayRef) -> Result<ArrayRef> {
    Ok(match a.data_type() {
        ArrowType::Utf8View => Arc::new(a.as_string_view().iter().collect::<StringArray>()),
        ArrowType::BinaryView => Arc::new(a.as_binary_view().iter().collect::<BinaryArray>()),
        _ => Arc::clone(a),
    })
}

/// `a` with offset arrays converted to views (Spark's carriers).
pub(super) fn to_view(a: &ArrayRef) -> Result<ArrayRef> {
    Ok(match a.data_type() {
        ArrowType::Utf8 => Arc::new(a.as_string::<i32>().iter().collect::<StringViewArray>()),
        ArrowType::LargeUtf8 => Arc::new(a.as_string::<i64>().iter().collect::<StringViewArray>()),
        ArrowType::Binary => Arc::new(a.as_binary::<i32>().iter().collect::<BinaryViewArray>()),
        ArrowType::Utf8View | ArrowType::BinaryView => Arc::clone(a),
        other => return Err(internal(&format!("{other} is not a string or binary"))),
    })
}

fn is_string(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8
    )
}

/// `s LIKE pattern [ESCAPE c]`, with a literal pattern compiled once.
#[derive(Debug)]
struct Like {
    s: Arc<dyn PhysExpr>,
    pattern: Arc<dyn PhysExpr>,
    escape: char,
}

pub(super) fn like(mut args: Vec<Arc<dyn PhysExpr>>) -> Result<Arc<dyn PhysExpr>> {
    let escape = if args.len() == 3 {
        let e = args.pop().ok_or_else(|| internal("LIKE arity"))?;
        let v = e.eval(&RecordBatch::new_empty(Arc::new(
            arrow_schema::Schema::empty(),
        )));
        match v {
            Ok(Value::Scalar(s)) if is_string(s.data_type()) && s.is_valid(0) => {
                let text = str_value(s.as_ref(), 0);
                let mut chars = text.chars();
                match (chars.next(), chars.next()) {
                    (Some(c), None) => c,
                    _ => return Err(unsupported("a LIKE escape that is not one character")),
                }
            }
            _ => return Err(unsupported("a LIKE escape that is not a literal")),
        }
    } else {
        '\\'
    };
    let pattern = args.pop().ok_or_else(|| internal("LIKE arity"))?;
    let s = args.pop().ok_or_else(|| internal("LIKE arity"))?;
    if !is_string(s.data_type()) || !is_string(pattern.data_type()) {
        return Err(unsupported("LIKE over non-string values"));
    }
    Ok(Arc::new(Like { s, pattern, escape }))
}

impl PhysExpr for Like {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.s.nullable() || self.pattern.nullable()
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let (s, p) = (self.s.eval(batch)?, self.pattern.eval(batch)?);
        let scalar = s.is_scalar() && p.is_scalar();
        let rows = if scalar { 1 } else { batch.num_rows() };
        let (sa, pa) = (s.array(), p.array());
        let at = |v: &Value, i: usize| if v.is_scalar() { 0 } else { i };
        let mut values = Vec::with_capacity(rows);
        let mut valid = Vec::with_capacity(rows);
        let mut compiled: Option<(usize, LikePattern)> = None;
        for i in 0..rows {
            let (si, pi) = (at(&s, i), at(&p, i));
            let ok = sa.is_valid(si) && pa.is_valid(pi);
            valid.push(ok);
            if !ok {
                values.push(false);
                continue;
            }
            // Recompile only when the pattern row changes (once for a literal).
            if compiled.as_ref().is_none_or(|(row, _)| *row != pi) {
                let pattern = str_value(pa.as_ref(), pi);
                compiled = Some((pi, LikePattern::compile(pattern, self.escape, i)?));
            }
            let m = compiled
                .as_ref()
                .is_some_and(|(_, c)| c.matches(str_value(sa.as_ref(), si)));
            values.push(m);
        }
        let out: ArrayRef = Arc::new(BooleanArray::new(
            BooleanBuffer::from(values),
            Some(NullBuffer::from(valid)),
        ));
        Ok(if scalar {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}

/// `concat(s, ...)` / `||`: NULL if any argument is NULL.
#[derive(Debug)]
struct Concat {
    args: Vec<Arc<dyn PhysExpr>>,
    nullable: bool,
}

pub(super) fn concat(args: Vec<Arc<dyn PhysExpr>>, nullable: bool) -> Result<Arc<dyn PhysExpr>> {
    if args.is_empty() || !args.iter().all(|a| is_string(a.data_type())) {
        return Err(unsupported("concat of non-string values"));
    }
    Ok(Arc::new(Concat { args, nullable }))
}

impl PhysExpr for Concat {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Utf8View
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let values = self
            .args
            .iter()
            .map(|a| a.eval(batch))
            .collect::<Result<Vec<_>>>()?;
        let scalar = values.iter().all(Value::is_scalar);
        let arrays = values
            .iter()
            .map(|v| from_view(v.array()))
            .collect::<Result<Vec<_>>>()?;
        let parts: Vec<StrOperand<'_, i32>> = values
            .iter()
            .zip(&arrays)
            .enumerate()
            .map(|(i, (v, a))| {
                let s = a.as_string::<i32>();
                // The kernel needs one column operand.
                if v.is_scalar() && !(scalar && i == 0) {
                    StrOperand::Scalar(s.is_valid(0).then(|| s.value(0)))
                } else {
                    StrOperand::Array(s)
                }
            })
            .collect();
        let out: ArrayRef = Arc::new(string::concat(&parts)?);
        let out = to_view(&out)?;
        Ok(if scalar {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}
