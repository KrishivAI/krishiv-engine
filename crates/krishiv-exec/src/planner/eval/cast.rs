// SPDX-License-Identifier: Apache-2.0
//! `CAST` / `TRY_CAST` through the kernels' cast.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::DataType as ArrowType;
use krishiv_kernels::cast::{CastMode as KernelMode, CastOptions, FixedOffset, cast};
use krishiv_plan::CastMode;
use krishiv_types::DataType;

use super::nested_cast;
use super::strings::{from_view, to_view};
use super::{Input, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};

#[derive(Debug)]
struct Cast {
    e: Arc<dyn PhysExpr>,
    parts: CastParts,
    nullable: bool,
}

/// Whether a cast between `from` and `to` reads the session time zone.
fn needs_zone(from: &ArrowType, to: &ArrowType) -> bool {
    let zoned = |t: &ArrowType| matches!(t, ArrowType::Timestamp(_, Some(_)));
    (zoned(from) && !matches!(to, ArrowType::Int64 | ArrowType::Timestamp(..)))
        || (zoned(to) && !matches!(from, ArrowType::Int64 | ArrowType::Timestamp(..)))
}

/// A cast's plan data, without its child: what [`NullIf`](super::cond::NullIf)
/// needs to cast an already-evaluated array.
#[derive(Debug, Clone)]
pub(super) struct CastParts {
    /// The result type.
    pub(super) to: ArrowType,
    /// The kernels' target: `Utf8` where the result is a `Utf8View` string.
    pub(super) kernel_to: ArrowType,
    pub(super) mode: KernelMode,
    pub(super) zone: FixedOffset,
}

/// The cast of `from` to `to` in `mode`, with the session zone of `env`.
pub(super) fn parts(
    from: &ArrowType,
    to: ArrowType,
    mode: CastMode,
    env: &super::EvalEnv,
) -> Result<CastParts> {
    let mode = match mode {
        CastMode::Ansi => KernelMode::Ansi,
        CastMode::Legacy => KernelMode::Legacy,
        CastMode::Try => KernelMode::Try,
        _ => return Err(internal("a cast mode the analyzer did not pin")),
    };
    let zone = match (needs_zone(from, &to), env.zone) {
        (_, Some(offset)) => FixedOffset(offset),
        (false, None) => FixedOffset(0),
        (true, None) => {
            return Err(unsupported(
                "casts that read a session time zone other than a fixed offset",
            ));
        }
    };
    let kernel_to = match &to {
        ArrowType::Utf8View => ArrowType::Utf8,
        ArrowType::BinaryView => ArrowType::Binary,
        other => other.clone(),
    };
    Ok(CastParts {
        to,
        kernel_to,
        mode,
        zone,
    })
}

impl CastParts {
    /// The cast of an element, value or field to `to`, in the same mode
    /// and zone.
    pub(super) fn child(&self, to: &ArrowType) -> Self {
        let kernel_to = match to {
            ArrowType::Utf8View => ArrowType::Utf8,
            ArrowType::BinaryView => ArrowType::Binary,
            other => other.clone(),
        };
        Self {
            to: to.clone(),
            kernel_to,
            mode: self.mode,
            zone: self.zone,
        }
    }
}

/// `a` through `p`.
pub(super) fn apply(p: &CastParts, a: &ArrayRef) -> Result<ArrayRef> {
    // Every value of a `void` array is NULL, of any type.
    if a.data_type() == &ArrowType::Null {
        return Ok(arrow_array::new_null_array(&p.to, a.len()));
    }
    if nested_cast::is_nested(a.data_type()) || nested_cast::is_nested(&p.to) {
        return nested_cast::cast(p, a);
    }
    let input = from_view(a)?;
    let opts = CastOptions {
        mode: p.mode,
        time_zone: &p.zone,
    };
    let out = cast(input.as_ref(), &p.kernel_to, &opts)?;
    let out: ArrayRef = match (out.data_type(), &p.to) {
        (a, b) if a == b => out,
        (ArrowType::Timestamp(..), ArrowType::Timestamp(..)) => {
            let data = out.to_data().into_builder().data_type(p.to.clone());
            arrow_array::make_array(data.build().map_err(|e| internal(&e.to_string()))?)
        }
        _ => to_view(&out)?,
    };
    if out.data_type() != &p.to {
        return Err(internal(&format!(
            "a cast to {} produced {}",
            p.to,
            out.data_type()
        )));
    }
    Ok(out)
}

pub(super) fn compile(
    e: Arc<dyn PhysExpr>,
    to: &DataType,
    mode: CastMode,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    let parts = parts(e.data_type(), to.to_arrow(), mode, input.env)?;
    // Every cast to a narrower or textual type can produce NULL in legacy
    // and `TRY_CAST` modes; ANSI keeps the input's nullability.
    let nullable = e.nullable() || parts.mode != KernelMode::Ansi;
    Ok(Arc::new(Cast { e, parts, nullable }))
}

impl PhysExpr for Cast {
    fn data_type(&self) -> &ArrowType {
        &self.parts.to
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let v = self.e.eval(batch)?;
        let out = apply(&self.parts, v.array())?;
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}
