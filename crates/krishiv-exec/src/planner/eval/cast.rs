// SPDX-License-Identifier: Apache-2.0
//! `CAST` / `TRY_CAST` through the kernels' cast.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::DataType as ArrowType;
use krishiv_kernels::cast::{CastMode as KernelMode, CastOptions, FixedOffset, cast};
use krishiv_plan::CastMode;
use krishiv_types::DataType;

use super::strings::{from_view, to_view};
use super::{Input, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};

#[derive(Debug)]
struct Cast {
    e: Arc<dyn PhysExpr>,
    to: ArrowType,
    /// The kernels' target: `Utf8` where the result is a `Utf8View` string.
    kernel_to: ArrowType,
    mode: KernelMode,
    zone: FixedOffset,
    nullable: bool,
}

/// Whether a cast between `from` and `to` reads the session time zone.
fn needs_zone(from: &ArrowType, to: &ArrowType) -> bool {
    let zoned = |t: &ArrowType| matches!(t, ArrowType::Timestamp(_, Some(_)));
    (zoned(from) && !matches!(to, ArrowType::Int64 | ArrowType::Timestamp(..)))
        || (zoned(to) && !matches!(from, ArrowType::Int64 | ArrowType::Timestamp(..)))
}

pub(super) fn compile(
    e: Arc<dyn PhysExpr>,
    to: &DataType,
    mode: CastMode,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    let mode = match mode {
        CastMode::Ansi => KernelMode::Ansi,
        CastMode::Legacy => KernelMode::Legacy,
        CastMode::Try => KernelMode::Try,
        _ => return Err(internal("a cast mode the analyzer did not pin")),
    };
    let target = to.to_arrow();
    let zone = match (needs_zone(e.data_type(), &target), input.env.zone) {
        (_, Some(offset)) => FixedOffset(offset),
        (false, None) => FixedOffset(0),
        (true, None) => {
            return Err(unsupported(
                "casts that read a session time zone other than a fixed offset",
            ));
        }
    };
    let kernel_to = match &target {
        ArrowType::Utf8View => ArrowType::Utf8,
        ArrowType::BinaryView => ArrowType::Binary,
        other => other.clone(),
    };
    // Every cast to a narrower or textual type can produce NULL in legacy
    // and `TRY_CAST` modes; ANSI keeps the input's nullability.
    let nullable = e.nullable() || mode != KernelMode::Ansi;
    Ok(Arc::new(Cast {
        e,
        to: target,
        kernel_to,
        mode,
        zone,
        nullable,
    }))
}

impl PhysExpr for Cast {
    fn data_type(&self) -> &ArrowType {
        &self.to
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let v = self.e.eval(batch)?;
        let input = from_view(v.array())?;
        let opts = CastOptions {
            mode: self.mode,
            time_zone: &self.zone,
        };
        let out = cast(input.as_ref(), &self.kernel_to, &opts)?;
        let out: ArrayRef = match (out.data_type(), &self.to) {
            (a, b) if a == b => out,
            (ArrowType::Timestamp(..), ArrowType::Timestamp(..)) => {
                let data = out.to_data().into_builder().data_type(self.to.clone());
                arrow_array::make_array(data.build().map_err(|e| internal(&e.to_string()))?)
            }
            _ => to_view(&out)?,
        };
        if out.data_type() != &self.to {
            return Err(internal(&format!(
                "a cast to {} produced {}",
                self.to,
                out.data_type()
            )));
        }
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}
