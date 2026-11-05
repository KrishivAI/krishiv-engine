// SPDX-License-Identifier: Apache-2.0
//! `CAST` into and out of `VARIANT` (Spark 4.2's `Cast` with a variant on
//! either side): out of a variant through the kernels' `VariantGet.cast`
//! (`INVALID_VARIANT_CAST` unless `TRY_CAST`, whatever ANSI says, as Spark
//! fails on error in every mode but `TRY`), into a variant through
//! `castToVariant`.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::DataType as ArrowType;
use krishiv_kernels::datetime::ZoneRef;
use krishiv_kernels::variant::{cast_from_variant, cast_to_variant};
use krishiv_plan::CastMode;
use krishiv_types::DataType;

use super::{EvalEnv, PhysExpr, Value};
use crate::error::Result;

/// Whether a cast from `from` (an Arrow type) to `to` involves a variant.
pub(super) fn is_variant_cast(from: &ArrowType, to: &DataType) -> bool {
    *to == DataType::Variant || krishiv_types::arrow::is_variant_struct(from)
}

#[derive(Debug)]
struct VariantCast {
    e: Arc<dyn PhysExpr>,
    /// The operand's Spark type.
    from: DataType,
    to: DataType,
    arrow: ArrowType,
    fail_on_error: bool,
    zone: ZoneRef,
}

/// `CAST(e AS to)` where `e` or `to` is a variant.
pub(super) fn compile(
    e: Arc<dyn PhysExpr>,
    to: &DataType,
    mode: CastMode,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let from = DataType::from_arrow(e.data_type())?;
    let zone = env.zone_rules()?;
    Ok(Arc::new(VariantCast {
        e,
        from,
        to: to.clone(),
        arrow: to.to_arrow(),
        fail_on_error: mode != CastMode::Try,
        zone,
    }))
}

impl PhysExpr for VariantCast {
    fn data_type(&self) -> &ArrowType {
        &self.arrow
    }

    fn nullable(&self) -> bool {
        // A variant null casts to SQL NULL.
        true
    }

    fn is_constant(&self) -> bool {
        self.e.is_constant()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.e.eval(batch, partition)?;
        let out = if self.from == DataType::Variant {
            cast_from_variant(v.array().as_ref(), &self.to, self.fail_on_error, &self.zone)?
        } else {
            cast_to_variant(v.array().as_ref(), &self.from)?
        };
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}
