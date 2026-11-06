// SPDX-License-Identifier: Apache-2.0
//! `CAST` into and out of `VARIANT` (Spark 4.2's `Cast` with a variant on
//! either side, at any depth): out of a variant through the kernels'
//! `VariantGet.cast` (`INVALID_VARIANT_CAST` unless `TRY_CAST`, whatever
//! ANSI says, as Spark fails on error in every mode but `TRY`), into a
//! variant through `castToVariant`. A variant inside an array, struct or map
//! (`CAST(array(v) AS ARRAY<INT>)`) is cast element by element; the parts
//! without a variant go through the ordinary cast.

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::{Array, ArrayRef, ListArray, MapArray, RecordBatch, StructArray};
use arrow_schema::DataType as ArrowType;
use krishiv_kernels::datetime::ZoneRef;
use krishiv_kernels::variant::{cast_from_variant, cast_to_variant};
use krishiv_plan::CastMode;
use krishiv_types::DataType;

use super::{EvalEnv, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::internal;

fn has_variant(t: &DataType) -> bool {
    t.any_type(&|x| *x == DataType::Variant)
}

/// Whether a cast from `from` (an Arrow type) to `to` involves a variant.
pub(super) fn is_variant_cast(from: &ArrowType, to: &DataType) -> bool {
    has_variant(to) || DataType::from_arrow(from).is_ok_and(|t| has_variant(&t))
}

#[derive(Debug)]
struct VariantCast {
    e: Arc<dyn PhysExpr>,
    /// The operand's Spark type.
    from: DataType,
    to: DataType,
    arrow: ArrowType,
    mode: CastMode,
    env: EvalEnv,
    zone: ZoneRef,
}

/// `CAST(e AS to)` where `e` or `to` holds a variant.
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
        mode,
        env: env.clone(),
        zone,
    }))
}

impl VariantCast {
    /// `a` (of `from`) as `to`.
    fn cast(&self, a: &ArrayRef, from: &DataType, to: &DataType) -> Result<ArrayRef> {
        if from == to {
            return Ok(Arc::clone(a));
        }
        if *from == DataType::Variant {
            return cast_from_variant(a.as_ref(), to, self.mode != CastMode::Try, &self.zone);
        }
        if *to == DataType::Variant {
            return cast_to_variant(a.as_ref(), from);
        }
        if !has_variant(from) && !has_variant(to) {
            let parts = super::cast::parts(a.data_type(), to.to_arrow(), self.mode, &self.env)?;
            return super::cast::apply(&parts, a);
        }
        let shape = || internal(&format!("a cast of {} to {}", from.sql(), to.sql()));
        Ok(match (from, to) {
            (DataType::Array(f), DataType::Array(t)) => {
                let l = a.as_list_opt::<i32>().ok_or_else(shape)?;
                let ArrowType::List(field) = to.to_arrow() else {
                    return Err(shape());
                };
                let values = self.cast(l.values(), f.element_type(), t.element_type())?;
                Arc::new(
                    ListArray::try_new(field, l.offsets().clone(), values, l.nulls().cloned())
                        .map_err(|e| internal(&e.to_string()))?,
                )
            }
            (DataType::Struct(f), DataType::Struct(t)) if f.fields().len() == t.fields().len() => {
                let s = a.as_struct_opt().ok_or_else(shape)?;
                let ArrowType::Struct(fields) = to.to_arrow() else {
                    return Err(shape());
                };
                let children = s
                    .columns()
                    .iter()
                    .zip(f.fields().iter().zip(t.fields()))
                    .map(|(c, (x, y))| self.cast(c, x.data_type(), y.data_type()))
                    .collect::<Result<Vec<_>>>()?;
                Arc::new(
                    StructArray::try_new_with_length(fields, children, s.nulls().cloned(), s.len())
                        .map_err(|e| internal(&e.to_string()))?,
                )
            }
            (DataType::Map(f), DataType::Map(t)) => {
                let m = a.as_map_opt().ok_or_else(shape)?;
                let ArrowType::Map(entries, sorted) = to.to_arrow() else {
                    return Err(shape());
                };
                let ArrowType::Struct(kv) = entries.data_type() else {
                    return Err(shape());
                };
                let keys = self.cast(m.keys(), f.key_type(), t.key_type())?;
                let values = self.cast(m.values(), f.value_type(), t.value_type())?;
                let entry = StructArray::try_new(kv.clone(), vec![keys, values], None)
                    .map_err(|e| internal(&e.to_string()))?;
                Arc::new(
                    MapArray::try_new(
                        Arc::clone(&entries),
                        m.offsets().clone(),
                        entry,
                        m.nulls().cloned(),
                        sorted,
                    )
                    .map_err(|e| internal(&e.to_string()))?,
                )
            }
            _ => return Err(shape()),
        })
    }
}

impl PhysExpr for VariantCast {
    fn data_type(&self) -> &ArrowType {
        &self.arrow
    }

    fn nullable(&self) -> bool {
        // A variant null casts to SQL NULL; a value cast into a variant is
        // NULL only where it was.
        has_variant(&self.from) || self.e.nullable()
    }

    fn is_constant(&self) -> bool {
        self.e.is_constant()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.e.eval(batch, partition)?;
        let out = self.cast(v.array(), &self.from, &self.to)?;
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}
