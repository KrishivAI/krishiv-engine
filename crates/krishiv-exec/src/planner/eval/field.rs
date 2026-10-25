// SPDX-License-Identifier: Apache-2.0
//! Struct field access: the analyzer-resolved field (`GetStructField`) of a
//! struct value, over the kernels' [`krishiv_kernels::r#struct::get_field`].

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::DataType as ArrowType;

use super::{PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::internal;

/// The field `ordinal` of `base` (of Arrow struct type).
#[derive(Debug)]
struct GetField {
    base: Arc<dyn PhysExpr>,
    ordinal: usize,
    data_type: ArrowType,
    nullable: bool,
}

impl PhysExpr for GetField {
    fn data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        Ok(match self.base.eval(batch)? {
            Value::Array(a) => Value::Array(get(&a, self.ordinal)?),
            Value::Scalar(s) => Value::Scalar(get(&s, self.ordinal)?),
        })
    }
}

/// The field `ordinal` of the struct array `base`.
fn get(base: &arrow_array::ArrayRef, ordinal: usize) -> Result<arrow_array::ArrayRef> {
    krishiv_kernels::r#struct::get_field(base.as_ref(), ordinal)
        .map_err(|e| internal(&e.to_string()))
}

/// Compiles the field `ordinal` (of Arrow type `data_type`) of `base`.
pub(super) fn compile(
    base: Arc<dyn PhysExpr>,
    ordinal: usize,
    data_type: &ArrowType,
    nullable: bool,
) -> Result<Arc<dyn PhysExpr>> {
    if !matches!(base.data_type(), ArrowType::Struct(_)) {
        return Err(internal(&format!("a struct field of {}", base.data_type())));
    }
    Ok(Arc::new(GetField {
        base,
        ordinal,
        data_type: data_type.clone(),
        nullable,
    }))
}
