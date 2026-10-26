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

/// Compiles the field `ordinal` (of Arrow type `data_type`) of `base`. A
/// plan is data (it may come from the plan codec), so the ordinal and the
/// declared type are checked against the struct here, never per batch.
pub(super) fn compile(
    base: Arc<dyn PhysExpr>,
    ordinal: usize,
    data_type: &ArrowType,
    nullable: bool,
) -> Result<Arc<dyn PhysExpr>> {
    let ArrowType::Struct(fields) = base.data_type() else {
        return Err(internal(&format!("a struct field of {}", base.data_type())));
    };
    match fields.get(ordinal) {
        Some(f) if f.data_type() == data_type => {}
        Some(f) => {
            return Err(internal(&format!(
                "struct field {ordinal} declared as {data_type} is {}",
                f.data_type()
            )));
        }
        None => {
            return Err(internal(&format!(
                "struct field {ordinal} of a struct with {} fields",
                fields.len()
            )));
        }
    }
    Ok(Arc::new(GetField {
        base,
        ordinal,
        data_type: data_type.clone(),
        nullable,
    }))
}

#[cfg(test)]
mod tests {
    use arrow_schema::{Field, Fields};

    use super::*;

    fn base() -> Arc<dyn PhysExpr> {
        let fields: Fields = vec![
            Field::new("a", ArrowType::Int32, true),
            Field::new("b", ArrowType::Utf8, true),
        ]
        .into();
        Arc::new(super::super::Column {
            index: 0,
            data_type: ArrowType::Struct(fields),
            nullable: true,
        })
    }

    /// A decoded plan's ordinal and declared type are checked when the
    /// expression compiles, not per batch.
    #[test]
    fn compile_checks_the_ordinal_and_the_declared_type() {
        assert!(compile(base(), 1, &ArrowType::Utf8, true).is_ok());
        let err = compile(base(), 2, &ArrowType::Utf8, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("struct field 2 of a struct with 2 fields"), "{err}");
        let err = compile(base(), 0, &ArrowType::Utf8, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("struct field 0 declared as Utf8 is Int32"), "{err}");
        let int: Arc<dyn PhysExpr> = Arc::new(super::super::Column {
            index: 0,
            data_type: ArrowType::Int32,
            nullable: true,
        });
        let err = compile(int, 0, &ArrowType::Int32, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("a struct field of Int32"), "{err}");
    }
}
