// SPDX-License-Identifier: Apache-2.0
//! The execution binding of session UDFs: debugging, arity and batches.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int32Array};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::ErrorClass;
use krishiv_types::DataType;
use krishiv_udf_sdk::{
    ArgField, InvokeContext, NullHandling, ReturnField, ScalarUdf, Signature, UdfArg, UdfError,
    UdfInputType, UdfProperties,
};

use super::super::{ScalarFunction, bind_udf};
use super::{class, col, ints, lit};

/// Copies its integer argument.
#[derive(Debug)]
struct Identity;

fn int_sig() -> Signature {
    Signature::Exact(vec![UdfInputType::Exact(DataType::Integer)])
}

impl ScalarUdf for Identity {
    fn name(&self) -> &str {
        "identity_i32"
    }

    fn version(&self) -> u32 {
        3
    }

    fn signature(&self) -> &Signature {
        use std::sync::OnceLock;
        static SIG: OnceLock<Signature> = OnceLock::new();
        SIG.get_or_init(int_sig)
    }

    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            nullable: true,
        })
    }

    fn properties(&self) -> UdfProperties {
        UdfProperties {
            deterministic: true,
            null_handling: NullHandling::Strict,
        }
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        rows: usize,
        _: &InvokeContext,
    ) -> Result<ArrayRef, UdfError> {
        let input = args[0].array();
        let input = input
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| UdfError::new("identity_i32 needs INT"))?;
        assert_eq!(
            input.len(),
            if matches!(args[0], UdfArg::Scalar(_)) {
                1
            } else {
                rows
            }
        );
        Ok(Arc::new(input.clone()))
    }
}

fn bind(nullable: bool) -> Arc<dyn ScalarFunction> {
    let udf: Arc<dyn ScalarUdf> = Arc::new(Identity);
    bind_udf(
        &udf,
        &[ArrowType::Int32],
        &DataType::Integer,
        nullable,
        true,
        Some(0),
    )
    .unwrap()
}

#[test]
fn bound_functions_debug_name_and_types() {
    let bound = bind(true);
    assert_eq!(bound.name(), "identity_i32");
    assert_eq!(bound.return_type(), &ArrowType::Int32);
    let text = format!("{bound:?}");
    assert!(text.contains("UdfFunction"), "{text}");
    assert!(text.contains("identity_i32"), "{text}");
}

#[test]
fn arity_mismatches_are_internal_and_empty_batches_invoke_nothing() {
    let udf: Arc<dyn ScalarUdf> = Arc::new(Identity);
    assert!(bind_udf(&udf, &[], &DataType::Integer, true, true, Some(0)).is_err());
    let bound = bind(true);
    let err = bound.invoke(&[], 1).unwrap_err();
    assert_eq!(class(&err), ErrorClass::InternalError);
    let two = ints(&[Some(1), Some(2)]);
    let err = bound.invoke(&[lit(&two), lit(&two)], 1).unwrap_err();
    assert_eq!(class(&err), ErrorClass::InternalError);
    let out = bound.invoke(&[col(&ints(&[]))], 0).unwrap();
    assert_eq!(out.len(), 0);
}
