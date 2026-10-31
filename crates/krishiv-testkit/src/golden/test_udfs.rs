// SPDX-License-Identifier: Apache-2.0
//! The test functions Spark's `SQLQueryTestSuite` registers for the `udf/` folders
//! (`IntegratedUDFTestUtils`): `udf(x)` is an identity that keeps the argument's type, which is
//! why the goldens print `udf(count(1)):bigint`. Nulls pass through.

use krishiv_session::{
    ArgField, InvokeContext, ReturnField, ScalarUdf, Session, Signature, UdfArg, UdfError,
    UdfInputType,
};
use krishiv_types::DataType;

use super::run::EngineError;

/// `udf(x)`: returns `x` as it is.
#[derive(Debug)]
struct IdentityUdf {
    signature: Signature,
}

impl IdentityUdf {
    fn new() -> Self {
        Self {
            signature: Signature::Exact(vec![UdfInputType::Any]),
        }
    }
}

impl ScalarUdf for IdentityUdf {
    fn name(&self) -> &str {
        "udf"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_field(&self, args: &[ArgField]) -> Result<ReturnField, UdfError> {
        let arg = args
            .first()
            .ok_or_else(|| UdfError::new("udf takes one argument"))?;
        Ok(ReturnField {
            data_type: arg.data_type.clone().unwrap_or(DataType::Null),
            nullable: arg.nullable,
        })
    }

    fn invoke(
        &self,
        args: &[UdfArg<'_>],
        _rows: usize,
        _ctx: &InvokeContext,
    ) -> Result<arrow_array::ArrayRef, UdfError> {
        args.first()
            .map(|a| a.array())
            .ok_or_else(|| UdfError::new("udf takes one argument"))
    }
}

/// Registers the test functions named in `names` (see `TestKind::udfs`). A name the harness has
/// no implementation for (`udaf`, which needs aggregate UDFs, and the PostgreSQL helpers) is
/// left unregistered, so statements calling it fail with the engine's own answer.
///
/// # Errors
/// When the session refuses a registration.
pub fn register(session: &Session, names: &[String]) -> Result<(), EngineError> {
    for name in names {
        if name == "udf" {
            session
                .register_udf(IdentityUdf::new())
                .map_err(|e| EngineError {
                    class: Some(e.error_class().name().to_owned()),
                    sql_state: e.sql_state().map(str::to_owned),
                    message: e.message().to_owned(),
                })?;
        }
    }
    Ok(())
}
