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
        let array = args
            .first()
            .map(|a| a.array())
            .ok_or_else(|| UdfError::new("udf takes one argument"))?;
        Ok(plain_carrier(array))
    }
}

/// The session binds a STRING or BINARY result as `Utf8` or `Binary`, but hands a UDF the view
/// carriers; convert those two and return every other array as it is.
fn plain_carrier(array: arrow_array::ArrayRef) -> arrow_array::ArrayRef {
    use arrow_array::{Array, BinaryArray, StringArray};
    use std::sync::Arc;
    if let Some(v) = array
        .as_any()
        .downcast_ref::<arrow_array::StringViewArray>()
    {
        Arc::new(v.iter().collect::<StringArray>())
    } else if let Some(v) = array
        .as_any()
        .downcast_ref::<arrow_array::BinaryViewArray>()
    {
        Arc::new(v.iter().collect::<BinaryArray>())
    } else {
        array
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;

    #[tokio::test]
    async fn identity_keeps_string_and_binary_values() {
        let s = Session::new().unwrap();
        register(&s, &["udf".to_owned()]).unwrap();
        let batches = s
            .sql(
                "SELECT udf(s) AS s, udf(CAST(s AS BINARY)) AS b, udf(n) AS n \
                 FROM VALUES ('x', 1), (NULL, NULL) AS t(s, n)",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
        assert_eq!(rows, 2);
        let text = cells(&batches);
        assert_eq!(text, ["x|78|1", "null|null|null"]);
    }

    /// Row cells as text (`|`-joined), binary as hex.
    fn cells(batches: &[arrow_array::RecordBatch]) -> Vec<String> {
        use arrow_array::cast::AsArray;
        let mut out = Vec::new();
        for b in batches {
            let (s, bin, n) = (
                b.column(0).as_string_view(),
                b.column(1).as_binary_view(),
                b.column(2).as_primitive::<arrow_array::types::Int32Type>(),
            );
            for i in 0..b.num_rows() {
                let c0 = (!s.is_null(i)).then(|| s.value(i).to_owned());
                let c1 = (!bin.is_null(i)).then(|| {
                    bin.value(i)
                        .iter()
                        .map(|x| format!("{x:02x}"))
                        .collect::<String>()
                });
                let c2 = (!n.is_null(i)).then(|| n.value(i).to_string());
                let f = |c: Option<String>| c.unwrap_or_else(|| "null".to_owned());
                out.push(format!("{}|{}|{}", f(c0), f(c1), f(c2)));
            }
        }
        out
    }
}
