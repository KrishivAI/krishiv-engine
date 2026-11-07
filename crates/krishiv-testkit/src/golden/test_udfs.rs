// SPDX-License-Identifier: Apache-2.0
//! The test functions Spark's `SQLQueryTestSuite` registers for the `udf/` folders
//! (`IntegratedUDFTestUtils`): `udf(x)` is an identity that keeps the argument's type, which is
//! why the goldens print `udf(count(1)):bigint`. Nulls pass through. `udaf(x)` (the Python
//! `pythonudaf`) counts the non-NULL values of its argument as a non-NULL `int`, and refuses
//! `DISTINCT`.

use krishiv_session::{
    AggregateProperties, AggregateUdf, ArgField, GroupsAccumulator, InvokeContext, ReturnField,
    ScalarUdf, Session, Signature, StateField, UdfArg, UdfError, UdfInputType,
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

/// `udaf(x)`: the number of non-NULL `x` of the group (the engine skips NULL rows, as for a
/// strict aggregate), `0` for none.
#[derive(Debug)]
struct CountUdaf {
    signature: Signature,
}

impl CountUdaf {
    fn new() -> Self {
        Self {
            signature: Signature::Exact(vec![UdfInputType::Any]),
        }
    }
}

impl AggregateUdf for CountUdaf {
    fn name(&self) -> &str {
        "udaf"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_field(&self, _args: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: DataType::Integer,
            nullable: false,
        })
    }

    fn state_fields(&self, _args: &[ArgField]) -> Result<Vec<StateField>, UdfError> {
        Ok(vec![StateField {
            name: "count".to_owned(),
            data_type: DataType::Long,
            nullable: false,
        }])
    }

    fn properties(&self) -> AggregateProperties {
        AggregateProperties::DEFAULT.with_distinct(false)
    }

    fn accumulator(
        &self,
        _args: &[ArgField],
        _ctx: &InvokeContext,
    ) -> Result<Box<dyn GroupsAccumulator>, UdfError> {
        Ok(Box::new(Counts(Vec::new())))
    }
}

/// [`CountUdaf`]'s per-group counts.
#[derive(Debug)]
struct Counts(Vec<i64>);

impl GroupsAccumulator for Counts {
    fn resize(&mut self, groups: usize) {
        self.0.resize(groups, 0);
    }

    fn update(&mut self, _args: &[UdfArg<'_>], groups: &[u32]) -> Result<(), UdfError> {
        for &g in groups {
            self.0[g as usize] += 1;
        }
        Ok(())
    }

    fn merge(&mut self, states: &[arrow_array::ArrayRef], groups: &[u32]) -> Result<(), UdfError> {
        use arrow_array::cast::AsArray;
        let counts = states
            .first()
            .and_then(|s| s.as_primitive_opt::<arrow_array::types::Int64Type>())
            .ok_or_else(|| UdfError::new("udaf state is one bigint column"))?;
        for (i, &g) in groups.iter().enumerate() {
            self.0[g as usize] += counts.value(i);
        }
        Ok(())
    }

    fn state(&self, groups: &[u32]) -> Result<Vec<arrow_array::ArrayRef>, UdfError> {
        Ok(vec![std::sync::Arc::new(
            arrow_array::Int64Array::from_iter_values(groups.iter().map(|&g| self.0[g as usize])),
        )])
    }

    fn evaluate(&self, groups: &[u32]) -> Result<arrow_array::ArrayRef, UdfError> {
        let values = groups
            .iter()
            .map(|&g| i32::try_from(self.0[g as usize]).map_err(|_| UdfError::new("udaf overflow")))
            .collect::<Result<Vec<i32>, _>>()?;
        Ok(std::sync::Arc::new(arrow_array::Int32Array::from(values)))
    }

    fn size(&self) -> usize {
        self.0.capacity() * size_of::<i64>()
    }
}

/// Registers the test functions named in `names` (see `TestKind::udfs`). A name the harness has
/// no implementation for (the PostgreSQL helpers) is left unregistered, so statements calling it
/// fail with the engine's own answer.
///
/// # Errors
/// When the session refuses a registration.
pub fn register(session: &Session, names: &[String]) -> Result<(), EngineError> {
    for name in names {
        let registered = match name.as_str() {
            "udf" => session.register_udf(IdentityUdf::new()),
            "udaf" => session.register_udaf(CountUdaf::new()),
            _ => continue,
        };
        registered.map_err(|e| EngineError {
            class: Some(e.error_class().name().to_owned()),
            sql_state: e.sql_state().map(str::to_owned),
            message: e.message().to_owned(),
        })?;
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

    #[tokio::test]
    async fn udaf_counts_non_null_values_per_group() {
        use arrow_array::cast::AsArray;
        let s = Session::new().unwrap();
        register(&s, &["udaf".to_owned()]).unwrap();
        let batches = s
            .sql(
                "SELECT a, udaf(b) AS c FROM VALUES (1, 1), (1, 2), (2, NULL), (NULL, 1) \
                 AS t(a, b) GROUP BY a ORDER BY a",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut got = Vec::new();
        for b in &batches {
            let a = b.column(0).as_primitive::<arrow_array::types::Int32Type>();
            let c = b.column(1).as_primitive::<arrow_array::types::Int32Type>();
            for i in 0..b.num_rows() {
                got.push(((!a.is_null(i)).then(|| a.value(i)), c.value(i)));
            }
        }
        assert_eq!(got, [(None, 1), (Some(1), 2), (Some(2), 0)]);
        let err = match s
            .sql("SELECT udaf(DISTINCT b) FROM VALUES (1) AS t(b)")
            .await
        {
            Ok(df) => df.collect().await.unwrap_err(),
            Err(e) => e,
        };
        assert_eq!(
            err.error_class().name(),
            "INVALID_SQL_SYNTAX.FUNCTION_WITH_UNSUPPORTED_SYNTAX"
        );
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
