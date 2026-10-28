// SPDX-License-Identifier: Apache-2.0
//! The execution binding of session UDFs: [`bind_udf`] turns a
//! [`ScalarUdf`](krishiv_udf_sdk::ScalarUdf) and its bound argument types
//! into a [`ScalarFunction`] the physical planner calls once per batch.
//!
//! The wrapper mirrors [`Bound`](super::Bound): it checks the
//! invocation against the binding (argument count and types), computes one
//! row and broadcasts it when every argument is a literal, and checks the
//! result's length, type and nullability. Where it differs is the contract
//! with user code: every user error, contract break or panic becomes
//! Spark's `FAILED_EXECUTE_UDF` (SQLSTATE 39000), and a
//! [`Strict`](krishiv_udf_sdk::NullHandling::Strict) UDF never sees a null
//! row (null inputs give null without invoking, as Spark's strict
//! functions do).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, UInt32Array, new_null_array};
use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::{FailedExecuteUdf, KrishivError};
use krishiv_types::DataType;
use krishiv_udf_sdk::guard_call;

use super::ScalarFunction;
use super::args::{Arg, broadcast, internal};

/// Binds the session UDF `udf` to `args` (the bound Arrow type of each
/// argument, in order): what the evaluator calls per batch.
///
/// `data_type` and `nullable` are the result the analyzer typed for the
/// same call (from [`ScalarUdf::return_field`](krishiv_udf_sdk::ScalarUdf::return_field)
/// over the same argument types); the wrapper checks every result against
/// them. `ansi` and `time_zone` reach the UDF in its
/// [`InvokeContext`](krishiv_udf_sdk::InvokeContext).
///
/// # Errors
/// `INTERNAL_ERROR` for an arity the UDF's signature does not take (the
/// analyzer checked it for the same version, so this is an engine bug).
pub fn bind_udf(
    udf: &Arc<dyn krishiv_udf_sdk::ScalarUdf>,
    args: &[ArrowType],
    data_type: &DataType,
    nullable: bool,
    ansi: bool,
    time_zone: Option<i32>,
) -> Result<Arc<dyn ScalarFunction>, KrishivError> {
    // Every trait call goes through `guard_call`: a panicking UDF becomes
    // `FAILED_EXECUTE_UDF`, never an unwinding engine.
    let name = guard_call("unknown", "name", || udf.name().to_owned())
        .map(|n| n.to_lowercase())
        .map_err(|e| failed_execute("unknown", "unknown", "unknown", e.message()))?;
    let version = guard_call(&name, "version", || udf.version())
        .map_err(|e| failed_execute(&name, "unknown", "unknown", e.message()))?;
    let signature = guard_call(&name, "signature", || udf.signature().clone())
        .map_err(|e| failed_execute(&name, "unknown", "unknown", e.message()))?;
    if !signature.accepts(args.len()) {
        return Err(internal(format!(
            "udf {name} bound to {} arguments",
            args.len()
        )));
    }
    let signature_sql = signature.inputs_sql();
    let strict = guard_call(&name, "properties", || udf.properties())
        .map(|p| matches!(p.null_handling, krishiv_udf_sdk::NullHandling::Strict))
        .map_err(|e| failed_execute(&name, &signature_sql, "unknown", e.message()))?;
    let _ = version;
    Ok(Arc::new(UdfBound {
        udf: Arc::clone(udf),
        name,
        version,
        signature_sql,
        args: args.to_vec(),
        return_type: plain_carrier(data_type),
        nullable,
        strict,
        ansi,
        time_zone,
    }))
}

/// `FAILED_EXECUTE_UDF` for `name` (`signature` => `result`) and `reason`.
pub fn failed_execute(name: &str, signature: &str, result: &str, reason: &str) -> KrishivError {
    KrishivError::from(FailedExecuteUdf {
        function_name: name.to_owned(),
        signature: signature.to_owned(),
        result: result.to_owned(),
        reason: reason.to_owned(),
    })
}

/// The plain Arrow carrier of `t`: what UDFs return (`Utf8` for `STRING`,
/// `Binary` for `BINARY`, the canonical type otherwise). The evaluator
/// converts strings and binaries to views itself, as for built-ins.
fn plain_carrier(t: &DataType) -> ArrowType {
    match t {
        DataType::String => ArrowType::Utf8,
        DataType::Binary => ArrowType::Binary,
        _ => t.to_arrow(),
    }
}

/// A bound session UDF over its argument types: what the evaluator calls
/// per batch.
struct UdfBound {
    udf: Arc<dyn krishiv_udf_sdk::ScalarUdf>,
    name: String,
    version: u32,
    signature_sql: String,
    args: Vec<ArrowType>,
    return_type: ArrowType,
    nullable: bool,
    strict: bool,
    ansi: bool,
    time_zone: Option<i32>,
}

impl std::fmt::Debug for UdfBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UdfFunction")
            .field("name", &self.name)
            .field("version", &self.version)
            .field("args", &self.args)
            .field("return_type", &self.return_type)
            .finish_non_exhaustive()
    }
}

impl ScalarFunction for UdfBound {
    fn name(&self) -> &str {
        &self.name
    }

    fn return_type(&self) -> &ArrowType {
        &self.return_type
    }

    fn invoke(&self, args: &[Arg], rows: usize) -> Result<ArrayRef, KrishivError> {
        if args.len() != self.args.len() {
            return Err(internal(format!(
                "{} bound to {} arguments called with {}",
                self.name,
                self.args.len(),
                args.len()
            )));
        }
        for (a, t) in args.iter().zip(&self.args) {
            a.check(t, rows, &self.name)?;
        }
        // No rows: nothing is computed, so nothing raises.
        if rows == 0 {
            return Ok(new_null_array(&self.return_type, 0));
        }
        let strict = self.strict.then(|| strict_mask(args));
        // A null in every row: null without invoking, so a strict UDF
        // never sees a null (a null literal is every row). Narrowed so the
        // match below needs no second `AllNull` arm.
        enum Narrow {
            None,
            Some(BooleanBuffer),
        }
        let strict: Option<Narrow> = match strict {
            Some(Strict::AllNull) => return Ok(new_null_array(&self.return_type, rows)),
            Some(Strict::None) => Some(Narrow::None),
            Some(Strict::Some(valid)) => Some(Narrow::Some(valid)),
            None => None,
        };
        // Every argument a literal: compute one row, then repeat it.
        if args.iter().all(|a| matches!(a, Arg::Scalar(_))) {
            let out = self.invoke_udf(args, 1)?;
            self.check(&out, 1)?;
            if rows == 1 {
                return Ok(out);
            }
            return broadcast(&out, rows).map_err(|e| internal(e.to_string()));
        }
        match strict {
            // No nulls (or not strict): invoke over the batch.
            Some(Narrow::None) | None => {
                let out = self.invoke_udf(args, rows)?;
                self.check(&out, rows)?;
                Ok(out)
            }
            // Some nulls: invoke over the non-null rows, scatter back.
            Some(Narrow::Some(valid)) => {
                let kept: Vec<usize> = valid
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| *v)
                    .map(|(i, _)| i)
                    .collect();
                let compacted: Vec<Arg> = args
                    .iter()
                    .map(|a| match a {
                        Arg::Scalar(one) => Ok(Arg::Scalar(Arc::clone(one))),
                        Arg::Array(column) => take(column, &kept).map(Arg::Array),
                    })
                    .collect::<Result<_, _>>()?;
                let compacted_udf: Vec<krishiv_udf_sdk::UdfArg<'_>> =
                    compacted.iter().map(arg_of).collect();
                let out = self.call(&compacted_udf, kept.len())?;
                self.check(&out, kept.len())?;
                scatter(&out, &valid)
            }
        }
    }
}

/// The null mask of `args` over `rows` rows: every row null, none, or some.
enum Strict {
    AllNull,
    None,
    Some(BooleanBuffer),
}

/// The union of the null masks of `args`: a scalar null is every row null.
/// Validity comes from `logical_nulls`, so a logical null (an untyped
/// `NULL` literal's `NullArray`, a dictionary or a run-end array) counts
/// as null too.
fn strict_mask(args: &[Arg]) -> Strict {
    let mut valid: Option<BooleanBuffer> = None;
    for a in args {
        match a {
            Arg::Scalar(one) if one.logical_nulls().is_some_and(|n| n.is_null(0)) => {
                return Strict::AllNull;
            }
            Arg::Scalar(_) => {}
            Arg::Array(column) => {
                let Some(nulls) = column.logical_nulls() else {
                    continue;
                };
                // A validity buffer over the whole batch.
                let column_valid = nulls.inner().clone();
                valid = Some(match valid {
                    None => column_valid,
                    Some(have) => &have & &column_valid,
                });
            }
        }
    }
    let Some(valid) = valid else {
        return Strict::None;
    };
    if valid.count_set_bits() == 0 {
        Strict::AllNull
    } else if valid.len() == valid.count_set_bits() {
        Strict::None
    } else {
        Strict::Some(valid)
    }
}

/// `a` as a UDF argument.
fn arg_of(a: &Arg) -> krishiv_udf_sdk::UdfArg<'_> {
    match a {
        Arg::Array(column) => krishiv_udf_sdk::UdfArg::Column(column),
        Arg::Scalar(one) => krishiv_udf_sdk::UdfArg::Scalar(one),
    }
}

impl UdfBound {
    /// Invokes the UDF over `args` (each a column of `rows` values or a
    /// one-row literal), catching panics. Every user error or panic is
    /// `FAILED_EXECUTE_UDF`.
    fn invoke_udf(&self, args: &[Arg], rows: usize) -> Result<ArrayRef, KrishivError> {
        let udf_args: Vec<krishiv_udf_sdk::UdfArg<'_>> = args.iter().map(arg_of).collect();
        self.call(&udf_args, rows)
    }

    /// Calls the UDF implementation through `guard_call`: every user error
    /// or panic is `FAILED_EXECUTE_UDF`.
    fn call(
        &self,
        args: &[krishiv_udf_sdk::UdfArg<'_>],
        rows: usize,
    ) -> Result<ArrayRef, KrishivError> {
        let ctx = krishiv_udf_sdk::InvokeContext {
            partition: 0,
            ansi: self.ansi,
            time_zone: self.time_zone,
        };
        let udf = Arc::clone(&self.udf);
        let name = self.name.clone();
        match guard_call(&name, "invoke", || udf.invoke(args, rows, &ctx)) {
            Ok(Ok(out)) => Ok(out),
            Ok(Err(e)) | Err(e) => Err(self.execute(e.message().to_owned())),
        }
    }

    /// Checks the UDF's result: `rows` values of the declared type, and no
    /// nulls when the call is declared non-nullable. A break is the UDF's
    /// fault, so `FAILED_EXECUTE_UDF`, never silent.
    fn check(&self, out: &ArrayRef, rows: usize) -> Result<(), KrishivError> {
        if out.len() != rows || out.data_type() != &self.return_type {
            return Err(self.execute(format!(
                "returned {} rows of {} for {rows} rows of {}",
                out.len(),
                out.data_type(),
                self.return_type
            )));
        }
        if !self.nullable && out.null_count() > 0 {
            return Err(self.execute(format!(
                "returned {} nulls but is declared non-nullable",
                out.null_count()
            )));
        }
        Ok(())
    }

    /// `FAILED_EXECUTE_UDF` for this call and `reason`.
    fn execute(&self, reason: String) -> KrishivError {
        FailedExecuteUdf {
            function_name: self.name.clone(),
            signature: self.signature_sql.clone(),
            result: self.udf_return_sql(),
            reason,
        }
        .into()
    }

    /// The declared result type as Spark prints it.
    fn udf_return_sql(&self) -> String {
        // The declared Arrow carrier inverts to the declared Spark type.
        DataType::from_arrow(&self.return_type)
            .map(|t| t.sql())
            .unwrap_or_else(|_| self.return_type.to_string())
    }
}

/// `column` at `kept`, in order.
fn take(column: &ArrayRef, kept: &[usize]) -> Result<ArrayRef, KrishivError> {
    let indices: UInt32Array = kept
        .iter()
        .map(|i| u32::try_from(*i).map_err(|_| internal("a batch too large to mask".to_owned())))
        .collect::<Result<_, _>>()?;
    arrow_select::take::take(column.as_ref(), &indices, None).map_err(|e| internal(e.to_string()))
}

/// `values` (one per set bit of `valid`, in order) spread over its rows,
/// NULL where `valid` is unset. The caller checks the row count first
/// ([`UdfBound::check`]), so a mismatch here is an engine bug.
fn scatter(values: &ArrayRef, valid: &BooleanBuffer) -> Result<ArrayRef, KrishivError> {
    debug_assert_eq!(
        values.len(),
        valid.count_set_bits(),
        "a UDF over masked rows returned the wrong row count"
    );
    let mut next = 0_u32;
    let indices: Vec<Option<u32>> = valid
        .iter()
        .map(|keep| {
            keep.then(|| {
                let at = next;
                next += 1;
                at
            })
        })
        .collect();
    debug_assert_eq!(
        usize::try_from(next).ok(),
        Some(values.len()),
        "a UDF over masked rows returned the wrong row count"
    );
    let indices = UInt32Array::from_iter(indices);
    arrow_select::take::take(values.as_ref(), &indices, None).map_err(|e| internal(e.to_string()))
}
