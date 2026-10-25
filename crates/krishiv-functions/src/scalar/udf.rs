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
    let name = match guard_call("unknown", "name", || udf.name().to_owned()) {
        Ok(n) => n.to_lowercase(),
        Err(e) => {
            return Err(failed_execute("unknown", "unknown", "unknown", e.message()));
        }
    };
    let version = match guard_call(&name, "version", || udf.version()) {
        Ok(v) => v,
        Err(e) => {
            return Err(failed_execute(&name, "unknown", "unknown", e.message()));
        }
    };
    let signature = match guard_call(&name, "signature", || udf.signature().clone()) {
        Ok(s) => s,
        Err(e) => {
            return Err(failed_execute(&name, "unknown", "unknown", e.message()));
        }
    };
    if !signature.accepts(args.len()) {
        return Err(internal(format!(
            "udf {name} bound to {} arguments",
            args.len()
        )));
    }
    let signature_sql = signature.inputs_sql();
    let strict = match guard_call(&name, "properties", || udf.properties()) {
        Ok(p) => matches!(p.null_handling, krishiv_udf_sdk::NullHandling::Strict),
        Err(e) => {
            return Err(failed_execute(
                &name,
                &signature_sql,
                "unknown",
                e.message(),
            ));
        }
    };
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
fn failed_execute(name: &str, signature: &str, result: &str, reason: &str) -> KrishivError {
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
        // never sees a null (a null literal is every row).
        if matches!(strict, Some(Strict::AllNull)) {
            return Ok(new_null_array(&self.return_type, rows));
        }
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
            Some(Strict::None) | None => {
                let out = self.invoke_udf(args, rows)?;
                self.check(&out, rows)?;
                Ok(out)
            }
            // Already returned above; kept for exhaustiveness.
            Some(Strict::AllNull) => Ok(new_null_array(&self.return_type, rows)),
            // Some nulls: invoke over the non-null rows, scatter back.
            Some(Strict::Some(valid)) => {
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
fn strict_mask(args: &[Arg]) -> Strict {
    let mut valid: Option<BooleanBuffer> = None;
    for a in args {
        match a {
            Arg::Scalar(one) if one.is_null(0) => return Strict::AllNull,
            Arg::Scalar(_) => {}
            Arg::Array(column) => {
                let Some(nulls) = column.nulls() else {
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
/// NULL where `valid` is unset.
fn scatter(values: &ArrayRef, valid: &BooleanBuffer) -> Result<ArrayRef, KrishivError> {
    if values.len() != valid.count_set_bits() {
        return Err(internal(
            "a UDF over masked rows returned the wrong row count".to_owned(),
        ));
    }
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
    if usize::try_from(next).ok() != Some(values.len()) {
        return Err(internal(
            "a UDF over masked rows returned the wrong row count".to_owned(),
        ));
    }
    let indices = UInt32Array::from_iter(indices);
    arrow_select::take::take(values.as_ref(), &indices, None).map_err(|e| internal(e.to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::PoisonError;

    use arrow_array::{Int32Array, StringArray};
    use krishiv_common::error::ErrorClass;
    use krishiv_types::DataType;
    use krishiv_udf_sdk::{
        ArgField, InvokeContext, NullHandling, ReturnField, ScalarUdf, Signature, UdfArg, UdfError,
        UdfInputType, UdfProperties,
    };

    use super::*;

    fn int_arg(values: Vec<Option<i32>>) -> ArrayRef {
        Arc::new(Int32Array::from(values))
    }

    fn lit_arg(value: i32) -> ArrayRef {
        Arc::new(Int32Array::from(vec![value]))
    }

    fn int_sig() -> Signature {
        Signature::Exact(vec![UdfInputType::Exact(DataType::Integer)])
    }

    #[derive(Debug, Default)]
    struct Calls {
        invocations: usize,
        rows: Vec<usize>,
    }

    /// Copies its integer argument, recording invocations.
    #[derive(Debug)]
    struct Identity {
        calls: Mutex<Calls>,
        signature: Signature,
        nullable: bool,
        strict: bool,
    }

    impl Identity {
        fn new(signature: Signature, nullable: bool, strict: bool) -> Self {
            Self {
                calls: Mutex::new(Calls::default()),
                signature,
                nullable,
                strict,
            }
        }
    }

    impl ScalarUdf for Identity {
        fn name(&self) -> &str {
            "identity_i32"
        }

        fn version(&self) -> u32 {
            3
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
            Ok(ReturnField {
                data_type: DataType::Integer,
                nullable: self.nullable,
            })
        }

        fn properties(&self) -> UdfProperties {
            UdfProperties {
                deterministic: true,
                null_handling: if self.strict {
                    NullHandling::Strict
                } else {
                    NullHandling::Custom
                },
            }
        }

        fn invoke(
            &self,
            args: &[UdfArg<'_>],
            rows: usize,
            _: &InvokeContext,
        ) -> Result<ArrayRef, UdfError> {
            let mut calls = self.calls.lock().unwrap_or_else(PoisonError::into_inner);
            calls.invocations += 1;
            calls.rows.push(rows);
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

    fn bind(udf: &Arc<dyn krishiv_udf_sdk::ScalarUdf>, nullable: bool) -> Arc<dyn ScalarFunction> {
        bind_udf(
            udf,
            &[ArrowType::Int32],
            &DataType::Integer,
            nullable,
            true,
            Some(0),
        )
        .unwrap()
    }

    #[test]
    fn columns_invoke_once_per_batch() {
        let udf: Arc<dyn krishiv_udf_sdk::ScalarUdf> =
            Arc::new(Identity::new(int_sig(), true, true));
        let bound = bind(&udf, true);
        assert_eq!(bound.name(), "identity_i32");
        assert_eq!(bound.return_type(), &ArrowType::Int32);
        let out = bound
            .invoke(&[Arg::Array(int_arg(vec![Some(1), None, Some(3)]))], 3)
            .unwrap();
        // Strict: the null row is null without invoking over it.
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!((out.value(0), out.is_null(1), out.value(2)), (1, true, 3));
    }

    #[test]
    fn all_null_strict_never_invokes() {
        let inner = Arc::new(Identity::new(int_sig(), true, true));
        let udf = Arc::clone(&inner) as Arc<dyn krishiv_udf_sdk::ScalarUdf>;
        let bound = bind(&udf, true);
        let out = bound
            .invoke(&[Arg::Array(int_arg(vec![None, None]))], 2)
            .unwrap();
        assert_eq!(out.null_count(), 2);
        // A null literal is every row null, also for all-literal calls.
        let out = bound
            .invoke(
                &[Arg::Scalar(Arc::new(new_null_array(
                    &arrow_schema::DataType::Int32,
                    1,
                )))],
                2,
            )
            .unwrap();
        assert_eq!(out.null_count(), 2);
        // Neither path invoked the UDF.
        assert_eq!(
            inner
                .calls
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .invocations,
            0
        );
    }

    #[test]
    fn all_scalar_broadcasts_one_row() {
        let inner = Arc::new(Identity::new(int_sig(), true, true));
        let udf = Arc::clone(&inner) as Arc<dyn krishiv_udf_sdk::ScalarUdf>;
        let bound = bind(&udf, true);
        let out = bound.invoke(&[Arg::Scalar(lit_arg(7))], 4).unwrap();
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(out.values(), &[7, 7, 7, 7]);
        let calls = inner.calls.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(
            (calls.invocations, calls.rows.as_slice()),
            (1, [1].as_slice())
        );
    }

    #[test]
    fn custom_nulls_reach_the_udf() {
        let udf: Arc<dyn krishiv_udf_sdk::ScalarUdf> =
            Arc::new(Identity::new(int_sig(), true, false));
        let bound = bind(&udf, true);
        let out = bound
            .invoke(&[Arg::Array(int_arg(vec![Some(1), None]))], 2)
            .unwrap();
        // The identity copies the null through itself.
        assert_eq!(out.null_count(), 1);
    }

    #[test]
    fn empty_batches_invoke_nothing() {
        let udf: Arc<dyn krishiv_udf_sdk::ScalarUdf> =
            Arc::new(Identity::new(int_sig(), true, true));
        let bound = bind(&udf, true);
        let out = bound.invoke(&[Arg::Array(int_arg(vec![]))], 0).unwrap();
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn arity_mismatches_are_internal() {
        let udf: Arc<dyn krishiv_udf_sdk::ScalarUdf> =
            Arc::new(Identity::new(int_sig(), true, true));
        assert!(bind_udf(&udf, &[], &DataType::Integer, true, true, Some(0)).is_err());
        let bound = bind(&udf, true);
        assert_eq!(
            bound.invoke(&[], 1).unwrap_err().error_class(),
            ErrorClass::InternalError
        );
        assert_eq!(
            bound
                .invoke(&[Arg::Scalar(lit_arg(1)), Arg::Scalar(lit_arg(2))], 1)
                .unwrap_err()
                .error_class(),
            ErrorClass::InternalError
        );
    }

    #[derive(Debug)]
    struct Broken {
        what: &'static str,
    }

    impl ScalarUdf for Broken {
        fn name(&self) -> &str {
            "broken"
        }

        fn version(&self) -> u32 {
            1
        }

        fn signature(&self) -> &Signature {
            use std::sync::OnceLock;
            static SIG: OnceLock<Signature> = OnceLock::new();
            SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(DataType::Integer)]))
        }

        fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
            Ok(ReturnField {
                data_type: DataType::Integer,
                nullable: false,
            })
        }

        fn invoke(
            &self,
            args: &[UdfArg<'_>],
            rows: usize,
            _: &InvokeContext,
        ) -> Result<ArrayRef, UdfError> {
            match self.what {
                "type" => Ok(Arc::new(StringArray::from(vec!["x"; rows]))),
                "length" => Ok(int_arg(vec![Some(1)])),
                "null" => Ok(int_arg(vec![None; rows])),
                "error" => Err(UdfError::new("nope")),
                "panic" => panic!("boom"),
                "message" => Err(UdfError::from("detailed reason")),
                _ => {
                    let input = args[0].array();
                    Ok(input)
                }
            }
        }
    }

    fn broken(what: &'static str) -> Arc<dyn ScalarFunction> {
        let udf: Arc<dyn krishiv_udf_sdk::ScalarUdf> = Arc::new(Broken { what });
        bind_udf(
            &udf,
            &[ArrowType::Int32],
            &DataType::Integer,
            false,
            true,
            Some(0),
        )
        .unwrap()
    }

    fn invoke(bound: &Arc<dyn ScalarFunction>, rows: usize) -> Result<ArrayRef, KrishivError> {
        bound.invoke(&[Arg::Array(int_arg(vec![Some(1); rows]))], rows)
    }

    #[test]
    fn contract_breaks_fail_as_failed_execute_udf() {
        for what in ["type", "length", "null"] {
            let err = invoke(&broken(what), 3).unwrap_err();
            assert_eq!(err.error_class(), ErrorClass::FailedExecuteUdf, "{what}");
            assert!(err.to_string().contains("broken"), "{what}: {err}");
        }
    }

    #[test]
    fn user_errors_and_panics_fail_as_failed_execute_udf() {
        for what in ["error", "panic"] {
            let err = invoke(&broken(what), 2).unwrap_err();
            assert_eq!(err.error_class(), ErrorClass::FailedExecuteUdf, "{what}");
            assert!(err.to_string().contains("broken"), "{what}: {err}");
        }
        let err = invoke(&broken("message"), 2).unwrap_err();
        assert!(err.to_string().contains("detailed reason"), "{err}");
        let err = invoke(&broken("panic"), 2).unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
    }

    #[test]
    fn failed_execute_udf_names_signature_and_result() {
        let udf: Arc<dyn krishiv_udf_sdk::ScalarUdf> = Arc::new(Broken { what: "error" });
        let bound = bind_udf(
            &udf,
            &[ArrowType::Int32],
            &DataType::Integer,
            true,
            true,
            Some(0),
        )
        .unwrap();
        let err = invoke(&bound, 1).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("broken"), "{text}");
        assert!(text.contains("\"INT\""), "{text}");
        assert!(text.contains("nope"), "{text}");
    }

    #[derive(Debug)]
    struct PanicMeta {
        what: &'static str,
    }

    impl ScalarUdf for PanicMeta {
        fn name(&self) -> &str {
            "panic_meta"
        }

        fn version(&self) -> u32 {
            1
        }

        fn signature(&self) -> &Signature {
            if self.what == "signature" {
                panic!("signature went bang");
            }
            use std::sync::OnceLock;
            static SIG: OnceLock<Signature> = OnceLock::new();
            SIG.get_or_init(|| Signature::Exact(vec![UdfInputType::Exact(DataType::Integer)]))
        }

        fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
            Ok(ReturnField {
                data_type: DataType::Integer,
                nullable: true,
            })
        }

        fn properties(&self) -> UdfProperties {
            if self.what == "properties" {
                panic!("properties went bang");
            }
            UdfProperties::default()
        }

        fn invoke(
            &self,
            args: &[UdfArg<'_>],
            _: usize,
            _: &InvokeContext,
        ) -> Result<ArrayRef, UdfError> {
            Ok(args[0].array())
        }
    }

    #[test]
    fn panicking_signature_and_properties_become_failed_execute_udf() {
        for what in ["signature", "properties"] {
            let udf: Arc<dyn krishiv_udf_sdk::ScalarUdf> = Arc::new(PanicMeta { what });
            let err = bind_udf(
                &udf,
                &[ArrowType::Int32],
                &DataType::Integer,
                true,
                true,
                Some(0),
            )
            .unwrap_err();
            assert_eq!(err.error_class(), ErrorClass::FailedExecuteUdf, "{what}");
            assert!(err.to_string().contains("went bang"), "{what}: {err}");
            assert!(err.to_string().contains("panic_meta"), "{what}: {err}");
        }
    }
}
