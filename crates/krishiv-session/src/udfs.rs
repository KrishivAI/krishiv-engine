// SPDX-License-Identifier: Apache-2.0
//! Session functions: Spark's built-ins plus the session's registered
//! native Rust UDFs.
//!
//! A [`SessionFunctions`] wraps the built-in [`BuiltinFunctions`] (or a
//! custom registry) and the session's [`UdfLibrary`]. Bare names resolve
//! built-ins first (Spark's search path: `system.builtin` before
//! `system.session`), so registering a name like `upper` does not shadow
//! the built-in; `system.builtin.f` reaches built-ins only, while
//! `system.session.f` and `session.f` reach UDFs only. Unknown names keep
//! today's `UNRESOLVED_ROUTINE`.

use std::sync::Arc;

use krishiv_common::error::{FailedExecuteUdf, KrishivError};
use krishiv_plan::analyzer::{
    ArgType, Arity, BuiltinFunctions, CallTypes, FunctionEntry, FunctionLookup, FunctionRegistry,
    InputType, Inputs, SparkProfile,
};
use krishiv_plan::{FunctionId, FunctionKind};
use krishiv_udf_sdk::{
    ArgField, NullHandling, ScalarUdf, Signature, UdfInputType, UdfLibrary, guard_call,
};

/// The function registry of a session: built-ins plus its UDFs.
#[derive(Clone)]
pub struct SessionFunctions {
    inner: Arc<dyn FunctionRegistry>,
    udfs: Arc<UdfLibrary>,
}

impl std::fmt::Debug for SessionFunctions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionFunctions")
            .field("udfs", &self.udfs)
            .finish_non_exhaustive()
    }
}

impl SessionFunctions {
    /// Built-ins plus `udfs`.
    #[must_use]
    pub fn new(udfs: Arc<UdfLibrary>) -> Self {
        Self {
            inner: Arc::new(BuiltinFunctions::new()),
            udfs,
        }
    }

    /// `inner` plus `udfs` (for sessions with a custom registry). Names
    /// arrive lower-cased, as Spark resolves function names
    /// case-insensitively.
    #[must_use]
    pub fn with_registry(inner: Arc<dyn FunctionRegistry>, udfs: Arc<UdfLibrary>) -> Self {
        Self { inner, udfs }
    }

    /// The session's UDF library.
    #[must_use]
    pub fn library(&self) -> &Arc<UdfLibrary> {
        &self.udfs
    }

    /// The [`FunctionEntry`] of the UDF named `name` (already lower-case),
    /// or [`FunctionLookup::NotFound`]. A signature form the engine does
    /// not implement is [`FunctionLookup::KnownUnsupported`], as for a
    /// built-in it lacks. Every UDF trait call goes through
    /// [`guard_call`], so a panicking `signature`, `properties`,
    /// `return_field` or `version` becomes `FAILED_EXECUTE_UDF` rather
    /// than unwinding the analyzer.
    fn udf_entry(&self, name: &str) -> FunctionLookup {
        let Some(udf) = self.udfs.get(name) else {
            return FunctionLookup::NotFound;
        };
        // `version` is part of the plan identity: a panic fails the call
        // rather than the analyzer.
        let version = match guard_call(name, "version", || udf.version()) {
            Ok(v) => v,
            Err(e) => return FunctionLookup::Found(failing_entry(name, 0, "unknown", e.message())),
        };
        let signature = match guard_call(name, "signature", || udf.signature().clone()) {
            Ok(s) => s,
            Err(e) => {
                return FunctionLookup::Found(failing_entry(name, version, "unknown", e.message()));
            }
        };
        let Some((arity, inputs)) = signature_parts(&signature) else {
            return FunctionLookup::KnownUnsupported;
        };
        let properties = match guard_call(name, "properties", || udf.properties()) {
            Ok(p) => p,
            Err(e) => {
                return FunctionLookup::Found(failing_entry(
                    name,
                    version,
                    &signature.inputs_sql(),
                    e.message(),
                ));
            }
        };
        let strict = matches!(properties.null_handling, NullHandling::Strict);
        let deterministic = properties.deterministic;
        let entry_udf = Arc::clone(&udf);
        let entry_name = name.to_owned();
        let signature_sql = signature.inputs_sql();
        let result = Arc::new(move |call: &CallTypes<'_>| {
            let fields: Vec<ArgField> = call
                .args()
                .iter()
                .map(|a| ArgField {
                    data_type: a.data_type.clone(),
                    nullable: a.nullable,
                })
                .collect();
            // A strict UDF never sees a null row (null inputs give null
            // without invoking), so its result is nullable whenever any
            // argument may be null, even when it declares non-nullable
            // results.
            let any_nullable = fields.iter().any(|f| f.nullable);
            let widen = strict && any_nullable;
            let entry_name = entry_name.clone();
            let signature_sql = signature_sql.clone();
            match guard_call(&entry_name, "return_field", || {
                entry_udf.return_field(&fields)
            }) {
                Ok(Ok(r)) => Ok(ArgType {
                    data_type: Some(r.data_type),
                    nullable: r.nullable || widen,
                }),
                Ok(Err(e)) | Err(e) => Err(KrishivError::from(FailedExecuteUdf {
                    function_name: entry_name,
                    signature: signature_sql,
                    result: "unknown".to_owned(),
                    reason: e.message().to_owned(),
                })),
            }
        });
        FunctionLookup::Found(FunctionEntry {
            id: FunctionId::Udf {
                name: name.to_owned(),
                version,
            },
            kind: FunctionKind::Scalar,
            arity,
            inputs,
            result,
            deterministic,
        })
    }
}

/// A UDF entry that always fails with `FAILED_EXECUTE_UDF`: what a UDF
/// whose `signature`, `properties` or `version` panics resolves to, so
/// the panic surfaces as an error at analysis rather than unwinding it.
fn failing_entry(name: &str, version: u32, signature: &str, reason: &str) -> FunctionEntry {
    let entry_name = name.to_owned();
    let signature_sql = signature.to_owned();
    let reason = reason.to_owned();
    let result = Arc::new(move |_: &CallTypes<'_>| {
        Err(KrishivError::from(FailedExecuteUdf {
            function_name: entry_name.clone(),
            signature: signature_sql.clone(),
            result: "unknown".to_owned(),
            reason: reason.clone(),
        }))
    });
    FunctionEntry {
        id: FunctionId::Udf {
            name: name.to_owned(),
            version,
        },
        kind: FunctionKind::Scalar,
        arity: Arity::any(),
        inputs: Inputs::Any,
        result,
        // Never folded, moved or duplicated: the error surfaces at execution.
        deterministic: false,
    }
}

/// Whether `a` and `b` name the same qualifier, ASCII case-insensitively
/// (function names are).
fn same(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

impl FunctionRegistry for SessionFunctions {
    fn lookup(&self, name: &[String], profile: SparkProfile) -> FunctionLookup {
        let lower: Vec<String> = name.iter().map(|p| p.to_lowercase()).collect();
        match lower.as_slice() {
            [f] => match self.inner.lookup(std::slice::from_ref(f), profile) {
                // Built-ins win over a same-name session UDF.
                FunctionLookup::Found(entry) => FunctionLookup::Found(entry),
                FunctionLookup::KnownUnsupported => FunctionLookup::KnownUnsupported,
                // Not found (or a future registry answer): a UDF may be.
                FunctionLookup::NotFound | _ => self.udf_entry(f),
            },
            // `session.f`: UDFs only.
            [q, f] if same(q, "session") => self.udf_entry(f),
            // `system.builtin.f`: built-ins only.
            [s, q, f] if same(s, "system") && same(q, "builtin") => {
                match self.inner.lookup(std::slice::from_ref(f), profile) {
                    FunctionLookup::Found(entry) => FunctionLookup::Found(entry),
                    other => other,
                }
            }
            // `system.session.f`: UDFs only.
            [s, q, f] if same(s, "system") && same(q, "session") => self.udf_entry(f),
            _ => FunctionLookup::NotFound,
        }
    }

    fn spark_deterministic(&self, name: &str) -> Option<bool> {
        if let Some(known) = self.inner.spark_deterministic(name) {
            return Some(known);
        }
        let udf = self.udfs.get(name)?;
        guard_call(name, "properties", || udf.properties())
            .ok()
            .map(|p| p.deterministic)
    }

    fn resolve_udf(&self, name: &str, version: u32) -> Option<Arc<dyn ScalarUdf>> {
        self.udfs.get_versioned(name, version)
    }

    fn version(&self) -> u64 {
        self.inner
            .version()
            .wrapping_add(self.udfs.version().wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }
}

/// The [`Arity`] and [`Inputs`] of `signature`: the analyzer's implicit
/// casts come from these, so a UDF casts like a built-in with the same
/// declared inputs. [`Signature::Coercible`] positions match strictly
/// (like [`InputType::Strict`]); the other forms map as written. `None`
/// for a signature form the engine does not implement yet.
fn signature_parts(signature: &Signature) -> Option<(Arity, Inputs)> {
    match signature {
        Signature::Exact(types) => {
            let mapped = types.iter().map(to_input).collect::<Option<Vec<_>>>()?;
            Some((
                exact_arity(types.len(), types.len().to_string()),
                Inputs::Expects(mapped),
            ))
        }
        Signature::Coercible(types) => {
            let mapped = types
                .iter()
                .map(|t| to_input(t).map(to_strict))
                .collect::<Option<Vec<_>>>()?;
            Some((
                exact_arity(types.len(), types.len().to_string()),
                Inputs::Expects(mapped),
            ))
        }
        Signature::Variadic { element, min_arity } => {
            let mut counts = [false; Arity::PROBED + 1];
            for (n, slot) in counts.iter_mut().enumerate() {
                *slot = n >= *min_arity;
            }
            Some((
                Arity::new(counts, signature.arity_text()),
                Inputs::Expects(vec![to_input(element)?]),
            ))
        }
        // A future signature form: not implemented here.
        _ => None,
    }
}

/// [`Arity`] accepting exactly `n` arguments (`WRONG_NUM_ARGS`
/// otherwise). Argument counts above [`Arity::PROBED`] are not
/// representable; such a signature never matches (see
/// [`Signature::accepts`]).
fn exact_arity(n: usize, expected: String) -> Arity {
    let mut counts = [false; Arity::PROBED + 1];
    if n <= Arity::PROBED {
        counts[n] = true;
    }
    Arity::new(counts, expected)
}

/// [`Signature::Exact`] positions of a [`Signature::Coercible`] match
/// strictly.
fn to_strict(input: InputType) -> InputType {
    match input {
        InputType::Exact(t) => InputType::Strict(t),
        other => other,
    }
}

/// A UDF input type as the analyzer's coercion sees it. `None` for a
/// form the engine does not implement yet.
fn to_input(t: &UdfInputType) -> Option<InputType> {
    Some(match t {
        UdfInputType::Any => InputType::Any,
        UdfInputType::Exact(t) => InputType::Exact(t.clone()),
        UdfInputType::Strict(t) => InputType::Strict(t.clone()),
        UdfInputType::Numeric { intervals } => InputType::Numeric {
            intervals: *intervals,
        },
        UdfInputType::OneOf(types) => InputType::OneOf(types.clone()),
        // A future input-type form: not implemented here.
        _ => return None,
    })
}
