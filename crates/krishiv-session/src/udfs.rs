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
use krishiv_udf_sdk::{ArgField, ScalarUdf, Signature, UdfInputType, UdfLibrary};

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
    /// built-in it lacks.
    fn udf_entry(&self, name: &str) -> FunctionLookup {
        let Some(udf) = self.udfs.get(name) else {
            return FunctionLookup::NotFound;
        };
        let Some((arity, inputs)) = signature_parts(udf.signature()) else {
            return FunctionLookup::KnownUnsupported;
        };
        let entry_udf = Arc::clone(&udf);
        let entry_name = name.to_owned();
        let signature_sql = udf.signature().inputs_sql();
        let result = Arc::new(move |call: &CallTypes<'_>| {
            let fields: Vec<ArgField> = call
                .args()
                .iter()
                .map(|a| ArgField {
                    data_type: a.data_type.clone(),
                    nullable: a.nullable,
                })
                .collect();
            entry_udf
                .return_field(&fields)
                .map(|r| ArgType {
                    data_type: Some(r.data_type),
                    nullable: r.nullable,
                })
                .map_err(|e| {
                    KrishivError::from(FailedExecuteUdf {
                        function_name: entry_name.clone(),
                        signature: signature_sql.clone(),
                        result: "unknown".to_owned(),
                        reason: e.message().to_owned(),
                    })
                })
        });
        FunctionLookup::Found(FunctionEntry {
            id: FunctionId::Udf {
                name: name.to_owned(),
                version: udf.version(),
            },
            kind: FunctionKind::Scalar,
            arity,
            inputs,
            result,
            deterministic: udf.properties().deterministic,
        })
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
        self.inner
            .spark_deterministic(name)
            .or_else(|| self.udfs.get(name).map(|u| u.properties().deterministic))
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

#[cfg(test)]
mod tests {
    use krishiv_plan::analyzer::BuiltinFunctions;
    use krishiv_types::DataType;

    use super::*;

    #[derive(Debug)]
    struct Doubles {
        version: u32,
    }

    impl ScalarUdf for Doubles {
        fn name(&self) -> &str {
            "doubles"
        }

        fn version(&self) -> u32 {
            self.version
        }

        fn signature(&self) -> &Signature {
            use std::sync::OnceLock;
            static SIG: OnceLock<Signature> = OnceLock::new();
            SIG.get_or_init(|| Signature::Coercible(vec![UdfInputType::Exact(DataType::Integer)]))
        }

        fn return_field(
            &self,
            _: &[ArgField],
        ) -> Result<krishiv_udf_sdk::ReturnField, krishiv_udf_sdk::UdfError> {
            Ok(krishiv_udf_sdk::ReturnField {
                data_type: DataType::Integer,
                nullable: true,
            })
        }

        fn invoke(
            &self,
            _: &[krishiv_udf_sdk::UdfArg<'_>],
            _: usize,
            _: &krishiv_udf_sdk::InvokeContext,
        ) -> Result<arrow_array::ArrayRef, krishiv_udf_sdk::UdfError> {
            Err(krishiv_udf_sdk::UdfError::new("test only"))
        }
    }

    fn functions() -> SessionFunctions {
        SessionFunctions::new(Arc::new(UdfLibrary::new()))
    }

    /// A UDF named like a built-in, for the precedence tests.
    #[derive(Debug)]
    struct UpperShadow;

    impl ScalarUdf for UpperShadow {
        fn name(&self) -> &str {
            "upper"
        }

        fn version(&self) -> u32 {
            1
        }

        fn signature(&self) -> &Signature {
            use std::sync::OnceLock;
            static SIG: OnceLock<Signature> = OnceLock::new();
            SIG.get_or_init(|| Signature::Coercible(vec![UdfInputType::Exact(DataType::String)]))
        }

        fn return_field(
            &self,
            _: &[ArgField],
        ) -> Result<krishiv_udf_sdk::ReturnField, krishiv_udf_sdk::UdfError> {
            Ok(krishiv_udf_sdk::ReturnField {
                data_type: DataType::String,
                nullable: true,
            })
        }

        fn invoke(
            &self,
            _: &[krishiv_udf_sdk::UdfArg<'_>],
            _: usize,
            _: &krishiv_udf_sdk::InvokeContext,
        ) -> Result<arrow_array::ArrayRef, krishiv_udf_sdk::UdfError> {
            Err(krishiv_udf_sdk::UdfError::new("test only"))
        }
    }

    #[test]
    fn bare_names_prefer_builtins() {
        let f = functions();
        // `upper` is a built-in: with no UDF registered, the built-in wins.
        let found = f.lookup(&["upper".to_owned()], SparkProfile::V4_2);
        assert!(matches!(found, FunctionLookup::Found(_)));
        // Registering `upper` does not shadow it.
        f.library().register(Arc::new(UpperShadow));
        let found = f.lookup(&["upper".to_owned()], SparkProfile::V4_2);
        assert!(matches!(
            found,
            FunctionLookup::Found(FunctionEntry {
                id: FunctionId::Registry { .. },
                ..
            })
        ));
        // But `system.session.upper` reaches the UDF.
        let found = f.lookup(
            &[
                "system".to_owned(),
                "session".to_owned(),
                "upper".to_owned(),
            ],
            SparkProfile::V4_2,
        );
        assert!(matches!(
            found,
            FunctionLookup::Found(FunctionEntry {
                id: FunctionId::Udf { .. },
                ..
            })
        ));
        // And `system.builtin.upper` reaches the built-in only.
        let found = f.lookup(
            &[
                "system".to_owned(),
                "builtin".to_owned(),
                "upper".to_owned(),
            ],
            SparkProfile::V4_2,
        );
        assert!(matches!(
            found,
            FunctionLookup::Found(FunctionEntry {
                id: FunctionId::Registry { .. },
                ..
            })
        ));
        // `session.upper` reaches the UDF only.
        let found = f.lookup(
            &["session".to_owned(), "upper".to_owned()],
            SparkProfile::V4_2,
        );
        assert!(matches!(
            found,
            FunctionLookup::Found(FunctionEntry {
                id: FunctionId::Udf { .. },
                ..
            })
        ));
    }

    #[test]
    fn resolutions_carry_name_and_version() {
        let f = functions();
        assert!(matches!(
            f.lookup(&["doubles".to_owned()], SparkProfile::V4_2),
            FunctionLookup::NotFound
        ));
        f.library().register(Arc::new(Doubles { version: 2 }));
        let FunctionLookup::Found(entry) = f.lookup(&["DOUBLES".to_owned()], SparkProfile::V4_2)
        else {
            panic!("a registered UDF resolves");
        };
        assert_eq!(
            entry.id,
            FunctionId::Udf {
                name: "doubles".to_owned(),
                version: 2,
            }
        );
        assert_eq!(entry.kind, FunctionKind::Scalar);
        assert!(entry.arity.accepts(1) && !entry.arity.accepts(2));
        assert!(entry.deterministic);
        assert!(f.resolve_udf("doubles", 2).is_some());
        assert!(f.resolve_udf("doubles", 1).is_none());
        // Re-registering replaces and bumps the registry version.
        let before = f.version();
        f.library().register(Arc::new(Doubles { version: 3 }));
        assert!(f.version() != before);
        let FunctionLookup::Found(entry) = f.lookup(&["doubles".to_owned()], SparkProfile::V4_2)
        else {
            panic!("a re-registered UDF resolves");
        };
        assert_eq!(
            entry.id,
            FunctionId::Udf {
                name: "doubles".to_owned(),
                version: 3,
            }
        );
        assert!(f.resolve_udf("doubles", 2).is_none());
    }

    #[test]
    fn unknown_and_qualified_builtins_resolve_as_before() {
        let f = functions();
        assert!(matches!(
            f.lookup(&["nosuchfn".to_owned()], SparkProfile::V4_2),
            FunctionLookup::NotFound
        ));
        // `system.builtin.nosuchfn` is not found either (built-ins only).
        assert!(matches!(
            f.lookup(
                &[
                    "system".to_owned(),
                    "builtin".to_owned(),
                    "nosuchfn".to_owned()
                ],
                SparkProfile::V4_2,
            ),
            FunctionLookup::NotFound
        ));
        // Anything else qualified is not found.
        assert!(matches!(
            f.lookup(&["mydb".to_owned(), "myfn".to_owned()], SparkProfile::V4_2,),
            FunctionLookup::NotFound
        ));
        assert!(matches!(
            f.lookup(
                &[
                    "a".to_owned(),
                    "b".to_owned(),
                    "c".to_owned(),
                    "d".to_owned()
                ],
                SparkProfile::V4_2
            ),
            FunctionLookup::NotFound
        ));
        // Spark's determinism table still answers.
        assert_eq!(f.spark_deterministic("upper"), Some(true));
        let plain = BuiltinFunctions::new();
        assert_eq!(f.version(), plain.version().wrapping_add(0));
    }
}
