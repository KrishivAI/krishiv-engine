// SPDX-License-Identifier: Apache-2.0
//! The production [`FunctionRegistry`]: Spark's built-in functions from
//! `krishiv-functions`.

use std::sync::Arc;

use krishiv_functions::registry::{self, Lookup, Registry};

use super::{FunctionEntry, FunctionLookup, FunctionRegistry, SparkProfile};
use crate::FunctionId;

/// The built-in functions (`krishiv_functions::registry::builtins`).
#[derive(Debug, Clone, Copy)]
pub struct BuiltinFunctions(&'static Registry);

impl BuiltinFunctions {
    /// The process-wide built-ins.
    #[must_use]
    pub fn new() -> Self {
        Self(registry::builtins())
    }
}

impl Default for BuiltinFunctions {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionRegistry for BuiltinFunctions {
    fn lookup(&self, name: &[String], profile: SparkProfile) -> FunctionLookup {
        // Built-ins are unqualified (`builtin.f` / `system.builtin.f` come
        // with catalog functions).
        let [name] = name else {
            return FunctionLookup::NotFound;
        };
        match self.0.lookup(name, profile) {
            Lookup::Found(def) => FunctionLookup::Found(FunctionEntry {
                id: FunctionId::Registry {
                    name: def.name.to_owned(),
                    id: def.id,
                },
                kind: def.kind,
                arity: def.arity.clone(),
                inputs: def.inputs.clone(),
                result: Arc::clone(&def.result),
                deterministic: def.deterministic,
            }),
            Lookup::KnownUnsupported => FunctionLookup::KnownUnsupported,
            _ => FunctionLookup::NotFound,
        }
    }

    fn version(&self) -> u64 {
        self.0.version()
    }
}

/// One named parameter of a function Spark documents with named
/// parameters: its name and default.
pub(super) struct NamedParam {
    /// The parameter name, as Spark spells it.
    pub name: &'static str,
    /// The default, filled in when the call omits the parameter.
    pub default: NamedDefault,
}

/// The default of a named parameter.
#[derive(Clone, Copy)]
pub(super) enum NamedDefault {
    /// No default: the call must supply the parameter.
    Required,
    /// `NULL`, retaining the input (as `mask`'s `otherChar` does).
    Null,
    /// A string literal default (as `mask`'s replacements do).
    Text(&'static str),
}

/// The named-parameter signature of the functions Spark 4.2 documents with
/// named parameters that the engine accepts named arguments for: `mask`
/// (implemented) and `count_min_sketch` (known but unimplemented, so its
/// calls reorder and then fail as unsupported). Any other function rejects
/// named arguments with `NAMED_PARAMETERS_NOT_SUPPORTED`.
pub(super) fn named_signature(name: &str) -> Option<&'static [NamedParam]> {
    use NamedDefault::{Null, Required, Text};
    const MASK: &[NamedParam] = &[
        NamedParam {
            name: "str",
            default: Required,
        },
        NamedParam {
            name: "upperChar",
            default: Text("X"),
        },
        NamedParam {
            name: "lowerChar",
            default: Text("x"),
        },
        NamedParam {
            name: "digitChar",
            default: Text("n"),
        },
        NamedParam {
            name: "otherChar",
            default: Null,
        },
    ];
    const COUNT_MIN_SKETCH: &[NamedParam] = &[
        NamedParam {
            name: "column",
            default: Required,
        },
        NamedParam {
            name: "epsilon",
            default: Required,
        },
        NamedParam {
            name: "confidence",
            default: Required,
        },
        NamedParam {
            name: "seed",
            default: Required,
        },
    ];
    match name.to_ascii_lowercase().as_str() {
        "mask" => Some(MASK),
        "count_min_sketch" => Some(COUNT_MIN_SKETCH),
        _ => None,
    }
}
