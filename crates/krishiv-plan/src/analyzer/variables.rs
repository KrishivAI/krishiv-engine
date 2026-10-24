// SPDX-License-Identifier: Apache-2.0
//! Session variables: named, typed values a query reads through the
//! [`AnalysisContext`].
//!
//! A variable is declared with a type (or infers one from its default),
//! holds one value of that type, and resolves in expressions after
//! columns and outer references: `v`, `session.v` and `system.session.v`
//! all name the variable `v`. Anything else (a four-part name, a name
//! whose qualifier is not `session` / `system.session`) never names one,
//! so it stays an unresolved column.

use std::collections::BTreeMap;

use krishiv_types::DataType;

use crate::Literal;

/// A session variable: its declared or inferred type and its current
/// value, always of that type (assignment casts keep them in step).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionVariable {
    /// The declared or inferred type.
    pub data_type: DataType,
    /// The current value, of [`Self::data_type`].
    pub value: Literal,
}

/// The variables one query resolves, with the session's variable version
/// they were taken at. The version enters the context fingerprint, so a
/// resolved plan is never reused after a variable changed.
#[derive(Debug, Clone, Default)]
pub struct VariableSnapshot {
    /// The variables by lower-case name.
    pub variables: BTreeMap<String, SessionVariable>,
    /// The session's variable version when taken.
    pub version: u64,
}

impl VariableSnapshot {
    /// The variable `parts` names (`v`, `session.v` or
    /// `system.session.v`, in any letter case), or `None` when the parts
    /// are not a variable's shape or name no variable.
    #[must_use]
    pub fn get(&self, parts: &[String]) -> Option<&SessionVariable> {
        let name = match parts {
            [v] => v,
            [s, v] if s.eq_ignore_ascii_case("session") => v,
            [system, s, v]
                if system.eq_ignore_ascii_case("system") && s.eq_ignore_ascii_case("session") =>
            {
                v
            }
            _ => return None,
        };
        self.variables.get(&name.to_lowercase())
    }
}
