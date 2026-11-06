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
use std::sync::Arc;

use krishiv_types::DataType;

use crate::{Expr, ExprNode, Literal};

/// A session variable: its declared or inferred type and its current
/// value, always of that type (assignment casts keep them in step).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionVariable {
    /// The declared or inferred type.
    pub data_type: DataType,
    /// The current value, of [`Self::data_type`].
    pub value: VariableValue,
}

/// A variable's value: a literal, or an `ARRAY`, `STRUCT` or `MAP` value held
/// as its unresolved constructor (`array(..)`, `named_struct(..)`, `map(..)`
/// over literals, cast to the declared type), which each reading query
/// resolves like any expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariableValue {
    /// A scalar value (or the NULL of any type).
    Scalar(Literal),
    /// A nested value as its constructor.
    Nested(Arc<Expr>),
}

impl From<Literal> for VariableValue {
    fn from(l: Literal) -> Self {
        Self::Scalar(l)
    }
}

impl VariableValue {
    /// The literal of a scalar value; `None` for a nested one.
    #[must_use]
    pub fn as_literal(&self) -> Option<&Literal> {
        match self {
            Self::Scalar(l) => Some(l),
            Self::Nested(_) => None,
        }
    }

    /// Spark's `Literal.sql` of the value (`ARRAY('a', 'b')`), as variable
    /// references show it in names.
    #[must_use]
    pub fn sql(&self) -> String {
        fn render(e: &Expr) -> String {
            match &e.node {
                // A typed NULL element prints with its type, as Spark's `Literal.sql`.
                ExprNode::Literal(Literal::Null(Some(t))) => format!("CAST(NULL AS {})", t.sql()),
                ExprNode::Literal(l) => l.to_string(),
                ExprNode::Cast { expr, .. } => render(expr),
                ExprNode::Function(f) => format!(
                    "{}({})",
                    f.name.to_uppercase(),
                    f.args
                        .iter()
                        .map(|a| render(a))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                _ => String::new(),
            }
        }
        match self {
            Self::Scalar(l) => l.to_string(),
            Self::Nested(e) => render(e),
        }
    }
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
    /// The local variables of the SQL script being run, outermost frame first; empty outside a
    /// script.
    pub locals: Vec<LocalFrame>,
}

/// The local variables of one `BEGIN ... END` body, or of one `FOR` loop's row.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LocalFrame {
    /// The lower-case label of the compound or loop, or the `FOR` variable; it qualifies the
    /// frame's variables (`label.v`).
    pub label: Option<String>,
    /// The frame's variables by lower-case name.
    pub variables: BTreeMap<String, SessionVariable>,
}

impl VariableSnapshot {
    /// The variable `parts` names, or `None` when the parts are not a variable's shape or name no
    /// variable. In any letter case:
    /// - `v`: the innermost local frame holding `v`, else the session variable;
    /// - `q.v`: `v` of the innermost local frame labelled `q`; with no such frame and `q` =
    ///   `session`, the session variable; a frame that lacks `v` ends the search;
    /// - `system.session.v`: the session variable.
    #[must_use]
    pub fn get(&self, parts: &[String]) -> Option<&SessionVariable> {
        let lower = |s: &str| s.to_lowercase();
        match parts {
            [v] => {
                let v = lower(v);
                self.locals
                    .iter()
                    .rev()
                    .find_map(|frame| frame.variables.get(&v))
                    .or_else(|| self.variables.get(&v))
            }
            [q, v] => {
                let (q, v) = (lower(q), lower(v));
                match self
                    .locals
                    .iter()
                    .rev()
                    .find(|frame| frame.label.as_deref() == Some(q.as_str()))
                {
                    Some(frame) => frame.variables.get(&v),
                    None if q == "session" => self.variables.get(&v),
                    None => None,
                }
            }
            _ => {
                let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
                let name = variable_name(&refs)?;
                self.variables.get(&name.to_lowercase())
            }
        }
    }
}

impl VariableSnapshot {
    /// A struct variable a prefix of `parts` names, with the prefix's length:
    /// `v.f`, `session.v.f` and `label.v.f` read the field `f` of the
    /// variable (the longest prefix wins). Only nested struct values have
    /// fields, so any other variable is never a prefix.
    #[must_use]
    pub fn get_struct_prefix(&self, parts: &[String]) -> Option<(&SessionVariable, usize)> {
        (1..parts.len()).rev().find_map(|k| {
            self.get(&parts[..k])
                .filter(|v| {
                    matches!(v.value, VariableValue::Nested(_))
                        && matches!(v.data_type, DataType::Struct(_))
                })
                .map(|v| (v, k))
        })
    }
}

/// The unqualified variable name `parts` names (`v`, `session.v` or
/// `system.session.v`, in any letter case), or `None` when the parts are
/// not a variable's shape. The single source of truth for that shape:
/// execution resolves names through this too.
#[must_use]
pub fn variable_name<'a>(parts: &[&'a str]) -> Option<&'a str> {
    match parts {
        [v] => Some(v),
        [s, v] if s.eq_ignore_ascii_case("session") => Some(v),
        [system, s, v]
            if system.eq_ignore_ascii_case("system") && s.eq_ignore_ascii_case("session") =>
        {
            Some(v)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(n: i32) -> SessionVariable {
        SessionVariable {
            data_type: DataType::Integer,
            value: Literal::Integer(n).into(),
        }
    }

    fn names(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    fn value(snapshot: &VariableSnapshot, parts: &[&str]) -> Option<i32> {
        match snapshot.get(&names(parts))?.value.as_literal()? {
            Literal::Integer(n) => Some(*n),
            _ => None,
        }
    }

    fn snapshot() -> VariableSnapshot {
        let frame = |label: Option<&str>, vars: &[(&str, i32)]| LocalFrame {
            label: label.map(str::to_owned),
            variables: vars
                .iter()
                .map(|(k, v)| ((*k).to_owned(), var(*v)))
                .collect(),
        };
        VariableSnapshot {
            variables: [("x".to_owned(), var(1)), ("sv".to_owned(), var(7))].into(),
            version: 3,
            locals: vec![
                frame(Some("outer"), &[("x", 2), ("y", 20)]),
                frame(Some("inner"), &[("x", 3)]),
                frame(None, &[("z", 30)]),
            ],
        }
    }

    #[test]
    fn a_bare_name_is_the_innermost_local_then_the_session_variable() {
        let s = snapshot();
        assert_eq!(value(&s, &["x"]), Some(3));
        assert_eq!(value(&s, &["X"]), Some(3));
        assert_eq!(value(&s, &["y"]), Some(20));
        assert_eq!(value(&s, &["z"]), Some(30));
        assert_eq!(value(&s, &["sv"]), Some(7));
        assert_eq!(value(&s, &["nope"]), None);
    }

    #[test]
    fn a_qualified_name_picks_the_frame_of_its_label() {
        let s = snapshot();
        assert_eq!(value(&s, &["outer", "x"]), Some(2));
        assert_eq!(value(&s, &["OUTER", "Y"]), Some(20));
        assert_eq!(value(&s, &["inner", "x"]), Some(3));
        // The frame that has the label but lacks the name ends the search.
        assert_eq!(value(&s, &["inner", "y"]), None);
        assert_eq!(value(&s, &["unknown", "x"]), None);
    }

    #[test]
    fn session_names_reach_the_session_variable_only() {
        let s = snapshot();
        assert_eq!(value(&s, &["session", "x"]), Some(1));
        assert_eq!(value(&s, &["system", "session", "x"]), Some(1));
        assert_eq!(value(&s, &["System", "Session", "sv"]), Some(7));
        assert_eq!(value(&s, &["a", "b", "c"]), None);
        // A frame labelled `session` is not allowed by the checks; if one existed it would win.
        let mut t = snapshot();
        t.locals.push(LocalFrame {
            label: Some("session".to_owned()),
            variables: [("x".to_owned(), var(99))].into(),
        });
        assert_eq!(value(&t, &["session", "x"]), Some(99));
        assert_eq!(value(&t, &["system", "session", "x"]), Some(1));
    }

    #[test]
    fn without_locals_the_snapshot_behaves_as_before() {
        let s = VariableSnapshot {
            variables: [("v".to_owned(), var(5))].into(),
            version: 1,
            locals: Vec::new(),
        };
        assert_eq!(value(&s, &["v"]), Some(5));
        assert_eq!(value(&s, &["session", "v"]), Some(5));
        assert_eq!(value(&s, &["x", "v"]), None);
    }
}
