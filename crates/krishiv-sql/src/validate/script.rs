// SPDX-License-Identifier: Apache-2.0
//! The structural checks Spark makes on a script before anything runs: labels, where declarations
//! and handlers may stand, conditions and handler conditions.
//!
//! One walk in source order; the first violation is the error. Names compare after case folding
//! (lower case unless `spark.sql.caseSensitive`). Parameter spellings follow Spark's
//! `SqlScriptingErrors`; the phase and the parameters are pinned by the Spark record in
//! `tests/spark-golden/v4.2.0/scripting`.

use krishiv_common::error::{
    DuplicateConditionInScope, DuplicateExceptionHandlerCondition,
    DuplicateExceptionHandlerSqlstate, EndLabelWithoutBeginLabel, ErrorClass,
    InvalidErrorConditionDeclarationNotAtStartOfCompoundStatement,
    InvalidHandlerDeclarationConditionNotFound,
    InvalidHandlerDeclarationDuplicateConditionInHandlerDeclaration,
    InvalidHandlerDeclarationDuplicateSqlstateInHandlerDeclaration,
    InvalidHandlerDeclarationInvalidConditionCombination,
    InvalidHandlerDeclarationWrongPlaceOfDeclaration, InvalidLabelUsageDoesNotExist,
    InvalidLabelUsageIterateInCompound, InvalidSqlstate,
    InvalidVariableDeclarationNotAllowedInScope, InvalidVariableDeclarationOnlyAtBeginning,
    KrishivError, LabelOrForVariableAlreadyExists, LabelOrForVariableNameForbidden, LabelsMismatch,
};

use crate::ParserOptions;
use crate::ast::{
    Compound, DeclareCondition, DeclareVariable, Handler, HandlerBody, HandlerCondition, Ident,
    ScriptStatement as S, Span,
};
use crate::error::ParseError;

type R = Result<(), ParseError>;

fn at(error: impl Into<KrishivError>, span: Span) -> ParseError {
    ParseError {
        error: error.into(),
        span,
    }
}

/// How a name is quoted in an error parameter.
fn quoted(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// What a label in scope names.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// A `BEGIN ... END` body.
    Compound,
    /// A loop, or the variable of a `FOR`.
    Loop,
}

struct Checker {
    case_sensitive: bool,
    /// Labels and `FOR` variables of the enclosing statements, innermost last.
    labels: Vec<(String, Scope)>,
    /// Conditions declared in the enclosing compounds, innermost last.
    conditions: Vec<Vec<String>>,
}

/// Checks the structure of the script `c`.
pub(super) fn check(c: &Compound, opts: ParserOptions) -> R {
    let mut checker = Checker {
        case_sensitive: opts.case_sensitive,
        labels: Vec::new(),
        conditions: Vec::new(),
    };
    checker.compound(c)
}

impl Checker {
    fn fold(&self, name: &str) -> String {
        if self.case_sensitive {
            name.to_owned()
        } else {
            name.to_lowercase()
        }
    }

    /// Opens the scope of a label or `FOR` variable.
    fn open(&mut self, label: &Ident, scope: Scope) -> R {
        let name = self.fold(&label.value);
        if name == "session" || name == "system" {
            return Err(at(
                LabelOrForVariableNameForbidden {
                    label: quoted(&label.value),
                },
                label.span,
            ));
        }
        if self.labels.iter().any(|(l, _)| *l == name) {
            return Err(at(
                LabelOrForVariableAlreadyExists {
                    label: quoted(&label.value),
                },
                label.span,
            ));
        }
        self.labels.push((name, scope));
        Ok(())
    }

    /// The end label of a statement that began with `label`.
    fn end_label(&self, label: Option<&Ident>, end: Option<&Ident>) -> R {
        match (label, end) {
            (None, Some(end)) => Err(at(
                EndLabelWithoutBeginLabel {
                    end_label: quoted(&end.value),
                },
                end.span,
            )),
            (Some(begin), Some(end)) if self.fold(&begin.value) != self.fold(&end.value) => {
                Err(at(
                    LabelsMismatch {
                        begin_label: quoted(&begin.value),
                        end_label: quoted(&end.value),
                    },
                    end.span,
                ))
            }
            _ => Ok(()),
        }
    }

    fn compound(&mut self, c: &Compound) -> R {
        self.end_label(c.label.as_ref(), c.end_label.as_ref())?;
        if let Some(label) = &c.label {
            self.open(label, Scope::Compound)?;
        }
        self.conditions.push(Vec::new());
        let result = self.compound_body(c);
        self.conditions.pop();
        if c.label.is_some() {
            self.labels.pop();
        }
        result
    }

    /// The items of a compound in order: declarations first, then handlers, then statements.
    fn compound_body(&mut self, c: &Compound) -> R {
        let mut seen_handler = false;
        let mut seen_statement = false;
        let mut handled: Vec<(bool, String)> = Vec::new();
        for item in &c.body {
            match item {
                S::Declare(d) => {
                    if seen_statement || seen_handler {
                        return Err(at(
                            InvalidVariableDeclarationOnlyAtBeginning {
                                var_name: variable_name(d),
                            },
                            d.span,
                        ));
                    }
                }
                S::Condition(cond) => {
                    if seen_statement || seen_handler {
                        return Err(at(
                            InvalidErrorConditionDeclarationNotAtStartOfCompoundStatement {
                                condition_name: quoted(&cond.name.value),
                            },
                            cond.span,
                        ));
                    }
                    self.condition(cond)?;
                }
                S::Handler(h) => {
                    if seen_statement {
                        return Err(at(
                            InvalidHandlerDeclarationWrongPlaceOfDeclaration {},
                            h.span,
                        ));
                    }
                    self.handler(h, &mut handled)?;
                    seen_handler = true;
                }
                other => {
                    seen_statement = true;
                    self.statement(other)?;
                }
            }
        }
        Ok(())
    }

    fn condition(&mut self, cond: &DeclareCondition) -> R {
        if let Some((state, span)) = &cond.sqlstate {
            check_sqlstate(state, *span)?;
        }
        let name = self.fold(&cond.name.value);
        let declared = self.conditions.last_mut();
        if let Some(declared) = declared {
            if declared.contains(&name) {
                return Err(at(
                    DuplicateConditionInScope {
                        condition: quoted(&cond.name.value),
                    },
                    cond.span,
                ));
            }
            declared.push(name);
        }
        Ok(())
    }

    fn handler(&mut self, h: &Handler, handled: &mut Vec<(bool, String)>) -> R {
        let mut names: Vec<(String, String)> = Vec::new();
        let mut states: Vec<String> = Vec::new();
        let mut broad = false;
        for condition in &h.conditions {
            match condition {
                HandlerCondition::SqlException(_) | HandlerCondition::NotFound(_) => broad = true,
                HandlerCondition::SqlState(state, span) => {
                    check_sqlstate(state, *span)?;
                    if states.contains(state) {
                        return Err(at(
                            InvalidHandlerDeclarationDuplicateSqlstateInHandlerDeclaration {
                                sql_state: state.clone(),
                            },
                            *span,
                        ));
                    }
                    states.push(state.clone());
                }
                HandlerCondition::Name(name) => {
                    let folded = self.fold(&name.value);
                    if names.iter().any(|(f, _)| *f == folded) {
                        return Err(at(
                            InvalidHandlerDeclarationDuplicateConditionInHandlerDeclaration {
                                condition: quoted(&name.value),
                            },
                            name.span,
                        ));
                    }
                    if !self.known_condition(&folded, &name.value) {
                        return Err(at(
                            InvalidHandlerDeclarationConditionNotFound {
                                condition: quoted(&name.value),
                            },
                            name.span,
                        ));
                    }
                    names.push((folded, name.value.clone()));
                }
            }
        }
        // `SQLEXCEPTION` and `NOT FOUND` may stand together, but not beside a name or a SQLSTATE.
        if broad && (!names.is_empty() || !states.is_empty()) {
            return Err(at(
                InvalidHandlerDeclarationInvalidConditionCombination {},
                h.span,
            ));
        }
        // Two handlers of one compound for the same condition or SQLSTATE.
        for (folded, written) in &names {
            if handled.contains(&(false, folded.clone())) {
                return Err(at(
                    DuplicateExceptionHandlerCondition {
                        condition: quoted(written),
                    },
                    h.span,
                ));
            }
            handled.push((false, folded.clone()));
        }
        for state in &states {
            if handled.contains(&(true, state.clone())) {
                return Err(at(
                    DuplicateExceptionHandlerSqlstate {
                        sql_state: state.clone(),
                    },
                    h.span,
                ));
            }
            handled.push((true, state.clone()));
        }
        match &h.body {
            HandlerBody::Compound(c) => self.compound(c),
            HandlerBody::Sql(_) | HandlerBody::Assign(_) => Ok(()),
        }
    }

    /// Whether `name` is a condition declared in an enclosing compound, or an error class (or the
    /// class before the first dot of one).
    fn known_condition(&self, folded: &str, written: &str) -> bool {
        if self
            .conditions
            .iter()
            .any(|c| c.iter().any(|d| d == folded))
        {
            return true;
        }
        let upper = written.to_uppercase();
        ErrorClass::ALL.iter().any(|class| {
            let n = class.name();
            n == upper || n.split('.').next() == Some(upper.as_str())
        })
    }

    fn statements(&mut self, items: &[crate::ast::ScriptStatement]) -> R {
        for item in items {
            // Declarations and handlers belong to the start of a `BEGIN ... END`.
            match item {
                S::Declare(d) => {
                    return Err(at(
                        InvalidVariableDeclarationNotAllowedInScope {
                            var_name: variable_name(d),
                        },
                        d.span,
                    ));
                }
                S::Condition(c) => {
                    return Err(at(
                        InvalidErrorConditionDeclarationNotAtStartOfCompoundStatement {
                            condition_name: quoted(&c.name.value),
                        },
                        c.span,
                    ));
                }
                S::Handler(h) => {
                    return Err(at(
                        InvalidHandlerDeclarationWrongPlaceOfDeclaration {},
                        h.span,
                    ));
                }
                other => self.statement(other)?,
            }
        }
        Ok(())
    }

    fn statement(&mut self, item: &crate::ast::ScriptStatement) -> R {
        match item {
            S::Sql(_)
            | S::Assign(_)
            | S::Cursor(_)
            | S::Declare(_)
            | S::Condition(_)
            | S::Handler(_) => Ok(()),
            S::Compound(c) => self.compound(c),
            S::If(i) => {
                for b in &i.branches {
                    self.statements(&b.body)?;
                }
                i.else_body
                    .as_deref()
                    .map_or(Ok(()), |e| self.statements(e))
            }
            S::Case(c) => {
                for b in &c.branches {
                    self.statements(&b.body)?;
                }
                c.else_body
                    .as_deref()
                    .map_or(Ok(()), |e| self.statements(e))
            }
            S::While(w) => self.looped(w.label.as_ref(), w.end_label.as_ref(), None, &w.body),
            S::Repeat(r) => self.looped(r.label.as_ref(), r.end_label.as_ref(), None, &r.body),
            S::Loop(l) => self.looped(l.label.as_ref(), l.end_label.as_ref(), None, &l.body),
            S::For(f) => self.looped(
                f.label.as_ref(),
                f.end_label.as_ref(),
                f.variable.as_ref(),
                &f.body,
            ),
            S::Leave(j) | S::Iterate(j) => {
                let leave = matches!(item, S::Leave(_));
                let name = self.fold(&j.label.value);
                match self.labels.iter().rev().find(|(l, _)| *l == name) {
                    None => Err(at(
                        InvalidLabelUsageDoesNotExist {
                            label_name: quoted(&j.label.value),
                            statement_type: if leave { "LEAVE" } else { "ITERATE" }.to_owned(),
                        },
                        j.span,
                    )),
                    Some((_, Scope::Compound)) if !leave => Err(at(
                        InvalidLabelUsageIterateInCompound {
                            label_name: quoted(&j.label.value),
                        },
                        j.span,
                    )),
                    Some(_) => Ok(()),
                }
            }
        }
    }

    /// A loop with its label and, for `FOR`, its variable.
    fn looped(
        &mut self,
        label: Option<&Ident>,
        end: Option<&Ident>,
        variable: Option<&Ident>,
        body: &[crate::ast::ScriptStatement],
    ) -> R {
        self.end_label(label, end)?;
        let mut opened = 0;
        let mut result = Ok(());
        for (name, scope) in [(label, Scope::Loop), (variable, Scope::Loop)] {
            if let Some(name) = name {
                match self.open(name, scope) {
                    Ok(()) => opened += 1,
                    Err(e) => {
                        result = Err(e);
                        break;
                    }
                }
            }
        }
        if result.is_ok() {
            result = self.statements(body);
        }
        for _ in 0..opened {
            self.labels.pop();
        }
        result
    }
}

/// The names of a declaration as an error parameter: the first one, as Spark names it.
fn variable_name(d: &DeclareVariable) -> String {
    d.names.first().map_or_else(String::new, |n| {
        n.parts
            .iter()
            .map(|p| quoted(&p.value))
            .collect::<Vec<_>>()
            .join(".")
    })
}

/// `INVALID_SQLSTATE` unless `state` is 5 letters or digits not starting `00`, `01` or `XX`.
fn check_sqlstate(state: &str, span: Span) -> R {
    // Spark accepts lower case ('abc12'); it refuses the success, warning and exception classes.
    let valid = state.len() == 5
        && state.bytes().all(|b| b.is_ascii_alphanumeric())
        && !["00", "01", "XX"].iter().any(|p| state.starts_with(p));
    if valid {
        Ok(())
    } else {
        Err(at(
            InvalidSqlstate {
                sql_state: state.to_owned(),
            },
            span,
        ))
    }
}
