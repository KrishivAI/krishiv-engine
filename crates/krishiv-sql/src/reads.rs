// SPDX-License-Identifier: Apache-2.0
//! The reads of a statement (P3-T063 N3, design `P3-T063-analyze-api.md` §3.2): every query a
//! statement runs internally, whole `SELECT`s or not, each analyzable on its own, so that its
//! securables are the statement's reads.
//!
//! [`statement_reads`] extends [`crate::statement_queries`] with the shapes whose reads are local to
//! the statement (they follow from its own text, with no state of the session):
//! - a multi-table `INSERT` with a shared `FROM`: each clause reads as `FROM source SELECT ...`
//!   ([`shared_from_query`], the form the session runs);
//! - `UPDATE`: each assigned column (`SELECT c FROM t`), and its new values and `WHERE` together
//!   (`SELECT values FROM t [AS alias] [WHERE ...]`), the queries the session analyzes before it finds
//!   that the table cannot change, subqueries included;
//! - `DELETE`: `SELECT TRUE FROM t [AS alias] [WHERE ...]` (without a `WHERE` too: a run asks `Select`
//!   on the table).
//!
//! What stays `None` (the statement is `partial`), and why:
//! - `MERGE` whose `ON`, a clause condition, a `SET` value or an inserted value holds a subquery
//!   (fail closed, review m1). Spark 4.2.0 accepts such subqueries at analysis (`MergeIntoTable` is
//!   `SupportsSubquery`, `v2Commands.scala:1081-1092`) and refuses the conditions' ones only when it
//!   rewrites the MERGE for a row-level-capable table (`UNSUPPORTED_MERGE_CONDITION.SUBQUERY`,
//!   `RewriteMergeIntoTable.scala:500-518`); Krishiv has no such table, and its run of a MERGE does not
//!   analyze those expressions, so there is no query here that a run also asks about.
//! - `INSERT ... REPLACE WHERE|ON` with a subquery: the same fail-closed rule (a run refuses `REPLACE`).
//! - `EXECUTE IMMEDIATE`: its reads are those of the statement its text holds, which the session parses
//!   (literal text) or cannot know (text from a variable).
//! - a script, `DECLARE ... DEFAULT` and `SET VAR a = expr`: their reads are expressions evaluated
//!   against session variables, not queries (until scripting analysis lands).

use std::borrow::Cow;

use crate::ast::{
    DdlName, Expr, ExprKind, FromClause, FromStatement, Ident, Insert, Literal, Merge, MergeAction,
    MergeInsert, MergeSource, Query, Relation, Select, SelectItem, SetExpr, SetExprKind, Span,
    Statement, TableAlias, TableFactor, TableFactorKind,
};
use crate::classify::statement_queries;

/// The queries `statement` runs to read (see the module documentation), or `None` when they are not
/// known from the statement alone. Borrowed queries are the statement's own; owned ones are built
/// from its expressions, whose spans still index the statement's text.
#[must_use]
pub fn statement_reads(statement: &Statement) -> Option<Vec<Cow<'_, Query>>> {
    match statement {
        Statement::Explain(e) => statement_reads(&e.statement),
        Statement::Insert(i) => {
            if replace_reads_a_relation(i) {
                return None;
            }
            Some(match &i.source {
                None => i.clauses.iter().map(|c| Cow::Borrowed(&*c.query)).collect(),
                Some(source) => i
                    .clauses
                    .iter()
                    .map(|c| Cow::Owned(shared_from_query(source, &c.query)))
                    .collect(),
            })
        }
        Statement::Update(u) => {
            let mut out: Vec<Cow<'_, Query>> = u
                .assignments
                .iter()
                .map(|a| {
                    let column = Expr {
                        kind: ExprKind::Column(a.target.parts.clone()),
                        span: a.target.span,
                    };
                    Cow::Owned(select_from(&u.table, None, vec![column], None))
                })
                .collect();
            let values: Vec<Expr> = u
                .assignments
                .iter()
                .map(|a| &a.value)
                .filter(|v| !is_bare_default(v))
                .cloned()
                .collect();
            out.push(Cow::Owned(select_from(
                &u.table,
                u.alias.as_ref(),
                values,
                u.where_clause.as_ref(),
            )));
            Some(out)
        }
        // Without a `WHERE` too: a run analyzes `SELECT 1 FROM t`, so it asks `Select` on `t`.
        Statement::Delete(d) => Some(vec![Cow::Owned(select_from(
            &d.table,
            d.alias.as_ref(),
            Vec::new(),
            d.where_clause.as_ref(),
        ))]),
        Statement::Merge(m) => {
            if merge_reads_a_relation(m) {
                return None;
            }
            Some(match &m.source {
                MergeSource::Query { query, .. } => vec![Cow::Borrowed(&**query)],
                MergeSource::Table { .. } => Vec::new(),
            })
        }
        other => statement_queries(other).map(|q| q.into_iter().map(Cow::Borrowed).collect()),
    }
}

/// The query a clause of a multi-table `INSERT` runs: `FROM source <clause query>`, its `WITH` kept
/// outside.
#[must_use]
pub fn shared_from_query(source: &FromClause, query: &Query) -> Query {
    let mut body = query.clone();
    let with = body.with.take();
    let span = query.span;
    Query {
        with,
        body: SetExpr {
            kind: SetExprKind::FromStatement(Box::new(FromStatement {
                from: source.clone(),
                bodies: vec![body],
                span,
            })),
            span,
        },
        ..bare_query(span)
    }
}

/// Whether a `REPLACE WHERE` or `REPLACE ON` condition of the `INSERT` `i` reads a relation of its own
/// (such an `INSERT` fails closed, as `MERGE` does).
pub(crate) fn replace_reads_a_relation(i: &Insert) -> bool {
    i.clauses
        .iter()
        .flat_map(|c| c.replace_where.iter().chain(c.replace_on.iter()))
        .any(reads_a_relation)
}

/// Whether an expression of the MERGE `m` (its `ON`, a clause condition, a `SET` value or an inserted
/// value) reads a relation of its own (see the module documentation: such a MERGE fails closed).
pub(crate) fn merge_reads_a_relation(m: &Merge) -> bool {
    let mut exprs: Vec<&Expr> = vec![&m.on];
    for clause in &m.clauses {
        exprs.extend(clause.condition.iter());
        match &clause.action {
            MergeAction::Update(assignments) => {
                exprs.extend(assignments.iter().map(|a| &a.value));
            }
            MergeAction::Insert(MergeInsert::Values { values, .. }) => exprs.extend(values.iter()),
            MergeAction::UpdateStar
            | MergeAction::Delete
            | MergeAction::Insert(MergeInsert::Star) => {}
        }
    }
    exprs.into_iter().any(reads_a_relation)
}

/// Whether the expression tree `e` reads a relation of its own: a subquery (scalar, `EXISTS`, `IN`) or a
/// `TABLE(...)` argument. Subquery plans are not entered: finding one is enough.
fn reads_a_relation(e: &Expr) -> bool {
    if matches!(
        e.kind,
        ExprKind::Subquery(_)
            | ExprKind::Exists(_)
            | ExprKind::InSubquery { .. }
            | ExprKind::TableArgument(_)
    ) {
        return true;
    }
    let mut found = false;
    e.for_each_child(&mut |c| found = found || reads_a_relation(c));
    found
}

/// Whether `e` is a bare `DEFAULT` value (`SET c = DEFAULT`, `(DEFAULT)`, any case, not quoted): it
/// reads the column's default, not a column.
///
/// ```
/// use krishiv_sql::{ParserOptions, is_bare_default, parse_named_expression};
///
/// let e = |sql: &str| parse_named_expression(sql, &ParserOptions::default()).unwrap().expr;
/// assert!(is_bare_default(&e("(default)")));
/// assert!(!is_bare_default(&e("`default`")));
/// ```
#[must_use]
pub fn is_bare_default(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Nested(inner) => is_bare_default(inner),
        ExprKind::Column(parts) => {
            matches!(&parts[..], [p] if !p.quoted && p.value.eq_ignore_ascii_case("default"))
        }
        _ => false,
    }
}

/// `SELECT items FROM table [AS alias] [WHERE condition]`; `SELECT TRUE ...` without items.
fn select_from(
    table: &DdlName,
    alias: Option<&Ident>,
    items: Vec<Expr>,
    condition: Option<&Expr>,
) -> Query {
    let span = table.span();
    let mut projection: Vec<SelectItem> = items
        .into_iter()
        .map(|expr| SelectItem {
            span: expr.span,
            expr,
            alias: None,
        })
        .collect();
    if projection.is_empty() {
        projection.push(SelectItem {
            expr: Expr {
                kind: ExprKind::Literal(Literal::Boolean(true)),
                span,
            },
            alias: None,
            span,
        });
    }
    let kind = match table {
        DdlName::Parts(name) => TableFactorKind::Table {
            name: name.clone(),
            temporal: None,
            options: Vec::new(),
        },
        DdlName::Dynamic(name) => TableFactorKind::DynamicTable {
            name: name.clone(),
            temporal: None,
            options: Vec::new(),
        },
    };
    let primary = TableFactor {
        kind,
        sample: None,
        alias: alias.map(|a| TableAlias {
            name: Some(a.clone()),
            columns: Vec::new(),
            span: a.span,
        }),
        span,
    };
    let from = FromClause {
        relations: vec![Relation {
            lateral: false,
            primary,
            extensions: Vec::new(),
            span,
        }],
        lateral_views: Vec::new(),
        pivot: None,
        unpivot: None,
        span,
    };
    let select = Select {
        hints: Vec::new(),
        quantifier: None,
        projection,
        from: Some(from),
        lateral_views: Vec::new(),
        selection: condition.cloned(),
        group_by: None,
        having: None,
        having_as_where: false,
        windows: Vec::new(),
        qualify: None,
        span,
    };
    Query {
        body: SetExpr {
            kind: SetExprKind::Select(Box::new(select)),
            span,
        },
        ..bare_query(span)
    }
}

/// A query with no clauses around its body (a placeholder body to be replaced).
fn bare_query(span: Span) -> Query {
    Query {
        with: None,
        body: SetExpr {
            kind: SetExprKind::Values(crate::ast::Values {
                rows: Vec::new(),
                alias: None,
                span,
            }),
            span,
        },
        order_by: Vec::new(),
        order_span: None,
        organization_span: None,
        cluster_by: Vec::new(),
        distribute_by: Vec::new(),
        sort_by: Vec::new(),
        windows: Vec::new(),
        limit: None,
        offset: None,
        span,
    }
}
