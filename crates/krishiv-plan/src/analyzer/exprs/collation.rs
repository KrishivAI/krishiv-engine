// SPDX-License-Identifier: Apache-2.0
//! The single collation of a call's string arguments (Spark's `CollationTypeCoercion`, ADR 0043,
//! design `P2-collations.md` section 3.2).
//!
//! Each string argument has a collation and a strength: *explicit* (`COLLATE`), *implicit* (a
//! column) or *default* (a literal). The strongest wins; two different explicit collations are
//! `COLLATION_MISMATCH.EXPLICIT`, two different implicit ones are indeterminate
//! (`INDETERMINATE_COLLATION_IN_EXPRESSION`). The call then runs on plain strings (the carrier is
//! the same) and a string result takes the collation.

use krishiv_common::error::{
    CollationMismatchExplicit, DatatypeMismatchUnexpectedInputType,
    IndeterminateCollationInExpression,
};
use krishiv_types::{Collation, CollationFamily};

use super::*;

/// How firmly an operand carries its collation, weakest first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Strength {
    Default,
    Implicit,
    Explicit,
}

/// Why the operands have no single collation.
#[derive(Debug, PartialEq)]
enum Conflict {
    /// Different explicit collations, in order of appearance.
    Explicit(Vec<Collation>),
    /// Different implicit collations.
    Indeterminate,
}

/// Calls that compare their string arguments, or key on them. They keep the collated types so
/// the execution guard sees them (units 3a and 3b lift it).
const COMPARING: &[&str] = &[
    "==",
    "<=>",
    "<",
    "<=",
    ">",
    ">=",
    "in",
    "between",
    "greatest",
    "least",
    "nullif",
    "min",
    "max",
    "min_by",
    "max_by",
    "mode",
    "collect_set",
    "listagg",
    "approx_count_distinct",
];

/// Calls that fail on operands without one collation (Spark's `canContainIndeterminateCollation`);
/// any other call over such operands runs on the plain strings.
const FAILS_INDETERMINATE: &[&str] = &[
    "contains",
    "startswith",
    "endswith",
    "find_in_set",
    "instr",
    "locate",
    "position",
    "replace",
    "substring_index",
    "translate",
    "trim",
    "ltrim",
    "rtrim",
    "btrim",
    "upper",
    "ucase",
    "lower",
    "lcase",
    "initcap",
];

/// Calls that search a string and so refuse an accent-insensitive ICU collation.
const SEARCHES: &[&str] = &[
    "contains",
    "startswith",
    "endswith",
    "instr",
    "locate",
    "position",
    "replace",
    "split_part",
    "str_to_map",
    "substring_index",
    "translate",
];

/// Calls that read the type of their argument: it stays collated.
const INSPECTS: &[&str] = &["collation", "typeof"];

fn is_string(t: Option<&DataType>) -> bool {
    matches!(t, Some(DataType::String | DataType::CollatedString(_)))
}

/// The collation and strength `e` carries, when it is a string.
fn strength_of(e: &Expr) -> Option<(Collation, Strength)> {
    match &e.node {
        ExprNode::Collate { collation, .. } => Some((*collation, Strength::Explicit)),
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
            match &c.data_type {
                DataType::String => Some((Collation::UTF8_BINARY, Strength::Implicit)),
                DataType::CollatedString(c) => Some((*c, Strength::Implicit)),
                _ => None,
            }
        }
        ExprNode::Literal(Literal::String(_)) => Some((Collation::UTF8_BINARY, Strength::Default)),
        ExprNode::Alias { expr, .. } => strength_of(expr),
        ExprNode::Cast { expr, to, .. } if is_string(Some(to)) => strength_of(expr),
        ExprNode::Function(_) | ExprNode::Resolved(ResolvedExpr::Function(_)) => {
            let mut kids = Vec::new();
            e.for_each_child(&mut |k| kids.extend(strength_of(k)));
            merge(&kids).ok().flatten()
        }
        _ => None,
    }
}

/// The collation of operands with these strengths.
fn merge(items: &[(Collation, Strength)]) -> Result<Option<(Collation, Strength)>, Conflict> {
    let distinct = |s: Strength| {
        let mut out: Vec<Collation> = Vec::new();
        for (c, x) in items {
            if *x == s && !out.contains(c) {
                out.push(*c);
            }
        }
        out
    };
    let explicit = distinct(Strength::Explicit);
    if explicit.len() > 1 {
        return Err(Conflict::Explicit(explicit));
    }
    if let Some(c) = explicit.first() {
        return Ok(Some((*c, Strength::Explicit)));
    }
    let implicit = distinct(Strength::Implicit);
    if implicit.len() > 1 {
        return Err(Conflict::Indeterminate);
    }
    if let Some(c) = implicit.first() {
        return Ok(Some((*c, Strength::Implicit)));
    }
    Ok(distinct(Strength::Default)
        .first()
        .map(|c| (*c, Strength::Default)))
}

impl Resolver<'_> {
    /// `expr_node_inner`, with the collation rule for a function call over collated strings.
    pub(super) fn expr_node(
        &mut self,
        e: &Arc<Expr>,
        kids: &[Bound],
        in_agg: bool,
        window_fn: bool,
        cx: &Cx<'_>,
    ) -> Result<Bound, AnalysisError> {
        let ExprNode::Function(f) = &e.node else {
            return self.expr_node_inner(e, kids, in_agg, window_fn, cx);
        };
        let n = f.args.len().min(kids.len());
        let name = f.name.to_ascii_lowercase();
        if f.distinct
            || COMPARING.contains(&name.as_str())
            || INSPECTS.contains(&name.as_str())
            || !kids[..n]
                .iter()
                .any(|k| matches!(k.ty.data_type, Some(DataType::CollatedString(_))))
        {
            return self.expr_node_inner(e, kids, in_agg, window_fn, cx);
        }
        let items: Vec<(Collation, Strength)> = kids[..n]
            .iter()
            .filter(|k| is_string(k.ty.data_type.as_ref()))
            .filter_map(|k| strength_of(&k.shown()))
            .collect();
        let chosen = match merge(&items) {
            Ok(c) => c.map(|(c, _)| c),
            Err(Conflict::Explicit(list)) => {
                let types: Vec<String> = list
                    .iter()
                    .map(|c| format!("\"STRING COLLATE {}\"", c.name()))
                    .collect();
                return Err(AnalysisError::new(
                    CollationMismatchExplicit {
                        explicit_types: types.join(", "),
                    },
                    e.origin,
                ));
            }
            Err(Conflict::Indeterminate) if !FAILS_INDETERMINATE.contains(&name.as_str()) => None,
            Err(Conflict::Indeterminate) => {
                let shown = with_children(e, kids[..n].iter().map(Bound::shown).collect());
                return Err(AnalysisError::new(
                    IndeterminateCollationInExpression {
                        expr: format!("\"{}\"", pretty(&shown, self.ctx.conf.session_zone)),
                    },
                    e.origin,
                ));
            }
        };
        if SEARCHES.contains(&name.as_str())
            && matches!(
                chosen.map(Collation::family),
                Some(CollationFamily::Icu { ai: true, .. })
            )
            && let Some(i) = kids[..n]
                .iter()
                .position(|k| matches!(k.ty.data_type, Some(DataType::CollatedString(_))))
        {
            let shown = with_children(e, kids[..n].iter().map(Bound::shown).collect());
            let zone = self.ctx.conf.session_zone;
            return Err(AnalysisError::new(
                DatatypeMismatchUnexpectedInputType {
                    sql_expr: format!("\"{}\"", pretty(&shown, zone)),
                    param_index: coerce::ordinal(i),
                    required_type: "\"STRING\"".to_owned(),
                    input_sql: format!("\"{}\"", pretty(&kids[i].shown(), zone)),
                    input_type: coerce::quoted_type(kids[i].ty.data_type.as_ref()),
                },
                e.origin,
            ));
        }
        // The call runs on plain strings: the carrier is one.
        let plain: Vec<Bound> = kids
            .iter()
            .map(|k| {
                let mut k = k.clone();
                if matches!(k.ty.data_type, Some(DataType::CollatedString(_))) {
                    k.ty.data_type = Some(DataType::String);
                }
                k
            })
            .collect();
        let mut bound = self.expr_node_inner(e, &plain, in_agg, window_fn, cx)?;
        if let Some(c) = chosen
            && !c.is_utf8_binary()
            && bound.ty.data_type == Some(DataType::String)
        {
            let to = DataType::collated_string(c);
            let shown = bound.shown();
            bound.expr = Arc::new(
                Expr::new(ExprNode::Cast {
                    expr: Arc::clone(&bound.expr),
                    to: to.clone(),
                    mode: CastMode::Ansi,
                })
                .with_origin(e.origin),
            );
            bound.shown = Some(shown);
            bound.ty.data_type = Some(to);
        }
        Ok(bound)
    }
}
