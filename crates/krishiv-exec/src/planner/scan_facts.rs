// SPDX-License-Identifier: Apache-2.0
//! What a scan's planner facts record (P3-T073 missing statistics, P3-T025 partition advice): the table's
//! statistics as the provider reported them and the columns the pushed filter compares with a constant by
//! `=` or `IN`. Read from the expression, not from its text, so a history that masks literals still has them.

use std::sync::Arc;

use krishiv_optimizer::TableStats;
use krishiv_plan::{Builtin, Expr, ExprNode, FunctionId, ResolvedExpr, ResolvedFunction};

use super::build::util::source_name;
use super::explain::{NodeFacts, TableFacts};

/// The facts of a scan of `scan` whose provider reported `stats`, estimated at `estimated_rows` rows, with
/// the columns its pushed filter compares by equality.
pub(super) fn of(
    scan: &krishiv_plan::Scan,
    stats: Option<TableStats>,
    estimated_rows: f64,
    filter_columns: Vec<String>,
) -> NodeFacts {
    let name = source_name(scan);
    let named = matches!(scan.source, krishiv_plan::ReadSource::Table { .. });
    match stats {
        None => NodeFacts {
            estimated_rows: Some(estimated_rows),
            tables_without_stats: vec![name],
            ..NodeFacts::default()
        },
        Some(s) => {
            let mut distinct: Vec<(String, u64)> = s
                .columns
                .iter()
                .filter_map(|(c, st)| Some((c.clone(), st.distinct?)))
                .collect();
            distinct.sort();
            NodeFacts {
                estimated_rows: Some(estimated_rows),
                table: named.then_some(TableFacts {
                    name,
                    rows: s.rows,
                    distinct,
                    filter_columns,
                }),
                ..NodeFacts::default()
            }
        }
    }
}

/// The columns `predicate` compares with constants by `=`, `<=>` or `IN` in a top-level conjunct, sorted and
/// without repeats. A comparison of two columns, a range and anything under an `OR` are not partition pruning.
pub(super) fn equality_columns(predicate: &Arc<Expr>) -> Vec<String> {
    let mut conjuncts = Vec::new();
    super::join::conjuncts_of(predicate, &mut conjuncts);
    let mut out: Vec<String> = conjuncts.iter().filter_map(|c| compared(c)).collect();
    out.sort();
    out.dedup();
    out
}

fn column(e: &Expr) -> Option<String> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.name.clone()),
        _ => None,
    }
}

fn constant(e: &Expr) -> bool {
    matches!(e.node, ExprNode::Literal(_))
}

fn compared(e: &Expr) -> Option<String> {
    let ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
        id: FunctionId::Builtin(b),
        args,
        ..
    })) = &e.node
    else {
        return None;
    };
    match (b, args.as_slice()) {
        (Builtin::Eq | Builtin::EqNullSafe, [l, r]) => match (column(l), column(r)) {
            (Some(c), None) if constant(r) => Some(c),
            (None, Some(c)) if constant(l) => Some(c),
            _ => None,
        },
        (Builtin::In, [x, list @ ..]) if list.iter().all(|v| constant(v)) => column(x),
        _ => None,
    }
}
