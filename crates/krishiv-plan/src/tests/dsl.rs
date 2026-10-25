// SPDX-License-Identifier: Apache-2.0
//! A small expression and plan DSL for hand-built plans, shaped as Spark's
//! parser shapes them.

use std::sync::Arc;

use krishiv_types::{DataType, DecimalType};

use crate::{
    Builtin, Expr, ExprNode, FunctionCall, JoinCondition, JoinKind, Literal, LogicalPlan,
    SortOrder, Star, SubqueryKind,
};

/// A column; dots separate name parts.
pub(super) fn c(name: &str) -> Expr {
    Expr::column(name.split('.'))
}

pub(super) fn int(v: i32) -> Expr {
    Literal::Integer(v).into()
}

pub(super) fn s(v: &str) -> Expr {
    Literal::String(v.to_owned()).into()
}

/// A decimal literal as Spark types it: `dec("0.05")` is `decimal(2,2)`.
pub(super) fn dec(text: &str) -> Expr {
    let (int_part, frac) = text.split_once('.').unwrap_or((text, ""));
    let unscaled: i128 = format!("{int_part}{frac}").parse().unwrap();
    let scale = u32::try_from(frac.len()).unwrap();
    let digits = u32::try_from(unscaled.unsigned_abs().to_string().len()).unwrap();
    let data_type = DecimalType::try_new(digits.max(scale), scale).unwrap();
    Literal::Decimal {
        unscaled,
        data_type,
    }
    .into()
}

/// `CAST('yyyy-mm-dd' AS date)`.
pub(super) fn date(v: &str) -> Expr {
    s(v).cast(DataType::Date)
}

pub(super) fn f(name: &str, args: Vec<Expr>) -> Expr {
    Expr::function(name, args)
}

/// `name(args)`, each with its `name => ` prefix when one is given.
pub(super) fn named_call(name: &str, args: &[(Option<&str>, Expr)]) -> Expr {
    Expr::new(ExprNode::Function(FunctionCall {
        name: name.to_owned(),
        qualifier: Vec::new(),
        args: args.iter().map(|(_, e)| Arc::new(e.clone())).collect(),
        arg_names: args.iter().map(|(n, _)| n.map(str::to_owned)).collect(),
        distinct: false,
        filter: None,
    }))
}

/// `qualifier.name(args)`.
pub(super) fn qualified_call(qualifier: &[&str], name: &str, args: Vec<Expr>) -> Expr {
    Expr::new(ExprNode::Function(FunctionCall {
        name: name.to_owned(),
        qualifier: qualifier.iter().map(|q| (*q).to_owned()).collect(),
        args: args.into_iter().map(Arc::new).collect(),
        arg_names: Vec::new(),
        distinct: false,
        filter: None,
    }))
}

pub(super) fn op(b: Builtin, l: Expr, r: Expr) -> Expr {
    Expr::builtin(b, vec![l, r])
}

pub(super) fn eq(l: Expr, r: Expr) -> Expr {
    op(Builtin::Eq, l, r)
}

/// `l <> r`, which Spark's parser and PySpark both write `not(l == r)`.
pub(super) fn ne(l: Expr, r: Expr) -> Expr {
    not(eq(l, r))
}

pub(super) fn lt(l: Expr, r: Expr) -> Expr {
    op(Builtin::Lt, l, r)
}

pub(super) fn le(l: Expr, r: Expr) -> Expr {
    op(Builtin::LtEq, l, r)
}

pub(super) fn gt(l: Expr, r: Expr) -> Expr {
    op(Builtin::Gt, l, r)
}

pub(super) fn ge(l: Expr, r: Expr) -> Expr {
    op(Builtin::GtEq, l, r)
}

pub(super) fn add(l: Expr, r: Expr) -> Expr {
    op(Builtin::Add, l, r)
}

pub(super) fn sub(l: Expr, r: Expr) -> Expr {
    op(Builtin::Subtract, l, r)
}

pub(super) fn mul(l: Expr, r: Expr) -> Expr {
    op(Builtin::Multiply, l, r)
}

pub(super) fn div(l: Expr, r: Expr) -> Expr {
    op(Builtin::Divide, l, r)
}

pub(super) fn like(l: Expr, pattern: &str) -> Expr {
    op(Builtin::Like, l, s(pattern))
}

pub(super) fn not(e: Expr) -> Expr {
    Expr::builtin(Builtin::Not, vec![e])
}

pub(super) fn or(l: Expr, r: Expr) -> Expr {
    any(vec![l, r])
}

/// `a AND b AND ...` as Spark's parser builds it: a balanced tree
/// (`AstBuilder.visitLogicalBinary`).
pub(super) fn and(terms: Vec<Expr>) -> Expr {
    balanced(Builtin::And, terms)
}

/// `a OR b OR ...`, balanced as [`and`].
pub(super) fn any(terms: Vec<Expr>) -> Expr {
    balanced(Builtin::Or, terms)
}

pub(super) fn balanced(name: Builtin, mut terms: Vec<Expr>) -> Expr {
    fn reduce(name: Builtin, terms: &mut Vec<Option<Expr>>, low: usize, high: usize) -> Expr {
        if low == high {
            return terms[low].take().unwrap();
        }
        let mid = low + (high - low) / 2;
        let (l, r) = if high - low == 1 {
            (terms[low].take().unwrap(), terms[high].take().unwrap())
        } else {
            (
                reduce(name, terms, low, mid),
                reduce(name, terms, mid + 1, high),
            )
        };
        op(name, l, r)
    }
    let n = terms.len();
    let mut slots: Vec<Option<Expr>> = terms.drain(..).map(Some).collect();
    reduce(name, &mut slots, 0, n - 1)
}

pub(super) fn in_list(v: Expr, items: Vec<Expr>) -> Expr {
    Expr::builtin(Builtin::In, std::iter::once(v).chain(items).collect())
}

pub(super) fn between(v: Expr, lo: Expr, hi: Expr) -> Expr {
    Expr::builtin(Builtin::Between, vec![v, lo, hi])
}

/// `CASE WHEN cond THEN then ELSE otherwise END`.
pub(super) fn when(cond: Expr, then: Expr, otherwise: Expr) -> Expr {
    Expr::builtin(Builtin::When, vec![cond, then, otherwise])
}

pub(super) fn star() -> Expr {
    Expr::new(ExprNode::Star(Star {
        qualifier: Vec::new(),
        except: Vec::new(),
        plan_id: None,
    }))
}

pub(super) fn count_star() -> Expr {
    f("count", vec![star()])
}

pub(super) fn sum(e: Expr) -> Expr {
    f("sum", vec![e])
}

pub(super) fn avg(e: Expr) -> Expr {
    f("avg", vec![e])
}

/// `extract(year FROM e)`.
pub(super) fn year(e: Expr) -> Expr {
    f("extract", vec![s("year"), e])
}

/// `l_extendedprice * (1 - l_discount)`.
pub(super) fn disc_price() -> Expr {
    mul(c("l_extendedprice"), sub(int(1), c("l_discount")))
}

pub(super) fn scalar(plan: LogicalPlan) -> Expr {
    Expr::subquery(SubqueryKind::Scalar, plan)
}

pub(super) fn exists(plan: LogicalPlan) -> Expr {
    Expr::subquery(SubqueryKind::Exists, plan)
}

pub(super) fn in_subquery(v: Expr, plan: LogicalPlan) -> Expr {
    Expr::subquery(SubqueryKind::In(vec![Arc::new(v)]), plan)
}

pub(super) fn asc(e: Expr) -> SortOrder {
    SortOrder::asc(e)
}

pub(super) fn desc(e: Expr) -> SortOrder {
    SortOrder::desc(e)
}

/// `FROM a, b, ...`; an item `"nation n1"` is aliased.
pub(super) fn from(items: &[&str]) -> LogicalPlan {
    items
        .iter()
        .map(|item| match item.split_once(' ') {
            Some((table, alias)) => LogicalPlan::table([table]).alias(alias),
            None => LogicalPlan::table([*item]),
        })
        .reduce(|l, r| l.join(r, JoinKind::Inner, JoinCondition::None))
        .unwrap()
}

pub(super) fn cols(names: &[&str]) -> Vec<Expr> {
    names.iter().map(|n| c(n)).collect()
}

pub(super) const COUNTRY_CODES: [&str; 7] = ["13", "31", "23", "29", "30", "18", "17"];

pub(super) fn cntrycode_in() -> Expr {
    in_list(
        f("substring", vec![c("c_phone"), int(1), int(2)]),
        COUNTRY_CODES.iter().map(|v| s(v)).collect(),
    )
}
