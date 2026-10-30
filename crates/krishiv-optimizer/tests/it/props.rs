// SPDX-License-Identifier: Apache-2.0
//! Random filter/join/aggregate queries: the optimised plan keeps its
//! output, is a fixed point, returns the same rows as the unoptimised one,
//! and leaves no pushable predicate above the lowest operator it may
//! legally reach.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_optimizer::{optimize, optimize_with};
use krishiv_plan::{
    AttrId, Builtin, Expr, ExprNode, FunctionId, Grouping, JoinCondition, JoinKind, LogicalPlan,
    PlanNode, ResolvedExpr,
};
use proptest::prelude::*;

use crate::support::eval::V;
use crate::support::{self, Rng, data, eval};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ty {
    Int,
    Str,
}

/// A generator of random queries over `t(a int, b string, c bigint)` and
/// `u(a int, d string)`.
pub(crate) struct Gen {
    r: Rng,
    alias: usize,
}

type Scope = Vec<(String, Ty)>;

impl Gen {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            r: Rng(seed | 1),
            alias: 0,
        }
    }

    fn chance(&mut self, percent: i64) -> bool {
        self.r.range(1, 100) <= percent
    }

    fn col(&mut self, scope: &Scope, ty: Option<Ty>) -> Option<(String, Ty)> {
        let fit: Vec<_> = scope
            .iter()
            .filter(|(_, t)| ty.is_none_or(|x| x == *t))
            .collect();
        if fit.is_empty() {
            return None;
        }
        let i = self.r.range(0, i64::try_from(fit.len()).unwrap() - 1);
        Some(fit[usize::try_from(i).unwrap()].clone())
    }

    fn lit(&mut self, ty: Ty) -> String {
        match ty {
            Ty::Int => self.r.range(-2, 3).to_string(),
            Ty::Str => format!("'{}'", self.r.pick(&["x", "y", "z"])),
        }
    }

    /// A query and its output columns `c0, c1, ...`.
    pub(crate) fn query(&mut self, depth: u32) -> (String, Scope) {
        let (from, scope) = self.from(depth);
        let filter = if self.chance(70) {
            format!(" WHERE {}", self.pred(&scope, 2))
        } else {
            String::new()
        };
        if self.chance(25) {
            let (key, ty) = self.col(&scope, None).unwrap();
            let sql =
                format!("SELECT {key} AS c0, count(*) AS c1 FROM {from}{filter} GROUP BY {key}");
            return (sql, vec![("c0".into(), ty), ("c1".into(), Ty::Int)]);
        }
        let mut items = Vec::new();
        let mut out = Vec::new();
        for i in 0..self.r.range(1, 3) {
            let (c, ty) = self.col(&scope, None).unwrap();
            let e = match ty {
                Ty::Int if self.chance(30) => format!("{c} + 1"),
                _ => c,
            };
            items.push(format!("{e} AS c{i}"));
            out.push((format!("c{i}"), ty));
        }
        (
            format!("SELECT {} FROM {from}{filter}", items.join(", ")),
            out,
        )
    }

    fn from(&mut self, depth: u32) -> (String, Scope) {
        let roll = if depth == 0 { 0 } else { self.r.range(0, 2) };
        if roll == 2 {
            let (left, ls) = self.from(depth - 1);
            let (right, rs) = self.relation(depth - 1);
            let kinds = [
                "JOIN",
                "LEFT JOIN",
                "RIGHT JOIN",
                "FULL JOIN",
                "CROSS JOIN",
                "LEFT SEMI JOIN",
                "LEFT ANTI JOIN",
            ];
            let kind = kinds[usize::try_from(self.r.range(0, 6)).unwrap()];
            let both: Scope = ls.iter().chain(&rs).cloned().collect();
            let on = if kind == "CROSS JOIN" {
                String::new()
            } else {
                let key = match (self.col(&ls, Some(Ty::Int)), self.col(&rs, Some(Ty::Int))) {
                    (Some((l, _)), Some((r, _))) => format!("{l} = {r}"),
                    _ => self.pred(&both, 1),
                };
                let extra = if self.chance(50) {
                    format!(" AND {}", self.pred(&both, 1))
                } else {
                    String::new()
                };
                format!(" ON {key}{extra}")
            };
            let scope = if kind.contains("SEMI") || kind.contains("ANTI") {
                ls
            } else {
                both
            };
            return (format!("{left} {kind} {right}{on}"), scope);
        }
        self.relation(depth)
    }

    /// A table or a derived table, under a fresh alias.
    fn relation(&mut self, depth: u32) -> (String, Scope) {
        self.alias += 1;
        let x = format!("x{}", self.alias);
        if depth > 0 && self.chance(40) {
            let (sql, out) = self.query(depth - 1);
            let scope = out
                .into_iter()
                .map(|(c, t)| (format!("{x}.{c}"), t))
                .collect();
            return (format!("({sql}) AS {x}"), scope);
        }
        let (table, cols): (&str, &[(&str, Ty)]) = if self.chance(50) {
            ("t", &[("a", Ty::Int), ("b", Ty::Str), ("c", Ty::Int)])
        } else {
            ("u", &[("a", Ty::Int), ("d", Ty::Str)])
        };
        let scope = cols.iter().map(|(c, t)| (format!("{x}.{c}"), *t)).collect();
        (format!("{table} AS {x}"), scope)
    }

    fn pred(&mut self, scope: &Scope, depth: u32) -> String {
        let (c, ty) = self.col(scope, None).unwrap();
        let roll = self.r.range(0, if depth == 0 { 4 } else { 7 });
        match roll {
            0 | 1 => {
                let op = self.r.pick(&["=", "<", "<=", ">", ">=", "<>", "<=>"]);
                let rhs = match self.col(scope, Some(ty)) {
                    Some((other, _)) if self.chance(40) => other,
                    _ => self.lit(ty),
                };
                format!("{c} {op} {rhs}")
            }
            2 => format!("{c} IS {}NULL", if self.chance(50) { "NOT " } else { "" }),
            3 => format!("{c} IN ({}, {})", self.lit(ty), self.lit(ty)),
            4 => format!("coalesce({c}, {}) = {}", self.lit(ty), self.lit(ty)),
            5 => format!("NOT ({})", self.pred(scope, depth - 1)),
            6 => format!(
                "({} AND {})",
                self.pred(scope, depth - 1),
                self.pred(scope, depth - 1)
            ),
            _ => format!(
                "({} OR {})",
                self.pred(scope, depth - 1),
                self.pred(scope, depth - 1)
            ),
        }
    }

    /// A random query, sometimes ordered and limited.
    pub(crate) fn top(&mut self) -> String {
        let (sql, out) = self.query(3);
        if self.chance(30) {
            let keys: Vec<_> = out.iter().map(|(c, _)| c.as_str()).collect();
            return format!(
                "SELECT * FROM ({sql}) ORDER BY {} LIMIT {}",
                keys.join(", "),
                self.r.range(0, 4)
            );
        }
        sql
    }
}

fn refs(e: &Expr, out: &mut HashSet<AttrId>) {
    if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &e.node {
        out.insert(c.id);
    }
    e.for_each_child(&mut |c| refs(c, out));
}

fn conjuncts(e: &Arc<Expr>, out: &mut Vec<Arc<Expr>>) {
    match op(e) {
        Some((Builtin::And, args)) => args.iter().for_each(|a| conjuncts(a, out)),
        _ => out.push(Arc::clone(e)),
    }
}

fn op(e: &Expr) -> Option<(Builtin, &[Arc<Expr>])> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => match f.id {
            FunctionId::Builtin(b) => Some((b, &f.args)),
            _ => None,
        },
        _ => None,
    }
}

fn ids(p: &LogicalPlan) -> HashSet<AttrId> {
    p.resolved_output().unwrap().iter().map(|c| c.id).collect()
}

/// Whether `pred` is certainly false or NULL when a column of `side` is
/// NULL: a null-intolerant comparison over a column of `side`.
fn rejects_nulls(pred: &Expr, side: &HashSet<AttrId>) -> bool {
    match op(pred) {
        Some((
            Builtin::Eq | Builtin::Lt | Builtin::LtEq | Builtin::Gt | Builtin::GtEq | Builtin::In,
            args,
        )) => {
            matches!(&args[0].node, ExprNode::Resolved(ResolvedExpr::Column(c)) if side.contains(&c.id))
        }
        Some((Builtin::IsNotNull, [a])) => {
            matches!(&a.node, ExprNode::Resolved(ResolvedExpr::Column(c)) if side.contains(&c.id))
        }
        _ => false,
    }
}

/// Whether filter conjunct `pred` could legally move below `child`.
///
/// A common-subexpression projection is opaque to predicate pushdown (see
/// `pushdown`): pushing anything below it would evaluate the shared
/// expression again at each use, so every conjunct above such a projection
/// stays put.
fn pushable(pred: &Expr, child: &LogicalPlan) -> bool {
    let mut r = HashSet::new();
    refs(pred, &mut r);
    if let PlanNode::Project { exprs, .. } = &child.node {
        let cse = exprs.iter().any(|e| match &e.node {
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => {
                output.name.starts_with("cse_")
            }
            _ => false,
        });
        if cse {
            return false;
        }
    }
    match &child.node {
        PlanNode::Filter { .. }
        | PlanNode::Project { .. }
        | PlanNode::Sort { .. }
        | PlanNode::SubqueryAlias { .. }
        | PlanNode::WithCte { .. }
        | PlanNode::Shift { .. } => true,
        PlanNode::Aggregate {
            grouping: Grouping::GroupBy(keys),
            aggregates,
            ..
        } if !keys.is_empty() => {
            let grouped: HashSet<AttrId> = aggregates
                .iter()
                .filter_map(|e| match &e.node {
                    ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
                    ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
                        let mut x = String::new();
                        x.push_str(&expr.to_string());
                        (!["count(", "sum(", "max(", "min(", "avg("]
                            .iter()
                            .any(|f| x.contains(f)))
                        .then_some(output.id)
                    }
                    _ => None,
                })
                .collect();
            let own = ids(child);
            let mine: HashSet<_> = r.intersection(&own).copied().collect();
            !mine.is_empty() && mine.is_subset(&grouped)
        }
        PlanNode::Join {
            left, right, kind, ..
        } => {
            let (lo, ro) = (ids(left), ids(right));
            match kind {
                JoinKind::Inner | JoinKind::Cross | JoinKind::LeftSemi | JoinKind::LeftAnti => true,
                JoinKind::LeftOuter => r.is_subset(&lo) || rejects_nulls(pred, &ro),
                JoinKind::RightOuter => r.is_subset(&ro) || rejects_nulls(pred, &lo),
                _ => rejects_nulls(pred, &lo) || rejects_nulls(pred, &ro),
            }
        }
        _ => false,
    }
}

/// Every filter conjunct sits at the lowest operator it may reach, and no
/// join condition conjunct could move into an input.
pub(crate) fn no_pushdown_missed(root: &LogicalPlan, sql: &str) {
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        p.for_each_child(&mut |c| stack.push(c));
        match &p.node {
            PlanNode::Filter { input, predicate } => {
                let mut cs = Vec::new();
                conjuncts(predicate, &mut cs);
                for c in cs {
                    assert!(!pushable(&c, input), "{sql}\n{c} could move below\n{p}");
                }
            }
            PlanNode::Join {
                left,
                right,
                kind,
                condition: JoinCondition::On(cond),
            } => {
                let (lo, ro) = (ids(left), ids(right));
                let mut cs = Vec::new();
                conjuncts(cond, &mut cs);
                for c in cs {
                    let mut r = HashSet::new();
                    refs(&c, &mut r);
                    let (l, rr) = (r.is_subset(&lo), r.is_subset(&ro));
                    let movable = match kind {
                        JoinKind::Inner | JoinKind::Cross | JoinKind::LeftSemi => l || rr,
                        JoinKind::LeftOuter | JoinKind::LeftAnti => rr,
                        JoinKind::RightOuter => l,
                        _ => false,
                    };
                    assert!(
                        !movable,
                        "{sql}\njoin condition {c} could move into an input\n{p}"
                    );
                }
            }
            _ => {}
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn random_queries_reach_the_lowest_operator(seed in any::<u64>()) {
        let ctx = support::ctx();
        let sql = Gen::new(seed).top();
        let plan = support::resolve(&sql, &ctx);
        let optimized = optimize(&plan).unwrap();
        prop_assert_eq!(optimized.root().resolved_output(), plan.root().resolved_output());
        let again = optimize(&optimized).unwrap();
        prop_assert!(again.structurally_eq(&optimized), "{sql}: not a fixed point");
        no_pushdown_missed(optimized.root(), &sql);
        let db = data::small(seed);
        prop_assert_eq!(eval::rows(&db, optimized.root()), eval::rows(&db, plan.root()), "{}", sql);
    }
}

/// Seed 3096671712105701144 of `random_queries_reach_the_lowest_operator`:
/// join ordering regroups `x2 x x3 x x4` so the `WHERE` disjunction sits on
/// a join with `x3 x x4` as one side, which implies `(x4.a IN (-2, 2) or
/// coalesce(x3.a, 3) = 2) or x3.a >= 1` there. Derived per side, a second
/// optimization added that predicate (the first had derived it for the
/// written grouping, where it implied nothing), so the plan was not a fixed
/// point. Implied filters are now derived per join input, which regrouping
/// does not change.
#[test]
fn implied_filters_reach_a_fixed_point() {
    let sql = "SELECT x1.c0 AS c0, count(*) AS c1 FROM (SELECT x2.a AS c0, x2.d AS c1 \
               FROM u AS x2 CROSS JOIN t AS x3 JOIN u AS x4 ON x3.c = x4.a AND x4.d <> x2.d \
               WHERE ((x4.a IN (-2, 2) OR coalesce(x3.a, 3) = 2) OR (x3.a >= 1 AND x2.d IS NULL))) \
               AS x1 GROUP BY x1.c0";
    let plan = support::resolve(sql, &support::ctx());
    let optimized = optimize(&plan).unwrap();
    let again = optimize(&optimized).unwrap();
    assert!(
        again.structurally_eq(&optimized),
        "not a fixed point:\n{}\n{}",
        optimized.root(),
        again.root()
    );
    // (c0, c1) per data seed, from Spark 4.2.0 on the same rows.
    let n = |v: i64| V::int(v);
    let expected = [
        (
            1,
            vec![vec![n(-1), n(3)], vec![n(0), n(6)], vec![n(1), n(6)]],
        ),
        (
            2,
            vec![vec![V::Null, n(2)], vec![n(-1), n(4)], vec![n(3), n(6)]],
        ),
        (
            3_096_671_712_105_701_144,
            vec![vec![V::Null, n(8)], vec![n(1), n(4)]],
        ),
    ];
    for (seed, rows) in expected {
        let db = data::small(seed);
        let got = eval::rows(&db, optimized.root());
        assert_eq!(got, eval::rows(&db, plan.root()));
        let mut want = rows;
        want.sort();
        assert_eq!(got, want, "seed {seed}");
    }
}

/// Seed 11717397470216288774 of `random_queries_reach_the_lowest_operator`:
/// the first run derives `coalesce(x2.c + 1, -1) = 2 or x2.b < 'y'` from the
/// join condition below the join and common-subexpression elimination then
/// puts `x2.c + 1` into a `cse_*` projection above that filter. The second
/// run derived the same predicate over the `cse_*` column; nothing moves
/// below that projection, so it stacked above it next to the equal filter
/// already under it. A predicate the filters directly under a
/// common-subexpression projection enforce is now dropped there.
#[test]
fn a_derived_filter_is_not_stacked_over_a_common_subexpression_projection() {
    let sql = Gen::new(11_717_397_470_216_288_774).top();
    let plan = support::resolve(&sql, &support::ctx());
    let optimized = optimize(&plan).unwrap();
    let again = optimize(&optimized).unwrap();
    assert!(
        again.structurally_eq(&optimized),
        "{sql}: not a fixed point:\n{}\n{}",
        optimized.root(),
        again.root()
    );
    let db = data::small(11_717_397_470_216_288_774);
    assert_eq!(
        eval::rows(&db, optimized.root()),
        eval::rows(&db, plan.root()),
        "{sql}"
    );
}

/// Seed 3574806867649217214 of `random_queries_reach_the_lowest_operator`:
/// the aggregate's input is already a semi join on its key (the `LEFT SEMI
/// JOIN` the query writes, its right side a shifted instance of `u`). The
/// restriction pass took that for its own work on the first run and
/// skipped; pruning then put a projection above the shift, and the second
/// run restricted the aggregate, so the plan was not a fixed point. An
/// existing semi join on the keys now counts whatever its right side looks
/// like.
#[test]
fn restriction_reaches_a_fixed_point() {
    let sql = "SELECT x1.c0 AS c0, x4.d AS c1, x1.c0 AS c2 FROM (SELECT x2.a AS c0, count(*) AS c1 \
               FROM u AS x2 LEFT SEMI JOIN u AS x3 ON x2.a = x3.a AND x2.d IS NOT NULL \
               WHERE x2.a IS NOT NULL GROUP BY x2.a) AS x1 LEFT JOIN u AS x4 ON x1.c0 = x4.a \
               WHERE x4.d = x4.d";
    let ctx = support::ctx();
    let plan = support::resolve(sql, &ctx);
    let optimized = optimize(&plan).unwrap();
    let again = optimize(&optimized).unwrap();
    assert!(
        again.structurally_eq(&optimized),
        "not a fixed point:\n{}\n{}",
        optimized.root(),
        again.root()
    );
    for seed in [1, 2, 3_574_806_867_649_217_214] {
        let db = data::small(seed);
        assert_eq!(
            eval::rows(&db, optimized.root()),
            eval::rows(&db, plan.root()),
            "seed {seed}"
        );
    }
}

/// Found by `random_queries_reach_the_lowest_operator`: the aggregate `x2`
/// sits in `u CROSS JOIN x2` and the equality to `x4` on the join above, so
/// the first run did not restrict it; join ordering then joined `x2` and
/// `x4` directly and the second run did. The restriction now reaches an
/// aggregate through inner joins, whatever order the query wrote.
#[test]
fn restriction_through_inner_joins_reaches_a_fixed_point() {
    let sql = "SELECT x1.d AS c0, count(*) AS c1 FROM u AS x1 CROSS JOIN (SELECT x3.a AS c0, \
               count(*) AS c1 FROM t AS x3 GROUP BY x3.a) AS x2 JOIN (SELECT x5.c0 + 1 AS c0, \
               x5.c0 AS c1 FROM (SELECT x6.c + 1 AS c0 FROM t AS x6 WHERE x6.a IN (-1, 2)) AS x5 \
               WHERE x5.c0 <=> 2) AS x4 ON x2.c0 = x4.c1 GROUP BY x1.d";
    let ctx = support::ctx();
    let plan = support::resolve(sql, &ctx);
    let optimized = optimize(&plan).unwrap();
    let again = optimize(&optimized).unwrap();
    assert!(
        again.structurally_eq(&optimized),
        "not a fixed point:\n{}\n{}",
        optimized.root(),
        again.root()
    );
    let text = optimized.root().to_string();
    assert!(text.contains("LeftSemi"), "not restricted:\n{text}");
    for seed in 1..=4 {
        let db = data::small(seed);
        assert_eq!(
            eval::rows(&db, optimized.root()),
            eval::rows(&db, plan.root()),
            "seed {seed}"
        );
    }
}

/// Seed 66213344527430992 of `random_queries_reach_the_lowest_operator`:
/// before join ordering the aggregate `x1` joins plain `u` (no selective
/// domain); ordering then joins it to `u` already filtered by the
/// `IN (-1, -2)` side, and only a second optimization restricted it.
/// Restriction and join ordering now alternate until neither changes the
/// plan.
#[test]
fn restriction_after_join_ordering_reaches_a_fixed_point() {
    let sql = "SELECT * FROM (SELECT x1.c0 AS c0, count(*) AS c1 FROM (SELECT x2.c AS c0, \
               count(*) AS c1 FROM t AS x2 WHERE (x2.a >= 3 OR coalesce(x2.c, 3) = -2) \
               GROUP BY x2.c) AS x1 RIGHT JOIN u AS x3 ON x1.c0 = x3.a RIGHT JOIN u AS x4 \
               ON x3.a = x4.a AND x4.a IN (-1, -2) WHERE x1.c0 <> 0 GROUP BY x1.c0) \
               ORDER BY c0, c1 LIMIT 4";
    let ctx = support::ctx();
    let plan = support::resolve(sql, &ctx);
    let optimized = optimize_with(&plan, &support::AllDistinct).unwrap();
    let again = optimize_with(&optimized, &support::AllDistinct).unwrap();
    assert!(
        again.structurally_eq(&optimized),
        "not a fixed point:\n{}\n{}",
        optimized.root(),
        again.root()
    );
    let text = optimized.root().to_string();
    assert!(text.contains("LeftSemi"), "not restricted:\n{text}");
    for seed in [1, 2, 66_213_344_527_430_992] {
        let db = data::small(seed);
        assert_eq!(
            eval::rows(&db, optimized.root()),
            eval::rows(&db, plan.root()),
            "seed {seed}"
        );
    }
}
