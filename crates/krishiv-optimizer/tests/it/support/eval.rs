// SPDX-License-Identifier: Apache-2.0
//! A row-at-a-time reference interpreter for resolved plans, for
//! differential tests of the optimizer: plain nested loops, SQL
//! three-valued logic, exact fixed-point numbers. Inner and cross join
//! trees are enumerated by backtracking, checking each predicate as soon as
//! the columns it reads are bound, so unoptimised comma joins stay
//! tractable on small data. Unsupported constructs panic.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_plan::{
    AttrId, Builtin, CastMode, Expr, ExprNode, FunctionId, FunctionKind, Grouping, JoinCondition,
    JoinKind, Literal, LogicalPlan, NullOrdering, PlanNode, ReadSource, ResolvedExpr,
    SortDirection, SubqueryKind,
};
use krishiv_types::DataType;

/// Fixed-point scale of every number.
pub(crate) const SCALE: i128 = 1_000_000;

/// A value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum V {
    /// NULL.
    Null,
    /// A boolean.
    Bool(bool),
    /// A number, times [`SCALE`].
    Num(i128),
    /// A string.
    Str(Arc<str>),
    /// Days since 1970-01-01.
    Date(i32),
}

impl V {
    /// An integer.
    pub(crate) fn int(v: i64) -> Self {
        Self::Num(i128::from(v) * SCALE)
    }

    /// A decimal with two digits after the point (`cents / 100`).
    pub(crate) fn cents(cents: i64) -> Self {
        Self::Num(i128::from(cents) * (SCALE / 100))
    }

    fn truth(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            Self::Null => None,
            v => panic!("not a boolean: {v:?}"),
        }
    }
}

/// Tables by name: column names and rows.
#[derive(Debug, Default)]
pub(crate) struct Db {
    /// Name → (columns, rows).
    pub(crate) tables: HashMap<String, (Vec<String>, Vec<Vec<V>>)>,
}

/// A relation: its attributes and rows.
#[derive(Debug, Clone)]
pub(crate) struct Rel {
    /// Attribute per column.
    pub(crate) cols: Vec<AttrId>,
    /// The rows.
    pub(crate) rows: Vec<Vec<V>>,
}

type Env = HashMap<AttrId, V>;

/// The rows `plan` returns over `db`, sorted.
pub(crate) fn rows(db: &Db, plan: &LogicalPlan) -> Vec<Vec<V>> {
    let mut rows = run(db, plan).rows;
    rows.sort();
    rows
}

/// Runs `plan` over `db`.
pub(crate) fn run(db: &Db, plan: &LogicalPlan) -> Rel {
    Interp {
        db,
        ctes: HashMap::new(),
        cache: HashMap::new(),
    }
    .plan(plan, &Env::new())
}

struct Interp<'a> {
    db: &'a Db,
    ctes: HashMap<Vec<AttrId>, Rel>,
    /// Subquery results by plan address and the values of its outer
    /// references.
    cache: HashMap<(*const LogicalPlan, Vec<V>), Rel>,
}

/// Where each attribute sits in a row.
fn index(cols: &[AttrId]) -> HashMap<AttrId, usize> {
    cols.iter().enumerate().map(|(i, c)| (*c, i)).collect()
}

/// The column ids an expression reads: its columns, and the outer
/// references of its subquery plans.
fn column_refs(e: &Expr, out: &mut Vec<AttrId>) {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => out.push(c.id),
        ExprNode::Subquery { plan, .. } => outer_refs(plan, out),
        _ => {}
    }
    e.for_each_child(&mut |c| column_refs(c, out));
}

/// The outer references anywhere in `p`.
fn outer_refs(p: &LogicalPlan, out: &mut Vec<AttrId>) {
    let mut exprs: Vec<&Expr> = Vec::new();
    let mut plans = vec![p];
    while let Some(p) = plans.pop() {
        p.for_each_expr(&mut |e| exprs.push(e));
        p.for_each_child(&mut |c| plans.push(c));
        while let Some(e) = exprs.pop() {
            match &e.node {
                ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => out.push(c.id),
                ExprNode::Subquery { plan, .. } => plans.push(plan),
                _ => {}
            }
            e.for_each_child(&mut |c| exprs.push(c));
        }
    }
}

fn conjuncts(e: &Arc<Expr>, out: &mut Vec<Arc<Expr>>) {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.id == FunctionId::Builtin(Builtin::And) =>
        {
            f.args.iter().for_each(|a| conjuncts(a, out));
        }
        _ => out.push(Arc::clone(e)),
    }
}

impl Interp<'_> {
    fn plan(&mut self, p: &LogicalPlan, outer: &Env) -> Rel {
        match &p.node {
            PlanNode::Scan(s) => {
                let ReadSource::Table { name, .. } = &s.source else {
                    panic!("files")
                };
                let table = name.last().unwrap();
                let (cols, rows) = &self.db.tables[table];
                let pick: Vec<usize> = s
                    .output
                    .iter()
                    .map(|c| cols.iter().position(|n| *n == c.name).unwrap())
                    .collect();
                Rel {
                    cols: s.output.iter().map(|c| c.id).collect(),
                    rows: rows
                        .iter()
                        .map(|r| pick.iter().map(|i| r[*i].clone()).collect())
                        .collect(),
                }
            }
            PlanNode::Project { input, exprs } => {
                let rel = self.plan(input, outer);
                let ix = index(&rel.cols);
                let rows = rel
                    .rows
                    .iter()
                    .map(|r| exprs.iter().map(|e| self.eval(e, &ix, r, outer)).collect())
                    .collect();
                Rel {
                    cols: exprs.iter().map(|e| output(e)).collect(),
                    rows,
                }
            }
            PlanNode::Expand {
                input,
                projections,
                output,
            } => {
                let rel = self.plan(input, outer);
                let ix = index(&rel.cols);
                let rows = rel
                    .rows
                    .iter()
                    .flat_map(|r| {
                        projections
                            .iter()
                            .map(|row| row.iter().map(|e| self.eval(e, &ix, r, outer)).collect())
                            .collect::<Vec<_>>()
                    })
                    .collect();
                Rel {
                    cols: output.iter().map(|c| c.id).collect(),
                    rows,
                }
            }
            PlanNode::Filter { input, predicate } => {
                let mut preds = Vec::new();
                conjuncts(predicate, &mut preds);
                if let PlanNode::Join {
                    kind: JoinKind::Inner | JoinKind::Cross,
                    ..
                } = &input.node
                {
                    return self.inner_joins(input, preds, outer);
                }
                let rel = self.plan(input, outer);
                let ix = index(&rel.cols);
                let rows = rel
                    .rows
                    .into_iter()
                    .filter(|r| self.eval(predicate, &ix, r, outer) == V::Bool(true))
                    .collect();
                Rel {
                    cols: rel.cols,
                    rows,
                }
            }
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => {
                let Grouping::GroupBy(keys) = grouping else {
                    panic!("grouping sets")
                };
                let rel = self.plan(input, outer);
                let ix = index(&rel.cols);
                let mut groups: Vec<Vec<Vec<V>>> = Vec::new();
                let mut at: HashMap<Vec<V>, usize> = HashMap::new();
                for r in rel.rows {
                    let k: Vec<V> = keys.iter().map(|e| self.eval(e, &ix, &r, outer)).collect();
                    let g = *at.entry(k).or_insert_with(|| {
                        groups.push(Vec::new());
                        groups.len() - 1
                    });
                    groups[g].push(r);
                }
                if keys.is_empty() && groups.is_empty() {
                    groups.push(Vec::new());
                }
                let rows = groups
                    .iter()
                    .map(|g| {
                        aggregates
                            .iter()
                            .map(|e| self.aggregate(e, &ix, g, outer))
                            .collect()
                    })
                    .collect();
                Rel {
                    cols: aggregates.iter().map(|e| output(e)).collect(),
                    rows,
                }
            }
            PlanNode::Join { .. } => match &p.node {
                PlanNode::Join {
                    kind: JoinKind::Inner | JoinKind::Cross,
                    ..
                } => self.inner_joins(p, Vec::new(), outer),
                _ => self.outer_join(p, outer),
            },
            PlanNode::Sort { input, order, .. } => {
                let rel = self.plan(input, outer);
                let ix = index(&rel.cols);
                let mut keyed: Vec<(Vec<V>, Vec<V>)> = rel
                    .rows
                    .into_iter()
                    .map(|r| {
                        let k = order.iter().map(|o| self.eval(&o.expr, &ix, &r, outer));
                        (k.collect(), r)
                    })
                    .collect();
                keyed.sort_by(|(a, _), (b, _)| {
                    for ((x, y), o) in a.iter().zip(b).zip(order) {
                        let c = match (x, y) {
                            (V::Null, V::Null) => std::cmp::Ordering::Equal,
                            (V::Null, _) => match o.nulls {
                                NullOrdering::NullsFirst => std::cmp::Ordering::Less,
                                NullOrdering::NullsLast => std::cmp::Ordering::Greater,
                            },
                            (_, V::Null) => match o.nulls {
                                NullOrdering::NullsFirst => std::cmp::Ordering::Greater,
                                NullOrdering::NullsLast => std::cmp::Ordering::Less,
                            },
                            _ if o.direction == SortDirection::Descending => y.cmp(x),
                            _ => x.cmp(y),
                        };
                        if c.is_ne() {
                            return c;
                        }
                    }
                    std::cmp::Ordering::Equal
                });
                Rel {
                    cols: rel.cols,
                    rows: keyed.into_iter().map(|(_, r)| r).collect(),
                }
            }
            PlanNode::Limit { input, limit } => {
                let mut rel = self.plan(input, outer);
                let V::Num(n) = self.eval(limit, &HashMap::new(), &[], outer) else {
                    panic!("limit")
                };
                rel.rows.truncate(usize::try_from(n / SCALE).unwrap());
                rel
            }
            PlanNode::Offset { input, offset } => {
                let mut rel = self.plan(input, outer);
                let V::Num(n) = self.eval(offset, &HashMap::new(), &[], outer) else {
                    panic!("offset")
                };
                let skip = usize::try_from(n / SCALE).unwrap().min(rel.rows.len());
                rel.rows.drain(..skip);
                rel
            }
            PlanNode::SubqueryAlias { input, .. } => self.plan(input, outer),
            PlanNode::WithCte { input, ctes } => {
                for c in ctes {
                    let rel = self.plan(&c.plan, outer);
                    self.ctes.insert(rel.cols.clone(), rel);
                }
                self.plan(input, outer)
            }
            PlanNode::CteRef(r) => {
                let def = &self.ctes[&r.source];
                let pick: Vec<usize> = r
                    .source
                    .iter()
                    .map(|s| def.cols.iter().position(|c| c == s).unwrap())
                    .collect();
                Rel {
                    cols: r.output.iter().map(|c| c.id).collect(),
                    rows: def
                        .rows
                        .iter()
                        .map(|row| pick.iter().map(|i| row[*i].clone()).collect())
                        .collect(),
                }
            }
            PlanNode::Shift { input, offset } => {
                let mut rel = self.plan(input, outer);
                rel.cols = rel.cols.iter().map(|c| AttrId(c.0 + offset)).collect();
                rel
            }
            other => panic!("unsupported relation {other:?}"),
        }
    }

    /// An inner/cross join tree under `extra` predicates, by backtracking.
    fn inner_joins(&mut self, p: &LogicalPlan, extra: Vec<Arc<Expr>>, outer: &Env) -> Rel {
        let mut leaves = Vec::new();
        let mut preds = extra;
        let mut stack = vec![p];
        while let Some(n) = stack.pop() {
            match &n.node {
                PlanNode::Join {
                    left,
                    right,
                    kind: JoinKind::Inner | JoinKind::Cross,
                    condition,
                } => {
                    if let JoinCondition::On(c) = condition {
                        conjuncts(c, &mut preds);
                    }
                    stack.push(right);
                    stack.push(left);
                }
                _ => leaves.push(self.plan(n, outer)),
            }
        }
        let mut cols = Vec::new();
        let mut at_level: Vec<Vec<Arc<Expr>>> = vec![Vec::new(); leaves.len()];
        let mut bound_after = Vec::new();
        for l in &leaves {
            cols.extend(l.cols.iter().copied());
            bound_after.push(cols.clone());
        }
        for pred in preds {
            let mut refs = Vec::new();
            column_refs(&pred, &mut refs);
            let level = bound_after
                .iter()
                .position(|b| refs.iter().all(|r| b.contains(r) || !cols.contains(r)))
                .unwrap();
            at_level[level].push(pred);
        }
        let ixs: Vec<_> = bound_after.iter().map(|b| index(b)).collect();
        let mut rows = Vec::new();
        let mut row = Vec::new();
        self.enumerate(0, &leaves, &at_level, &ixs, &mut row, &mut rows, outer);
        Rel { cols, rows }
    }

    #[expect(clippy::too_many_arguments, reason = "test recursion state")]
    fn enumerate(
        &mut self,
        level: usize,
        leaves: &[Rel],
        preds: &[Vec<Arc<Expr>>],
        ixs: &[HashMap<AttrId, usize>],
        row: &mut Vec<V>,
        out: &mut Vec<Vec<V>>,
        outer: &Env,
    ) {
        if level == leaves.len() {
            out.push(row.clone());
            return;
        }
        for r in &leaves[level].rows {
            let len = row.len();
            row.extend(r.iter().cloned());
            if preds[level]
                .iter()
                .all(|p| self.eval(p, &ixs[level], row, outer) == V::Bool(true))
            {
                self.enumerate(level + 1, leaves, preds, ixs, row, out, outer);
            }
            row.truncate(len);
        }
    }

    fn outer_join(&mut self, p: &LogicalPlan, outer: &Env) -> Rel {
        let PlanNode::Join {
            left,
            right,
            kind,
            condition,
        } = &p.node
        else {
            unreachable!()
        };
        let (l, r) = (self.plan(left, outer), self.plan(right, outer));
        let cols: Vec<AttrId> = l.cols.iter().chain(&r.cols).copied().collect();
        let ix = index(&cols);
        let nulls = |n: usize| vec![V::Null; n];
        let mut rows = Vec::new();
        let mut right_hit = vec![false; r.rows.len()];
        for lr in &l.rows {
            let mut hit = false;
            for (j, rr) in r.rows.iter().enumerate() {
                let row: Vec<V> = lr.iter().chain(rr).cloned().collect();
                let ok = match condition {
                    JoinCondition::On(c) => self.eval(c, &ix, &row, outer) == V::Bool(true),
                    _ => true,
                };
                if ok {
                    hit = true;
                    right_hit[j] = true;
                    if !matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
                        rows.push(row);
                    }
                }
            }
            match kind {
                JoinKind::LeftSemi if hit => rows.push(lr.clone()),
                JoinKind::LeftAnti if !hit => rows.push(lr.clone()),
                JoinKind::LeftOuter | JoinKind::FullOuter if !hit => {
                    rows.push(lr.iter().cloned().chain(nulls(r.cols.len())).collect());
                }
                _ => {}
            }
        }
        if matches!(kind, JoinKind::RightOuter | JoinKind::FullOuter) {
            for (j, rr) in r.rows.iter().enumerate() {
                if !right_hit[j] {
                    rows.push(
                        nulls(l.cols.len())
                            .into_iter()
                            .chain(rr.iter().cloned())
                            .collect(),
                    );
                }
            }
        }
        let cols = if matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
            l.cols
        } else {
            cols
        };
        Rel { cols, rows }
    }

    /// An aggregate output entry over one group's rows.
    fn aggregate(&mut self, e: &Expr, ix: &HashMap<AttrId, usize>, g: &[Vec<V>], outer: &Env) -> V {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
                let mut vals: Vec<V> = Vec::new();
                for r in g {
                    if let Some(c) = &f.filter
                        && self.eval(c, ix, r, outer) != V::Bool(true)
                    {
                        continue;
                    }
                    let v = match f.args.first() {
                        Some(a) => self.eval(a, ix, r, outer),
                        None => V::Bool(true),
                    };
                    if v != V::Null {
                        vals.push(v);
                    }
                }
                if f.distinct {
                    vals.sort();
                    vals.dedup();
                }
                let nums = || {
                    vals.iter().map(|v| match v {
                        V::Num(n) => *n,
                        v => panic!("not a number: {v:?}"),
                    })
                };
                match f.id.name() {
                    "count" => V::int(i64::try_from(vals.len()).unwrap()),
                    _ if vals.is_empty() => V::Null,
                    "sum" => V::Num(nums().sum()),
                    "avg" => V::Num(nums().sum::<i128>() / i128::try_from(vals.len()).unwrap()),
                    "min" => vals.iter().min().unwrap().clone(),
                    "max" => vals.iter().max().unwrap().clone(),
                    "any_value" | "nondeterministic_first" => vals[0].clone(),
                    other => panic!("aggregate {other}"),
                }
            }
            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => {
                self.aggregate(expr, ix, g, outer)
            }
            ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                let args: Vec<V> = f
                    .args
                    .iter()
                    .map(|a| self.aggregate(a, ix, g, outer))
                    .collect();
                apply(f.id.name(), &args)
            }
            ExprNode::Cast { expr, to, mode } => {
                cast(self.aggregate(expr, ix, g, outer), to, *mode)
            }
            _ => match g.first() {
                Some(r) => self.eval(e, ix, r, outer),
                None => self.eval(e, &HashMap::new(), &[], outer),
            },
        }
    }

    fn eval(&mut self, e: &Expr, ix: &HashMap<AttrId, usize>, row: &[V], outer: &Env) -> V {
        match &e.node {
            ExprNode::Literal(l) => literal(l),
            ExprNode::Resolved(ResolvedExpr::Column(c)) => match ix.get(&c.id) {
                Some(i) => row[*i].clone(),
                None => outer
                    .get(&c.id)
                    .cloned()
                    .unwrap_or_else(|| panic!("unbound {c:?}")),
            },
            ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => outer
                .get(&c.id)
                .cloned()
                .unwrap_or_else(|| panic!("unbound outer {c:?}")),
            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => self.eval(expr, ix, row, outer),
            ExprNode::Cast { expr, to, mode } => cast(self.eval(expr, ix, row, outer), to, *mode),
            ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                let name = f.id.name();
                let short = |s: &mut Self, i: usize| s.eval(&f.args[i], ix, row, outer).truth();
                match name {
                    "and" => match short(self, 0) {
                        Some(false) => V::Bool(false),
                        a => match (a, short(self, 1)) {
                            (_, Some(false)) => V::Bool(false),
                            (Some(true), Some(true)) => V::Bool(true),
                            _ => V::Null,
                        },
                    },
                    "or" => match short(self, 0) {
                        Some(true) => V::Bool(true),
                        a => match (a, short(self, 1)) {
                            (_, Some(true)) => V::Bool(true),
                            (Some(false), Some(false)) => V::Bool(false),
                            _ => V::Null,
                        },
                    },
                    _ => {
                        let args: Vec<V> = f
                            .args
                            .iter()
                            .map(|a| self.eval(a, ix, row, outer))
                            .collect();
                        apply(name, &args)
                    }
                }
            }
            ExprNode::Subquery { kind, plan } => {
                let mut env = outer.clone();
                for (id, i) in ix {
                    env.insert(*id, row[*i].clone());
                }
                let mut refs = Vec::new();
                outer_refs(plan, &mut refs);
                refs.sort();
                refs.dedup();
                let values = refs
                    .iter()
                    .map(|id| env.get(id).cloned().unwrap_or(V::Null));
                let key = (std::ptr::from_ref(&**plan), values.collect());
                if !self.cache.contains_key(&key) {
                    let rel = self.plan(plan, &env);
                    self.cache.insert(key.clone(), rel);
                }
                let rel = self.cache[&key].clone();
                match kind {
                    SubqueryKind::Scalar => {
                        assert!(rel.rows.len() <= 1, "scalar subquery returned several rows");
                        rel.rows.first().map_or(V::Null, |r| r[0].clone())
                    }
                    SubqueryKind::Exists => V::Bool(!rel.rows.is_empty()),
                    SubqueryKind::In(values) => {
                        let v: Vec<V> = values
                            .iter()
                            .map(|x| self.eval(x, ix, row, outer))
                            .collect();
                        let mut result = V::Bool(false);
                        for r in &rel.rows {
                            let eq: Vec<V> = v
                                .iter()
                                .zip(r)
                                .map(|(a, b)| apply("==", &[a.clone(), b.clone()]))
                                .collect();
                            if eq.iter().all(|x| *x == V::Bool(true)) {
                                return V::Bool(true);
                            }
                            if eq.iter().all(|x| *x != V::Bool(false)) {
                                result = V::Null;
                            }
                        }
                        result
                    }
                    _ => panic!("subquery kind"),
                }
            }
            other => panic!("unsupported expression {other:?}"),
        }
    }
}

fn output(e: &Expr) -> AttrId {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::Alias { output: c, .. }) => c.id,
        other => panic!("not an output: {other:?}"),
    }
}

fn literal(l: &Literal) -> V {
    match l {
        Literal::Null(_) => V::Null,
        Literal::Boolean(b) => V::Bool(*b),
        Literal::Byte(v) => V::int(i64::from(*v)),
        Literal::Short(v) => V::int(i64::from(*v)),
        Literal::Integer(v) => V::int(i64::from(*v)),
        Literal::Long(v) => V::int(*v),
        Literal::Decimal {
            unscaled,
            data_type,
        } => {
            let s = u32::from(data_type.scale());
            V::Num(if s <= 6 {
                unscaled * 10_i128.pow(6 - s)
            } else {
                unscaled / 10_i128.pow(s - 6)
            })
        }
        Literal::String(s) => V::Str(s.as_str().into()),
        Literal::Date(d) => V::Date(*d),
        other => panic!("literal {other:?}"),
    }
}

/// Days since 1970-01-01 of a civil date.
pub(crate) fn days(y: i64, m: i64, d: i64) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    i32::try_from(era * 146_097 + doe - 719_468).unwrap()
}

fn year(days: i32) -> i64 {
    let z = i64::from(days) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    yoe + era * 400 + i64::from(m <= 2)
}

fn cast(v: V, to: &DataType, _mode: CastMode) -> V {
    match (v, to) {
        (V::Null, _) => V::Null,
        (V::Str(s), DataType::Date) => {
            let p: Vec<i64> = s.split('-').map(|x| x.trim().parse().unwrap()).collect();
            V::Date(days(p[0], p[1], p[2]))
        }
        (V::Str(s), DataType::Integer | DataType::Long | DataType::Decimal(_)) => {
            // Invalid input is NULL here; ANSI's error is not modelled.
            let (int, frac) = s.trim().split_once('.').unwrap_or((s.trim(), ""));
            let digits = format!("{frac:0<6}");
            match (int.parse::<i128>(), digits.get(..6).map(str::parse::<i128>)) {
                (Ok(i), Some(Ok(f))) if frac.len() <= 6 => {
                    let n = if to == &DataType::Integer || to == &DataType::Long {
                        0
                    } else {
                        f
                    };
                    V::Num(i * SCALE + if s.trim().starts_with('-') { -n } else { n })
                }
                _ => V::Null,
            }
        }
        (V::Num(n), DataType::Integer | DataType::Long | DataType::Short | DataType::Byte) => {
            V::Num(n / SCALE * SCALE)
        }
        (V::Num(n), DataType::String) => V::Str(format!("{}", n / SCALE).into()),
        (v @ V::Num(_), DataType::Double | DataType::Float | DataType::Decimal(_))
        | (v @ V::Date(_), DataType::Date)
        | (v @ V::Str(_), DataType::String) => v,
        (v, t) => panic!("cast {v:?} to {t:?}"),
    }
}

fn like(s: &str, p: &str) -> bool {
    fn go(s: &[char], p: &[char]) -> bool {
        match p.split_first() {
            None => s.is_empty(),
            Some(('%', rest)) => (0..=s.len()).any(|i| go(&s[i..], rest)),
            Some(('_', rest)) => !s.is_empty() && go(&s[1..], rest),
            Some((c, rest)) => s.first() == Some(c) && go(&s[1..], rest),
        }
    }
    let (s, p): (Vec<char>, Vec<char>) = (s.chars().collect(), p.chars().collect());
    go(&s, &p)
}

/// A scalar function over evaluated arguments (`and`/`or` are handled by
/// the caller, which short-circuits).
fn apply(name: &str, a: &[V]) -> V {
    use std::cmp::Ordering;
    let cmp = |test: fn(Ordering) -> bool| match (&a[0], &a[1]) {
        (V::Null, _) | (_, V::Null) => V::Null,
        (x, y) => V::Bool(test(x.cmp(y))),
    };
    let num = |f: fn(i128, i128) -> Option<i128>| match (&a[0], &a[1]) {
        (V::Num(x), V::Num(y)) => f(*x, *y).map_or(V::Null, V::Num),
        (V::Null, _) | (_, V::Null) => V::Null,
        (x, y) => panic!("{name}({x:?}, {y:?})"),
    };
    match name {
        "==" => cmp(Ordering::is_eq),
        "<" => cmp(Ordering::is_lt),
        "<=" => cmp(Ordering::is_le),
        ">" => cmp(Ordering::is_gt),
        ">=" => cmp(Ordering::is_ge),
        "<=>" => V::Bool(a[0] == a[1]),
        "and" | "or" => unreachable!("short-circuited"),
        "not" => match &a[0] {
            V::Bool(b) => V::Bool(!b),
            _ => V::Null,
        },
        "+" => num(|x, y| Some(x + y)),
        "-" => num(|x, y| Some(x - y)),
        "*" => num(|x, y| Some(x * y / SCALE)),
        "/" => num(|x, y| (y != 0).then(|| x * SCALE / y)),
        "%" => num(|x, y| (y != 0).then(|| x % y)),
        // Integral division: the quotient truncated to a whole number.
        "div" => num(|x, y| (y != 0).then(|| x / y * SCALE)),
        "negative" => match &a[0] {
            V::Num(x) => V::Num(-x),
            _ => V::Null,
        },
        "isnull" => V::Bool(a[0] == V::Null),
        "isnotnull" => V::Bool(a[0] != V::Null),
        "in" => {
            if a[0] == V::Null {
                return V::Null;
            }
            let mut out = V::Bool(false);
            for x in &a[1..] {
                match apply("==", &[a[0].clone(), x.clone()]) {
                    V::Bool(true) => return V::Bool(true),
                    V::Null => out = V::Null,
                    _ => {}
                }
            }
            out
        }
        "between" => {
            let ge = apply(">=", &a[..2]);
            let le = apply("<=", &[a[0].clone(), a[2].clone()]);
            match (ge, le) {
                (V::Bool(false), _) | (_, V::Bool(false)) => V::Bool(false),
                (V::Bool(true), V::Bool(true)) => V::Bool(true),
                _ => V::Null,
            }
        }
        "like" => match (&a[0], &a[1]) {
            (V::Str(s), V::Str(p)) => V::Bool(like(s, p)),
            _ => V::Null,
        },
        "when" => {
            for pair in a.chunks(2) {
                match pair {
                    [c, v] if *c == V::Bool(true) => return v.clone(),
                    [otherwise] => return otherwise.clone(),
                    _ => {}
                }
            }
            V::Null
        }
        "concat" => {
            if a.contains(&V::Null) {
                return V::Null;
            }
            V::Str(
                a.iter()
                    .map(|v| match v {
                        V::Str(s) => s.to_string(),
                        v => panic!("concat {v:?}"),
                    })
                    .collect::<String>()
                    .into(),
            )
        }
        "coalesce" => a
            .iter()
            .find(|v| **v != V::Null)
            .cloned()
            .unwrap_or(V::Null),
        "upper" => match &a[0] {
            V::Str(s) => V::Str(s.to_uppercase().into()),
            _ => V::Null,
        },
        "extract" => match (&a[0], &a[1]) {
            (V::Str(f), V::Date(d)) if f.eq_ignore_ascii_case("year") => V::int(year(*d)),
            (_, V::Null) => V::Null,
            (f, d) => panic!("extract({f:?}, {d:?})"),
        },
        "substring" => match (&a[0], &a[1], &a[2]) {
            (V::Str(s), V::Num(p), V::Num(l)) => {
                let start = usize::try_from((p / SCALE).max(1) - 1).unwrap();
                let len = usize::try_from((l / SCALE).max(0)).unwrap();
                V::Str(s.chars().skip(start).take(len).collect::<String>().into())
            }
            _ => V::Null,
        },
        other => panic!("function {other}"),
    }
}
