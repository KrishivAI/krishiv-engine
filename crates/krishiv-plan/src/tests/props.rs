// SPDX-License-Identifier: Apache-2.0
//! Property tests of the canonical encoding and the structural hash (owner
//! decision 4): equal for equal plans, different after any single-field
//! mutation, unaffected by origins and consistent `plan_id` renumbering.
//!
//! Random plans are built from a byte string ([`Gen`]), so the same bytes
//! build the same plan twice, and with `remap` the same plan with every
//! origin changed and every `plan_id` renumbered.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::secret::{OptionValue, SecretString, SecretValue};
use krishiv_types::{DataType, DecimalType};
use proptest::prelude::*;
use proptest::sample::Index;

use crate::{
    AttrId, BoundColumn, Builtin, CastMode, ColumnRef, Cte, CteRef, DayTimeField, Expr, ExprNode,
    FrameBound, FrameUnits, FunctionCall, FunctionId, FunctionKind, Grouping, JoinCondition,
    JoinKind, JoinStrategy, Literal, LogicalPlan, NullOrdering, Origin, Plan, PlanId, PlanNode,
    ReadSource, ResolvedExpr, ResolvedFunction, Scan, SetOperation, SortDirection, SortOrder,
    SourceId, Span, SqlArgs, Star, SubqueryKind, TableVersion, WindowExpr, WindowFrame, WindowSpec,
    WindowSpecRef, YearMonthField,
};

// ---- generator ------------------------------------------------------------

/// Builds plans from bytes; past the end every choice is 0.
struct Gen<'a> {
    bytes: &'a [u8],
    pos: usize,
    remap: bool,
}

/// The bijection applied to `plan_id`s under `remap`.
fn renumber(id: PlanId) -> PlanId {
    PlanId(100 - id.0)
}

fn remap(o: Origin) -> Origin {
    let mut out = match o.sql {
        Some((source, s)) => Origin::sql(
            SourceId(source.0 + 7),
            Span::new(s.start + 1000, s.end + 2000),
        ),
        None => Origin::sql(SourceId(9), Span::new(1, 2)),
    };
    out.plan_id = o.plan_id.map(renumber);
    out
}

impl Gen<'_> {
    fn pick(&mut self, n: u8) -> u8 {
        let b = self.bytes.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b % n
    }

    fn flag(&mut self) -> bool {
        self.pick(2) == 1
    }

    fn many<T>(&mut self, min: u8, max: u8, mut f: impl FnMut(&mut Self) -> T) -> Vec<T> {
        let n = min + self.pick(max - min + 1);
        (0..n).map(|_| f(self)).collect()
    }

    fn origin(&mut self) -> Origin {
        let mut o = Origin::UNKNOWN;
        if self.flag() {
            o.plan_id = Some(PlanId(i64::from(self.pick(6))));
        }
        if self.flag() {
            let start = u32::from(self.pick(50));
            o.sql = Some((
                SourceId(u32::from(self.pick(3))),
                Span::new(start, start + u32::from(self.pick(50))),
            ));
        }
        if self.remap { remap(o) } else { o }
    }

    fn plan_ref(&mut self) -> Option<PlanId> {
        let id = self.flag().then(|| PlanId(i64::from(self.pick(8))));
        if self.remap { id.map(renumber) } else { id }
    }

    fn name(&mut self) -> String {
        ["a", "b", "c"][usize::from(self.pick(3))].to_owned()
    }

    fn data_type(&mut self) -> DataType {
        match self.pick(6) {
            0 => DataType::Integer,
            1 => DataType::Long,
            2 => DataType::String,
            3 => DataType::Date,
            4 => DataType::TimestampNtz,
            _ => {
                let p = u32::from(1 + self.pick(38));
                DataType::Decimal(DecimalType::try_new(p, p / 2).unwrap())
            }
        }
    }

    fn literal(&mut self) -> Literal {
        let v = self.pick(255);
        let ym = [YearMonthField::Year, YearMonthField::Month];
        let dt = [
            DayTimeField::Day,
            DayTimeField::Hour,
            DayTimeField::Minute,
            DayTimeField::Second,
        ];
        match self.pick(16) {
            0 => Literal::Null(self.flag().then(|| self.data_type())),
            1 => Literal::Boolean(v.is_multiple_of(2)),
            2 => Literal::Byte(i8::from_le_bytes([v])),
            3 => Literal::Short(i16::from(v) - 100),
            4 => Literal::Integer(i32::from(v) - 100),
            5 => Literal::Long(i64::from(v) << 40),
            6 => Literal::Float([f32::NAN, 0.0, -0.0, 1.5][usize::from(v % 4)]),
            7 => Literal::Double([f64::NAN, 0.0, -0.0, 1.5][usize::from(v % 4)]),
            8 => Literal::Decimal {
                unscaled: i128::from(v) - 100,
                data_type: DecimalType::try_new(10, u32::from(v % 3)).unwrap(),
            },
            9 => Literal::String("ab".chars().cycle().take(usize::from(v % 4)).collect()),
            10 => Literal::Binary(vec![v; usize::from(v % 3)]),
            11 => Literal::Date(i32::from(v)),
            12 => Literal::Timestamp(i64::from(v)),
            13 => Literal::TimestampNtz(i64::from(v)),
            14 => {
                let start = ym[usize::from(self.pick(2))];
                Literal::YearMonthInterval {
                    months: i32::from(v),
                    start,
                    end: start.max(ym[usize::from(self.pick(2))]),
                }
            }
            _ => {
                let start = dt[usize::from(self.pick(4))];
                Literal::DayTimeInterval {
                    micros: i64::from(v),
                    start,
                    end: start.max(dt[usize::from(self.pick(4))]),
                }
            }
        }
    }

    fn bound(&mut self) -> BoundColumn {
        BoundColumn {
            id: AttrId(u64::from(self.pick(4))),
            name: self.name(),
            data_type: self.data_type(),
            nullable: self.flag(),
        }
    }

    fn args(&mut self, depth: u8) -> (Vec<Arc<Expr>>, bool, Option<Arc<Expr>>) {
        (
            self.many(0, 2, |g| Arc::new(g.expr(depth))),
            self.flag(),
            self.flag().then(|| Arc::new(self.expr(depth))),
        )
    }

    fn expr(&mut self, depth: u8) -> Expr {
        let origin = self.origin();
        let kinds = if depth == 0 { 5 } else { 12 };
        let d = depth.saturating_sub(1);
        let node = match self.pick(kinds) {
            0 => ExprNode::Column(ColumnRef {
                name: self.many(1, 2, Gen::name),
                plan_id: self.plan_ref(),
            }),
            1 => ExprNode::Literal(self.literal()),
            2 => ExprNode::Star(Star {
                qualifier: self.many(0, 1, Gen::name),
                except: Vec::new(),
                plan_id: self.plan_ref(),
            }),
            3 => ExprNode::Resolved(ResolvedExpr::Column(self.bound())),
            4 => ExprNode::Resolved(ResolvedExpr::OuterColumn(self.bound())),
            5 => {
                let name = ["+", "sum", "and", "in"][usize::from(self.pick(4))].to_owned();
                let (args, distinct, filter) = self.args(d);
                ExprNode::Function(FunctionCall {
                    name,
                    args,
                    distinct,
                    filter,
                })
            }
            6 => ExprNode::Cast {
                expr: Arc::new(self.expr(d)),
                to: self.data_type(),
                mode: [
                    CastMode::Default,
                    CastMode::Legacy,
                    CastMode::Ansi,
                    CastMode::Try,
                ][usize::from(self.pick(4))],
            },
            7 => ExprNode::Alias {
                expr: Arc::new(self.expr(d)),
                name: self.name(),
            },
            8 => {
                let id = if self.flag() {
                    FunctionId::Builtin(Builtin::ALL[usize::from(self.pick(25))])
                } else {
                    FunctionId::Registry {
                        name: self.name(),
                        id: u32::from(self.pick(9)),
                    }
                };
                let kind = [
                    FunctionKind::Scalar,
                    FunctionKind::Aggregate,
                    FunctionKind::Window,
                    FunctionKind::Generator,
                ][usize::from(self.pick(4))];
                let (args, distinct, filter) = self.args(d);
                ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                    id,
                    kind,
                    args,
                    distinct,
                    filter,
                    data_type: self.data_type(),
                    nullable: self.flag(),
                    deterministic: self.flag(),
                }))
            }
            9 => ExprNode::Resolved(ResolvedExpr::Alias {
                expr: Arc::new(self.expr(d)),
                output: self.bound(),
            }),
            11 => ExprNode::Window(WindowExpr {
                function: Arc::new(self.expr(d)),
                window: if self.flag() {
                    WindowSpecRef::Named(self.name())
                } else {
                    WindowSpecRef::Spec(WindowSpec {
                        partition_by: self.many(0, 2, |g| Arc::new(g.expr(d))),
                        order_by: self.orders(d),
                        frame: self.flag().then(|| WindowFrame {
                            units: if self.flag() {
                                FrameUnits::Rows
                            } else {
                                FrameUnits::Range
                            },
                            lower: self.frame_bound(d),
                            upper: self.frame_bound(d),
                        }),
                    })
                },
            }),
            _ => ExprNode::Subquery {
                kind: match self.pick(3) {
                    0 => SubqueryKind::Scalar,
                    1 => SubqueryKind::Exists,
                    _ => SubqueryKind::In(self.many(1, 2, |g| Arc::new(g.expr(d)))),
                },
                plan: Arc::new(self.plan(1)),
            },
        };
        Expr { node, origin }
    }

    fn orders(&mut self, depth: u8) -> Vec<SortOrder> {
        self.many(0, 2, |g| SortOrder {
            expr: Arc::new(g.expr(depth)),
            direction: if g.flag() {
                SortDirection::Ascending
            } else {
                SortDirection::Descending
            },
            nulls: if g.flag() {
                NullOrdering::NullsFirst
            } else {
                NullOrdering::NullsLast
            },
        })
    }

    fn frame_bound(&mut self, depth: u8) -> FrameBound {
        match self.pick(4) {
            0 => FrameBound::UnboundedPreceding,
            1 => FrameBound::CurrentRow,
            2 => FrameBound::UnboundedFollowing,
            _ => FrameBound::Offset(Arc::new(self.expr(depth))),
        }
    }

    fn exprs(&mut self) -> Vec<Arc<Expr>> {
        self.many(0, 2, |g| Arc::new(g.expr(2)))
    }

    fn source(&mut self) -> ReadSource {
        let options = self
            .many(0, 1, |g| (g.name(), g.option()))
            .into_iter()
            .collect();
        if self.flag() {
            ReadSource::Table {
                name: self.many(1, 2, Gen::name),
                options,
            }
        } else {
            ReadSource::Files {
                format: self.name(),
                paths: self.many(1, 2, Gen::option),
                options,
            }
        }
    }

    fn option(&mut self) -> OptionValue {
        let v = self.name();
        match self.pick(4) {
            0 => OptionValue::Plain(v),
            1 => OptionValue::Secret(SecretValue::Inline(SecretString::new(v))),
            2 => OptionValue::reference(self.name(), v),
            _ => OptionValue::from_user("url", format!("https://u:{v}@h/?sig={}", self.name())),
        }
    }

    fn input(&mut self, depth: u8) -> Arc<LogicalPlan> {
        Arc::new(self.plan(depth))
    }

    fn plan(&mut self, depth: u8) -> LogicalPlan {
        let origin = self.origin();
        let kinds = if depth == 0 { 4 } else { 24 };
        let d = depth.saturating_sub(1);
        let node = match self.pick(kinds) {
            0 => PlanNode::Read(self.source()),
            1 => PlanNode::Sql {
                query: self.name(),
                args: match self.pick(3) {
                    0 => SqlArgs::None,
                    1 => SqlArgs::Positional(self.many(1, 2, |g| Arc::new(g.expr(0)))),
                    _ => SqlArgs::Named(
                        self.many(1, 2, |g| (g.name(), Arc::new(g.expr(0))))
                            .into_iter()
                            .collect(),
                    ),
                },
            },
            2 => PlanNode::Scan(Scan {
                source: self.source(),
                output: self.many(0, 2, Gen::bound),
                version: self.flag().then(|| TableVersion(i64::from(self.pick(9)))),
            }),
            3 => PlanNode::CteRef(CteRef {
                name: self.name(),
                source: self.many(0, 2, |g| AttrId(u64::from(g.pick(9)))),
                output: self.many(0, 2, Gen::bound),
            }),
            14 => PlanNode::OneRowRelation,
            12 => PlanNode::Shift {
                input: self.input(d),
                offset: u64::from(self.pick(9)),
            },
            13 => PlanNode::Project {
                input: self.input(d),
                exprs: self.exprs(),
            },
            4 => PlanNode::WithColumns {
                input: self.input(d),
                columns: self.many(0, 2, |g| (g.name(), Arc::new(g.expr(2)))),
            },
            5 => PlanNode::Filter {
                input: self.input(d),
                predicate: Arc::new(self.expr(2)),
            },
            6 => PlanNode::Aggregate {
                input: self.input(d),
                grouping: match self.pick(4) {
                    0 => Grouping::GroupBy(self.exprs()),
                    1 => Grouping::Rollup(self.exprs()),
                    2 => Grouping::Cube(self.exprs()),
                    _ => Grouping::GroupingSets {
                        exprs: self.exprs(),
                        sets: self.many(0, 2, Gen::exprs),
                    },
                },
                aggregates: self.exprs(),
            },
            7 => PlanNode::Join {
                left: self.input(d),
                right: self.input(d),
                kind: [
                    JoinKind::Inner,
                    JoinKind::LeftOuter,
                    JoinKind::RightOuter,
                    JoinKind::FullOuter,
                    JoinKind::LeftSemi,
                    JoinKind::LeftAnti,
                    JoinKind::Cross,
                ][usize::from(self.pick(7))],
                condition: match self.pick(3) {
                    0 => JoinCondition::None,
                    1 => JoinCondition::On(Arc::new(self.expr(2))),
                    _ => JoinCondition::Using(self.many(1, 2, Gen::name)),
                },
            },
            8 => PlanNode::Sort {
                input: self.input(d),
                order: self.orders(2),
                global: self.flag(),
            },
            9 => PlanNode::Limit {
                input: self.input(d),
                limit: Arc::new(self.expr(1)),
            },
            10 => PlanNode::SubqueryAlias {
                input: self.input(d),
                alias: self.name(),
                columns: self.many(0, 1, Gen::name),
            },
            15 => PlanNode::Hint {
                input: self.input(d),
                name: self.name(),
                params: self.exprs(),
            },
            16 => PlanNode::JoinHint {
                input: self.input(d),
                strategy: [
                    JoinStrategy::Broadcast,
                    JoinStrategy::ShuffleMerge,
                    JoinStrategy::ShuffleHash,
                    JoinStrategy::ShuffleReplicateNl,
                ][usize::from(self.pick(4))],
            },
            17 => PlanNode::Union {
                inputs: self.many(1, 3, |g| g.input(d)),
                by_name: self.flag(),
                allow_missing_columns: self.flag(),
            },
            18 => PlanNode::Distinct {
                input: self.input(d),
            },
            19 => PlanNode::SetOp {
                left: self.input(d),
                right: self.input(d),
                op: if self.flag() {
                    SetOperation::Intersect
                } else {
                    SetOperation::Except
                },
                all: self.flag(),
            },
            20 => PlanNode::ReplicateRows {
                input: self.input(d),
                count: Arc::new(self.expr(1)),
            },
            21 => PlanNode::Offset {
                input: self.input(d),
                offset: Arc::new(self.expr(1)),
            },
            22 => PlanNode::Window {
                input: self.input(d),
                exprs: self.exprs(),
                partition_by: self.exprs(),
                order_by: self.orders(2),
            },
            23 => PlanNode::Expand {
                input: self.input(d),
                projections: self.many(1, 2, Gen::exprs),
                output: self.many(0, 2, Gen::bound),
            },
            _ => PlanNode::WithCte {
                input: self.input(d),
                ctes: self.many(1, 2, |g| Cte {
                    name: g.name(),
                    plan: g.input(d),
                }),
            },
        };
        LogicalPlan { node, origin }
    }
}

pub(super) fn build(bytes: &[u8], remap: bool) -> Plan {
    let root = Gen {
        bytes,
        pos: 0,
        remap,
    }
    .plan(3);
    Plan::try_new(root).unwrap()
}

// ---- single-field mutation ------------------------------------------------

/// Counts mutation sites down to the one to mutate (`k == 0`).
struct Walk {
    k: usize,
}

impl Walk {
    fn hit(&mut self) -> bool {
        if self.k == 0 {
            return true;
        }
        self.k -= 1;
        false
    }
}

/// One mutation site: when it is the chosen one, apply `$m` and return true.
macro_rules! site {
    ($w:ident, $m:expr) => {
        if $w.hit() {
            $m;
            return true;
        }
    };
}

fn other_type(t: &DataType) -> DataType {
    if *t == DataType::Integer {
        DataType::Long
    } else {
        DataType::Integer
    }
}

fn flip<T: PartialEq>(v: &mut T, a: T, b: T) {
    *v = if *v == a { b } else { a };
}

fn m_string(s: &mut String, w: &mut Walk) -> bool {
    site!(w, s.push('x'));
    false
}

/// Rewrites one element, or pushes one.
fn m_strings(v: &mut Vec<String>, w: &mut Walk) -> bool {
    site!(w, v.push("x".to_owned()));
    v.iter_mut().any(|s| m_string(s, w))
}

/// The same kind of option value with different contents.
fn other_value(v: &OptionValue) -> OptionValue {
    match v {
        OptionValue::Plain(s) => OptionValue::Plain(format!("{s}x")),
        OptionValue::Secret(SecretValue::Inline(s)) => OptionValue::Secret(SecretValue::Inline(
            SecretString::new(format!("{}x", s.expose_secret())),
        )),
        OptionValue::Secret(SecretValue::Reference { scope, key }) => {
            OptionValue::reference(scope.clone(), format!("{key}x"))
        }
        OptionValue::Url(u) => {
            OptionValue::from_user("url", format!("{}x", u.expose_secret().expose_secret()))
        }
    }
}

/// The value as another kind: plain becomes secret, anything else plain.
fn other_kind(v: &OptionValue) -> OptionValue {
    match v {
        OptionValue::Plain(s) => {
            OptionValue::Secret(SecretValue::Inline(SecretString::new(s.clone())))
        }
        OptionValue::Secret(SecretValue::Inline(s)) => {
            OptionValue::Plain(s.expose_secret().to_owned())
        }
        OptionValue::Secret(SecretValue::Reference { key, .. }) => OptionValue::Plain(key.clone()),
        OptionValue::Url(u) => OptionValue::Plain(u.expose_secret().expose_secret().to_owned()),
    }
}

/// Rewrites one value (contents or kind), or pushes one.
fn m_values(v: &mut Vec<OptionValue>, w: &mut Walk) -> bool {
    site!(w, v.push(OptionValue::Plain("x".to_owned())));
    for value in v.iter_mut() {
        site!(w, *value = other_value(value));
        site!(w, *value = other_kind(value));
    }
    false
}

fn m_options(m: &mut BTreeMap<String, OptionValue>, w: &mut Walk) -> bool {
    site!(
        w,
        m.insert("new".to_owned(), OptionValue::Plain("x".to_owned()))
    );
    let keys: Vec<String> = m.keys().cloned().collect();
    for k in keys {
        if let Some(v) = m.get_mut(&k) {
            site!(w, *v = other_value(v));
            site!(w, *v = other_kind(v));
        }
        site!(w, {
            let v = m.remove(&k).unwrap_or(OptionValue::Plain(String::new()));
            m.insert(format!("{k}x"), v);
        });
    }
    false
}

fn m_type(t: &mut DataType, w: &mut Walk) -> bool {
    site!(w, *t = other_type(t));
    false
}

fn m_bound(c: &mut BoundColumn, w: &mut Walk) -> bool {
    site!(w, c.id.0 += 1);
    site!(w, c.name.push('x'));
    site!(w, c.nullable = !c.nullable);
    m_type(&mut c.data_type, w)
}

fn m_literal(v: &mut Literal, w: &mut Walk) -> bool {
    site!(w, flip(v, Literal::Integer(7), Literal::Integer(8)));
    match v {
        Literal::Null(t) => site!(
            w,
            *t = if t.is_some() {
                None
            } else {
                Some(DataType::Long)
            }
        ),
        Literal::Boolean(b) => site!(w, *b = !*b),
        Literal::Byte(x) => site!(w, *x = x.wrapping_add(1)),
        Literal::Short(x) => site!(w, *x = x.wrapping_add(1)),
        Literal::Integer(x) | Literal::Date(x) => site!(w, *x = x.wrapping_add(1)),
        Literal::Long(x) | Literal::Timestamp(x) | Literal::TimestampNtz(x) => {
            site!(w, *x = x.wrapping_add(1));
        }
        Literal::Float(x) => site!(w, *x = f32::from_bits(x.to_bits() ^ 1)),
        Literal::Double(x) => site!(w, *x = f64::from_bits(x.to_bits() ^ 1)),
        Literal::Decimal {
            unscaled,
            data_type,
        } => {
            site!(w, *unscaled += 1);
            site!(w, *data_type = DecimalType::try_new(12, 3).unwrap());
        }
        Literal::String(s) => site!(w, s.push('x')),
        Literal::Binary(b) => site!(w, b.push(0)),
        Literal::YearMonthInterval { months, start, end } => {
            site!(w, *months += 1);
            site!(w, flip(start, YearMonthField::Year, YearMonthField::Month));
            site!(w, flip(end, YearMonthField::Year, YearMonthField::Month));
        }
        Literal::DayTimeInterval { micros, start, end } => {
            site!(w, *micros += 1);
            site!(w, flip(start, DayTimeField::Day, DayTimeField::Hour));
            site!(w, flip(end, DayTimeField::Second, DayTimeField::Minute));
        }
    }
    false
}

fn m_exprs(v: &mut Vec<Arc<Expr>>, w: &mut Walk) -> bool {
    site!(w, v.push(Arc::new(Literal::Integer(0).into())));
    v.iter_mut().any(|e| m_expr(e, w))
}

fn m_filter(filter: &mut Option<Arc<Expr>>, w: &mut Walk) -> bool {
    let toggled = match filter {
        Some(_) => None,
        None => Some(Arc::new(Literal::Boolean(true).into())),
    };
    site!(w, *filter = toggled);
    filter.as_mut().is_some_and(|e| m_expr(e, w))
}

fn m_expr(e: &mut Arc<Expr>, w: &mut Walk) -> bool {
    let e = Arc::make_mut(e);
    match &mut e.node {
        ExprNode::Column(ColumnRef { name, .. }) => m_strings(name, w),
        ExprNode::Literal(v) => m_literal(v, w),
        ExprNode::TypedLiteral { value, .. } => m_string(value, w),
        ExprNode::Star(Star {
            qualifier, except, ..
        }) => m_strings(qualifier, w) || except.iter_mut().any(|path| m_strings(path, w)),
        ExprNode::Function(FunctionCall {
            name,
            args,
            distinct,
            filter,
        }) => {
            site!(w, name.push('x'));
            site!(w, *distinct = !*distinct);
            m_exprs(args, w) || m_filter(filter, w)
        }
        ExprNode::Cast { expr, to, mode } => {
            site!(w, flip(mode, CastMode::Try, CastMode::Ansi));
            m_type(to, w) || m_expr(expr, w)
        }
        ExprNode::Alias { expr, name } => m_string(name, w) || m_expr(expr, w),
        ExprNode::Subquery { kind, plan } => {
            let other = match kind {
                SubqueryKind::Scalar => SubqueryKind::Exists,
                SubqueryKind::Exists | SubqueryKind::In(_) => SubqueryKind::Scalar,
            };
            site!(w, *kind = other);
            let values = match kind {
                SubqueryKind::In(v) => m_exprs(v, w),
                SubqueryKind::Scalar | SubqueryKind::Exists => false,
            };
            values || m_plan(plan, w)
        }
        ExprNode::Window(WindowExpr { function, window }) => {
            let other = match window {
                WindowSpecRef::Named(_) => WindowSpecRef::Spec(WindowSpec {
                    partition_by: Vec::new(),
                    order_by: Vec::new(),
                    frame: None,
                }),
                WindowSpecRef::Spec(_) => WindowSpecRef::Named("x".to_owned()),
            };
            site!(w, *window = other);
            let inside = match window {
                WindowSpecRef::Named(name) => m_string(name, w),
                WindowSpecRef::Spec(spec) => m_window_spec(spec, w),
            };
            inside || m_expr(function, w)
        }
        ExprNode::Resolved(r) => match r {
            ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c) => m_bound(c, w),
            ResolvedExpr::Function(ResolvedFunction {
                id,
                kind,
                args,
                distinct,
                filter,
                data_type,
                nullable,
                deterministic: _,
            }) => {
                let other = match id {
                    FunctionId::Builtin(Builtin::Eq) => FunctionId::Builtin(Builtin::Lt),
                    FunctionId::Builtin(_) => FunctionId::Builtin(Builtin::Eq),
                    FunctionId::Registry { name, id } => FunctionId::Registry {
                        name: format!("{name}x"),
                        id: *id,
                    },
                };
                site!(w, *id = other);
                site!(w, flip(kind, FunctionKind::Scalar, FunctionKind::Window));
                site!(w, *distinct = !*distinct);
                site!(w, *nullable = !*nullable);
                m_type(data_type, w) || m_exprs(args, w) || m_filter(filter, w)
            }
            ResolvedExpr::Alias { expr, output } => m_bound(output, w) || m_expr(expr, w),
        },
    }
}

fn m_orders(order: &mut Vec<SortOrder>, w: &mut Walk) -> bool {
    site!(
        w,
        order.push(SortOrder::asc(Expr::from(Literal::Integer(0))))
    );
    order.iter_mut().any(|o| {
        use NullOrdering::{NullsFirst, NullsLast};
        use SortDirection::{Ascending, Descending};
        site!(w, flip(&mut o.direction, Ascending, Descending));
        site!(w, flip(&mut o.nulls, NullsFirst, NullsLast));
        m_expr(&mut o.expr, w)
    })
}

fn m_bound_kind(b: &mut FrameBound, w: &mut Walk) -> bool {
    let other = match b {
        FrameBound::CurrentRow => FrameBound::UnboundedPreceding,
        _ => FrameBound::CurrentRow,
    };
    site!(w, *b = other);
    match b {
        FrameBound::Offset(e) => m_expr(e, w),
        _ => false,
    }
}

fn m_window_spec(spec: &mut WindowSpec, w: &mut Walk) -> bool {
    let frame = match spec.frame {
        Some(_) => None,
        None => Some(WindowFrame::running(FrameUnits::Rows)),
    };
    site!(w, spec.frame = frame);
    let in_frame = spec.frame.as_mut().is_some_and(|f| {
        site!(w, flip(&mut f.units, FrameUnits::Rows, FrameUnits::Range));
        m_bound_kind(&mut f.lower, w) || m_bound_kind(&mut f.upper, w)
    });
    in_frame || m_exprs(&mut spec.partition_by, w) || m_orders(&mut spec.order_by, w)
}

fn m_source(s: &mut ReadSource, w: &mut Walk) -> bool {
    let other = match s {
        ReadSource::Table { name, options } => ReadSource::Files {
            format: "parquet".to_owned(),
            paths: name.iter().cloned().map(OptionValue::Plain).collect(),
            options: options.clone(),
        },
        ReadSource::Files { paths, options, .. } => ReadSource::Table {
            name: paths.iter().map(ToString::to_string).collect(),
            options: options.clone(),
        },
    };
    site!(w, *s = other);
    match s {
        ReadSource::Table { name, options } => m_strings(name, w) || m_options(options, w),
        ReadSource::Files {
            format,
            paths,
            options,
        } => m_string(format, w) || m_values(paths, w) || m_options(options, w),
    }
}

fn m_columns(v: &mut Vec<(String, Arc<Expr>)>, w: &mut Walk) -> bool {
    site!(
        w,
        v.push(("x".to_owned(), Arc::new(Literal::Integer(0).into())))
    );
    v.iter_mut().any(|(n, e)| m_string(n, w) || m_expr(e, w))
}

fn m_grouping(g: &mut Grouping, w: &mut Walk) -> bool {
    let other = match g {
        Grouping::GroupBy(v) => Grouping::Rollup(v.clone()),
        Grouping::Rollup(v) | Grouping::Cube(v) => Grouping::GroupBy(v.clone()),
        Grouping::GroupingSets { .. } => Grouping::GroupBy(Vec::new()),
    };
    site!(w, *g = other);
    match g {
        Grouping::GroupBy(v) | Grouping::Rollup(v) | Grouping::Cube(v) => m_exprs(v, w),
        Grouping::GroupingSets { exprs, sets } => {
            site!(w, sets.push(Vec::new()));
            m_exprs(exprs, w) || sets.iter_mut().any(|s| m_exprs(s, w))
        }
    }
}

/// Applies the `w.k`-th single-field mutation of `p`; false if `p` has
/// fewer sites. Origins and `plan_id` values are not sites.
fn m_plan(p: &mut Arc<LogicalPlan>, w: &mut Walk) -> bool {
    let p = Arc::make_mut(p);
    match &mut p.node {
        PlanNode::OneRowRelation => false,
        PlanNode::Read(s) => m_source(s, w),
        PlanNode::Sql { query, args } => {
            site!(w, query.push('x'));
            let other = match args {
                SqlArgs::None => SqlArgs::Positional(Vec::new()),
                SqlArgs::Positional(_) | SqlArgs::Named(_) => SqlArgs::None,
            };
            site!(w, *args = other);
            match args {
                SqlArgs::None => false,
                SqlArgs::Positional(v) => m_exprs(v, w),
                SqlArgs::Named(m) => {
                    let keys: Vec<String> = m.keys().cloned().collect();
                    for k in keys {
                        site!(w, {
                            let v = m.remove(&k).unwrap();
                            m.insert(format!("{k}x"), v);
                        });
                    }
                    m.values_mut().any(|e| m_expr(e, w))
                }
            }
        }
        PlanNode::Scan(Scan {
            source,
            output,
            version,
        }) => {
            site!(
                w,
                *version = Some(TableVersion(version.map_or(0, |v| v.0 + 1)))
            );
            site!(
                w,
                output.push(output.first().cloned().unwrap_or(BoundColumn {
                    id: AttrId(0),
                    name: "x".to_owned(),
                    data_type: DataType::Integer,
                    nullable: false,
                }))
            );
            m_source(source, w) || output.iter_mut().any(|c| m_bound(c, w))
        }
        PlanNode::CteRef(CteRef {
            name,
            source,
            output,
        }) => {
            site!(w, source.push(AttrId(99)));
            site!(
                w,
                output.push(output.first().cloned().unwrap_or(BoundColumn {
                    id: AttrId(0),
                    name: "x".to_owned(),
                    data_type: DataType::Integer,
                    nullable: false,
                }))
            );
            m_string(name, w) || output.iter_mut().any(|c| m_bound(c, w))
        }
        PlanNode::Shift { input, offset } => {
            site!(w, *offset += 1);
            m_plan(input, w)
        }
        PlanNode::Project { input, exprs } => m_exprs(exprs, w) || m_plan(input, w),
        PlanNode::WithColumns { input, columns } => m_columns(columns, w) || m_plan(input, w),
        PlanNode::Filter {
            input,
            predicate: e,
        }
        | PlanNode::Limit { input, limit: e }
        | PlanNode::Offset { input, offset: e }
        | PlanNode::ReplicateRows { input, count: e } => m_expr(e, w) || m_plan(input, w),
        PlanNode::Expand {
            input,
            projections,
            output,
        } => {
            site!(w, projections.push(Vec::new()));
            site!(
                w,
                output.push(output.first().cloned().unwrap_or(BoundColumn {
                    id: AttrId(0),
                    name: "x".to_owned(),
                    data_type: DataType::Integer,
                    nullable: false,
                }))
            );
            projections.iter_mut().any(|p| m_exprs(p, w))
                || output.iter_mut().any(|c| m_bound(c, w))
                || m_plan(input, w)
        }
        PlanNode::SetOp {
            left,
            right,
            op,
            all,
        } => {
            site!(w, flip(op, SetOperation::Intersect, SetOperation::Except));
            site!(w, *all = !*all);
            m_plan(left, w) || m_plan(right, w)
        }
        PlanNode::Aggregate {
            input,
            grouping,
            aggregates,
        } => m_grouping(grouping, w) || m_exprs(aggregates, w) || m_plan(input, w),
        PlanNode::Join {
            left,
            right,
            kind,
            condition,
        } => {
            site!(w, flip(kind, JoinKind::Inner, JoinKind::Cross));
            let other = match condition {
                JoinCondition::None => JoinCondition::Using(vec!["x".to_owned()]),
                JoinCondition::On(_) | JoinCondition::Using(_) | JoinCondition::Natural => {
                    JoinCondition::None
                }
            };
            site!(w, *condition = other);
            let inside = match condition {
                JoinCondition::On(e) => m_expr(e, w),
                JoinCondition::Using(cols) => m_strings(cols, w),
                JoinCondition::None | JoinCondition::Natural => false,
            };
            inside || m_plan(left, w) || m_plan(right, w)
        }
        PlanNode::Sort {
            input,
            order,
            global,
        } => {
            site!(w, *global = !*global);
            m_orders(order, w) || m_plan(input, w)
        }
        PlanNode::Window {
            input,
            exprs,
            partition_by,
            order_by,
        } => {
            m_exprs(exprs, w)
                || m_exprs(partition_by, w)
                || m_orders(order_by, w)
                || m_plan(input, w)
        }
        PlanNode::SubqueryAlias {
            input,
            alias,
            columns,
        } => m_string(alias, w) || m_strings(columns, w) || m_plan(input, w),
        PlanNode::Hint {
            input,
            name,
            params,
        } => m_string(name, w) || m_exprs(params, w) || m_plan(input, w),
        PlanNode::JoinHint { input, strategy } => {
            site!(
                w,
                flip(strategy, JoinStrategy::Broadcast, JoinStrategy::ShuffleHash)
            );
            m_plan(input, w)
        }
        PlanNode::Union {
            inputs,
            by_name,
            allow_missing_columns,
        } => {
            site!(w, *by_name = !*by_name);
            site!(w, *allow_missing_columns = !*allow_missing_columns);
            site!(w, inputs.push(Arc::new(LogicalPlan::table(["x"]))));
            inputs.iter_mut().any(|i| m_plan(i, w))
        }
        PlanNode::Distinct { input } => m_plan(input, w),
        PlanNode::WithCte { input, ctes } => {
            site!(
                w,
                ctes.push(Cte {
                    name: "x".to_owned(),
                    plan: Arc::new(LogicalPlan::table(["x"])),
                })
            );
            let any = ctes
                .iter_mut()
                .any(|c| m_string(&mut c.name, w) || m_plan(&mut c.plan, w));
            any || m_plan(input, w)
        }
    }
}

/// The number of mutation sites in `p`.
fn sites(p: &Plan) -> usize {
    let mut w = Walk { k: usize::MAX };
    m_plan(&mut Arc::clone(p.root()), &mut w);
    usize::MAX - w.k
}

// ---- properties -----------------------------------------------------------

fn bytes() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 0..400)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn equal_plans_have_equal_keys(b in bytes()) {
        let (p, q) = (build(&b, false), build(&b, false));
        prop_assert!(p.structurally_eq(&q));
        prop_assert_eq!(p.structural_hash(), q.structural_hash());
        let hashes = |p: &Plan| p.subtree_hashes().into_iter().map(|(_, h)| h).collect::<Vec<_>>();
        prop_assert_eq!(hashes(&p), hashes(&q));
    }

    #[test]
    fn any_single_field_mutation_changes_the_key(b in bytes(), at in any::<Index>()) {
        let p = build(&b, false);
        // A lone `OneRowRelation` has no field to change.
        prop_assume!(sites(&p) > 0);
        let mut root = Arc::clone(p.root());
        let mut walk = Walk { k: at.index(sites(&p)) };
        prop_assert!(m_plan(&mut root, &mut walk));
        let mutated = Plan::try_new(root).unwrap();
        prop_assert!(!mutated.structurally_eq(&p));
        prop_assert_ne!(mutated.structural_hash(), p.structural_hash(), "{:?}", mutated);
    }

    #[test]
    fn origins_and_plan_ids_do_not_change_the_key(b in bytes()) {
        let (p, moved) = (build(&b, false), build(&b, true));
        prop_assert!(p.structurally_eq(&moved));
        prop_assert_eq!(p.structural_hash(), moved.structural_hash());
    }
}
