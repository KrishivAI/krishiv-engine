// SPDX-License-Identifier: Apache-2.0
//! The canonical encoding and the structural hash (ADR 0003 #6, owner
//! decisions 3 and 4).
//!
//! The canonical encoding is a byte string written field by field with
//! fixed tags, little-endian fixed-width integers and length-prefixed
//! strings, so it is the same on every platform, toolchain and process.
//! It leaves out origins (spans, `plan_id`s). It starts with
//! [`ENCODING_VERSION`] and [`crate::IR_VERSION`]; a change to the layout
//! bumps the version.
//!
//! A `plan_id` reference (`df["a"]`) is encoded by where its candidate
//! targets are, relative to the operator that holds the reference: every
//! relation carrying that `plan_id` in the operator's subtree (Spark
//! resolves DataFrame columns in that subtree), as a carrier record that
//! mirrors the subtree's shape and marks each carrier. A reference inside a
//! subquery with no carrier there is looked up in the subtrees of the
//! enclosing operators, innermost first, and encoded as (levels out,
//! carriers). A reference found nowhere is encoded by order of first
//! appearance among the operator's unfound references. So consistently
//! renumbered plans encode the same, and two plans with equal encodings
//! resolve every reference alike under any rule that depends only on where
//! the carriers are (the nearest one, or an error when there are several).
//!
//! Plans and expressions can share subtrees through `Arc` (a DAG), so the
//! encoding is a canonical DAG rather than an expanded tree: one record per
//! distinct relation or expression (its own fields, with child references
//! as record indices), records in post-order of first appearance, equal
//! records stored once. Structurally equal plans have equal encodings
//! however their subtrees are shared, and the encoding and every key take
//! time linear in the distinct nodes.
//!
//! The structural hash is a Merkle hash: each record's digest is XXH3-128
//! (seeded with [`ENCODING_VERSION`]) of its own fields with child
//! references replaced by the children's digests. Every subtree's key is
//! computed once, bottom-up.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use twox_hash::XxHash3_128;

use crate::{BoundColumn, Scan};
use crate::{
    CastMode, ColumnRef, DayTimeField, Expr, ExprNode, FunctionCall, FunctionKind, Grouping,
    JoinCondition, JoinKind, Literal, LogicalPlan, NullOrdering, Plan, PlanId, PlanNode,
    ReadSource, ResolvedExpr, ResolvedFunction, SortDirection, SqlArgs, Star, SubqueryKind,
    YearMonthField,
};

/// The version of the canonical encoding layout.
pub const ENCODING_VERSION: u16 = 2;

/// A 128-bit structural key of a plan or subtree: XXH3-128 over the
/// canonical encoding (see the module docs). Equal plans, ignoring origins,
/// have equal keys in every process and on every platform. A cache that
/// finds a key confirms the hit with [`Plan::structurally_eq`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StructuralHash(u128);

impl StructuralHash {
    /// The hash value.
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
}

impl fmt::Display for StructuralHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

impl Plan {
    /// The canonical encoding of the plan, origins excluded: the header,
    /// the records, then the root record's index.
    #[must_use]
    pub fn canonical_encoding(&self) -> Vec<u8> {
        let mut keys = Keys::default();
        let (root, _) = keys.plan(self.root(), &[], 0);
        let mut out = Vec::new();
        out.extend_from_slice(&ENCODING_VERSION.to_le_bytes());
        out.extend_from_slice(&self.ir_version().to_le_bytes());
        put_len(&mut out, keys.records.len());
        for record in &keys.records {
            put_len(&mut out, record.len());
            out.extend_from_slice(record);
        }
        out.extend_from_slice(&root.to_le_bytes());
        out
    }

    /// The structural hash of the plan.
    #[must_use]
    pub fn structural_hash(&self) -> StructuralHash {
        StructuralHash(Keys::default().plan(self.root(), &[], 0).1)
    }

    /// Whether the plans are equal ignoring origins and `plan_id` values:
    /// their canonical encodings are equal.
    #[must_use]
    pub fn structurally_eq(&self, other: &Self) -> bool {
        self.canonical_encoding() == other.canonical_encoding()
    }

    /// The structural hash of every distinct relation of the main query
    /// (not of subquery plans), in pre-order, root first; a relation shared
    /// by several parents appears once. All are computed in one bottom-up
    /// pass.
    #[must_use]
    pub fn subtree_hashes(&self) -> Vec<(&LogicalPlan, StructuralHash)> {
        let mut keys = Keys::default();
        keys.plan(self.root(), &[], 0);
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = vec![&**self.root()];
        let mut next = Vec::new();
        while let Some(p) = stack.pop() {
            if !seen.insert(std::ptr::from_ref(p)) {
                continue;
            }
            if let Some(&(_, digest)) = keys.plans.get(&(std::ptr::from_ref(p), 0)) {
                out.push((p, StructuralHash(digest)));
            }
            p.for_each_child(&mut |c| next.push(&**c));
            stack.extend(next.drain(..).rev());
        }
        out
    }
}

/// A record's index and digest.
type Key = (u64, u128);

/// The state of one encoding: interned records and memoised keys.
#[derive(Default)]
struct Keys {
    /// The carrier record of a `plan_id` in a subtree; `None` when no
    /// relation in it carries the `plan_id`.
    carriers: HashMap<(*const LogicalPlan, PlanId), Option<Key>>,
    records: Vec<Vec<u8>>,
    index: HashMap<Vec<u8>, u64>,
    /// Relation keys per (relation, enclosing-operator context).
    plans: HashMap<(*const LogicalPlan, usize), Key>,
    /// Expression keys per (expression, operator pass).
    exprs: HashMap<(*const Expr, usize), Key>,
    /// Contexts and operator passes handed out so far.
    counter: usize,
}

/// The operator whose expressions are being encoded.
struct Op<'a> {
    node: &'a LogicalPlan,
    /// Enclosing operators, innermost last (for subquery plans).
    outer: Vec<&'a LogicalPlan>,
    /// `plan_id`s found nowhere, in order of first appearance.
    unbound: Vec<PlanId>,
    pass: usize,
}

impl Keys {
    fn next_id(&mut self) -> usize {
        self.counter += 1;
        self.counter
    }

    /// The carrier record of `id` in the subtree of `p`: whether `p`
    /// carries `id`, then each child's carrier record or "none". It mirrors
    /// the subtree's shape, so it fixes every carrier's position. `None`
    /// when no relation in the subtree carries `id`. Memoised per relation,
    /// so it is linear in distinct relations.
    fn carriers(&mut self, p: &LogicalPlan, id: PlanId) -> Option<Key> {
        let key = (std::ptr::from_ref(p), id);
        if let Some(&found) = self.carriers.get(&key) {
            return found;
        }
        let mut children = Vec::new();
        p.for_each_child(&mut |c| children.push(Arc::clone(c)));
        let marks: Vec<Option<Key>> = children.iter().map(|c| self.carriers(c, id)).collect();
        let carries = p.origin.plan_id == Some(id);
        if !carries && marks.iter().all(Option::is_none) {
            self.carriers.insert(key, None);
            return None;
        }
        let mut rec = vec![b'C', u8::from(carries)];
        put_len(&mut rec, marks.len());
        let mut dig = rec.clone();
        for mark in marks {
            match mark {
                None => {
                    rec.push(0);
                    dig.push(0);
                }
                Some((index, digest)) => {
                    rec.push(1);
                    rec.extend_from_slice(&index.to_le_bytes());
                    dig.push(1);
                    dig.extend_from_slice(&digest.to_le_bytes());
                }
            }
        }
        let found = Some(self.intern(rec, &dig));
        self.carriers.insert(key, found);
        found
    }

    /// Where `id` may point from `op`: (levels out, carrier record) at the
    /// innermost level with a carrier, level 0 being `op` itself.
    fn locate(
        &mut self,
        id: PlanId,
        op: &LogicalPlan,
        outer: &[&LogicalPlan],
    ) -> Option<(usize, Key)> {
        std::iter::once(op)
            .chain(outer.iter().rev().copied())
            .enumerate()
            .find_map(|(level, p)| Some((level, self.carriers(p, id)?)))
    }

    /// The `plan_id`s that `op`'s expressions reference but that are found
    /// nowhere, in order of first appearance.
    fn unbound(&mut self, op: &LogicalPlan, outer: &[&LogicalPlan]) -> Vec<PlanId> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = Vec::new();
        op.for_each_expr(&mut |e| stack.push(&**e));
        stack.reverse();
        let mut next = Vec::new();
        while let Some(e) = stack.pop() {
            if !seen.insert(std::ptr::from_ref(e)) {
                continue;
            }
            let id = match &e.node {
                ExprNode::Column(c) => c.plan_id,
                ExprNode::Star(s) => s.plan_id,
                _ => None,
            };
            if let Some(id) = id
                && !out.contains(&id)
                && self.locate(id, op, outer).is_none()
            {
                out.push(id);
            }
            e.for_each_child(&mut |c| next.push(&**c));
            stack.extend(next.drain(..).rev());
        }
        out
    }

    fn intern(&mut self, record: Vec<u8>, hashed: &[u8]) -> Key {
        let digest = XxHash3_128::oneshot_with_seed(u64::from(ENCODING_VERSION), hashed);
        let index = match self.index.get(&record) {
            Some(&i) => i,
            None => {
                let i = u64::try_from(self.records.len()).unwrap_or(u64::MAX);
                self.records.push(record.clone());
                self.index.insert(record, i);
                i
            }
        };
        (index, digest)
    }

    /// The key of relation `p` under the enclosing operators `outer`
    /// (context `ctx`).
    fn plan<'a>(&mut self, p: &'a LogicalPlan, outer: &[&'a LogicalPlan], ctx: usize) -> Key {
        if let Some(&key) = self.plans.get(&(std::ptr::from_ref(p), ctx)) {
            return key;
        }
        let op = Op {
            node: p,
            outer: outer.to_vec(),
            unbound: self.unbound(p, outer),
            pass: self.next_id(),
        };
        let mut enc = Encoder::new(self, &op, b'P');
        enc.plan_fields(p);
        let mut children = Vec::new();
        p.for_each_child(&mut |c| children.push(&**c));
        for c in children {
            let key = enc.keys.plan(c, outer, ctx);
            enc.child(key);
        }
        let (record, hashed) = (enc.rec, enc.dig);
        let key = self.intern(record, &hashed);
        self.plans.insert((std::ptr::from_ref(p), ctx), key);
        key
    }

    /// The key of expression `e` of operator `op`.
    fn expr<'a>(&mut self, e: &'a Expr, op: &Op<'a>) -> Key {
        if let Some(&key) = self.exprs.get(&(std::ptr::from_ref(e), op.pass)) {
            return key;
        }
        let mut enc = Encoder::new(self, op, b'E');
        enc.expr_fields(e);
        let (record, hashed) = (enc.rec, enc.dig);
        let key = self.intern(record, &hashed);
        self.exprs.insert((std::ptr::from_ref(e), op.pass), key);
        key
    }
}

/// Writes one record: its own fields, child references as indices in
/// `rec` and as digests in `dig`.
struct Encoder<'k, 'a> {
    keys: &'k mut Keys,
    op: &'k Op<'a>,
    rec: Vec<u8>,
    dig: Vec<u8>,
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}

impl<'k, 'a> Encoder<'k, 'a> {
    fn new(keys: &'k mut Keys, op: &'k Op<'a>, kind: u8) -> Self {
        Self {
            keys,
            op,
            rec: vec![kind],
            dig: vec![kind],
        }
    }

    fn put(&mut self, b: &[u8]) {
        self.rec.extend_from_slice(b);
        self.dig.extend_from_slice(b);
    }

    /// A reference to a child record.
    fn child(&mut self, (index, digest): Key) {
        self.rec.extend_from_slice(&index.to_le_bytes());
        self.dig.extend_from_slice(&digest.to_le_bytes());
    }

    /// A child expression, as a reference to its record.
    fn expr(&mut self, e: &'a Expr) {
        let key = self.keys.expr(e, self.op);
        self.child(key);
    }

    fn u8(&mut self, v: u8) {
        self.put(&[v]);
    }

    fn bool(&mut self, v: bool) {
        self.u8(u8::from(v));
    }

    fn u32(&mut self, v: u32) {
        self.put(&v.to_le_bytes());
    }

    fn i64(&mut self, v: i64) {
        self.put(&v.to_le_bytes());
    }

    fn len(&mut self, len: usize) {
        self.put(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
    }

    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }

    fn bytes(&mut self, b: &[u8]) {
        self.len(b.len());
        self.put(b);
    }

    fn strs(&mut self, v: &[String]) {
        self.len(v.len());
        v.iter().for_each(|s| self.str(s));
    }

    fn options(&mut self, m: &std::collections::BTreeMap<String, String>) {
        self.len(m.len());
        for (k, v) in m {
            self.str(k);
            self.str(v);
        }
    }

    fn data_type(&mut self, t: &krishiv_types::DataType) {
        self.str(&t.simple_string());
    }

    fn exprs(&mut self, v: &'a [Arc<Expr>]) {
        self.len(v.len());
        v.iter().for_each(|e| self.expr(e));
    }

    fn plan_fields(&mut self, p: &'a LogicalPlan) {
        match &p.node {
            PlanNode::Read(s) => {
                self.u8(1);
                self.source(s);
            }
            PlanNode::Sql { query, args } => {
                self.u8(2);
                self.str(query);
                match args {
                    SqlArgs::None => self.u8(0),
                    SqlArgs::Positional(v) => {
                        self.u8(1);
                        self.exprs(v);
                    }
                    SqlArgs::Named(m) => {
                        self.u8(2);
                        self.len(m.len());
                        for (k, e) in m {
                            self.str(k);
                            self.expr(e);
                        }
                    }
                }
            }
            PlanNode::Scan(Scan {
                source,
                output,
                version,
            }) => {
                self.u8(3);
                self.source(source);
                self.len(output.len());
                output.iter().for_each(|c| self.bound(c));
                match version {
                    None => self.u8(0),
                    Some(v) => {
                        self.u8(1);
                        self.i64(v.0);
                    }
                }
            }
            PlanNode::Project { exprs, .. } => {
                self.u8(4);
                self.exprs(exprs);
            }
            PlanNode::WithColumns { columns, .. } => {
                self.u8(5);
                self.len(columns.len());
                for (name, e) in columns {
                    self.str(name);
                    self.expr(e);
                }
            }
            PlanNode::Filter { predicate, .. } => {
                self.u8(6);
                self.expr(predicate);
            }
            PlanNode::Aggregate {
                grouping,
                aggregates,
                ..
            } => {
                self.u8(7);
                match grouping {
                    Grouping::GroupBy(v) => {
                        self.u8(1);
                        self.exprs(v);
                    }
                    Grouping::Rollup(v) => {
                        self.u8(2);
                        self.exprs(v);
                    }
                    Grouping::Cube(v) => {
                        self.u8(3);
                        self.exprs(v);
                    }
                    Grouping::GroupingSets { exprs, sets } => {
                        self.u8(4);
                        self.exprs(exprs);
                        self.len(sets.len());
                        sets.iter().for_each(|s| self.exprs(s));
                    }
                }
                self.exprs(aggregates);
            }
            PlanNode::Join {
                kind, condition, ..
            } => {
                self.u8(8);
                self.u8(match kind {
                    JoinKind::Inner => 1,
                    JoinKind::LeftOuter => 2,
                    JoinKind::RightOuter => 3,
                    JoinKind::FullOuter => 4,
                    JoinKind::LeftSemi => 5,
                    JoinKind::LeftAnti => 6,
                    JoinKind::Cross => 7,
                });
                match condition {
                    JoinCondition::None => self.u8(0),
                    JoinCondition::On(e) => {
                        self.u8(1);
                        self.expr(e);
                    }
                    JoinCondition::Using(cols) => {
                        self.u8(2);
                        self.strs(cols);
                    }
                }
            }
            PlanNode::Sort { order, global, .. } => {
                self.u8(9);
                self.bool(*global);
                self.len(order.len());
                for o in order {
                    self.u8(match o.direction {
                        SortDirection::Ascending => 1,
                        SortDirection::Descending => 2,
                    });
                    self.u8(match o.nulls {
                        NullOrdering::NullsFirst => 1,
                        NullOrdering::NullsLast => 2,
                    });
                    self.expr(&o.expr);
                }
            }
            PlanNode::Limit { limit, .. } => {
                self.u8(10);
                self.expr(limit);
            }
            PlanNode::SubqueryAlias { alias, columns, .. } => {
                self.u8(11);
                self.str(alias);
                self.strs(columns);
            }
            PlanNode::WithCte { ctes, .. } => {
                self.u8(12);
                self.len(ctes.len());
                ctes.iter().for_each(|c| self.str(&c.name));
            }
        }
    }

    fn source(&mut self, s: &ReadSource) {
        match s {
            ReadSource::Table { name, options } => {
                self.u8(1);
                self.strs(name);
                self.options(options);
            }
            ReadSource::Files {
                format,
                paths,
                options,
            } => {
                self.u8(2);
                self.str(format);
                self.strs(paths);
                self.options(options);
            }
        }
    }

    fn bound(&mut self, c: &BoundColumn) {
        self.put(&c.id.0.to_le_bytes());
        self.str(&c.name);
        self.data_type(&c.data_type);
        self.bool(c.nullable);
    }

    /// Encodes where `id` points (see the module docs).
    fn plan_ref(&mut self, id: Option<PlanId>) {
        let Some(id) = id else {
            return self.u8(0);
        };
        match self.keys.locate(id, self.op.node, &self.op.outer) {
            Some((0, carriers)) => {
                self.u8(1);
                self.child(carriers);
            }
            Some((level, carriers)) => {
                self.u8(2);
                self.len(level);
                self.child(carriers);
            }
            None => {
                let k = self
                    .op
                    .unbound
                    .iter()
                    .position(|&u| u == id)
                    .unwrap_or(usize::MAX);
                self.u8(3);
                self.len(k);
            }
        }
    }

    fn call(&mut self, args: &'a [Arc<Expr>], distinct: bool, filter: Option<&'a Arc<Expr>>) {
        self.exprs(args);
        self.bool(distinct);
        match filter {
            None => self.u8(0),
            Some(e) => {
                self.u8(1);
                self.expr(e);
            }
        }
    }

    fn expr_fields(&mut self, e: &'a Expr) {
        match &e.node {
            ExprNode::Column(ColumnRef { name, plan_id }) => {
                self.u8(1);
                self.strs(name);
                self.plan_ref(*plan_id);
            }
            ExprNode::Literal(v) => {
                self.u8(2);
                self.literal(v);
            }
            ExprNode::Star(Star { qualifier, plan_id }) => {
                self.u8(3);
                self.strs(qualifier);
                self.plan_ref(*plan_id);
            }
            ExprNode::Function(FunctionCall {
                name,
                args,
                distinct,
                filter,
            }) => {
                self.u8(4);
                self.str(name);
                self.call(args, *distinct, filter.as_ref());
            }
            ExprNode::Cast { expr, to, mode } => {
                self.u8(5);
                self.data_type(to);
                self.u8(match mode {
                    CastMode::Default => 1,
                    CastMode::Legacy => 2,
                    CastMode::Ansi => 3,
                    CastMode::Try => 4,
                });
                self.expr(expr);
            }
            ExprNode::Alias { expr, name } => {
                self.u8(6);
                self.str(name);
                self.expr(expr);
            }
            ExprNode::Subquery { kind, plan } => {
                self.u8(7);
                match kind {
                    SubqueryKind::Scalar => self.u8(1),
                    SubqueryKind::Exists => self.u8(2),
                    SubqueryKind::In(values) => {
                        self.u8(3);
                        self.exprs(values);
                    }
                }
                let mut outer = self.op.outer.clone();
                outer.push(self.op.node);
                let ctx = self.keys.next_id();
                let key = self.keys.plan(plan, &outer, ctx);
                self.child(key);
            }
            ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                self.u8(8);
                self.bound(c);
            }
            ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => {
                self.u8(9);
                self.bound(c);
            }
            ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                id,
                kind,
                args,
                distinct,
                filter,
                data_type,
                nullable,
            })) => {
                self.u8(10);
                // The canonical name only: a registry id is never part of
                // a key (see `FunctionId`).
                self.str(id.name());
                self.u8(match kind {
                    FunctionKind::Scalar => 1,
                    FunctionKind::Aggregate => 2,
                    FunctionKind::Window => 3,
                    FunctionKind::Generator => 4,
                });
                self.data_type(data_type);
                self.bool(*nullable);
                self.call(args, *distinct, filter.as_ref());
            }
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
                self.u8(11);
                self.bound(output);
                self.expr(expr);
            }
        }
    }

    fn literal(&mut self, v: &Literal) {
        match v {
            Literal::Null(t) => {
                self.u8(1);
                match t {
                    None => self.u8(0),
                    Some(t) => {
                        self.u8(1);
                        self.data_type(t);
                    }
                }
            }
            Literal::Boolean(b) => {
                self.u8(2);
                self.bool(*b);
            }
            Literal::Byte(x) => {
                self.u8(3);
                self.put(&x.to_le_bytes());
            }
            Literal::Short(x) => {
                self.u8(4);
                self.put(&x.to_le_bytes());
            }
            Literal::Integer(x) => {
                self.u8(5);
                self.put(&x.to_le_bytes());
            }
            Literal::Long(x) => {
                self.u8(6);
                self.i64(*x);
            }
            Literal::Float(x) => {
                self.u8(7);
                self.u32(x.to_bits());
            }
            Literal::Double(x) => {
                self.u8(8);
                self.put(&x.to_bits().to_le_bytes());
            }
            Literal::Decimal {
                unscaled,
                data_type,
            } => {
                self.u8(9);
                self.put(&unscaled.to_le_bytes());
                self.u8(data_type.precision());
                self.u8(data_type.scale());
            }
            Literal::String(s) => {
                self.u8(10);
                self.str(s);
            }
            Literal::Binary(b) => {
                self.u8(11);
                self.bytes(b);
            }
            Literal::Date(d) => {
                self.u8(12);
                self.put(&d.to_le_bytes());
            }
            Literal::Timestamp(us) => {
                self.u8(13);
                self.i64(*us);
            }
            Literal::TimestampNtz(us) => {
                self.u8(14);
                self.i64(*us);
            }
            Literal::YearMonthInterval { months, start, end } => {
                self.u8(15);
                self.put(&months.to_le_bytes());
                for f in [start, end] {
                    self.u8(match f {
                        YearMonthField::Year => 1,
                        YearMonthField::Month => 2,
                    });
                }
            }
            Literal::DayTimeInterval { micros, start, end } => {
                self.u8(16);
                self.i64(*micros);
                for f in [start, end] {
                    self.u8(match f {
                        DayTimeField::Day => 1,
                        DayTimeField::Hour => 2,
                        DayTimeField::Minute => 3,
                        DayTimeField::Second => 4,
                    });
                }
            }
        }
    }
}
