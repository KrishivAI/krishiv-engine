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
//! It never holds secret plaintext (ADR 0038). A `secret('scope', 'key')`
//! reference is encoded as its scope and key. An inline secret, given as an
//! option or found inside a URL option, is encoded as its keyed fingerprint
//! ([`krishiv_common::secret::SecretString::fingerprint`]): plans that differ
//! only in an inline secret get different keys, but the fingerprint key is
//! random per process, so the keys of such plans (only those) differ
//! between processes. That is intended: another credential may mean other
//! data or permissions, and a cache entry must not outlive its process's
//! view of it.
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
//! however their subtrees are shared.
//!
//! Cost: records and digests are linear in the distinct nodes. Encoding a
//! `plan_id` reference walks only the part of the referencing operator's
//! subtree that holds a carrier of that id: each relation keeps a bit set of
//! the ids carried below it (one word-wise merge per distinct relation), and
//! subtrees without the id are skipped. A chain of N operators each
//! referencing the previous one, the common PySpark `withColumn`/`df["x"]`
//! shape, is then O(N) walks plus O(N²/64) bit-set words; references whose
//! carriers sit deep under the operator still walk down to them. A benchmark
//! tracks the chain case.
//!
//! The structural hash is a Merkle hash: each record's digest is XXH3-128
//! (seeded with [`ENCODING_VERSION`]) of its own fields with child
//! references replaced by the children's digests. Every subtree's key is
//! computed once, bottom-up.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use krishiv_common::secret::{OptionValue, SecretUrl, SecretValue, UrlPart};
use twox_hash::XxHash3_128;

use crate::{BoundColumn, CteRef, Scan};
use crate::{
    CastMode, ColumnRef, DayTimeField, Expr, ExprNode, FieldOp, FunctionCall, Grouping,
    JoinCondition, JoinKind, Literal, LogicalPlan, NullOrdering, Plan, PlanId, PlanNode,
    ReadSource, ResolvedExpr, ResolvedFunction, SortDirection, SqlArgs, Star, SubqueryKind,
    YearMonthField,
};

/// The version of the canonical encoding layout.
pub const ENCODING_VERSION: u16 = 10;

/// A 128-bit structural key of a plan or subtree: XXH3-128 over the
/// canonical encoding (see the module docs). Equal plans, ignoring origins,
/// have equal keys in every process and on every platform, except plans
/// with inline secrets, whose keys hold only within a process. A cache that
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

/// A 128-bit identity of every distinct relation under `root` (subquery
/// plans included), by address: relations that are equal field by field,
/// `plan_id`s included (as values) and origins otherwise excluded, get one
/// identity. The analyzer resolves equal relations once, whether or not
/// they share an `Arc`, so equal input keys give equal resolved plans.
pub(crate) fn identity_keys(root: &LogicalPlan) -> HashMap<*const LogicalPlan, u128> {
    let mut keys = Keys {
        absolute: true,
        ..Keys::default()
    };
    keys.plan(root, &[], 0);
    keys.plans
        .into_iter()
        .map(|((p, _), (_, digest))| (p, digest))
        .collect()
}

/// A record's index and digest.
type Key = (u64, u128);

/// The state of one encoding: interned records and memoised keys.
#[derive(Default)]
struct Keys {
    /// The carrier record of a `plan_id` in a subtree; `None` when no
    /// relation in it carries the `plan_id`.
    carriers: HashMap<(*const LogicalPlan, PlanId), Option<Key>>,
    /// Per relation, the bit set of `plan_id`s carried anywhere in its
    /// subtree, so carrier walks skip subtrees that cannot hold the id.
    below: HashMap<*const LogicalPlan, Rc<[u64]>>,
    /// Dense bit index of each carried `plan_id`.
    id_bits: HashMap<PlanId, usize>,
    records: Vec<Vec<u8>>,
    index: HashMap<Vec<u8>, u64>,
    /// Relation keys per (relation, enclosing-operator context).
    plans: HashMap<(*const LogicalPlan, usize), Key>,
    /// Expression keys per (expression, operator pass).
    exprs: HashMap<(*const Expr, usize), Key>,
    /// Contexts and operator passes handed out so far.
    counter: usize,
    /// Identity mode (see [`identity_keys`]): `plan_id`s are encoded as
    /// themselves, not by where their carriers are.
    absolute: bool,
    /// Whether a relation's subtree (subquery plans included) holds a
    /// `plan_id` reference: only then does its key depend on the enclosing
    /// operators, so a subtree without one is encoded once.
    refs: HashMap<*const LogicalPlan, bool>,
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
    /// when no relation in the subtree carries `id`. Memoised per
    /// (relation, id), and only relations whose subtree holds the id are
    /// visited.
    fn carriers(&mut self, p: &LogicalPlan, id: PlanId) -> Option<Key> {
        let key = (std::ptr::from_ref(p), id);
        if let Some(&found) = self.carriers.get(&key) {
            return found;
        }
        // Pruning only: a subtree without the id yields `None` either way,
        // so the encoding is unchanged.
        if !self.carried_below(p, id) {
            self.carriers.insert(key, None);
            return None;
        }
        let mut children = Vec::new();
        p.for_each_child(&mut |c| children.push(Arc::clone(c)));
        let marks: Vec<Option<Key>> = children.iter().map(|c| self.carriers(c, id)).collect();
        // `carried_below` holds, so `p` carries `id` or some mark is set:
        // the record always marks at least one carrier.
        let carries = p.origin.plan_id == Some(id);
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

    /// Whether the subtree of `root` holds a `plan_id` reference, memoised
    /// per relation, without recursion. The walk stops at the first
    /// reference (a hit marks the relations on the path to it), so a chain
    /// of operators that each reference the previous one costs one step per
    /// operator; a subtree without a reference is scanned once.
    fn has_refs(&mut self, root: &LogicalPlan) -> bool {
        use crate::walk::{Node, children};
        if let Some(&known) = self.refs.get(&std::ptr::from_ref(root)) {
            return known;
        }
        let mut stack = vec![(Node::Plan(root), false)];
        let mut next = Vec::new();
        // Expressions finished without a reference (shared ones once).
        let mut clean: HashSet<*const Expr> = HashSet::new();
        while let Some((node, expanded)) = stack.pop() {
            if expanded {
                // Everything below finished without a reference.
                match node {
                    Node::Plan(p) => {
                        self.refs.entry(std::ptr::from_ref(p)).or_insert(false);
                    }
                    Node::Expr(e) => {
                        clean.insert(std::ptr::from_ref(e));
                    }
                }
                continue;
            }
            let hit = match node {
                Node::Plan(p) => self.refs.get(&std::ptr::from_ref(p)).copied(),
                Node::Expr(e) if clean.contains(&std::ptr::from_ref(e)) => Some(false),
                Node::Expr(e) => match &e.node {
                    ExprNode::Column(c) => c.plan_id.is_some().then_some(true),
                    ExprNode::Star(s) => s.plan_id.is_some().then_some(true),
                    _ => None,
                },
            };
            match hit {
                Some(true) => {
                    // The relations being scanned all contain this one.
                    for (n, done) in &stack {
                        if let (Node::Plan(p), true) = (n, done) {
                            self.refs.insert(std::ptr::from_ref(*p), true);
                        }
                    }
                    return true;
                }
                Some(false) => continue,
                None => {}
            }
            children(node, &mut next);
            stack.push((node, true));
            stack.extend(next.drain(..).map(|c| (c, false)));
        }
        self.refs
            .get(&std::ptr::from_ref(root))
            .copied()
            .unwrap_or(false)
    }

    /// Whether any relation in the subtree of `p` carries `id`.
    fn carried_below(&mut self, p: &LogicalPlan, id: PlanId) -> bool {
        let bits = self.ids_below(p);
        self.id_bits
            .get(&id)
            .is_some_and(|&b| bits.get(b / 64).is_some_and(|w| w & (1 << (b % 64)) != 0))
    }

    /// The bit set of `plan_id`s carried in the subtree of `p`, memoised per
    /// relation: one pass over the distinct relations, each merging its
    /// children's sets word by word.
    fn ids_below(&mut self, p: &LogicalPlan) -> Rc<[u64]> {
        let ptr = std::ptr::from_ref(p);
        if let Some(bits) = self.below.get(&ptr) {
            return Rc::clone(bits);
        }
        let mut bits: Vec<u64> = Vec::new();
        if let Some(id) = p.origin.plan_id {
            let next = self.id_bits.len();
            let b = *self.id_bits.entry(id).or_insert(next);
            bits.resize(b / 64 + 1, 0);
            bits[b / 64] |= 1 << (b % 64);
        }
        let mut children = Vec::new();
        p.for_each_child(&mut |c| children.push(Arc::clone(c)));
        for child in &children {
            let theirs = self.ids_below(child);
            if theirs.len() > bits.len() {
                bits.resize(theirs.len(), 0);
            }
            for (word, other) in bits.iter_mut().zip(theirs.iter()) {
                *word |= other;
            }
        }
        let bits: Rc<[u64]> = bits.into();
        self.below.insert(ptr, Rc::clone(&bits));
        bits
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
        let ctx = if self.has_refs(p) { ctx } else { 0 };
        if let Some(&key) = self.plans.get(&(std::ptr::from_ref(p), ctx)) {
            return key;
        }
        let unbound = if self.absolute {
            Vec::new()
        } else {
            self.unbound(p, outer)
        };
        let op = Op {
            node: p,
            outer: outer.to_vec(),
            unbound,
            pass: self.next_id(),
        };
        let mut enc = Encoder::new(self, &op, b'P');
        enc.plan_fields(p);
        if enc.keys.absolute {
            // The relation's own `plan_id` is part of its identity.
            match p.origin.plan_id {
                None => enc.u8(0),
                Some(id) => {
                    enc.u8(1);
                    enc.i64(id.0);
                }
            }
        }
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

    /// A star's `|> SET` assignments (out of line: `expr_fields` recurses,
    /// and its frame must stay small).
    #[inline(never)]
    fn replacements(&mut self, replace: &'a [(String, Arc<Expr>)]) {
        for (name, expr) in replace {
            self.str(name);
            self.expr(expr);
        }
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

    /// A function's identity in a key: its canonical name, never a registry
    /// id (see `FunctionId`), and for a UDF its version, so a re-registered
    /// UDF never shares a key with the version it replaced. Kept out of line
    /// so the recursive expression encoder's frame stays small.
    #[inline(never)]
    fn function_id(&mut self, id: &crate::FunctionId) {
        self.str(id.name());
        if let crate::FunctionId::Udf { version, .. } = id {
            self.u32(*version);
        }
    }

    fn bytes(&mut self, b: &[u8]) {
        self.len(b.len());
        self.put(b);
    }

    fn strs(&mut self, v: &[String]) {
        self.len(v.len());
        v.iter().for_each(|s| self.str(s));
    }

    fn options(&mut self, m: &BTreeMap<String, OptionValue>) {
        self.len(m.len());
        for (k, v) in m {
            self.str(k);
            self.option_value(v);
        }
    }

    /// An option value or path: a plain value as itself, a reference as its
    /// (scope, key), an inline secret (alone or inside a URL) as its keyed
    /// fingerprint, never its plaintext (ADR 0038).
    fn option_value(&mut self, v: &OptionValue) {
        match v {
            OptionValue::Plain(s) => {
                self.u8(1);
                self.str(s);
            }
            OptionValue::Secret(SecretValue::Reference { scope, key }) => {
                self.u8(2);
                self.str(scope);
                self.str(key);
            }
            OptionValue::Secret(SecretValue::Inline(s)) => {
                self.u8(3);
                self.put(&s.fingerprint());
            }
            OptionValue::Url(url) => {
                self.u8(4);
                self.len(url.parts().len());
                for part in url.parts() {
                    match part {
                        UrlPart::Text(s) => {
                            self.u8(1);
                            self.str(s);
                        }
                        UrlPart::Secret(s) => {
                            self.u8(2);
                            self.put(&s.fingerprint());
                        }
                    }
                }
            }
        }
    }

    fn data_type(&mut self, t: &krishiv_types::DataType) {
        self.str(&t.simple_string());
    }

    fn exprs(&mut self, v: &'a [Arc<Expr>]) {
        self.len(v.len());
        v.iter().for_each(|e| self.expr(e));
    }

    /// An unresolved struct update, out of line so [`Self::expr_fields`]
    /// keeps its frame: plans at the depth cap must still fit a 2 MiB
    /// stack (see `tests::depth`).
    #[inline(never)]
    fn update_fields(&mut self, e: &'a Expr) {
        if let ExprNode::UpdateFields { base, ops } = &e.node {
            self.u8(17);
            self.expr(base);
            self.len(ops.len());
            for op in ops {
                match op {
                    FieldOp::With { name, value } => {
                        self.u8(1);
                        self.str(name);
                        self.expr(value);
                    }
                    FieldOp::Drop { name } => {
                        self.u8(2);
                        self.str(name);
                    }
                }
            }
        }
    }

    fn plan_fields(&mut self, p: &'a LogicalPlan) {
        match &p.node {
            PlanNode::Read(s) => {
                self.u8(1);
                self.source(s);
            }
            PlanNode::OneRowRelation => self.u8(15),
            PlanNode::PartialAggregate {
                keys,
                aggregates,
                states,
                ..
            } => {
                self.u8(30);
                self.exprs(keys);
                self.exprs(aggregates);
                self.len(states.len());
                for c in states {
                    self.bound(c);
                }
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
                let tag: u8 = match kind {
                    JoinKind::Inner => 1,
                    JoinKind::LeftOuter => 2,
                    JoinKind::RightOuter => 3,
                    JoinKind::FullOuter => 4,
                    JoinKind::LeftSemi => 5,
                    JoinKind::LeftAnti => 6,
                    JoinKind::Cross => 7,
                    JoinKind::Existence(_) => 8,
                    JoinKind::Single => 9,
                };
                self.u8(tag);
                if let JoinKind::Existence(id) = kind {
                    self.put(&id.0.to_le_bytes());
                }
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
                    JoinCondition::Natural => self.u8(3),
                }
            }
            PlanNode::Sort { order, global, .. } => {
                self.u8(9);
                self.bool(*global);
                self.orders(order);
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
            PlanNode::CteRef(CteRef {
                name,
                source,
                output,
            }) => {
                self.u8(13);
                self.str(name);
                self.len(source.len());
                source.iter().for_each(|id| self.put(&id.0.to_le_bytes()));
                self.len(output.len());
                output.iter().for_each(|c| self.bound(c));
            }
            PlanNode::Shift { offset, .. } => {
                self.u8(14);
                self.put(&offset.to_le_bytes());
            }
            PlanNode::Hint { name, params, .. } => {
                self.u8(16);
                self.str(name);
                self.exprs(params);
            }
            PlanNode::PipeOperator { .. } => self.u8(31),
            PlanNode::JoinHint { strategy, .. } => {
                self.u8(17);
                self.str(strategy.name());
            }
            PlanNode::Union {
                inputs,
                by_name,
                allow_missing_columns,
            } => {
                self.u8(18);
                self.len(inputs.len());
                self.u8(u8::from(*by_name));
                self.u8(u8::from(*allow_missing_columns));
            }
            PlanNode::Distinct { .. } => self.u8(19),
            PlanNode::SetOp { op, all, .. } => {
                self.u8(20);
                self.u8(match op {
                    crate::SetOperation::Intersect => 0,
                    crate::SetOperation::Except => 1,
                });
                self.u8(u8::from(*all));
            }
            PlanNode::ReplicateRows { count, .. } => {
                self.u8(21);
                self.expr(count);
            }
            PlanNode::Offset { offset, .. } => {
                self.u8(22);
                self.expr(offset);
            }
            PlanNode::Window {
                exprs,
                partition_by,
                order_by,
                ..
            } => {
                self.u8(23);
                self.exprs(exprs);
                self.exprs(partition_by);
                self.orders(order_by);
            }
            PlanNode::Expand {
                projections,
                output,
                ..
            } => {
                self.u8(24);
                self.len(projections.len());
                projections.iter().for_each(|p| self.exprs(p));
                self.len(output.len());
                output.iter().for_each(|c| self.bound(c));
            }
            PlanNode::InlineTable { rows, columns } => {
                self.u8(25);
                self.len(rows.len());
                rows.iter().for_each(|r| self.exprs(r));
                self.strs(columns);
            }
            PlanNode::Pivot {
                pivot_columns,
                values,
                aggregates,
                ..
            } => {
                self.u8(26);
                self.exprs(pivot_columns);
                self.len(values.len());
                for v in values {
                    self.exprs(&v.exprs);
                    self.u8(u8::from(v.alias.is_some()));
                    if let Some(alias) = &v.alias {
                        self.str(alias);
                    }
                }
                self.exprs(aggregates);
            }
            PlanNode::Unpivot {
                value_columns,
                name_column,
                column_sets,
                include_nulls,
                ..
            } => {
                self.u8(27);
                self.strs(value_columns);
                self.str(name_column);
                self.len(column_sets.len());
                for s in column_sets {
                    self.exprs(&s.columns);
                    self.u8(u8::from(s.alias.is_some()));
                    if let Some(alias) = &s.alias {
                        self.str(alias);
                    }
                }
                self.u8(u8::from(*include_nulls));
            }
            PlanNode::UnionLoop {
                id,
                max_level,
                limit,
                output,
                ..
            } => {
                self.u8(31);
                self.put(&id.to_le_bytes());
                match limit {
                    None => self.u8(0),
                    Some(n) => {
                        self.u8(1);
                        self.put(&n.to_le_bytes());
                    }
                }
                match max_level {
                    None => self.u8(0),
                    Some(n) => {
                        self.u8(1);
                        self.put(&n.to_le_bytes());
                    }
                }
                self.len(output.len());
                output.iter().for_each(|c| self.bound(c));
            }
            PlanNode::UnionLoopRef { id, output } => {
                self.u8(32);
                self.put(&id.to_le_bytes());
                self.len(output.len());
                output.iter().for_each(|c| self.bound(c));
            }
            PlanNode::Generate {
                kind,
                outer,
                arg,
                output,
                names,
                qualifier,
                ..
            } => {
                self.u8(30);
                self.u8(match kind {
                    crate::GeneratorKind::Explode => 0,
                    crate::GeneratorKind::PosExplode => 1,
                    crate::GeneratorKind::Inline => 2,
                });
                self.u8(u8::from(*outer));
                self.expr(arg);
                self.len(output.len());
                output.iter().for_each(|c| self.bound(c));
                self.strs(names);
                match qualifier {
                    None => self.u8(0),
                    Some(q) => {
                        self.u8(1);
                        self.str(q);
                    }
                }
            }
            PlanNode::TableFunction { name, args } => {
                self.u8(28);
                self.str(name);
                self.exprs(args);
            }
            PlanNode::Range {
                start,
                end,
                step,
                num_slices,
                output,
            } => {
                self.u8(29);
                self.i64(*start);
                self.i64(*end);
                self.i64(*step);
                match num_slices {
                    None => self.u8(0),
                    Some(n) => {
                        self.u8(1);
                        self.put(&n.to_le_bytes());
                    }
                }
                self.bound(output);
            }
        }
    }

    fn orders(&mut self, order: &'a [crate::SortOrder]) {
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

    fn frame_bound(&mut self, b: &'a crate::FrameBound) {
        match b {
            crate::FrameBound::UnboundedPreceding => self.u8(1),
            crate::FrameBound::CurrentRow => self.u8(2),
            crate::FrameBound::UnboundedFollowing => self.u8(3),
            crate::FrameBound::Offset(e) => {
                self.u8(4);
                self.expr(e);
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
                self.len(paths.len());
                paths.iter().for_each(|p| self.option_value(p));
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
        if self.keys.absolute {
            self.u8(4);
            return self.i64(id.0);
        }
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

    /// The `WITHIN GROUP` orderings of a call node, as a length prefix
    /// (empty for almost every node). Takes the node (already bound by
    /// the caller) rather than the orderings so `expr_fields` grows no
    /// new local on its recursive path: a plan at `MAX_DEPTH` must
    /// survive this encoding on a 2 MiB stack.
    fn call_within_group(&mut self, e: &'a Expr) {
        // Both call sites are function arms, so this always matches;
        // the fallback keeps the helper total without growing the
        // callers' frames.
        let (ExprNode::Function(FunctionCall { within_group, .. })
        | ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction { within_group, .. }))) =
            &e.node
        else {
            return;
        };
        if within_group.is_empty() {
            self.len(0);
        } else {
            self.orders(within_group);
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
            ExprNode::Star(Star {
                qualifier,
                except,
                replace,
                position,
                plan_id,
            }) => {
                self.u8(3);
                self.strs(qualifier);
                self.len(except.len());
                except.iter().for_each(|path| self.strs(path));
                self.plan_ref(*plan_id);
                // Only a replacing star writes more, so the encoding of
                // every other plan is unchanged.
                if !replace.is_empty() {
                    self.len(replace.len());
                    self.replacements(replace);
                }
                if let Some(n) = position {
                    self.len(*n);
                }
            }
            ExprNode::Function(FunctionCall {
                name,
                qualifier,
                args,
                arg_names,
                distinct,
                filter,
                ..
            }) => {
                self.u8(4);
                self.str(name);
                self.strs(qualifier);
                self.len(arg_names.len());
                for n in arg_names {
                    match n {
                        None => self.u8(0),
                        Some(n) => {
                            self.u8(1);
                            self.str(n);
                        }
                    }
                }
                self.call(args, *distinct, filter.as_ref());
                self.call_within_group(e);
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
            ExprNode::MultiAlias { expr, names } => {
                self.u8(33);
                self.len(names.len());
                names.iter().for_each(|n| self.str(n));
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
                let ctx = if self.keys.absolute {
                    0
                } else {
                    self.keys.next_id()
                };
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
                // Follows from the identity (see `ResolvedFunction`).
                ..
            })) => {
                self.u8(10);
                self.function_id(id);
                self.u8(kind.code());
                self.data_type(data_type);
                self.bool(*nullable);
                self.call(args, *distinct, filter.as_ref());
                self.call_within_group(e);
            }
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
                self.u8(11);
                self.bound(output);
                self.expr(expr);
            }
            ExprNode::Lambda { params, body } => {
                self.u8(15);
                self.len(params.len());
                params.iter().for_each(|c| self.bound(c));
                self.expr(body);
            }
            ExprNode::Resolved(ResolvedExpr::LambdaVar(c)) => {
                self.u8(16);
                self.bound(c);
            }
            ExprNode::Window(crate::WindowExpr { function, window }) => {
                self.u8(12);
                self.expr(function);
                match window {
                    crate::WindowSpecRef::Named(name) => {
                        self.u8(1);
                        self.str(name);
                    }
                    crate::WindowSpecRef::Spec(spec) => {
                        self.u8(2);
                        self.exprs(&spec.partition_by);
                        self.orders(&spec.order_by);
                        match &spec.frame {
                            None => self.u8(0),
                            Some(f) => {
                                self.u8(match f.units {
                                    crate::FrameUnits::Rows => 1,
                                    crate::FrameUnits::Range => 2,
                                });
                                self.frame_bound(&f.lower);
                                self.frame_bound(&f.upper);
                            }
                        }
                    }
                }
            }
            ExprNode::TypedLiteral { kind, value } => {
                self.u8(13);
                self.u8(match kind {
                    crate::TypedLiteralKind::Date => 1,
                    crate::TypedLiteralKind::Timestamp => 2,
                    crate::TypedLiteralKind::TimestampNtz => 3,
                    crate::TypedLiteralKind::TimestampLtz => 4,
                    crate::TypedLiteralKind::Time => 5,
                });
                self.str(value);
            }
            ExprNode::Field { base, name } => {
                self.u8(14);
                self.str(name);
                self.expr(base);
            }
            ExprNode::Resolved(ResolvedExpr::StateInput { columns, raw_type }) => {
                self.u8(30);
                self.data_type(raw_type);
                self.exprs(columns);
            }
            ExprNode::Resolved(ResolvedExpr::Field {
                base,
                ordinal,
                name,
                data_type,
                nullable,
            }) => {
                self.u8(15);
                self.str(name);
                self.len(*ordinal);
                self.data_type(data_type);
                self.bool(*nullable);
                self.expr(base);
            }
            ExprNode::UpdateFields { .. } => self.update_fields(e),
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
            // A URL's credentials enter as keyed fingerprints, like an
            // option's (see the module docs).
            Literal::String(s) => match SecretUrl::parse(s) {
                Some(url) => {
                    self.u8(17);
                    self.option_value(&OptionValue::Url(url));
                }
                None => {
                    self.u8(10);
                    self.str(s);
                }
            },
            Literal::Binary(b) => {
                self.u8(11);
                self.bytes(b);
            }
            Literal::Map(pairs) => {
                self.u8(18);
                self.len(pairs.len());
                for (k, v) in pairs {
                    self.str(k);
                    self.str(v);
                }
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
