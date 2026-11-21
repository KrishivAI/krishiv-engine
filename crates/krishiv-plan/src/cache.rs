// SPDX-License-Identifier: Apache-2.0
//! Normalized plan hashes and relation-cache keys (P2-T094).
//!
//! [`normalized_plan_hash`] renumbers attribute ids in first-use order, so
//! two analyses of the same query hash equal however each numbered its
//! attributes. [`CacheKey`] adds the pins, the principal id and the policy
//! version: a relation cached under one principal is never substituted into
//! another principal's plan, and a policy-version change makes old entries
//! unreachable.
//!
//! The encoding is a byte string written field by field (length-prefixed
//! strings, little-endian integers, one tag per node and expression form),
//! hashed with XXH3-128. It leaves out origins and renumbers every
//! [`AttrId`] (with a [`PlanNode::Shift`] offset folded in) by first
//! encounter in a pre-order walk: children after the node's own expressions,
//! CTE definitions before the query, subquery plans where they appear.
//! Plans equal up to a consistent renaming of attribute ids hash equal;
//! different literals, names, types or shapes hash differently. Secrets
//! never enter in plaintext: inline secrets hash by their per-process
//! fingerprint, as in the canonical encoding. The hash is an in-process
//! key only (it is never stored in a plan or sent anywhere).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use krishiv_common::secret::{OptionValue, SecretValue, UrlPart};
use twox_hash::XxHash3_128;

use crate::analyzer::{PolicyVersion, TableIdent};
use crate::{
    AttrId, BoundColumn, Expr, ExprNode, FieldOp, Grouping, JoinCondition, JoinKind, LogicalPlan,
    Plan, PlanNode, ReadSource, ResolvedExpr, SqlArgs, TableVersion,
};

/// A cache key: the normalized hash, the pinned table versions, who runs
/// the query and the policy version they run under.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// [`normalized_plan_hash`] of the resolved plan.
    pub plan: u128,
    /// The pinned version per table.
    pub pins: BTreeMap<TableIdent, TableVersion>,
    /// [`Principal::id`](crate::analyzer::Principal::id).
    pub principal: String,
    /// The access-policy version.
    pub policy: PolicyVersion,
    /// The resolution configuration (ANSI mode, session time zone, ...): the
    /// same logical plan returns different values or errors under another.
    pub conf: crate::analyzer::ResolutionConf,
}

impl CacheKey {
    /// The key of an analyzed plan.
    #[must_use]
    pub fn of(analyzed: &crate::analyzer::AnalyzedPlan) -> Self {
        Self {
            plan: normalized_plan_hash(&analyzed.plan),
            pins: analyzed.pins.clone(),
            principal: analyzed.context.principal.clone(),
            policy: analyzed.context.policy,
            conf: analyzed.context.conf,
        }
    }
}

/// The normalized hash of a resolved plan: attribute ids renumbered in
/// first-use order, so two analyses of the same query hash equal.
#[must_use]
pub fn normalized_plan_hash(plan: &Plan) -> u128 {
    normalized_subtree_hash(plan.root())
}

/// The semantic hash of a resolved plan (Spark's `QueryPlan.semanticHash`
/// over the canonicalized plan, `QueryPlan.scala:708-749`): the normalized
/// hash without the cosmetic parts Spark's canonicalization erases. Column
/// and alias names are left out (`Alias(child, "")`, an `AttributeReference`
/// canonicalized as `"none"`, `namedExpressions.scala:326-327`) and a
/// `SubqueryAlias` is its input (`basicLogicalOperators.scala:2031`). Spark's
/// other rewrites (commutative operands sorted, projects that only rename
/// dropped, join hints dropped) are not made, so plans differing only in
/// those still hash differently: it may say "different" where Spark says
/// "same", never the reverse.
#[must_use]
pub fn semantic_plan_hash(plan: &Plan) -> u128 {
    let mut enc = Encoder {
        semantic: true,
        ..Encoder::default()
    };
    enc.plan(plan.root(), 0);
    XxHash3_128::oneshot(&enc.out)
}

/// The normalized hash of a resolved subtree, for subplan substitution: a
/// plan and a subplan of another query with the same result hash equal.
#[must_use]
pub fn normalized_subtree_hash(plan: &LogicalPlan) -> u128 {
    let mut enc = Encoder::default();
    enc.plan(plan, 0);
    XxHash3_128::oneshot(&enc.out)
}

/// [`normalized_subtree_hash`] without the scans' table versions: the same
/// plan over other versions of its tables hashes equal. A relation cache keys
/// its registrations by it and compares the versions apart, so a write to a
/// cached table keeps the registration (Spark's `recacheByPlan`).
#[must_use]
pub fn unversioned_subtree_hash(plan: &LogicalPlan) -> u128 {
    let mut enc = Encoder {
        unversioned: true,
        ..Encoder::default()
    };
    enc.plan(plan, 0);
    XxHash3_128::oneshot(&enc.out)
}

/// The normalized hash of an expression, except that the attribute ids in
/// `pinned` keep their raw values: two copies of one subquery resolved apart
/// hash equal only when they read the same enclosing-query columns in the
/// same places. Literals are hashed whole.
#[must_use]
pub(crate) fn pinned_expr_hash(e: &Expr, pinned: HashSet<u64>) -> u128 {
    let mut enc = Encoder {
        pinned,
        ..Encoder::default()
    };
    enc.expr(e, 0);
    XxHash3_128::oneshot(&enc.out)
}

/// Writes the normalized encoding: bytes plus the attribute-id renumbering.
#[derive(Debug, Default)]
struct Encoder {
    out: Vec<u8>,
    /// Raw ids written as themselves (outside any shift), marked by the top
    /// bit, which no normalized id or attribute id below `MAX_ID` has.
    pinned: HashSet<u64>,
    /// The normalized id per `(raw id, shift bias)`: a [`PlanNode::Shift`]
    /// instance reads as its own attributes, so the bias is part of the key.
    ids: HashMap<(u64, u64), u64>,
    next: u64,
    /// The normalized id per recursive-CTE loop id, in first-use order: the
    /// raw ids come from a process-wide counter, so two lowerings of one query differ in them.
    loops: HashMap<u32, u64>,
    /// The digest of each plan and expression node already encoded, by
    /// address and shift bias. A shared (`Arc`) subtree is reached once per
    /// path; the digest makes that one encoding of its body, and a node
    /// writes only its children's 16-byte digests. A second visit assigns no
    /// new attribute ids (the first did), so a separate copy of the same
    /// subtree has the same digest: shared and copied plans hash equal.
    seen: HashMap<(NodeKind, usize, u64), u128>,
    /// [`semantic_plan_hash`]: names and subquery aliases left out.
    semantic: bool,
    /// Leaves the scans' table versions out ([`unversioned_subtree_hash`]).
    unversioned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum NodeKind {
    Plan,
    Expr,
}

impl Encoder {
    /// The normalized id of `raw` under `bias`, assigned in first-use order.
    fn remap(&mut self, raw: AttrId, bias: u64) -> u64 {
        if bias == 0 && self.pinned.contains(&raw.0) {
            return raw.0 | (1 << 63);
        }
        let key = (raw.0, bias);
        if let Some(n) = self.ids.get(&key) {
            return *n;
        }
        let n = self.next;
        self.next += 1;
        self.ids.insert(key, n);
        n
    }

    fn loop_id(&mut self, raw: u32) -> u64 {
        let next = self.loops.len() as u64;
        *self.loops.entry(raw).or_insert(next)
    }

    fn num(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn inum(&mut self, v: i64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn flag(&mut self, v: bool) {
        self.out.push(u8::from(v));
    }

    fn tag(&mut self, name: &str) {
        self.num(name.len() as u64);
        self.out.extend_from_slice(name.as_bytes());
    }

    fn text(&mut self, s: &str) {
        self.num(s.len() as u64);
        self.out.extend_from_slice(s.as_bytes());
    }

    /// An Arrow schema as names, Spark types and nullability (never data).
    fn arrow_schema(&mut self, schema: &arrow_schema::Schema) {
        self.num(schema.fields().len() as u64);
        for f in schema.fields() {
            self.text(f.name());
            self.text(&crate::local::type_text(f));
            self.flag(f.is_nullable());
        }
    }

    fn debug<T: std::fmt::Debug>(&mut self, v: &T) {
        let s = format!("{v:?}");
        self.text(&s);
    }

    fn bound(&mut self, c: &BoundColumn, bias: u64) {
        let id = self.remap(c.id, bias);
        self.num(id);
        if !self.semantic {
            self.text(&c.name);
        }
        self.debug(&c.data_type);
        self.flag(c.nullable);
    }

    fn attr(&mut self, id: AttrId, bias: u64) {
        let id = self.remap(id, bias);
        self.num(id);
    }

    fn option_value(&mut self, v: &OptionValue) {
        match v {
            OptionValue::Plain(s) => {
                self.tag("plain");
                self.text(s);
            }
            OptionValue::Secret(s) => match s {
                SecretValue::Reference { scope, key } => {
                    self.tag("reference");
                    self.text(scope);
                    self.text(key);
                }
                // A keyed fingerprint, as in the canonical encoding:
                // equal secrets hash equal within a process, and the
                // plaintext never enters the key.
                SecretValue::Inline(s) => {
                    self.tag("inline");
                    self.out.extend_from_slice(&s.fingerprint());
                }
            },
            OptionValue::Url(u) => {
                self.tag("url");
                self.num(u.parts().len() as u64);
                for part in u.parts() {
                    match part {
                        UrlPart::Text(s) => {
                            self.tag("text");
                            self.text(s);
                        }
                        UrlPart::Secret(s) => {
                            self.tag("secret");
                            self.out.extend_from_slice(&s.fingerprint());
                        }
                    }
                }
            }
        }
    }

    fn source(&mut self, s: &ReadSource, _bias: u64) {
        match s {
            ReadSource::Table { name, options } => {
                self.tag("table");
                self.num(name.len() as u64);
                for part in name {
                    self.text(part);
                }
                self.num(options.len() as u64);
                for (key, value) in options {
                    self.text(key);
                    self.option_value(value);
                }
            }
            ReadSource::Local(l) => {
                self.tag("local");
                self.text(&l.relation().hash.hex());
                self.arrow_schema(l.schema());
                self.num(l.batches() as u64);
                self.num(l.rows());
            }
            ReadSource::Files {
                format,
                paths,
                options,
                schema,
            } => {
                self.tag("files");
                self.text(format);
                self.num(paths.len() as u64);
                for path in paths {
                    self.option_value(path);
                }
                self.num(options.len() as u64);
                for (key, value) in options {
                    self.text(key);
                    self.option_value(value);
                }
                // Only when declared, so a read that infers keeps its key.
                if let Some(schema) = schema {
                    self.tag("schema");
                    self.debug(schema);
                }
            }
        }
    }

    fn exprs(&mut self, exprs: &[Arc<Expr>], bias: u64) {
        self.num(exprs.len() as u64);
        for e in exprs {
            self.expr(e, bias);
        }
    }

    /// Writes `e` as the digest of its encoding, computed once per node.
    fn expr(&mut self, e: &Expr, bias: u64) {
        let key = (NodeKind::Expr, std::ptr::from_ref(e) as usize, bias);
        let digest = match self.seen.get(&key) {
            Some(d) => *d,
            None => {
                let outer = std::mem::take(&mut self.out);
                self.expr_body(e, bias);
                let d = XxHash3_128::oneshot(&std::mem::replace(&mut self.out, outer));
                self.seen.insert(key, d);
                d
            }
        };
        self.out.extend_from_slice(&digest.to_le_bytes());
    }

    /// A star's `|> SET` assignments (out of line: `expr_body` recurses,
    /// and its frame must stay small).
    #[inline(never)]
    fn replacements(&mut self, replace: &[(String, Arc<Expr>)], bias: u64) {
        for (name, expr) in replace {
            self.text(name);
            self.expr(expr, bias);
        }
    }

    fn expr_body(&mut self, e: &Expr, bias: u64) {
        match &e.node {
            ExprNode::Column(c) => {
                self.tag("column");
                self.num(c.name.len() as u64);
                for part in &c.name {
                    self.text(part);
                }
                self.flag(c.plan_id.is_some());
                if let Some(id) = c.plan_id {
                    self.inum(id.0);
                }
            }
            ExprNode::Literal(l) => {
                self.tag("literal");
                self.debug(l);
            }
            ExprNode::TypedLiteral { kind, value } => {
                self.tag("typed-literal");
                self.debug(kind);
                self.text(value);
            }
            ExprNode::Star(s) => {
                self.tag("star");
                self.num(s.qualifier.len() as u64);
                for part in &s.qualifier {
                    self.text(part);
                }
                self.num(s.except.len() as u64);
                for path in &s.except {
                    self.num(path.len() as u64);
                    for part in path {
                        self.text(part);
                    }
                }
                self.flag(s.plan_id.is_some());
                self.replacements(&s.replace, bias);
                self.num(s.position.map_or(0, |n| n as u64));
            }
            ExprNode::Function(f) => {
                self.tag("function");
                self.text(&f.name);
                self.num(f.qualifier.len() as u64);
                for part in &f.qualifier {
                    self.text(part);
                }
                self.exprs(&f.args, bias);
                self.num(f.arg_names.len() as u64);
                for name in &f.arg_names {
                    self.flag(name.is_some());
                    if let Some(name) = name {
                        self.text(name);
                    }
                }
                self.flag(f.distinct);
                self.flag(f.filter.is_some());
                if let Some(filter) = &f.filter {
                    self.expr(filter, bias);
                }
                self.within_group(&f.within_group, bias);
            }
            ExprNode::Lambda { params, body } => {
                self.tag("lambda");
                self.num(params.len() as u64);
                for param in params {
                    self.bound(param, bias);
                }
                self.expr(body, bias);
            }
            ExprNode::Field { base, name } => {
                self.tag("field");
                self.expr(base, bias);
                self.text(name);
            }
            ExprNode::UpdateFields { base, ops } => {
                self.tag("update-fields");
                self.expr(base, bias);
                self.num(ops.len() as u64);
                for op in ops {
                    match op {
                        FieldOp::With { name, value } => {
                            self.tag("with");
                            self.text(name);
                            self.expr(value, bias);
                        }
                        FieldOp::Drop { name } => {
                            self.tag("drop");
                            self.text(name);
                        }
                    }
                }
            }
            ExprNode::Cast { expr, to, mode } => {
                self.tag("cast");
                self.expr(expr, bias);
                self.debug(to);
                self.debug(mode);
            }
            ExprNode::StoredField { expr, comment } => {
                self.tag("stored_field");
                self.expr(expr, bias);
                self.debug(comment);
            }
            ExprNode::Collate {
                expr,
                collation,
                name,
            } => {
                self.tag("collate");
                self.expr(expr, bias);
                self.debug(collation);
                self.text(name);
            }
            ExprNode::Alias { expr, name } => {
                self.tag("alias");
                self.expr(expr, bias);
                self.text(name);
            }
            ExprNode::Pipe { expr, clause } => {
                self.tag("pipe");
                self.expr(expr, bias);
                self.debug(clause);
            }
            ExprNode::MultiAlias { expr, names } => {
                self.tag("multi-alias");
                self.expr(expr, bias);
                names.iter().for_each(|n| self.text(n));
            }
            ExprNode::Subquery { kind, plan } => {
                self.tag("subquery");
                match kind {
                    crate::SubqueryKind::Scalar => self.tag("scalar"),
                    crate::SubqueryKind::Exists => self.tag("exists"),
                    crate::SubqueryKind::In(values) => {
                        self.tag("in");
                        self.exprs(values, bias);
                    }
                    crate::SubqueryKind::Table(t) => {
                        self.tag("table");
                        self.exprs(&t.partition_by, bias);
                        self.num(t.order_by.len() as u64);
                        for o in &t.order_by {
                            self.order(o, bias);
                        }
                        self.flag(t.single_partition);
                    }
                }
                self.plan(plan, bias);
            }
            ExprNode::Window(w) => {
                self.tag("window");
                self.expr(&w.function, bias);
                match &w.window {
                    crate::WindowSpecRef::Spec(s) => {
                        self.tag("spec");
                        self.exprs(&s.partition_by, bias);
                        self.num(s.order_by.len() as u64);
                        for o in &s.order_by {
                            self.order(o, bias);
                        }
                        self.flag(s.frame.is_some());
                        if let Some(f) = &s.frame {
                            self.debug(&f.units);
                            self.debug(&f.lower);
                            self.debug(&f.upper);
                        }
                    }
                    crate::WindowSpecRef::Named(n) => {
                        self.tag("named");
                        self.text(n);
                    }
                }
            }
            ExprNode::Resolved(r) => self.resolved(r, bias),
        }
    }

    /// A call's `WITHIN GROUP` ordering (written only when there is one, so
    /// the keys of calls without it do not change).
    fn within_group(&mut self, orders: &[crate::SortOrder], bias: u64) {
        if orders.is_empty() {
            return;
        }
        self.tag("within-group");
        self.num(orders.len() as u64);
        for o in orders {
            self.order(o, bias);
        }
    }

    fn order(&mut self, o: &crate::SortOrder, bias: u64) {
        self.expr(&o.expr, bias);
        self.debug(&o.direction);
        self.debug(&o.nulls);
    }

    fn resolved(&mut self, r: &ResolvedExpr, bias: u64) {
        match r {
            ResolvedExpr::Column(c) => {
                self.tag("rcolumn");
                self.bound(c, bias);
            }
            ResolvedExpr::OuterColumn(c) => {
                self.tag("router");
                self.bound(c, bias);
            }
            ResolvedExpr::LambdaVar(c) => {
                self.tag("rlambda");
                self.bound(c, bias);
            }
            ResolvedExpr::Function(f) => {
                self.tag("rfunction");
                // The canonical name only: a registry handle is an
                // in-process lookup, never part of a key, as in the
                // canonical encoding. A session UDF's version is part of
                // its identity (a re-registration, or another unregistered
                // Python UDF of the same name, is another function: Spark
                // compares a `PythonUDF`'s pickled function).
                self.text(f.id.name());
                match &f.id {
                    crate::FunctionId::PythonUdf { version, .. }
                    | crate::FunctionId::Udf { version, .. } => self.num(u64::from(*version)),
                    _ => self.num(0),
                }
                self.debug(&f.kind);
                self.exprs(&f.args, bias);
                self.flag(f.distinct);
                self.flag(f.filter.is_some());
                if let Some(filter) = &f.filter {
                    self.expr(filter, bias);
                }
                self.within_group(&f.within_group, bias);
                self.debug(&f.data_type);
                self.flag(f.nullable);
            }
            ResolvedExpr::Field {
                base,
                ordinal,
                name,
                data_type,
                nullable,
            } => {
                self.tag("rfield");
                self.expr(base, bias);
                self.num(*ordinal as u64);
                self.text(name);
                self.debug(data_type);
                self.flag(*nullable);
            }
            ResolvedExpr::StateInput { columns, raw_type } => {
                self.tag("rstate");
                self.exprs(columns, bias);
                self.debug(raw_type);
            }
            ResolvedExpr::Alias { expr, output } => {
                self.tag("ralias");
                self.expr(expr, bias);
                self.bound(output, bias);
            }
            ResolvedExpr::EvalMode {
                ansi,
                expr,
                data_type,
                nullable,
            } => {
                self.tag("revalmode");
                self.debug(ansi);
                self.debug(data_type);
                self.debug(nullable);
                self.expr(expr, bias);
            }
        }
    }

    fn grouping(&mut self, g: &Grouping, bias: u64) {
        match g {
            Grouping::GroupBy(keys) => {
                self.tag("groupby");
                self.exprs(keys, bias);
            }
            Grouping::Rollup(keys) => {
                self.tag("rollup");
                self.exprs(keys, bias);
            }
            Grouping::Cube(keys) => {
                self.tag("cube");
                self.exprs(keys, bias);
            }
            Grouping::GroupingSets { exprs, sets } => {
                self.tag("grouping-sets");
                self.exprs(exprs, bias);
                self.num(sets.len() as u64);
                for set in sets {
                    self.exprs(set, bias);
                }
            }
        }
    }

    /// A relation node's Python UDF (by identity) and declared columns.
    fn python_udf(&mut self, udf: &crate::PythonUdfRef, output: &[BoundColumn], bias: u64) {
        self.debug(udf);
        self.num(output.len() as u64);
        for c in output {
            self.bound(c, bias);
        }
    }

    /// Writes `plan` as the digest of its encoding, computed once per node.
    fn plan(&mut self, plan: &LogicalPlan, bias: u64) {
        if self.semantic
            && let PlanNode::SubqueryAlias { input, .. } = &plan.node
        {
            return self.plan(input, bias);
        }
        let key = (NodeKind::Plan, std::ptr::from_ref(plan) as usize, bias);
        let digest = match self.seen.get(&key) {
            Some(d) => *d,
            None => {
                let outer = std::mem::take(&mut self.out);
                self.plan_body(plan, bias);
                let d = XxHash3_128::oneshot(&std::mem::replace(&mut self.out, outer));
                self.seen.insert(key, d);
                d
            }
        };
        self.out.extend_from_slice(&digest.to_le_bytes());
    }

    fn plan_body(&mut self, plan: &LogicalPlan, bias: u64) {
        match &plan.node {
            PlanNode::Read(s) => {
                self.tag("Read");
                self.source(s, bias);
            }
            PlanNode::Sql { query, args } => {
                self.tag("Sql");
                self.text(query);
                match args {
                    SqlArgs::None => self.tag("no-args"),
                    SqlArgs::Positional(v) => {
                        self.tag("positional");
                        self.exprs(v, bias);
                    }
                    SqlArgs::Named(m) => {
                        self.tag("named");
                        self.num(m.len() as u64);
                        for (key, value) in m {
                            self.text(key);
                            self.expr(value, bias);
                        }
                    }
                }
            }
            PlanNode::Scan(s) => {
                self.tag("Scan");
                self.source(&s.source, bias);
                self.num(s.output.len() as u64);
                for c in &s.output {
                    self.bound(c, bias);
                }
                if !self.unversioned {
                    self.flag(s.version.is_some());
                    if let Some(v) = &s.version {
                        self.inum(v.0);
                    }
                }
            }
            PlanNode::OneRowRelation => self.tag("OneRowRelation"),
            PlanNode::TableFunction { name, args } => {
                self.tag("TableFunction");
                self.text(name);
                self.exprs(args, bias);
            }
            PlanNode::Range {
                start,
                end,
                step,
                num_slices,
                output,
            } => {
                self.tag("Range");
                self.inum(*start);
                self.inum(*end);
                self.inum(*step);
                self.flag(num_slices.is_some());
                if let Some(n) = num_slices {
                    self.num(u64::from(*n));
                }
                self.bound(output, bias);
            }
            PlanNode::Project { input, exprs } => {
                self.tag("Project");
                self.exprs(exprs, bias);
                self.plan(input, bias);
            }
            PlanNode::WithColumns { input, columns } => {
                self.tag("WithColumns");
                self.num(columns.len() as u64);
                for (name, value) in columns {
                    self.text(name);
                    self.expr(value, bias);
                }
                self.plan(input, bias);
            }
            PlanNode::Filter { input, predicate } => {
                self.tag("Filter");
                self.expr(predicate, bias);
                self.plan(input, bias);
            }
            PlanNode::PartialAggregate {
                input,
                keys,
                aggregates,
                states,
            } => {
                self.tag("PartialAggregate");
                self.exprs(keys, bias);
                self.exprs(aggregates, bias);
                for c in states {
                    self.bound(c, bias);
                }
                self.plan(input, bias);
            }
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => {
                self.tag("Aggregate");
                self.grouping(grouping, bias);
                self.exprs(aggregates, bias);
                self.plan(input, bias);
            }
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => {
                self.tag("Join");
                if let JoinKind::Existence(id) = kind {
                    self.tag("existence");
                    self.attr(*id, bias);
                } else {
                    self.debug(kind);
                }
                match condition {
                    JoinCondition::None => self.tag("no-cond"),
                    JoinCondition::On(e) => {
                        self.tag("on");
                        self.expr(e, bias);
                    }
                    JoinCondition::Using(cols) => {
                        self.tag("using");
                        self.num(cols.len() as u64);
                        for c in cols {
                            self.text(c);
                        }
                    }
                    JoinCondition::Natural => self.tag("natural"),
                }
                self.plan(left, bias);
                self.plan(right, bias);
            }
            PlanNode::AsOfJoin {
                left,
                right,
                kind,
                condition,
                as_of,
            } => {
                self.tag("AsOfJoin");
                self.debug(kind);
                match condition {
                    JoinCondition::None => self.tag("no-cond"),
                    JoinCondition::On(e) => {
                        self.tag("on");
                        self.expr(e, bias);
                    }
                    JoinCondition::Using(cols) => {
                        self.tag("using");
                        self.num(cols.len() as u64);
                        for c in cols {
                            self.text(c);
                        }
                    }
                    JoinCondition::Natural => self.tag("natural"),
                }
                self.expr(&as_of.left, bias);
                self.expr(&as_of.right, bias);
                self.debug(&as_of.direction);
                self.flag(as_of.allow_exact);
                self.flag(as_of.as_written);
                self.flag(as_of.distance_on_every_pair);
                match &as_of.tolerance {
                    None => self.tag("no-tolerance"),
                    Some(t) => self.expr(t, bias),
                }
                match &as_of.distance {
                    None => self.tag("no-distance"),
                    Some(d) => self.expr(d, bias),
                }
                self.plan(left, bias);
                self.plan(right, bias);
            }
            PlanNode::Sort {
                input,
                order,
                global,
            } => {
                self.tag("Sort");
                self.num(order.len() as u64);
                for o in order {
                    self.order(o, bias);
                }
                self.flag(*global);
                self.plan(input, bias);
            }
            PlanNode::Limit { input, limit } => {
                self.tag("Limit");
                self.expr(limit, bias);
                self.plan(input, bias);
            }
            PlanNode::Offset { input, offset } => {
                self.tag("Offset");
                self.expr(offset, bias);
                self.plan(input, bias);
            }
            PlanNode::Tail { input, limit } => {
                self.tag("Tail");
                self.expr(limit, bias);
                self.plan(input, bias);
            }
            PlanNode::SubqueryAlias {
                input,
                alias,
                columns,
            } => {
                self.tag("SubqueryAlias");
                self.text(alias);
                self.num(columns.len() as u64);
                for c in columns {
                    self.text(c);
                }
                self.plan(input, bias);
            }
            PlanNode::WithCte { input, ctes } => {
                self.tag("WithCte");
                self.num(ctes.len() as u64);
                for cte in ctes {
                    self.text(&cte.name);
                    self.plan(&cte.plan, bias);
                }
                self.plan(input, bias);
            }
            PlanNode::CteRef(r) => {
                self.tag("CteRef");
                self.text(&r.name);
                self.num(r.source.len() as u64);
                for id in &r.source {
                    self.attr(*id, bias);
                }
                self.num(r.output.len() as u64);
                for c in &r.output {
                    self.bound(c, bias);
                }
            }
            PlanNode::Shift { input, offset } => {
                self.tag("Shift");
                self.num(*offset);
                self.plan(input, bias.wrapping_add(*offset));
            }
            PlanNode::Hint {
                input,
                name,
                params,
            } => {
                self.tag("Hint");
                self.text(name);
                self.exprs(params, bias);
                self.plan(input, bias);
            }
            PlanNode::PipeOperator { input } => {
                self.tag("PipeOperator");
                self.plan(input, bias);
            }
            PlanNode::ColumnNames { input, op } => {
                self.tag("ColumnNames");
                match op {
                    crate::ColumnNamesOp::ToDf(names) => {
                        self.num(0);
                        self.num(names.len() as u64);
                        for n in names {
                            self.text(n);
                        }
                    }
                    crate::ColumnNamesOp::Rename(pairs) => {
                        self.num(1);
                        self.num(pairs.len() as u64);
                        for (from, to) in pairs {
                            self.text(from);
                            self.text(to);
                        }
                    }
                    crate::ColumnNamesOp::Drop { names, columns } => {
                        self.num(2);
                        self.num(names.len() as u64);
                        for n in names {
                            self.text(n);
                        }
                        self.num(columns.len() as u64);
                        for n in columns {
                            self.text(n);
                        }
                    }
                    crate::ColumnNamesOp::DropNa {
                        subset,
                        min_non_nulls,
                    } => {
                        self.num(3);
                        self.num(subset.len() as u64);
                        for n in subset {
                            self.text(n);
                        }
                        self.num(min_non_nulls.map_or(0, |n| u64::from(n) + 1));
                    }
                    crate::ColumnNamesOp::Replace { cols, pairs } => {
                        self.num(5);
                        self.num(cols.len() as u64);
                        for n in cols {
                            self.text(n);
                        }
                        self.exprs(pairs, bias);
                    }
                    crate::ColumnNamesOp::FillNa { cols, values } => {
                        self.num(4);
                        self.num(cols.len() as u64);
                        for n in cols {
                            self.text(n);
                        }
                        self.exprs(values, bias);
                    }
                }
                self.plan(input, bias);
            }
            PlanNode::Repartition { input, exprs } => {
                self.tag("Repartition");
                self.exprs(exprs, bias);
                self.plan(input, bias);
            }
            PlanNode::CollectMetrics {
                input,
                name,
                metrics,
            } => {
                self.tag("CollectMetrics");
                self.text(name);
                self.exprs(metrics, bias);
                self.plan(input, bias);
            }
            PlanNode::JoinHint { input, strategy } => {
                self.tag("JoinHint");
                self.debug(strategy);
                self.plan(input, bias);
            }
            PlanNode::Union {
                inputs,
                by_name,
                allow_missing_columns,
            } => {
                self.tag("Union");
                self.flag(*by_name);
                self.flag(*allow_missing_columns);
                self.num(inputs.len() as u64);
                for input in inputs {
                    self.plan(input, bias);
                }
            }
            PlanNode::Distinct { input } => {
                self.tag("Distinct");
                self.plan(input, bias);
            }
            PlanNode::SetOp {
                left,
                right,
                op,
                all,
            } => {
                self.tag("SetOp");
                self.debug(op);
                self.flag(*all);
                self.plan(left, bias);
                self.plan(right, bias);
            }
            PlanNode::Window {
                input,
                exprs,
                partition_by,
                order_by,
            } => {
                self.tag("Window");
                self.exprs(exprs, bias);
                self.exprs(partition_by, bias);
                self.num(order_by.len() as u64);
                for o in order_by {
                    self.order(o, bias);
                }
                self.plan(input, bias);
            }
            PlanNode::MapPartitions {
                input,
                udf,
                output,
                barrier,
            } => {
                self.tag("MapPartitions");
                self.python_udf(udf, output, bias);
                self.flag(*barrier);
                self.plan(input, bias);
            }
            PlanNode::GroupMap {
                input,
                keys,
                udf,
                output,
            } => {
                self.tag("GroupMap");
                self.exprs(keys, bias);
                self.python_udf(udf, output, bias);
                self.plan(input, bias);
            }
            PlanNode::CoGroupMap {
                left,
                right,
                left_keys,
                right_keys,
                udf,
                output,
            } => {
                self.tag("CoGroupMap");
                self.exprs(left_keys, bias);
                self.exprs(right_keys, bias);
                self.python_udf(udf, output, bias);
                self.plan(left, bias);
                self.plan(right, bias);
            }
            PlanNode::PythonTableFunction {
                args,
                table_arg,
                udf,
                output,
                lateral,
                analyzed,
            } => {
                self.tag("PythonTableFunction");
                self.exprs(args, bias);
                self.python_udf(udf, output, bias);
                self.flag(table_arg.is_some());
                if let Some(t) = table_arg {
                    self.exprs(&t.partition_by, bias);
                    self.num(t.order_by.len() as u64);
                    for o in &t.order_by {
                        self.order(o, bias);
                    }
                    self.flag(t.single_partition);
                    self.exprs(&t.select, bias);
                    self.num(t.position as u64);
                    self.flag(t.name.is_some());
                    if let Some(n) = &t.name {
                        self.text(n);
                    }
                    self.plan(&t.input, bias);
                }
                self.flag(analyzed.is_some());
                if let Some(bytes) = analyzed {
                    self.num(bytes.len() as u64);
                    for b in bytes.iter() {
                        self.num(u64::from(*b));
                    }
                }
                self.flag(lateral.is_some());
                if let Some(l) = lateral {
                    self.plan(l, bias);
                }
            }
            PlanNode::ReplicateRows { input, count } => {
                self.tag("ReplicateRows");
                self.expr(count, bias);
                self.plan(input, bias);
            }
            PlanNode::Sample {
                input,
                bounds,
                seed,
            } => {
                self.tag("Sample");
                self.num(bounds.lower.to_bits());
                self.num(bounds.upper.to_bits());
                match seed {
                    Some(s) => {
                        self.tag("seed");
                        self.num(s.cast_unsigned());
                    }
                    None => self.tag("unseeded"),
                }
                self.plan(input, bias);
            }
            PlanNode::Generate {
                input,
                kind,
                outer,
                arg,
                output,
                names,
                qualifier,
            } => {
                self.tag("Generate");
                self.num(*kind as u64);
                self.flag(*outer);
                self.expr(arg, bias);
                self.num(output.len() as u64);
                for c in output {
                    self.bound(c, bias);
                }
                self.num(names.len() as u64);
                for n in names {
                    self.text(n);
                }
                self.flag(qualifier.is_some());
                if let Some(q) = qualifier {
                    self.text(q);
                }
                self.plan(input, bias);
            }
            PlanNode::UnionLoop {
                id,
                anchor,
                recursive,
                max_level,
                limit,
                output,
            } => {
                self.tag("UnionLoop");
                let id = self.loop_id(*id);
                self.num(id);
                self.flag(max_level.is_some());
                if let Some(n) = max_level {
                    self.num(u64::from(*n));
                }
                self.flag(limit.is_some());
                if let Some(n) = limit {
                    self.num(*n);
                }
                self.num(output.len() as u64);
                for c in output {
                    self.bound(c, bias);
                }
                self.plan(anchor, bias);
                self.plan(recursive, bias);
            }
            PlanNode::UnionLoopRef { id, output } => {
                self.tag("UnionLoopRef");
                let id = self.loop_id(*id);
                self.num(id);
                self.num(output.len() as u64);
                for c in output {
                    self.bound(c, bias);
                }
            }
            PlanNode::Expand {
                input,
                projections,
                output,
            } => {
                self.tag("Expand");
                self.num(projections.len() as u64);
                for projection in projections {
                    self.exprs(projection, bias);
                }
                self.num(output.len() as u64);
                for c in output {
                    self.bound(c, bias);
                }
                self.plan(input, bias);
            }
            PlanNode::Pivot {
                input,
                pivot_columns,
                values,
                aggregates,
            } => {
                self.tag("Pivot");
                self.exprs(pivot_columns, bias);
                self.num(values.len() as u64);
                for v in values {
                    self.exprs(&v.exprs, bias);
                    self.flag(v.alias.is_some());
                    if let Some(alias) = &v.alias {
                        self.text(alias);
                    }
                }
                self.exprs(aggregates, bias);
                self.plan(input, bias);
            }
            PlanNode::Unpivot {
                input,
                value_columns,
                name_column,
                column_sets,
                include_nulls,
            } => {
                self.tag("Unpivot");
                self.num(value_columns.len() as u64);
                for c in value_columns {
                    self.text(c);
                }
                self.text(name_column);
                self.num(column_sets.len() as u64);
                for set in column_sets {
                    self.num(set.columns.len() as u64);
                    for c in &set.columns {
                        self.expr(c, bias);
                    }
                    self.flag(set.alias.is_some());
                    if let Some(alias) = &set.alias {
                        self.text(alias);
                    }
                }
                self.flag(*include_nulls);
                self.plan(input, bias);
            }
            PlanNode::InlineTable { rows, columns } => {
                self.tag("InlineTable");
                self.num(columns.len() as u64);
                for c in columns {
                    self.text(c);
                }
                self.num(rows.len() as u64);
                for row in rows {
                    self.exprs(row, bias);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
