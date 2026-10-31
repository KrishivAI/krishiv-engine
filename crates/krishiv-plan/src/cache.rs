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

use std::collections::{BTreeMap, HashMap};
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

/// The normalized hash of a resolved subtree, for subplan substitution: a
/// plan and a subplan of another query with the same result hash equal.
#[must_use]
pub fn normalized_subtree_hash(plan: &LogicalPlan) -> u128 {
    let mut enc = Encoder::default();
    enc.plan(plan, 0);
    XxHash3_128::oneshot(&enc.out)
}

/// Writes the normalized encoding: bytes plus the attribute-id renumbering.
#[derive(Debug, Default)]
struct Encoder {
    out: Vec<u8>,
    /// The normalized id per `(raw id, shift bias)`: a [`PlanNode::Shift`]
    /// instance reads as its own attributes, so the bias is part of the key.
    ids: HashMap<(u64, u64), u64>,
    next: u64,
    /// The digest of each plan and expression node already encoded, by
    /// address and shift bias. A shared (`Arc`) subtree is reached once per
    /// path; the digest makes that one encoding of its body, and a node
    /// writes only its children's 16-byte digests. A second visit assigns no
    /// new attribute ids (the first did), so a separate copy of the same
    /// subtree has the same digest: shared and copied plans hash equal.
    seen: HashMap<(NodeKind, usize, u64), u128>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum NodeKind {
    Plan,
    Expr,
}

impl Encoder {
    /// The normalized id of `raw` under `bias`, assigned in first-use order.
    fn remap(&mut self, raw: AttrId, bias: u64) -> u64 {
        let key = (raw.0, bias);
        if let Some(n) = self.ids.get(&key) {
            return *n;
        }
        let n = self.next;
        self.next += 1;
        self.ids.insert(key, n);
        n
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

    fn debug<T: std::fmt::Debug>(&mut self, v: &T) {
        let s = format!("{v:?}");
        self.text(&s);
    }

    fn bound(&mut self, c: &BoundColumn, bias: u64) {
        let id = self.remap(c.id, bias);
        self.num(id);
        self.text(&c.name);
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
            ReadSource::Files {
                format,
                paths,
                options,
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
            ExprNode::Alias { expr, name } => {
                self.tag("alias");
                self.expr(expr, bias);
                self.text(name);
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
                // canonical encoding.
                self.text(f.id.name());
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

    /// Writes `plan` as the digest of its encoding, computed once per node.
    fn plan(&mut self, plan: &LogicalPlan, bias: u64) {
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
                self.flag(s.version.is_some());
                if let Some(v) = &s.version {
                    self.inum(v.0);
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
            PlanNode::ReplicateRows { input, count } => {
                self.tag("ReplicateRows");
                self.expr(count, bias);
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
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use krishiv_types::DataType;

    use crate::{Builtin, FunctionId, FunctionKind, Literal, ResolvedFunction};

    fn bound(id: u64, name: &str) -> BoundColumn {
        BoundColumn {
            id: AttrId(id),
            name: name.to_owned(),
            data_type: DataType::Integer,
            nullable: false,
        }
    }

    fn rcol(id: u64, name: &str) -> Arc<Expr> {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(bound(
            id, name,
        )))))
    }

    fn scan(ids: &[u64], names: &[&str]) -> LogicalPlan {
        let output = ids
            .iter()
            .zip(names.iter())
            .map(|(id, name)| bound(*id, name))
            .collect();
        LogicalPlan::new(PlanNode::Scan(crate::Scan {
            source: ReadSource::Table {
                name: vec!["t".to_owned()],
                options: BTreeMap::new(),
            },
            output,
            version: None,
        }))
    }

    fn project(input: LogicalPlan, cols: &[Arc<Expr>]) -> LogicalPlan {
        LogicalPlan::new(PlanNode::Project {
            input: Arc::new(input),
            exprs: cols.to_vec(),
        })
    }

    fn filter(input: LogicalPlan, predicate: Expr) -> LogicalPlan {
        LogicalPlan::new(PlanNode::Filter {
            input: Arc::new(input),
            predicate: Arc::new(predicate),
        })
    }

    fn gt(left: Arc<Expr>, value: i32) -> Expr {
        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                within_group: Vec::new(),
                id: FunctionId::Builtin(Builtin::Gt),
                kind: FunctionKind::Scalar,
                args: vec![left, Arc::new(Expr::literal(Literal::Integer(value)))],
                distinct: false,
                filter: None,
                data_type: DataType::Boolean,
                nullable: false,
                deterministic: true,
            },
        )))
    }

    fn hash(plan: &LogicalPlan) -> u128 {
        let plan = Plan::try_new(plan.clone()).unwrap();
        normalized_plan_hash(&plan)
    }

    #[test]
    fn renumbered_attribute_ids_hash_equal() {
        let a = project(scan(&[1, 2], &["a", "b"]), &[rcol(1, "a"), rcol(2, "b")]);
        let b = project(
            scan(&[41, 42], &["a", "b"]),
            &[rcol(41, "a"), rcol(42, "b")],
        );
        assert_eq!(hash(&a), hash(&b));
    }

    /// Calls that differ only in their `WITHIN GROUP` ordering hash apart.
    #[test]
    fn within_group_orderings_enter_the_hash() {
        use crate::{NullOrdering, SortDirection, SortOrder};
        let call = |col: u64, dir: SortDirection, nulls: NullOrdering, ordered: bool| {
            let within_group = if ordered {
                vec![SortOrder {
                    expr: rcol(col, "a"),
                    direction: dir,
                    nulls,
                }]
            } else {
                Vec::new()
            };
            let f = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    within_group,
                    id: FunctionId::Builtin(Builtin::Gt),
                    kind: FunctionKind::Aggregate,
                    args: vec![rcol(1, "a")],
                    distinct: false,
                    filter: None,
                    data_type: DataType::Boolean,
                    nullable: true,
                    deterministic: true,
                },
            ))));
            project(scan(&[1, 2], &["a", "b"]), &[f])
        };
        use NullOrdering::{NullsFirst, NullsLast};
        use SortDirection::{Ascending, Descending};
        let all = [
            call(1, Ascending, NullsFirst, true),
            call(1, Descending, NullsFirst, true),
            call(1, Ascending, NullsLast, true),
            call(2, Ascending, NullsFirst, true),
            call(1, Ascending, NullsFirst, false),
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(hash(a), hash(b));
            }
        }
        assert_eq!(hash(&all[0]), hash(&call(1, Ascending, NullsFirst, true)));
    }

    #[test]
    fn different_literals_hash_differently() {
        let a = filter(scan(&[1], &["a"]), gt(rcol(1, "a"), 1));
        let b = filter(scan(&[1], &["a"]), gt(rcol(1, "a"), 2));
        assert_ne!(hash(&a), hash(&b));
    }

    #[test]
    fn different_columns_hash_differently() {
        let a = project(scan(&[1, 2], &["a", "b"]), &[rcol(1, "a")]);
        let b = project(scan(&[1, 2], &["a", "b"]), &[rcol(2, "b")]);
        assert_ne!(hash(&a), hash(&b));
        // ... as do different names on the same ids.
        let c = project(scan(&[1], &["a"]), &[rcol(1, "a")]);
        let d = project(scan(&[1], &["z"]), &[rcol(1, "z")]);
        assert_ne!(hash(&c), hash(&d));
    }

    #[test]
    fn range_and_table_function_hash_every_field() {
        let range = |start: i64, end: i64, step: i64, slices: Option<u32>, id: u64| {
            LogicalPlan::new(PlanNode::Range {
                start,
                end,
                step,
                num_slices: slices,
                output: bound(id, "id"),
            })
        };
        let base = range(0, 10, 1, None, 1);
        assert_eq!(hash(&base), hash(&range(0, 10, 1, None, 1)));
        // Renumbered attribute ids normalise away.
        assert_eq!(hash(&base), hash(&range(0, 10, 1, None, 77)));
        for other in [
            range(1, 10, 1, None, 1),
            range(0, 11, 1, None, 1),
            range(0, 10, 2, None, 1),
            range(0, 10, 1, Some(4), 1),
        ] {
            assert_ne!(hash(&base), hash(&other));
        }
        assert_ne!(
            hash(&range(0, 10, 1, Some(4), 1)),
            hash(&range(0, 10, 1, Some(5), 1))
        );
        let call = |name: &str, args: Vec<Arc<Expr>>| {
            LogicalPlan::new(PlanNode::TableFunction {
                name: name.to_owned(),
                args,
            })
        };
        let one = Arc::new(Expr::literal(Literal::Integer(1)));
        assert_ne!(
            hash(&call("range", vec![])),
            hash(&call("sql_keywords", vec![]))
        );
        assert_ne!(
            hash(&call("range", vec![])),
            hash(&call("range", vec![one]))
        );
    }

    /// One plan per node form and one expression per expression form: each
    /// hashes the same on a second encoding and differently from every
    /// other, so no form shares a tag or drops a field.
    #[test]
    fn every_node_and_expression_form_hashes_distinctly() {
        use crate::{
            ColumnRef, Cte, CteRef, FrameBound, FrameUnits, FunctionCall, JoinKind, JoinStrategy,
            NullOrdering, PivotValue, PlanId, SetOperation, SortDirection, SortOrder, Star,
            SubqueryKind, TypedLiteralKind, UnpivotColumnSet, WindowExpr, WindowFrame, WindowSpec,
            WindowSpecRef,
        };

        let base = || Arc::new(scan(&[1], &["a"]));
        let a = || rcol(1, "a");
        let lit = |v: i32| Arc::new(Expr::literal(Literal::Integer(v)));
        let unresolved = |node: ExprNode| Arc::new(Expr::new(node));
        let call = |name: &str| FunctionCall {
            within_group: Vec::new(),
            name: name.to_owned(),
            qualifier: Vec::new(),
            args: vec![lit(1)],
            arg_names: vec![Some("x".to_owned())],
            distinct: true,
            filter: Some(lit(2)),
        };
        let project = |e: Arc<Expr>| {
            LogicalPlan::new(PlanNode::Project {
                input: base(),
                exprs: vec![e],
            })
        };
        let node = |n: PlanNode| LogicalPlan::new(n);
        let order = |dir: SortDirection| SortOrder {
            expr: a(),
            direction: dir,
            nulls: NullOrdering::NullsLast,
        };
        let spec = WindowSpec {
            partition_by: vec![a()],
            order_by: vec![order(SortDirection::Descending)],
            frame: Some(WindowFrame {
                units: FrameUnits::Rows,
                lower: FrameBound::UnboundedPreceding,
                upper: FrameBound::Offset(lit(2)),
            }),
        };
        let sub = |kind: SubqueryKind| ExprNode::Subquery { kind, plan: base() };
        let resolved_fn = |filter: Option<Arc<Expr>>| {
            Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    within_group: Vec::new(),
                    id: FunctionId::Builtin(Builtin::Gt),
                    kind: FunctionKind::Scalar,
                    args: vec![a(), lit(1)],
                    distinct: true,
                    filter,
                    data_type: DataType::Boolean,
                    nullable: true,
                    deterministic: true,
                },
            ))))
        };
        let resolved = |r: ResolvedExpr| Arc::new(Expr::new(ExprNode::Resolved(r)));
        let grouped = |g: Grouping| {
            node(PlanNode::Aggregate {
                input: base(),
                grouping: g,
                aggregates: vec![a()],
            })
        };
        let join = |kind: JoinKind, condition: JoinCondition| {
            node(PlanNode::Join {
                left: base(),
                right: Arc::new(scan(&[2], &["b"])),
                kind,
                condition,
            })
        };

        let exprs: Vec<(&str, Arc<Expr>)> = vec![
            (
                "column",
                unresolved(ExprNode::Column(ColumnRef {
                    name: vec!["t".into(), "a".into()],
                    plan_id: Some(PlanId(3)),
                })),
            ),
            (
                "column-bare",
                unresolved(ExprNode::Column(ColumnRef {
                    name: vec!["a".into()],
                    plan_id: None,
                })),
            ),
            ("literal", lit(7)),
            (
                "typed-literal",
                unresolved(ExprNode::TypedLiteral {
                    kind: TypedLiteralKind::Date,
                    value: "2024-01-02".into(),
                }),
            ),
            (
                "star",
                unresolved(ExprNode::Star(Star {
                    qualifier: vec!["t".into()],
                    except: vec![vec!["a".into()]],
                    plan_id: Some(PlanId(1)),
                })),
            ),
            ("function", unresolved(ExprNode::Function(call("f")))),
            ("function-other", unresolved(ExprNode::Function(call("g")))),
            (
                "lambda",
                unresolved(ExprNode::Lambda {
                    params: vec![bound(9, "x")],
                    body: lit(1),
                }),
            ),
            (
                "field",
                unresolved(ExprNode::Field {
                    base: a(),
                    name: "f".into(),
                }),
            ),
            (
                "cast",
                unresolved(ExprNode::Cast {
                    expr: a(),
                    to: DataType::Long,
                    mode: crate::CastMode::Ansi,
                }),
            ),
            (
                "alias",
                unresolved(ExprNode::Alias {
                    expr: a(),
                    name: "z".into(),
                }),
            ),
            ("scalar-subquery", unresolved(sub(SubqueryKind::Scalar))),
            ("exists-subquery", unresolved(sub(SubqueryKind::Exists))),
            (
                "in-subquery",
                unresolved(sub(SubqueryKind::In(vec![lit(1)]))),
            ),
            (
                "window-spec",
                unresolved(ExprNode::Window(WindowExpr {
                    function: unresolved(ExprNode::Function(call("sum"))),
                    window: WindowSpecRef::Spec(spec),
                })),
            ),
            (
                "window-named",
                unresolved(ExprNode::Window(WindowExpr {
                    function: unresolved(ExprNode::Function(call("sum"))),
                    window: WindowSpecRef::Named("w".into()),
                })),
            ),
            ("r-column", a()),
            (
                "r-outer",
                resolved(ResolvedExpr::OuterColumn(bound(1, "a"))),
            ),
            (
                "r-lambda-var",
                resolved(ResolvedExpr::LambdaVar(bound(1, "a"))),
            ),
            ("r-function", resolved_fn(Some(lit(3)))),
            ("r-function-unfiltered", resolved_fn(None)),
            (
                "r-field",
                resolved(ResolvedExpr::Field {
                    base: a(),
                    ordinal: 2,
                    name: "f".into(),
                    data_type: DataType::Integer,
                    nullable: true,
                }),
            ),
            (
                "r-alias",
                resolved(ResolvedExpr::Alias {
                    expr: a(),
                    output: bound(5, "al"),
                }),
            ),
        ];
        let mut plans: Vec<(String, LogicalPlan)> = exprs
            .into_iter()
            .map(|(name, e)| (format!("expr {name}"), project(e)))
            .collect();
        plans.extend(
            [
                ("one-row", node(PlanNode::OneRowRelation)),
                (
                    "table-function",
                    node(PlanNode::TableFunction {
                        name: "range".into(),
                        args: vec![lit(3)],
                    }),
                ),
                (
                    "range",
                    node(PlanNode::Range {
                        start: 0,
                        end: 5,
                        step: 1,
                        num_slices: Some(2),
                        output: bound(4, "id"),
                    }),
                ),
                (
                    "with-columns",
                    node(PlanNode::WithColumns {
                        input: base(),
                        columns: vec![("c".into(), a())],
                    }),
                ),
                (
                    "filter",
                    node(PlanNode::Filter {
                        input: base(),
                        predicate: lit(1),
                    }),
                ),
                ("group-by", grouped(Grouping::GroupBy(vec![a()]))),
                ("rollup", grouped(Grouping::Rollup(vec![a()]))),
                ("cube", grouped(Grouping::Cube(vec![a()]))),
                (
                    "grouping-sets",
                    grouped(Grouping::GroupingSets {
                        exprs: vec![a()],
                        sets: vec![vec![a()], vec![]],
                    }),
                ),
                ("join-none", join(JoinKind::Cross, JoinCondition::None)),
                (
                    "join-existence",
                    join(JoinKind::Existence(AttrId(1)), JoinCondition::On(lit(1))),
                ),
                ("join-on", join(JoinKind::Inner, JoinCondition::On(lit(1)))),
                (
                    "join-using",
                    join(JoinKind::LeftOuter, JoinCondition::Using(vec!["a".into()])),
                ),
                (
                    "join-natural",
                    join(JoinKind::FullOuter, JoinCondition::Natural),
                ),
                (
                    "sort",
                    node(PlanNode::Sort {
                        input: base(),
                        order: vec![order(SortDirection::Ascending)],
                        global: true,
                    }),
                ),
                (
                    "limit",
                    node(PlanNode::Limit {
                        input: base(),
                        limit: lit(5),
                    }),
                ),
                (
                    "offset",
                    node(PlanNode::Offset {
                        input: base(),
                        offset: lit(5),
                    }),
                ),
                (
                    "subquery-alias",
                    node(PlanNode::SubqueryAlias {
                        input: base(),
                        alias: "s".into(),
                        columns: vec!["c".into()],
                    }),
                ),
                (
                    "with-cte",
                    node(PlanNode::WithCte {
                        input: base(),
                        ctes: vec![Cte {
                            name: "c".into(),
                            plan: base(),
                        }],
                    }),
                ),
                (
                    "cte-ref",
                    node(PlanNode::CteRef(CteRef {
                        name: "c".into(),
                        source: vec![AttrId(1)],
                        output: vec![bound(2, "a")],
                    })),
                ),
                (
                    "shift",
                    node(PlanNode::Shift {
                        input: base(),
                        offset: 10,
                    }),
                ),
                (
                    "hint",
                    node(PlanNode::Hint {
                        input: base(),
                        name: "h".into(),
                        params: vec![lit(1)],
                    }),
                ),
                (
                    "pipe-operator",
                    node(PlanNode::PipeOperator { input: base() }),
                ),
                (
                    "join-hint",
                    node(PlanNode::JoinHint {
                        input: base(),
                        strategy: JoinStrategy::Broadcast,
                    }),
                ),
                (
                    "union",
                    node(PlanNode::Union {
                        inputs: vec![base(), base()],
                        by_name: true,
                        allow_missing_columns: true,
                    }),
                ),
                ("distinct", node(PlanNode::Distinct { input: base() })),
                (
                    "set-op",
                    node(PlanNode::SetOp {
                        left: base(),
                        right: Arc::new(scan(&[2], &["b"])),
                        op: SetOperation::Except,
                        all: true,
                    }),
                ),
                (
                    "window",
                    node(PlanNode::Window {
                        input: base(),
                        exprs: vec![a()],
                        partition_by: vec![a()],
                        order_by: vec![order(SortDirection::Ascending)],
                    }),
                ),
                (
                    "replicate-rows",
                    node(PlanNode::ReplicateRows {
                        input: base(),
                        count: lit(2),
                    }),
                ),
                (
                    "expand",
                    node(PlanNode::Expand {
                        input: base(),
                        projections: vec![vec![a()], vec![lit(1)]],
                        output: vec![bound(6, "e")],
                    }),
                ),
                (
                    "pivot",
                    node(PlanNode::Pivot {
                        input: base(),
                        pivot_columns: vec![a()],
                        values: vec![
                            PivotValue {
                                exprs: vec![lit(1)],
                                alias: Some("one".into()),
                            },
                            PivotValue {
                                exprs: vec![lit(2)],
                                alias: None,
                            },
                        ],
                        aggregates: vec![a()],
                    }),
                ),
                (
                    "unpivot",
                    node(PlanNode::Unpivot {
                        input: base(),
                        value_columns: vec!["v".into()],
                        name_column: "k".into(),
                        column_sets: vec![
                            UnpivotColumnSet {
                                columns: vec![a()],
                                alias: Some("x".into()),
                            },
                            UnpivotColumnSet {
                                columns: vec![a()],
                                alias: None,
                            },
                        ],
                        include_nulls: true,
                    }),
                ),
                (
                    "inline-table",
                    node(PlanNode::InlineTable {
                        rows: vec![vec![lit(1)]],
                        columns: vec!["c".into()],
                    }),
                ),
                ("scan-versioned", {
                    let mut s = scan(&[1], &["a"]);
                    if let PlanNode::Scan(sc) = &mut s.node {
                        sc.version = Some(crate::TableVersion(4));
                    }
                    s
                }),
                (
                    "sql",
                    node(PlanNode::Sql {
                        query: "SELECT 1".into(),
                        args: SqlArgs::Positional(vec![lit(1)]),
                    }),
                ),
                (
                    "sql-named",
                    node(PlanNode::Sql {
                        query: "SELECT 1".into(),
                        args: SqlArgs::Named(BTreeMap::from([("k".to_owned(), lit(1))])),
                    }),
                ),
                (
                    "sql-no-args",
                    node(PlanNode::Sql {
                        query: "SELECT 1".into(),
                        args: SqlArgs::None,
                    }),
                ),
                (
                    "read-files",
                    node(PlanNode::Read(ReadSource::Files {
                        format: "csv".into(),
                        paths: vec![
                            OptionValue::from_user("path", "/p"),
                            OptionValue::from_user("path", "s3://user:pw@host/k"),
                        ],
                        options: BTreeMap::from([
                            ("secret".to_owned(), OptionValue::reference("s", "k")),
                            (
                                "password".to_owned(),
                                OptionValue::from_user("password", "pw"),
                            ),
                        ]),
                    })),
                ),
                (
                    "read-table-options",
                    node(PlanNode::Read(ReadSource::Table {
                        name: vec!["t".into()],
                        options: BTreeMap::from([(
                            "k".to_owned(),
                            OptionValue::from_user("k", "v"),
                        )]),
                    })),
                ),
            ]
            .into_iter()
            .map(|(name, plan)| (name.to_owned(), plan)),
        );

        let mut seen: BTreeMap<u128, String> = BTreeMap::new();
        for (name, plan) in &plans {
            let h = normalized_subtree_hash(plan);
            assert_eq!(h, normalized_subtree_hash(plan), "{name} is stable");
            if let Some(other) = seen.insert(h, name.clone()) {
                panic!("{name} hashes like {other}");
            }
        }
    }

    #[test]
    fn subtree_hash_is_stable_and_differs_from_whole() {
        let sub = filter(scan(&[3], &["a"]), gt(rcol(3, "a"), 1));
        let same = filter(scan(&[90], &["a"]), gt(rcol(90, "a"), 1));
        assert_eq!(
            normalized_subtree_hash(&sub),
            normalized_subtree_hash(&same)
        );
        let whole = project(sub.clone(), &[rcol(3, "a")]);
        assert_ne!(normalized_subtree_hash(&sub), hash(&whole));
    }

    #[test]
    fn cache_keys_separate_principals_policies_and_pins() {
        let plan = Plan::try_new(project(scan(&[1], &["a"]), &[rcol(1, "a")])).unwrap();
        let context = || crate::analyzer::ContextFingerprint {
            profile: crate::analyzer::SparkProfile::default(),
            conf: crate::analyzer::ResolutionConf::default(),
            current: vec!["spark_catalog".to_owned(), "default".to_owned()],
            catalogs: 0,
            catalog_epochs: BTreeMap::new(),
            functions: 0,
            table_functions: 0,
            sql: None,
            principal: "a".to_owned(),
            policy: PolicyVersion(7),
            files: false,
            variables: 0,
        };
        let analyzed = |principal: &str, policy: u64| crate::analyzer::AnalyzedPlan {
            plan: plan.clone(),
            output: vec![bound(1, "a")],
            pins: BTreeMap::new(),
            securables: Vec::new(),
            input_key: plan.structural_hash(),
            context: crate::analyzer::ContextFingerprint {
                principal: principal.to_owned(),
                policy: PolicyVersion(policy),
                ..context()
            },
            query_text: None,
            reads_files: false,
            timings: crate::analyzer::AnalysisTimings::default(),
            warnings: Vec::new(),
        };
        let base = CacheKey::of(&analyzed("a", 7));
        assert_ne!(CacheKey::of(&analyzed("b", 7)), base, "principals");
        assert_ne!(CacheKey::of(&analyzed("a", 8)), base, "policy version");
        let mut pinned = analyzed("a", 7);
        pinned.pins.insert(
            TableIdent {
                catalog: "spark_catalog".to_owned(),
                namespace: vec!["default".to_owned()],
                name: "t".to_owned(),
            },
            TableVersion(3),
        );
        assert_ne!(CacheKey::of(&pinned), base, "pins");
        assert_eq!(CacheKey::of(&analyzed("a", 7)), base, "same inputs");
    }
}
