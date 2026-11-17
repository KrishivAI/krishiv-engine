// SPDX-License-Identifier: Apache-2.0
//! The Python UDF relations (design P3-T085 §3.1) as a front end builds
//! them: `mapInPandas` / `mapInArrow` ([`PlanNode::MapPartitions`]),
//! `groupBy().applyInPandas` / `applyInArrow` ([`PlanNode::GroupMap`]) and
//! `cogroup().applyInPandas` / `applyInArrow` ([`PlanNode::CoGroupMap`]),
//! over inputs that may still be unresolved (Spark Connect, P3-T014).
//!
//! The inputs resolve as any relation does; a grouped map's keys bind over
//! its input, a cogroup's over each side (Spark's `groupBy(cols)` on each
//! `Dataset`). A key may be anything a projection allows; an aggregate,
//! a window or a generator key is `KRISHIV_UNSUPPORTED_FEATURE` (Spark's
//! answer for each is not verified against a Spark run). The output is the UDF's declared
//! schema, the columns a front end gives the node: their ids are kept the
//! first time this analysis meets them and replaced by fresh ones after
//! (as a `Range` leaf's), so two translated nodes that carry the same ids
//! stay apart. The columns carry no qualifier (Spark's `toAttributes`).
//!
//! Spark's `CheckAnalysis` refuses a grouped or cogrouped map whose UDF is
//! nondeterministic (`INVALID_NON_DETERMINISTIC_EXPRESSIONS`, v4.3.0
//! L1123-1143: only `MapInPandas` and `MapInArrow` of the Python relations
//! may hold one), so `df.schema` and `explain` fail, not only an action:
//! [`crate::check_python_relation_determinism`] runs here, on the resolved
//! node.

use std::sync::Arc;

use krishiv_common::error::KrishivUnsupportedFeature;

use super::AnalysisError;
use super::exprs::Bound;
use super::ops::has_generator;
use super::resolve::{Attr, Env, Rel, Resolver};
use super::subquery::correlated_reference;
use crate::{
    AttrId, BoundColumn, Expr, LogicalPlan, Origin, PlanNode, PythonUdfKind, PythonUdfRef,
};

impl Resolver<'_> {
    /// `mapInPandas` / `mapInArrow`: every column of `input` goes to the UDF.
    #[inline(never)]
    pub(super) fn map_partitions(
        &mut self,
        o: Origin,
        input: &Arc<LogicalPlan>,
        udf: &PythonUdfRef,
        output: &[BoundColumn],
        barrier: bool,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let out = self.python_output(output);
        let node = PlanNode::MapPartitions {
            input: Arc::clone(&c.plan),
            udf: udf.clone(),
            output: out.iter().map(|a| a.col.clone()).collect(),
            barrier,
        };
        self.python_node(o, node, out, &[&c])
    }

    /// `groupBy(keys).applyInPandas` / `applyInArrow`.
    #[inline(never)]
    pub(super) fn group_map(
        &mut self,
        o: Origin,
        input: &Arc<LogicalPlan>,
        keys: &[Arc<Expr>],
        udf: &PythonUdfRef,
        output: &[BoundColumn],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let keys = self.python_keys(&c, keys, env, operator(udf), o)?;
        let out = self.python_output(output);
        let node = PlanNode::GroupMap {
            input: Arc::clone(&c.plan),
            keys,
            udf: udf.clone(),
            output: out.iter().map(|a| a.col.clone()).collect(),
        };
        self.python_node(o, node, out, &[&c])
    }

    /// `groupBy(..).cogroup(groupBy(..)).applyInPandas` / `applyInArrow`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the fields of the node, as the other node resolvers take them"
    )]
    #[inline(never)]
    pub(super) fn cogroup_map(
        &mut self,
        o: Origin,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        left_keys: &[Arc<Expr>],
        right_keys: &[Arc<Expr>],
        udf: &PythonUdfRef,
        output: &[BoundColumn],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let l = self.child(left, env)?;
        let r = self.child(right, env)?;
        let op = operator(udf);
        let left_keys = self.python_keys(&l, left_keys, env, op, o)?;
        let right_keys = self.python_keys(&r, right_keys, env, op, o)?;
        let out = self.python_output(output);
        let node = PlanNode::CoGroupMap {
            left: Arc::clone(&l.plan),
            right: Arc::clone(&r.plan),
            left_keys,
            right_keys,
            udf: udf.clone(),
            output: out.iter().map(|a| a.col.clone()).collect(),
        };
        self.python_node(o, node, out, &[&l, &r])
    }

    /// The grouping keys of one side, bound over `c`.
    fn python_keys(
        &mut self,
        c: &Rel,
        keys: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
        o: Origin,
    ) -> Result<Vec<Arc<Expr>>, AnalysisError> {
        let rels = [c];
        let cx = Self::cx(&c.out, &rels, env, op);
        let mut bounds: Vec<Bound> = Vec::with_capacity(keys.len());
        for k in keys {
            bounds.push(self.bind(k, &cx)?);
        }
        if bounds.iter().any(|b| b.outer.is_some()) {
            let shown: Vec<Arc<Expr>> = bounds.iter().map(Bound::shown).collect();
            return Err(correlated_reference(&shown, o));
        }
        // Spark puts the keys in a projection below the node (`groupBy(cols)` then
        // `flatMapGroupsIn*`), where a window or a generator key is extracted and an aggregate key fails
        // later with a class not verified against a Spark run: refused, naming the case, until goldens exist.
        let case = bounds.iter().find_map(|b| {
            if b.agg.is_some() {
                Some("an aggregate")
            } else if b.win.is_some() {
                Some("a window function")
            } else if has_generator(&b.expr) {
                Some("a generator")
            } else {
                None
            }
        });
        if let Some(case) = case {
            return Err(AnalysisError::new(
                KrishivUnsupportedFeature {
                    feature: format!("{case} in a grouping key of {op}"),
                },
                o,
            ));
        }
        Ok(bounds.into_iter().map(|b| b.expr).collect())
    }

    /// The node's declared columns, each with an id unique in this analysis.
    fn python_output(&mut self, output: &[BoundColumn]) -> Vec<Attr> {
        let qual: Arc<[String]> = Arc::from(Vec::new());
        output
            .iter()
            .map(|col| Attr {
                col: BoundColumn {
                    id: self.built_id(col.id),
                    ..col.clone()
                },
                qual: Arc::clone(&qual),
                source: None,
                duplicate: false,
            })
            .collect()
    }

    /// The resolved node, after Spark's determinism check.
    fn python_node(
        &mut self,
        o: Origin,
        node: PlanNode,
        out: Vec<Attr>,
        kids: &[&Rel],
    ) -> Result<Rel, AnalysisError> {
        crate::check_python_relation_determinism(&node).map_err(|e| AnalysisError::new(e, o))?;
        Ok(self.make(o, node, out, kids))
    }

    /// `id` the first time this analysis meets it among the ids front ends
    /// choose (`Range` leaves, Python relations), a fresh id after.
    pub(super) fn built_id(&mut self, id: AttrId) -> AttrId {
        if self.built_ids.insert(id) {
            id
        } else {
            let fresh = self.fresh();
            self.built_ids.insert(fresh);
            fresh
        }
    }
}

/// Spark's operator name for the node running `udf`, as its errors show it.
fn operator(udf: &PythonUdfRef) -> &'static str {
    use crate::PythonEvalType as T;
    match (udf.eval_type.kind(), udf.eval_type) {
        (PythonUdfKind::CogroupedMap, T::CogroupedMapArrow) => "FlatMapCoGroupsInArrow",
        (PythonUdfKind::CogroupedMap, _) => "FlatMapCoGroupsInPandas",
        (_, T::GroupedMapArrow | T::GroupedMapArrowIter) => "FlatMapGroupsInArrow",
        _ => "FlatMapGroupsInPandas",
    }
}
