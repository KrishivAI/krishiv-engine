// SPDX-License-Identifier: Apache-2.0
//! Grouping sets aggregate their input once: `Aggregate(Project*(Expand(X)))`
//! becomes the same aggregate merging the states of `PartialAggregate(X)`,
//! keyed by the columns of `X` that the expansion and everything above it
//! read (`krishiv.optimizer.preAggregateExpand`).
//!
//! A `ROLLUP` of `k` columns repeats each input row `k + 1` times; the partial
//! aggregate folds the input's rows into their finest groups first, and the
//! expansion repeats those groups instead (TPC-DS q22: 26.5M rows expanded to
//! 132.6M fed the aggregate; about 100K groups feed it now). Each group of a
//! grouping set is the union of the finest groups that the set's columns
//! cannot tell apart, so merging their states gives what folding the rows does.
//! The same holds for any expansion, so the rule also takes the rewrite of
//! several distinct calls (`ops/distinct.rs`): its regular calls' arguments
//! pass through every projection unchanged, and the keys are the distinct
//! arguments the expansion reads.
//!
//! A shape with no key stays: a keyless partial aggregate is a global one and
//! yields a row over empty input, where the grouped aggregate above yields none.
//!
//! The calls are the ones eager aggregation moves ([`refusal_of_call`]): the
//! same accumulators, whose result cannot depend on the order or the grouping
//! of their rows. No row is dropped on the way, so nothing reads the estimate:
//! where `X` does not reduce (a finest set that is nearly unique, as q67's),
//! the partial operator's adaptive bypass passes the rows on as their own
//! states after its first rows.

use std::sync::Arc;

use super::{
    Attrs, Estimator, Shape, attrs_of, column, contains, contains_floating, distinct_calls,
    references_into, refusal_of_call, states_of,
};
use krishiv_plan::{Grouping, LogicalPlan, PlanNode};

impl<'a> Shape<'a> {
    /// An aggregate over projections over one `Expand` of at least two
    /// projections; `join` is then the expansion's input. Any expansion
    /// matches: grouping sets, and the rewrite of several distinct calls
    /// (`ops/distinct.rs`), whose regular arguments pass through every
    /// projection unchanged.
    pub(super) fn over_expand(plan: &'a LogicalPlan) -> Option<Self> {
        let PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(keys),
            aggregates: outputs,
        } = &plan.node
        else {
            return None;
        };
        let mut spine = Vec::new();
        let mut at = input;
        loop {
            match &at.node {
                PlanNode::Project { input, .. } => {
                    spine.push(at.as_ref());
                    at = input;
                }
                PlanNode::Expand {
                    input, projections, ..
                } if projections.len() >= 2 => {
                    spine.push(at.as_ref());
                    return Some(Self {
                        plan,
                        keys,
                        outputs,
                        spine,
                        join: input,
                    });
                }
                _ => return None,
            }
        }
    }

    /// The aggregate merging the states of a partial aggregate below the
    /// expansion, or why it stays.
    pub(super) fn pre_aggregate(
        &self,
        est: &mut Estimator<'_>,
        next: &mut u64,
    ) -> Result<(Arc<LogicalPlan>, String), String> {
        let input = self.join;
        if matches!(input.node, PlanNode::PartialAggregate { .. }) {
            return Err("the expansion's input is already aggregated".to_owned());
        }
        let calls = distinct_calls(self.outputs);
        if calls.is_empty() {
            return Err("no aggregate call".to_owned());
        }
        let mut args = Attrs::default();
        for (_, f) in &calls {
            for a in &f.args {
                references_into(a, &mut args);
            }
        }
        let need = self.read_above(&args)?;
        let cols = attrs_of(input)?;
        if !args.iter().all(|a| contains(&cols, *a)) {
            return Err("an argument is computed in the expansion".to_owned());
        }
        let input_est = est.plan(input);
        for (_, f) in &calls {
            refusal_of_call(f, &input_est, false)?;
        }
        let keys: Vec<_> = cols
            .iter()
            .filter(|c| need.contains(&c.id))
            .cloned()
            .collect();
        // Without a key the partial aggregate is a global one, which yields a
        // state row over empty input; the expansion would turn it into a row
        // per grouping set where the grouped aggregate above yields none.
        if keys.is_empty() {
            return Err(
                "no key: a keyless partial aggregate yields a row over empty input".to_owned(),
            );
        }
        // Values that compare equal but read differently (`-0.0` and `0.0`,
        // strings equal under a collation) would merge before a grouping
        // expression could tell them apart.
        if let Some(k) = keys
            .iter()
            .find(|k| contains_floating(&k.data_type) || k.data_type.contains_collation())
        {
            return Err(format!(
                "{} is a floating point or collated key: equal values can still differ above",
                k.name
            ));
        }

        let (states, merged) = states_of(&calls, next)?;
        let partial = Arc::new(
            LogicalPlan::new(PlanNode::PartialAggregate {
                input: Arc::clone(input),
                keys: keys.iter().map(column).collect(),
                aggregates: calls.iter().map(|(c, _)| Arc::clone(c)).collect(),
                states: states.clone(),
            })
            .with_origin(input.origin),
        );
        let pushed = self.above(partial, &args, &states, &merged)?;
        let names: Vec<&str> = keys.iter().map(|k| k.name.as_str()).collect();
        Ok((
            pushed,
            format!(
                "pre-aggregated below the Expand by [{}]: rows {:.0}",
                names.join(", "),
                input_est.rows
            ),
        ))
    }
}
