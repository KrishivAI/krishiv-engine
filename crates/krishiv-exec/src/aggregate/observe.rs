// SPDX-License-Identifier: Apache-2.0
//! `Dataset.observe` at run time ([`PhysNode::CollectMetrics`]).
//!
//! [`CollectMetrics`] is a pass-through operator: each driver folds the
//! batches it is handed into the partial states of the metrics' aggregates
//! (the table of a global partial aggregate: one group, O(1) state, weights
//! respected) and forwards every batch unchanged, so what is counted is the
//! rows that flowed through it, as Spark's `CollectMetricsExec` counts the
//! rows its consumer pulled (a limit above stops the pull; here at batch
//! granularity). A driver's states go to the query when its local state
//! ends, whether the input ended, a limit above stopped the pull or the
//! query failed (`Drop`), as the task metric of that partition; the query
//! merges the states of an observation's drivers and a [`Finalizer`] turns
//! them into one row (`QueryContext::observed`).
//!
//! Two instances of one name in a plan (a self-join of an observed
//! relation) do not add up: the last one's states are the observation's
//! (Spark's `CollectMetricsExec.collect` keeps the last).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_types::batch::Batch;

use super::AggregateOutput;
use super::table::{AggCore, AggTable};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::hash::HashSeed;
use crate::operator::{LocalState, Operator, OperatorOutput};
use crate::operators::{Project, local_mut};

/// Orders the instances of a plan (instantiation is one thread, in plan
/// order): the highest is the last in the plan.
static INSTANCES: AtomicU64 = AtomicU64::new(0);

/// Turns the partial states of one observation into its row.
#[derive(Debug)]
pub(crate) struct Finalizer {
    name: String,
    /// This instance's place among the plan's.
    instance: u64,
    /// Merges the states of the drivers and gives the aggregates' results.
    core: Arc<AggCore>,
    /// The metrics over those results, one column each.
    post: Arc<Project>,
}

impl Finalizer {
    /// The observation's name.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// The instance's rank in its plan.
    pub(crate) fn instance(&self) -> u64 {
        self.instance
    }

    /// The metrics' row over `partials`, the states of every driver of the
    /// instance: the aggregates' zero state if there are none.
    pub(crate) fn finalize(&self, partials: &[RecordBatch]) -> Result<RecordBatch> {
        let mut table = AggTable::new(&self.core, &HashSeed::new(None))?;
        for states in partials {
            let p = table.prepare(&Batch::new(states.clone()), true)?;
            table.apply(&p)?;
        }
        let results = table.output_batch(&[0])?;
        let out = self.post.project(&Batch::new(results), 0)?;
        Ok(out.data().clone())
    }
}

/// The operator of a [`PhysNode::CollectMetrics`](crate::physical::PhysNode).
#[derive(Debug)]
pub(crate) struct CollectMetrics {
    /// The input's schema: rows pass through as they are.
    schema: SchemaRef,
    /// Computes the aggregates' arguments from a batch, if they are not
    /// plain input columns.
    pre: Option<Arc<Project>>,
    /// The partial aggregate (no keys).
    core: Arc<AggCore>,
    finalizer: Arc<Finalizer>,
}

impl CollectMetrics {
    /// The operator over rows of `schema`: `pre` computes the aggregates'
    /// arguments (`pre_schema`: its output, or `schema` without it),
    /// `partial` is the global aggregate's plan, `post` computes the
    /// metrics from its results.
    pub(crate) fn try_new(
        name: &str,
        schema: SchemaRef,
        pre: Option<Arc<Project>>,
        partial: super::AggregateSpec,
        final_spec: impl FnOnce(&SchemaRef) -> Result<super::AggregateSpec>,
        post: impl FnOnce(&SchemaRef) -> Result<Arc<Project>>,
    ) -> Result<Arc<Self>> {
        if partial.output != AggregateOutput::Partial || !partial.group_by.is_empty() {
            return Err(internal(
                "observed metrics are a global partial aggregate".to_owned(),
            ));
        }
        let core = Arc::new(AggCore::new(partial)?);
        let merge = Arc::new(AggCore::new(final_spec(&core.state_schema)?)?);
        let post = post(&merge.output_schema)?;
        let finalizer = Arc::new(Finalizer {
            name: name.to_owned(),
            instance: INSTANCES.fetch_add(1, Ordering::Relaxed),
            core: merge,
            post,
        });
        Ok(Arc::new(Self {
            schema,
            pre,
            core,
            finalizer,
        }))
    }
}

/// One driver's partial states, sent to the query when it ends.
struct Local {
    table: AggTable,
    query: Arc<crate::context::QueryContext>,
    finalizer: Arc<Finalizer>,
    partition: u32,
    sent: bool,
}

impl Local {
    /// Sends the states to the query, once.
    fn send(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.sent, true) {
            return Ok(());
        }
        let states = self.table.state_batch(&[0])?;
        self.query.observe_partial(&self.finalizer, states);
        Ok(())
    }
}

impl Drop for Local {
    fn drop(&mut self) {
        // The input ended, a limit above stopped the pull, or the query
        // failed: the rows seen so far count either way.
        if let Err(e) = self.send() {
            self.query.fail(e);
        }
    }
}

impl Operator for CollectMetrics {
    fn lane_safe(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "CollectMetrics"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(Local {
            table: AggTable::new(&self.core, &HashSeed::new(ctx.query().config().seed))?,
            query: Arc::clone(ctx.query()),
            finalizer: Arc::clone(&self.finalizer),
            partition: ctx.partition(),
            sent: false,
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        let l = local_mut::<Local>(local, "CollectMetrics")?;
        if input.num_rows() == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let args;
        let folded = match &self.pre {
            Some(pre) => {
                args = pre.project(input, l.partition)?;
                &args
            }
            None => input,
        };
        let p = l.table.prepare(folded, false)?;
        l.table.apply(&p)?;
        Ok(OperatorOutput::NeedMoreInput(Some(input.clone())))
    }

    fn finish(&self, local: &mut LocalState) -> Result<Option<Batch>> {
        local_mut::<Local>(local, "CollectMetrics")?.send()?;
        Ok(None)
    }
}
