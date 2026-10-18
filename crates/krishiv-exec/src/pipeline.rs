// SPDX-License-Identifier: Apache-2.0
//! Splitting a physical plan into pipelines (DuckDB `BuildPipelines`).
//!
//! The builder starts with the root pipeline, whose sink is the query's
//! result. Each plan node adds itself to the *current* pipeline:
//! - a scan sets the source: `b.set_source(scan)`;
//! - a streaming operator builds its child, then pushes itself:
//!   `child.build_pipelines(b)?; b.push_operator(op)`;
//! - a pipeline breaker (aggregate, sort) is the sink of a new child pipeline
//!   and the source of the current one:
//!   `let id = b.child_pipeline(agg.clone(), child)?; b.set_breaker_source(agg, id)`;
//! - a hash join builds its build side as a child pipeline ending in the
//!   join's build sink, then its probe side into the current pipeline, then
//!   pushes the probe operator with
//!   [`push_operator_reading`](PipelineBuilder::push_operator_reading), which
//!   hands it the build's finalized state.
//!
//! - a `UNION ALL` fans in: [`fan_in`](PipelineBuilder::fan_in) builds each
//!   input into a pipeline of its own ending in the current pipeline's
//!   sink, and the operators pushed after it go into every one of them
//!   (DuckDB's union pipelines). The executor starts these pipelines
//!   together and runs them as one: one sink state, finalized once, and
//!   one flush barrier per shared operator.
//!
//! A pipeline starts once every child pipeline it depends on has
//! finalized. The executor owns breaker state: it keeps each finalized
//! sink's global state, hands it to the reading source and drops it when the
//! query ends. Plan nodes are still per-query instances (an operator may keep
//! per-query counters in itself, like a streaming `LIMIT`), so a plan runs
//! once.

use std::sync::Arc;

use crate::context::OperatorId;
use crate::error::{Result, internal};
use crate::operator::{Operator, Sink, Source};

/// A physical plan node (implemented by the physical plan, P2-T074).
pub trait ExecNode: Send + Sync {
    /// Adds this node and its children to `b`.
    ///
    /// # Errors
    /// An error stops planning; the query is not started.
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()>;
}

/// One `Source → Operator* → Sink` chain.
pub(crate) struct Pipeline {
    pub(crate) source: (OperatorId, Arc<dyn Source>),
    pub(crate) operators: Vec<(OperatorId, Arc<dyn Operator>)>,
    /// Per operator: the breaker sink whose finalized state it reads.
    pub(crate) operator_inputs: Vec<Option<OperatorId>>,
    pub(crate) sink: (OperatorId, Arc<dyn Sink>),
    /// Pipelines that must finalize first.
    pub(crate) deps: Vec<usize>,
    /// Run one driver (the pipeline must keep its input order).
    pub(crate) preserve_order: bool,
    /// The breaker sink whose finalized global state the source reads.
    pub(crate) input: Option<OperatorId>,
}

struct Draft {
    /// The pipelines a fan-in below this one's next operators added: the
    /// operators pushed from now on go into them too.
    fanned: Vec<usize>,
    source: Option<(OperatorId, Arc<dyn Source>)>,
    input: Option<OperatorId>,
    operators: Vec<(OperatorId, Arc<dyn Operator>)>,
    operator_inputs: Vec<Option<OperatorId>>,
    sink: (OperatorId, Arc<dyn Sink>),
    deps: Vec<usize>,
    preserve_order: bool,
}

/// Every operator's name, by id.
pub(crate) type Names = Vec<(OperatorId, &'static str)>;

/// Collects the pipelines of a plan; see the module docs.
pub struct PipelineBuilder {
    drafts: Vec<Draft>,
    current: usize,
    next_id: u32,
    names: Names,
}

impl std::fmt::Debug for PipelineBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineBuilder")
            .field("pipelines", &self.drafts.len())
            .field("current", &self.current)
            .finish_non_exhaustive()
    }
}

impl PipelineBuilder {
    /// A builder whose root pipeline ends in `result`.
    pub(crate) fn new(result: Arc<dyn Sink>) -> Self {
        let mut b = Self {
            drafts: Vec::new(),
            current: 0,
            next_id: 0,
            names: Vec::new(),
        };
        let id = b.id(result.name());
        b.drafts.push(Draft {
            fanned: Vec::new(),
            source: None,
            input: None,
            operators: Vec::new(),
            operator_inputs: Vec::new(),
            sink: (id, result),
            deps: Vec::new(),
            preserve_order: false,
        });
        b
    }

    fn id(&mut self, name: &'static str) -> OperatorId {
        let id = OperatorId(self.next_id);
        self.next_id += 1;
        self.names.push((id, name));
        id
    }

    fn draft(&mut self) -> &mut Draft {
        &mut self.drafts[self.current]
    }

    /// Sets the current pipeline's source.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if it already has one.
    pub fn set_source(&mut self, source: Arc<dyn Source>) -> Result<OperatorId> {
        if self.draft().source.is_some() {
            return Err(internal("a pipeline has two sources".to_owned()));
        }
        let id = self.id(source.name());
        self.draft().source = Some((id, source));
        Ok(id)
    }

    /// Sets the current pipeline's source to the source side of a breaker
    /// whose sink side `sink` (the id [`child_pipeline`](Self::child_pipeline)
    /// returned) ends one of its child pipelines. The executor hands the
    /// sink's finalized global state to `source` through
    /// [`TaskContext::dependency`](crate::TaskContext::dependency) and drops
    /// it when the query ends. A breaker has two ids (sink side, source
    /// side): `EXPLAIN ANALYZE` maps both to its plan node.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the pipeline already has a source or `sink` does
    /// not end a child pipeline of it.
    pub fn set_breaker_source(
        &mut self,
        source: Arc<dyn Source>,
        sink: OperatorId,
    ) -> Result<OperatorId> {
        self.check_child(sink)?;
        let id = self.set_source(source)?;
        self.draft().input = Some(sink);
        Ok(id)
    }

    /// Appends a streaming operator to the current pipeline (after its
    /// source and the operators pushed before it).
    pub fn push_operator(&mut self, op: Arc<dyn Operator>) -> OperatorId {
        let id = self.id(op.name());
        let (last, rest) = (self.current, self.drafts[self.current].fanned.clone());
        for i in rest {
            let draft = &mut self.drafts[i];
            draft.operators.push((id, Arc::clone(&op)));
            draft.operator_inputs.push(None);
        }
        let draft = &mut self.drafts[last];
        draft.operators.push((id, op));
        draft.operator_inputs.push(None);
        id
    }

    /// The current pipeline and those a fan-in below it added.
    fn frontier(&self) -> Vec<usize> {
        let mut all = vec![self.current];
        all.extend(&self.drafts[self.current].fanned);
        all
    }

    /// As [`push_operator`](Self::push_operator), for an operator that reads
    /// the finalized global state of the breaker sink `sink` (a hash join's
    /// probe reads its build) through
    /// [`TaskContext::dependency`](crate::TaskContext::dependency).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if `sink` does not end a child pipeline of the
    /// current one.
    pub fn push_operator_reading(
        &mut self,
        op: Arc<dyn Operator>,
        sink: OperatorId,
    ) -> Result<OperatorId> {
        self.check_child(sink)?;
        let id = self.push_operator(op);
        for i in self.frontier() {
            if let Some(input) = self.drafts[i].operator_inputs.last_mut() {
                *input = Some(sink);
            }
        }
        Ok(id)
    }

    fn check_child(&self, sink: OperatorId) -> Result<()> {
        for i in self.frontier() {
            let deps = &self.drafts[i].deps;
            if !deps.iter().any(|&d| self.drafts[d].sink.0 == sink) {
                return Err(internal(format!("{sink:?} does not end a child pipeline")));
            }
        }
        Ok(())
    }

    /// Runs the current pipeline with one driver, keeping its input order
    /// (above a sort, in P2a).
    pub fn preserve_order(&mut self) {
        self.draft().preserve_order = true;
    }

    /// Builds `child` into a new pipeline ending in `sink`; the current
    /// pipeline starts only after it has finalized.
    ///
    /// # Errors
    /// Errors of `child`'s planning.
    pub fn child_pipeline(
        &mut self,
        sink: Arc<dyn Sink>,
        child: Arc<dyn ExecNode>,
    ) -> Result<OperatorId> {
        let id = self.id(sink.name());
        let index = self.drafts.len();
        self.drafts.push(Draft {
            fanned: Vec::new(),
            source: None,
            input: None,
            operators: Vec::new(),
            operator_inputs: Vec::new(),
            sink: (id, sink),
            deps: Vec::new(),
            preserve_order: false,
        });
        let parent = std::mem::replace(&mut self.current, index);
        let built = child.build_pipelines(self);
        self.current = parent;
        built?;
        for i in self.frontier() {
            self.drafts[i].deps.push(index);
        }
        Ok(id)
    }

    /// Fans `inputs` in (`UNION ALL`): the first is built into the current
    /// pipeline, each other one into a new pipeline that ends in the same
    /// sink and depends on what the current one depends on so far; the
    /// operators pushed after this call go into all of them.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` without inputs or if the current pipeline already
    /// has a source; errors of the inputs' planning.
    pub fn fan_in(&mut self, inputs: &[Arc<dyn ExecNode>]) -> Result<()> {
        let Some((first, rest)) = inputs.split_first() else {
            return Err(internal("a fan-in without inputs".to_owned()));
        };
        if self.draft().source.is_some() || !self.draft().operators.is_empty() {
            return Err(internal("a fan-in into a started pipeline".to_owned()));
        }
        let (sink, deps) = (self.draft().sink.clone(), self.draft().deps.clone());
        let order = self.draft().preserve_order;
        Arc::clone(first).build_pipelines(self)?;
        let main = self.current;
        let mut added = Vec::new();
        for input in rest {
            let index = self.drafts.len();
            self.drafts.push(Draft {
                fanned: Vec::new(),
                source: None,
                input: None,
                operators: Vec::new(),
                operator_inputs: Vec::new(),
                sink: sink.clone(),
                deps: deps.clone(),
                preserve_order: order,
            });
            self.current = index;
            let built = Arc::clone(input).build_pipelines(self);
            self.current = main;
            built?;
            added.push(index);
            added.extend(self.drafts[index].fanned.iter().copied());
        }
        self.draft().fanned.extend(added);
        Ok(())
    }

    /// The pipelines and every operator's name.
    pub(crate) fn finish(self) -> Result<(Vec<Pipeline>, Names)> {
        let pipelines = self
            .drafts
            .into_iter()
            .map(|d| {
                let source = d
                    .source
                    .ok_or_else(|| internal("a pipeline has no source".to_owned()))?;
                Ok(Pipeline {
                    source,
                    operators: d.operators,
                    operator_inputs: d.operator_inputs,
                    sink: d.sink,
                    deps: d.deps,
                    preserve_order: d.preserve_order,
                    input: d.input,
                })
            })
            .collect::<Result<_>>()?;
        Ok((pipelines, self.names))
    }
}
