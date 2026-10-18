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
//!   pushes the probe operator.
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
    pub(crate) sink: (OperatorId, Arc<dyn Sink>),
    /// Pipelines that must finalize first.
    pub(crate) deps: Vec<usize>,
    /// Run one driver (the pipeline must keep its input order).
    pub(crate) preserve_order: bool,
    /// The breaker sink whose finalized global state the source reads.
    pub(crate) input: Option<OperatorId>,
}

struct Draft {
    source: Option<(OperatorId, Arc<dyn Source>)>,
    input: Option<OperatorId>,
    operators: Vec<(OperatorId, Arc<dyn Operator>)>,
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
            source: None,
            input: None,
            operators: Vec::new(),
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
        let deps = &self.drafts[self.current].deps;
        let child = deps.iter().any(|&d| self.drafts[d].sink.0 == sink);
        if !child {
            return Err(internal(format!("{sink:?} does not end a child pipeline")));
        }
        let id = self.set_source(source)?;
        self.draft().input = Some(sink);
        Ok(id)
    }

    /// Appends a streaming operator to the current pipeline (after its
    /// source and the operators pushed before it).
    pub fn push_operator(&mut self, op: Arc<dyn Operator>) -> OperatorId {
        let id = self.id(op.name());
        self.draft().operators.push((id, op));
        id
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
            source: None,
            input: None,
            operators: Vec::new(),
            sink: (id, sink),
            deps: Vec::new(),
            preserve_order: false,
        });
        let parent = std::mem::replace(&mut self.current, index);
        let built = child.build_pipelines(self);
        self.current = parent;
        built?;
        self.draft().deps.push(index);
        Ok(id)
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
