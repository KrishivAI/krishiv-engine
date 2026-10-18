// SPDX-License-Identifier: Apache-2.0
//! The generic plan nodes the planner assembles operators with.

use std::sync::Arc;

use crate::error::Result;
use crate::operator::{Operator, Sink, Source};
use crate::pipeline::{ExecNode, PipelineBuilder};

/// A source: the start of the current pipeline.
pub(crate) struct SourceNode(pub(crate) Arc<dyn Source>);

impl ExecNode for SourceNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        b.set_source(Arc::clone(&self.0)).map(drop)
    }
}

/// A streaming operator over its input, in the input's pipeline.
pub(crate) struct OperatorNode {
    pub(crate) input: Arc<dyn ExecNode>,
    pub(crate) op: Arc<dyn Operator>,
}

impl ExecNode for OperatorNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        Arc::clone(&self.input).build_pipelines(b)?;
        b.push_operator(Arc::clone(&self.op));
        Ok(())
    }
}

/// A pipeline breaker: the sink of its input's pipeline and the source of
/// the current one; `ordered` keeps its output order (one driver above it).
pub(crate) struct BreakerNode {
    pub(crate) input: Arc<dyn ExecNode>,
    pub(crate) sink: Arc<dyn Sink>,
    pub(crate) source: Arc<dyn Source>,
    pub(crate) ordered: bool,
}

impl ExecNode for BreakerNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.child_pipeline(Arc::clone(&self.sink), Arc::clone(&self.input))?;
        b.set_breaker_source(Arc::clone(&self.source), sink)?;
        if self.ordered {
            b.preserve_order();
        }
        Ok(())
    }
}

/// `UNION ALL`: its inputs fan into the current pipeline (see
/// [`PipelineBuilder::fan_in`]).
pub(crate) struct FanInNode {
    pub(crate) inputs: Vec<Arc<dyn ExecNode>>,
}

impl ExecNode for FanInNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        b.fan_in(&self.inputs)
    }
}
