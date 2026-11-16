// SPDX-License-Identifier: Apache-2.0
//! [`PhysNode::TableWrite`](crate::physical::PhysNode::TableWrite) as operators: the input's
//! pipeline ends in the table provider's sink, and a breaker source after it returns no rows
//! (design `P3-PR-DML-row-level.md` §7). The sink commits when it finalizes, so the source runs
//! only after the commit.

use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use futures::future::BoxFuture;

use crate::context::TaskContext;
use crate::error::Result;
use crate::operator::{GlobalState, LocalState, Sink, Source};
use crate::pipeline::{ExecNode, PipelineBuilder};
use krishiv_types::batch::Batch;

/// `input` written through a table provider's `sink`.
pub(crate) struct TableWrite {
    pub(crate) input: Arc<dyn ExecNode>,
    pub(crate) sink: Arc<dyn Sink>,
}

impl std::fmt::Debug for TableWrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableWrite")
            .field("sink", &self.sink.name())
            .finish_non_exhaustive()
    }
}

impl ExecNode for TableWrite {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.child_pipeline(Arc::clone(&self.sink), Arc::clone(&self.input))?;
        b.set_breaker_source(Arc::new(Written) as Arc<dyn Source>, sink)?;
        Ok(())
    }
}

/// The result of a table write: no rows, no columns.
#[derive(Debug)]
struct Written;

/// The schema of a table write's result.
pub(crate) fn written_schema() -> SchemaRef {
    Arc::new(Schema::empty())
}

impl Source for Written {
    fn name(&self) -> &'static str {
        "TableWriteResult"
    }

    fn schema(&self) -> SchemaRef {
        written_schema()
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(()))
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn get_data<'a>(
        &'a self,
        _local: &'a mut LocalState,
        _global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(std::future::ready(Ok(None)))
    }
}
