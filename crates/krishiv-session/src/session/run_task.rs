// SPDX-License-Identifier: Apache-2.0
//! Runs one task of a physical plan in this process (tests of the session's
//! workers). A child of `session` so that it reads the session's private
//! state and keeps `session.rs` from growing (`cargo xtask file-size`).

use std::sync::Arc;

use arrow_array::RecordBatch;
use futures::StreamExt;
use krishiv_common::error::KrishivError;
use krishiv_exec::planner::PhysicalPlan;
use krishiv_exec::{ExecConfig, QueryContext, QueryId, WorkloadClass};

use super::{QueryEnv, Session};

impl Session {
    /// Runs `physical` as `task` in this process (tests): the worker of
    /// [`Self::worker`] instantiating with the task's scan splits, executed
    /// on the session's executor (I/O needs its runtime).
    pub(crate) async fn run_task(
        &self,
        physical: &PhysicalPlan,
        env: &QueryEnv,
        task: &krishiv_exec::physical::TaskSpec,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        use krishiv_exec::physical::instantiate_task;
        let worker = self.worker(env);
        let node = instantiate_task(physical, &worker, task)?;
        let state = Arc::clone(&self.state);
        let config = Arc::clone(&env.config);
        self.on_executor(async move {
            let id = QueryId(state.runtime.next_index());
            let pool = state.memory.child(&format!("query-task-{}", id.0), None);
            let query = QueryContext::new(
                id,
                WorkloadClass::Interactive,
                ExecConfig::from_config(&config),
                pool,
            );
            let mut stream = state
                .runtime
                .runner
                .execute(node, Arc::clone(&query))
                .await?;
            let mut out = Vec::new();
            while let Some(batch) = stream.next().await {
                out.push(batch?.into_parts().0);
            }
            query.finished().await;
            Ok(out)
        })
        .await
    }
}
