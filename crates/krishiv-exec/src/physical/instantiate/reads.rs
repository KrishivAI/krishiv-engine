// SPDX-License-Identifier: Apache-2.0
//! The rows a decided scan (a source that decided its filters once) read for one execution.

use super::*;

impl WorkerContext {
    /// The same context keeping the rows a decided scan reads in `reads`, shared by the execution's tasks.
    #[must_use]
    pub fn with_execution_reads(mut self, reads: Arc<crate::operators::ExecutionReads>) -> Self {
        self.execution_reads = Some(reads);
        self
    }

    /// Reads the decided scans of `plan` for this execution, each against the query's cancel
    /// ([`crate::operators::ExecutionReads::fill`]). A no-op for a context without execution reads.
    ///
    /// # Errors
    /// The read's error; `cancelled`'s when it completes first.
    pub async fn fill_reads(
        &self,
        plan: &PhysicalPlan,
        cancelled: impl std::future::Future<Output = krishiv_common::error::KrishivError>,
    ) -> Result<()> {
        match &self.execution_reads {
            Some(reads) => reads.fill(plan, &*self.tables, cancelled).await,
            None => Ok(()),
        }
    }

    /// Reads the decided scans of `plan` against the cancel of `query` (the class its cancellation carries),
    /// then awaits `next`; a failed read is the result and `next` never runs.
    ///
    /// # Errors
    /// The read's error, else `next`'s.
    pub async fn after_reads<T>(
        &self,
        plan: &PhysicalPlan,
        query: Arc<crate::context::QueryContext>,
        next: impl std::future::Future<Output = Result<T>>,
    ) -> Result<T> {
        self.fill_reads(plan, async move {
            query.cancelled().await;
            query.cancellation()
        })
        .await?;
        next.await
    }
}

impl Instantiation<'_> {
    /// The factory a scan reads through: the execution's read of a decided scan, else the table's own.
    pub(super) fn scan_factory(
        &self,
        s: &ScanNode,
    ) -> Result<Arc<dyn crate::operators::ScanPushdown>> {
        match &s.decided {
            Some(decided) if decided.is_undecided() => Err(internal(
                "a deciding scan planned without its decision".to_owned(),
            )),
            Some(decided) => {
                let Some(reads) = &self.worker.execution_reads else {
                    return Err(internal("a decided scan outside an execution".to_owned()));
                };
                reads.get(decided.id)
            }
            None => self.worker.tables.provider(&s.scan)?.scan(),
        }
    }
}
