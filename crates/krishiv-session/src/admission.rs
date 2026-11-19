// SPDX-License-Identifier: Apache-2.0
//! The admission seam (E5): every query asks an [`AdmissionController`] for a
//! [`Permit`] before it runs.
//!
//! The default controller admits everything at once. A platform supplies its
//! own through `SessionBuilder::admission_controller` (Trino-style resource
//! groups are one implementation). While `admit` is pending the query's row
//! in `system.queries` has the state `queued`; it becomes `running` when the
//! permit arrives, or `failed` with the controller's error class when the
//! controller refuses. Dropping the query while it is queued cancels it. The
//! permit lives as long as the query and releases its slot when dropped.
//!
//! A query that reads only `system.*` tables skips the controller (no slot, no budget, no wait,
//! straight to `running`), so an operator can always see why the engine is saturated. Writes and
//! queries that read anything else are admitted as usual.

use std::fmt;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_exec::WorkloadClass;
use krishiv_plan::analyzer::{BoxFuture, Principal};

use crate::explain::Estimate;

/// What a query asks of the admission controller.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct QueryRequest {
    /// Who runs the query.
    pub principal: Arc<Principal>,
    /// The session's tags when the query started (`system.queries.tags`).
    pub tags: Vec<(String, String)>,
    /// The workload class (every session path runs `Interactive` today).
    pub class: WorkloadClass,
    /// Who the client says it is (Connect `client_type`, a Flight SQL `user-agent`, or
    /// `SessionBuilder::source`); `None` when unknown. Client-asserted: a routing hint only.
    pub source: Option<String>,
    /// The statement's class, when the caller has classified it (`None` otherwise).
    pub statement_class: Option<krishiv_sql::StatementClass>,
    /// What the query is expected to cost (ADR 0041), from the optimized plan the query runs.
    /// Computed only when the installed controller's [`AdmissionController::uses_estimate`] is
    /// `true` (the default controller does not), and `None` means unknown: no estimate was
    /// computed or the plan could not be estimated, so a controller must treat `None` as "no
    /// information", never as "free". `DataFrameWriter::save_as_table` in `overwrite` mode is
    /// admitted before its query is planned, so its request always carries `None`.
    pub estimate: Option<Estimate>,
}

/// Permission to run one query: a slot, and optionally a memory budget for
/// it. The slot is given back when the permit is dropped, which the session
/// does when the query's stream ends, fails or is dropped.
#[must_use]
pub struct Permit {
    memory_budget: Option<usize>,
    slot: Option<Box<dyn std::any::Any + Send + Sync>>,
    group: Option<String>,
    priority: Option<u8>,
    class: Option<WorkloadClass>,
}

impl Permit {
    /// A permit with no budget and no slot to release.
    pub fn unlimited() -> Self {
        Self {
            memory_budget: None,
            slot: None,
            group: None,
            priority: None,
            class: None,
        }
    }

    /// A permit holding `slot`, which is dropped when the query ends. Give
    /// it a guard object that returns the slot to its group.
    pub fn with_slot(slot: impl Send + Sync + 'static) -> Self {
        Self {
            memory_budget: None,
            slot: Some(Box::new(slot)),
            group: None,
            priority: None,
            class: None,
        }
    }

    /// This permit with a per-query memory limit in bytes: the query's pool
    /// is created with it, so the query spills or fails at the limit.
    pub fn with_memory_budget(mut self, bytes: usize) -> Self {
        self.memory_budget = Some(bytes);
        self
    }

    /// This permit naming the resource group the controller placed the query in
    /// (`system.queries.resource_group`).
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    /// This permit with an abort priority (0 to 100): when the memory pool is exhausted the lowest
    /// priority query is aborted first (`MemoryPool::child_with_priority`).
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = Some(priority);
        self
    }

    /// This permit with the workload class the query runs as.
    pub fn with_class(mut self, class: WorkloadClass) -> Self {
        self.class = Some(class);
        self
    }

    /// The abort priority the controller set, if any.
    #[must_use]
    pub fn priority(&self) -> Option<u8> {
        self.priority
    }

    /// The workload class the controller set, if any.
    #[must_use]
    pub fn class(&self) -> Option<WorkloadClass> {
        self.class
    }

    /// The resource group the controller placed the query in, if it named one.
    #[must_use]
    pub fn group(&self) -> Option<&str> {
        self.group.as_deref()
    }

    /// The slot guard this permit holds, if it is a `T` (how a session reaches its controller's
    /// guard to report scanned bytes or reserved memory).
    #[must_use]
    pub fn slot<T: 'static>(&self) -> Option<&T> {
        self.slot.as_deref().and_then(|s| s.downcast_ref::<T>())
    }

    /// The per-query memory limit, if the controller set one.
    #[must_use]
    pub fn memory_budget(&self) -> Option<usize> {
        self.memory_budget
    }
}

impl fmt::Debug for Permit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Permit")
            .field("memory_budget", &self.memory_budget)
            .field("slot", &self.slot.is_some())
            .finish()
    }
}

/// Decides when a query may start.
pub trait AdmissionController: Send + Sync + fmt::Debug {
    /// Resolves with a [`Permit`] when `request` may run, or with the error
    /// that refuses it (the controller picks the error class). It may wait
    /// as long as it likes: the query shows as `queued` meanwhile, and
    /// dropping the query drops this future.
    fn admit<'a>(
        &'a self,
        request: &'a QueryRequest,
    ) -> BoxFuture<'a, Result<Permit, KrishivError>>;

    /// Whether this controller reads [`QueryRequest::estimate`]. When it returns `false` the
    /// session does not compute an estimate (planning cost) and the field is `None`. The default
    /// is `true`; [`Unlimited`] returns `false`.
    fn uses_estimate(&self) -> bool {
        true
    }
}

/// The default controller: admits every query immediately, with no budget.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unlimited;

impl AdmissionController for Unlimited {
    fn admit<'a>(&'a self, _: &'a QueryRequest) -> BoxFuture<'a, Result<Permit, KrishivError>> {
        Box::pin(std::future::ready(Ok(Permit::unlimited())))
    }

    fn uses_estimate(&self) -> bool {
        false
    }
}
