// SPDX-License-Identifier: Apache-2.0
//! Memory errors and their Spark error classes.

use krishiv_common::error::{
    ErrorClass, InternalErrorMemory, KrishivError, KrishivQueryAbortedOutOfMemory,
    UnableToAcquireMemory,
};

use crate::ConsumerId;

/// A reservation that could not be granted. Never a panic: every failure of
/// the pool or the arbitrator is one of these, and each maps to a Spark-style
/// error class and SQLSTATE through [`KrishivError`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MemoryError {
    /// The request does not fit and the arbitrator could not make room
    /// (`UNABLE_TO_ACQUIRE_MEMORY`, SQLSTATE 53200). Spark's message reports
    /// the bytes received, which is always 0: a request is granted whole or
    /// not at all. `available` is the free memory of the pool whose limit was
    /// hit, for diagnostics.
    #[error("{}", KrishivError::from(self))]
    Exhausted {
        /// Bytes requested.
        requested: usize,
        /// Free bytes in the limiting pool when the request failed.
        available: usize,
    },
    /// The arbitrator aborted this query to free memory
    /// (`KRISHIV_QUERY_ABORTED_OUT_OF_MEMORY`, SQLSTATE 53200). Every task of
    /// an aborted query sees this; only the reservation call that failed
    /// sees [`Exhausted`](Self::Exhausted).
    #[error("{}", KrishivError::from(self))]
    Aborted {
        /// Bytes the query held when it was chosen.
        held: usize,
    },
    /// The consumer is not registered with this pool: a caller bug, reported
    /// as `INTERNAL_ERROR_MEMORY` (SQLSTATE XX000) rather than a panic.
    #[error("{}", KrishivError::from(self))]
    UnknownConsumer {
        /// The unknown id.
        id: ConsumerId,
    },
    /// The arbitrator's clock cannot time its waits (for example the
    /// built-in timer thread could not be started), so the request failed
    /// before anything was spilled or aborted; a later request tries again
    /// (`INTERNAL_ERROR_MEMORY`, SQLSTATE XX000).
    #[error("{}", KrishivError::from(self))]
    TimerUnavailable {
        /// Why the clock is unavailable.
        reason: String,
    },
}

impl MemoryError {
    /// The error class this error is reported with.
    pub fn error_class(&self) -> ErrorClass {
        match self {
            MemoryError::Exhausted { .. } => ErrorClass::UnableToAcquireMemory,
            MemoryError::Aborted { .. } => ErrorClass::KrishivQueryAbortedOutOfMemory,
            MemoryError::UnknownConsumer { .. } | MemoryError::TimerUnavailable { .. } => {
                ErrorClass::InternalErrorMemory
            }
        }
    }
}

impl From<&MemoryError> for KrishivError {
    fn from(e: &MemoryError) -> Self {
        match e {
            MemoryError::Exhausted { requested, .. } => UnableToAcquireMemory {
                requested_bytes: requested.to_string(),
                received_bytes: "0".to_owned(),
            }
            .into(),
            MemoryError::Aborted { held } => KrishivQueryAbortedOutOfMemory {
                held_bytes: held.to_string(),
            }
            .into(),
            MemoryError::UnknownConsumer { id } => InternalErrorMemory {
                message: format!("memory consumer {} is not registered with this pool", id.0),
            }
            .into(),
            MemoryError::TimerUnavailable { reason } => InternalErrorMemory {
                message: format!("arbitration timer unavailable: {reason}"),
            }
            .into(),
        }
    }
}

impl From<MemoryError> for KrishivError {
    fn from(e: MemoryError) -> Self {
        Self::from(&e)
    }
}
