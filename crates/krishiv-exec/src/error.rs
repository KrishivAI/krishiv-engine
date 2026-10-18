// SPDX-License-Identifier: Apache-2.0
//! Errors of the execution layer. Every error is a [`KrishivError`]:
//!
//! | Situation | Error class |
//! |---|---|
//! | the query was cancelled (by the user, by dropping its result stream, or after another task failed) | `OPERATION_CANCELED` |
//! | a task panicked, a driver slice stalled in fail-on-stall mode, or an engine invariant broke | `INTERNAL_ERROR` |
//! | the memory arbitrator aborted the query, or memory ran out | the [`krishiv_memory::MemoryError`] class |

use krishiv_common::error::{InternalError, KrishivError, OperationCanceled};

/// Result alias for this crate.
pub type Result<T, E = KrishivError> = std::result::Result<T, E>;

/// `OPERATION_CANCELED`.
pub(crate) fn cancelled() -> KrishivError {
    OperationCanceled {}.into()
}

/// `INTERNAL_ERROR`, for a broken engine invariant (not a user error).
pub(crate) fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}
