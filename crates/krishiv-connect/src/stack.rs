// SPDX-License-Identifier: Apache-2.0
//! Running deep recursion on a thread with a big stack.
//!
//! Decoding, dropping and translating a message nested a thousand levels
//! deep recurses a thousand levels, about 3 KiB of stack per level in a debug
//! build: more than the 2 MiB of a test or async-runtime worker thread. A
//! message deeper than [`INLINE_LEVELS`] therefore runs on a thread of its
//! own whose stack fits [`crate::wire::MAX_NESTING`] levels whatever the
//! caller's stack is (the SQL parser does the same). The stack is virtual
//! memory: only the pages a deep message touches are ever mapped.

use krishiv_common::error::{InternalError, KrishivError};

/// Nesting handled on the caller's stack: 32 levels take about 100 KiB.
pub(crate) const INLINE_LEVELS: usize = 32;

/// Runs `f` on a thread with a stack for the deepest accepted message.
///
/// A panic of `f` is re-raised on the caller, as it would be had `f` run
/// there.
///
/// # Errors
/// `INTERNAL_ERROR` when the thread cannot be started.
pub(crate) fn on_deep_stack<R: Send>(f: impl FnOnce() -> R + Send) -> Result<R, KrishivError> {
    let stack = if cfg!(debug_assertions) {
        64 << 20
    } else {
        16 << 20
    };
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("krishiv-connect-deep".to_owned())
            .stack_size(stack)
            .spawn_scoped(scope, f)
            .map_err(|e| {
                KrishivError::from(InternalError {
                    message: format!("cannot start a thread for a deeply nested plan: {e}"),
                })
            })
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
    })
}

#[cfg(test)]
mod tests;
