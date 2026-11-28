// SPDX-License-Identifier: Apache-2.0
//! Running deep recursion on a thread with a big stack.
//!
//! Decoding, dropping and translating a message nested a thousand levels
//! deep recurses a thousand levels, about 3 KiB of stack per level in a debug
//! build: more than the 2 MiB of a test or async-runtime worker thread. A
//! message deeper than [`INLINE_LEVELS`] therefore runs on a thread of its
//! own ([`krishiv_common::stack::on_big_stack`], which the SQL lowering uses
//! too) whose stack fits [`crate::wire::MAX_NESTING`] levels whatever the
//! caller's stack is. The stack is virtual memory: only the pages a deep
//! message touches are ever mapped.

use krishiv_common::error::{InternalError, KrishivError};

/// Nesting handled on the caller's stack: 16 levels take about 50 KiB (4 on Windows, whose debug
/// frames are several times larger).
pub(crate) const INLINE_LEVELS: usize = if cfg!(windows) { 4 } else { 16 };

/// Runs `f` on a thread with a big stack. A panic of `f` is re-raised on the
/// caller, as it would be had `f` run there.
///
/// # Errors
/// `INTERNAL_ERROR` when the thread cannot be started.
pub(crate) fn on_deep_stack<R: Send>(f: impl FnOnce() -> R + Send) -> Result<R, KrishivError> {
    krishiv_common::stack::on_big_stack(f).ok_or_else(|| {
        KrishivError::from(InternalError {
            message: "cannot start a thread for a deeply nested plan".to_owned(),
        })
    })
}

/// A value (a deeply nested proto) that is dropped on a big-stack thread, wherever its owner is
/// dropped: an async worker holding it across an `await` cannot overflow its own stack by it.
pub struct Deep<T: Send>(Option<Box<T>>);

impl<T: Send> Deep<T> {
    /// Takes `value`.
    pub fn new(value: T) -> Self {
        Self(Some(Box::new(value)))
    }

    /// The value.
    #[must_use]
    pub fn get(&self) -> Option<&T> {
        self.0.as_deref()
    }

    /// Builds a value from this one on a big-stack thread (a deep clone, or a derived message).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when the thread cannot be started or the value is gone.
    pub fn map<U: Send>(&self, f: impl FnOnce(&T) -> U + Send) -> Result<Deep<U>, KrishivError>
    where
        T: Sync,
    {
        let value = self.0.as_deref().ok_or_else(|| {
            KrishivError::from(InternalError {
                message: "a deep value was already dropped".to_owned(),
            })
        })?;
        on_deep_stack(|| f(value)).map(Deep::new)
    }
}

impl<T: Send> Drop for Deep<T> {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            // A failed thread start leaks the value rather than recursing on a small stack.
            let _ = krishiv_common::stack::on_big_stack(move || drop(value));
        }
    }
}

impl<T: Send + Sync + Clone> Clone for Deep<T> {
    fn clone(&self) -> Self {
        // Cloned on a big stack too; a failed thread start leaves the copy empty (its users refuse it).
        let value = self
            .0
            .as_ref()
            .and_then(|v| krishiv_common::stack::on_big_stack(|| Box::new((**v).clone())));
        Self(value)
    }
}

impl<T: Send + std::fmt::Debug> std::fmt::Debug for Deep<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Deep(..)")
    }
}

impl<T: Send + PartialEq> PartialEq for Deep<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

#[cfg(test)]
mod tests;
