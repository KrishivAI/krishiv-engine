// SPDX-License-Identifier: Apache-2.0
//! Deep recursion without stack overflow.
//!
//! Recursive passes over plans and syntax trees nest at most `MAX_DEPTH`
//! levels, but a level's frame can be large (debug builds especially). A
//! pass that recurses calls [`on_big_stack`] every [`STEP`] levels, so it
//! never needs more than `STEP` frames of the caller's stack.

/// Levels a recursive pass may take on one stack segment.
pub const STEP: usize = 32;

type Inherited = std::sync::Arc<dyn std::any::Any + Send + Sync>;

thread_local! {
    static INHERITED: std::cell::RefCell<Option<Inherited>> = const { std::cell::RefCell::new(None) };
}

/// Runs `f` with `value` as this thread's inherited value, which every [`on_big_stack`] thread started
/// under it sees too (a caller's task-local state, such as a cancellation handle, does not cross to
/// the new thread by itself).
pub fn with_inherited<T>(value: Inherited, f: impl FnOnce() -> T) -> T {
    let previous = INHERITED.with(|c| c.replace(Some(value)));
    let out = f();
    INHERITED.with(|c| *c.borrow_mut() = previous);
    out
}

/// The value [`with_inherited`] set on this thread or on the thread that started it.
#[must_use]
pub fn inherited() -> Option<Inherited> {
    INHERITED.with(|c| c.borrow().clone())
}

/// Runs `f` on a new thread with a 64 MiB stack (16 MiB in release
/// builds), waiting for it; `None` when no thread can be started. A panic
/// in `f` is resumed on the calling thread.
pub fn on_big_stack<T: Send>(f: impl FnOnce() -> T + Send) -> Option<T> {
    let size = if cfg!(debug_assertions) {
        64 << 20
    } else {
        16 << 20
    };
    let inherited = inherited();
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .name("krishiv-deep".to_owned())
            .stack_size(size)
            .spawn_scoped(scope, move || {
                INHERITED.with(|c| *c.borrow_mut() = inherited);
                f()
            })
            .ok()?;
        Some(
            handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
        )
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_big_stack_thread_inherits_the_value() {
        let seen = super::with_inherited(std::sync::Arc::new(7_u32), || {
            super::on_big_stack(|| {
                super::inherited().and_then(|v| v.downcast_ref::<u32>().copied())
            })
        });
        assert_eq!(seen, Some(Some(7)));
        assert!(super::inherited().is_none());
    }

    #[test]
    fn runs_and_returns() {
        assert_eq!(super::on_big_stack(|| 6 * 7), Some(42));
    }
}
