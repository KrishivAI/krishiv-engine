// SPDX-License-Identifier: Apache-2.0
//! Deep recursion without stack overflow.
//!
//! Recursive passes over plans and syntax trees nest at most `MAX_DEPTH`
//! levels, but a level's frame can be large (debug builds especially). A
//! pass that recurses calls [`on_big_stack`] every [`STEP`] levels, so it
//! never needs more than `STEP` frames of the caller's stack.

/// Levels a recursive pass may take on one stack segment.
pub const STEP: usize = 32;

/// Runs `f` on a new thread with a 64 MiB stack (16 MiB in release
/// builds), waiting for it; `None` when no thread can be started. A panic
/// in `f` is resumed on the calling thread.
pub fn on_big_stack<T: Send>(f: impl FnOnce() -> T + Send) -> Option<T> {
    let size = if cfg!(debug_assertions) {
        64 << 20
    } else {
        16 << 20
    };
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .name("krishiv-deep".to_owned())
            .stack_size(size)
            .spawn_scoped(scope, f)
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
    fn runs_and_returns() {
        assert_eq!(super::on_big_stack(|| 6 * 7), Some(42));
    }
}
