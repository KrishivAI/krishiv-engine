// SPDX-License-Identifier: Apache-2.0
//! Tests of the deep-stack thread (`stack`).

use super::*;

#[test]
fn runs_the_closure_and_returns_its_value() {
    let caller = std::thread::current().id();
    let (value, worker) = on_deep_stack(|| (21 * 2, std::thread::current().id())).expect("runs");
    assert_eq!(value, 42);
    assert_ne!(worker, caller, "the closure runs on a thread of its own");
}

#[test]
fn borrows_from_the_caller() {
    let text = String::from("borrowed");
    let len = on_deep_stack(|| text.len()).expect("runs");
    assert_eq!(len, 8);
}

#[test]
fn a_panic_reaches_the_caller() {
    let outcome = std::panic::catch_unwind(|| on_deep_stack(|| -> u32 { panic!("boom") }));
    assert!(outcome.is_err());
}

#[test]
fn the_stack_fits_a_deep_recursion() {
    // 100,000 frames of a few dozen bytes each need several MiB: more than a
    // test thread has, so this only passes on the big stack.
    fn depth(n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            1 + std::hint::black_box(depth(n - 1))
        }
    }
    assert_eq!(on_deep_stack(|| depth(100_000)).expect("runs"), 100_000);
}
