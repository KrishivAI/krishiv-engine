// SPDX-License-Identifier: Apache-2.0
//! The registry functions declared leakproof (design `P3-E14-security-barrier.md` §4): a call
//! that, given arguments it never raises for, reveals nothing about its row except through its
//! value. A security barrier lets only leakproof expressions run on rows its policy rejects.
//!
//! A declaration is a reviewed security decision, separate from the never-raises allow-list of
//! `krishiv-plan` `raises.rs` (a performance list): adding a name there does not widen the
//! barrier. Sorted (a test keeps it so); every name is a registry entry.

/// The declared names, sorted.
pub(super) const LEAKPROOF: &[&str] = &[
    // NULL handling over its arguments: no error of its own, no side effect.
    "coalesce",
    // The `try_` arithmetic gives NULL where the operator raises.
    "try_add",
    "try_divide",
    "try_mod",
    "try_multiply",
    "try_subtract",
];

/// Whether `name` (lower case) is declared leakproof.
pub(super) fn declared(name: &str) -> bool {
    LEAKPROOF.binary_search(&name).is_ok()
}
