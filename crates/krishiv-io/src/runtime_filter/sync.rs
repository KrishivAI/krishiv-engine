// SPDX-License-Identifier: Apache-2.0
//! Synchronisation primitives of the publication protocol, swapped for
//! loom's under `--cfg loom` so the model in `tests.rs` explores every
//! interleaving of a publication with concurrent readers.

#[cfg(all(test, loom))]
pub(crate) use loom::sync::{
    Mutex, RwLock,
    atomic::{AtomicU64, Ordering},
};
#[cfg(not(all(test, loom)))]
pub(crate) use std::sync::{
    Mutex, RwLock,
    atomic::{AtomicU64, Ordering},
};
