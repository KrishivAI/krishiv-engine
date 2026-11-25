// SPDX-License-Identifier: Apache-2.0
//! The embedded engine of the npm package `krishiv` (design/node-engine-napi.md): a
//! Node-API addon built with napi-rs. `js/src/embedded.ts` wraps it as the package's
//! embedded backend; nothing here is public npm API.
//!
//! JavaScript surface: `NativeSession.open(conf)`, `execute(sql) -> NativeQuery`,
//! `setConf`, `getConf`, `close`; `NativeQuery.next()` (Arrow IPC chunks, `null` at the
//! end) and `cancel()`; `version()`. Failures, panics included, reject with an `Error`
//! whose message is the JSON of the engine error (`error.rs`).

mod error;
mod ipc;
mod query;
mod session;

pub use query::NativeQuery;
pub use session::NativeSession;

use napi_derive::napi;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The engine version of this binary; the package refuses a binary of another version.
#[napi]
#[must_use]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

#[cfg(test)]
mod tests;
