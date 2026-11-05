// SPDX-License-Identifier: Apache-2.0
//! Whose credentials a store's cached blocks belong to (E15).

use std::sync::atomic::{AtomicU64, Ordering};

/// Whose credentials a store reads with, the part of a cached block's key that keeps a byte from
/// being served to a query whose credentials could not have read it (E15): a cache hit skips the
/// store's own access check.
///
/// The default is [`CacheScope::Isolated`]: sharing across principals is something a caller
/// asks for, never something it gets by omission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheScope {
    /// One engine identity reads for every principal (a single-tenant compute): blocks are shared.
    /// The caller vouches that a store's credentials grant every principal the same reads.
    Shared,
    /// The credentials of one principal (its `Principal::id` or a token's subject): blocks are
    /// shared only with the same scope text.
    Principal(String),
    /// Credentials seen by this registration alone (a random nonce): blocks are never shared.
    Isolated,
}

impl CacheScope {
    /// The text joined to every block key. An isolated scope draws a fresh process-unique nonce.
    #[must_use]
    pub fn key_text(&self) -> String {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        match self {
            Self::Shared => "shared".to_owned(),
            Self::Principal(p) => format!("p:{p}"),
            Self::Isolated => format!("n:{}", NEXT.fetch_add(1, Ordering::Relaxed)),
        }
    }
}
