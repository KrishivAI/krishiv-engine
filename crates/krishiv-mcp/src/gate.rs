// SPDX-License-Identifier: Apache-2.0
//! The read-only gate (ADR 0025, design `P2-T095` section 2.3), three layers:
//! the statement must parse as exactly one statement of class `Read`
//! ([`check_read`]); the call's session denies every privilege but `Select`
//! through the normal access path ([`SetupLock`]); and the server has no
//! write tool.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use krishiv_common::error::{KrishivError, KrishivMcpStatementNotAllowed};
use krishiv_session::{
    AccessDecision, AccessPolicy, BoxFuture, PolicyVersion, Principal, Privilege, ReadOnlyPolicy,
    Securable, StatementClass,
};
use krishiv_sql::ParserOptions;

/// Parses `sql` as exactly one statement and allows it only when it is a
/// read (`EXPLAIN ANALYZE` of a write is a write).
///
/// # Errors
/// The parse error, or `KRISHIV_MCP_STATEMENT_NOT_ALLOWED` naming the class.
pub fn check_read(sql: &str) -> Result<(), KrishivError> {
    let statement =
        krishiv_sql::parse_statement(sql, &ParserOptions::default()).map_err(|e| e.error)?;
    match krishiv_sql::classify(&statement) {
        StatementClass::Read => Ok(()),
        class => Err(KrishivMcpStatementNotAllowed {
            statement_class: class.name().to_owned(),
        }
        .into()),
    }
}

/// `inner` while a session is being set up (its `--data` views are
/// registered), read-only once [`SetupLock::lock`] is called.
pub struct SetupLock {
    locked: AtomicBool,
    inner: Arc<dyn AccessPolicy>,
    read_only: ReadOnlyPolicy,
}

impl std::fmt::Debug for SetupLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SetupLock")
            .field("locked", &self.locked.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl SetupLock {
    /// Open over `inner` (path and other checks); read-only after `lock`.
    #[must_use]
    pub fn new(inner: Arc<dyn AccessPolicy>) -> Self {
        Self {
            locked: AtomicBool::new(false),
            read_only: ReadOnlyPolicy::new(Arc::clone(&inner)),
            inner,
        }
    }

    /// From now on only `Select` is allowed.
    pub fn lock(&self) {
        self.locked.store(true, Ordering::Release);
    }
}

impl AccessPolicy for SetupLock {
    fn version(&self) -> PolicyVersion {
        self.inner.version()
    }

    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        if self.locked.load(Ordering::Acquire) {
            self.read_only.check(principal, securable, privilege)
        } else {
            self.inner.check(principal, securable, privilege)
        }
    }
}

/// Whether the process runs as uid 0 (Linux; elsewhere `false`).
#[must_use]
pub fn is_root() -> bool {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("Uid:").map(str::to_owned))
        })
        .is_some_and(|uids| uids.split_whitespace().next() == Some("0"))
}
