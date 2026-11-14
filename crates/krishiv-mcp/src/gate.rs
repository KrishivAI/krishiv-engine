// SPDX-License-Identifier: Apache-2.0
//! The read-only gate (ADR 0025, design `P2-T095` section 2.3, `P3-T063-analyze-api.md` §3.4), four
//! layers:
//! 1. the statement must parse as exactly one statement of class `Read` ([`check_read`]: sync and
//!    session-free, before the call queues);
//! 2. its analysis, in the call's session, must be a complete read ([`check_analysis`]): class `READ`,
//!    not `partial`, needing no privilege outside [`READ_ONLY_PRIVILEGES`], and no user code run at
//!    analysis (a Python UDTF's `analyze`; MCP sessions have no UDF runtime today, but the gate does not
//!    rely on that);
//! 3. the call's session denies every privilege but `Select` and `Browse` through the normal access
//!    path ([`SetupLock`]), which also closes the gap between the analysis and the run (a catalog or
//!    policy change between the two);
//! 4. the server has no write tool.
//!
//! Cost of layer 2: the statement is analyzed twice per call (once here, once when it runs), so a
//! denied read is also audited twice (`krishiv::audit`); acceptable at MCP's rate of one running query.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use krishiv_common::error::{KrishivError, KrishivMcpStatementNotAllowed};
use krishiv_session::{
    AccessDecision, AccessPolicy, Analysis, AnalysisWarning, BoxFuture, PolicyVersion, Principal,
    Privilege, READ_ONLY_PRIVILEGES, ReadOnlyPolicy, Securable, Session, StatementClass,
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

/// Analyzes `sql` in `session` and allows it only when the analysis is a complete read (layer 2 of the
/// module doc). Async and bound to the session: it runs after the call's session is built.
///
/// # Errors
/// The analysis error (a denied read is `KRISHIV_INSUFFICIENT_PERMISSIONS`, as on a run), or
/// `KRISHIV_MCP_STATEMENT_NOT_ALLOWED` naming why.
pub async fn check_analysis(session: &Session, sql: &str) -> Result<(), KrishivError> {
    allows(&session.analyze(sql).await?)
}

/// Whether `analysis` is a complete read (see [`check_analysis`]).
fn allows(analysis: &Analysis) -> Result<(), KrishivError> {
    allows_parts(
        analysis.info.class,
        analysis.partial,
        analysis.info.securables.iter().map(|u| u.privilege),
        analysis
            .warnings
            .contains(&AnalysisWarning::UserCodeRanAtAnalysis),
    )
}

/// [`allows`] over the parts of an analysis: its class, whether it is partial, the privileges it
/// needs, and whether user code ran at analysis.
fn allows_parts(
    class: StatementClass,
    partial: bool,
    privileges: impl IntoIterator<Item = Privilege>,
    user_code_ran: bool,
) -> Result<(), KrishivError> {
    let refused = |why: &str| -> Result<(), KrishivError> {
        Err(KrishivMcpStatementNotAllowed {
            statement_class: why.to_owned(),
        }
        .into())
    };
    if class != StatementClass::Read {
        return refused(class.name());
    }
    if partial {
        return refused("READ, but its reads cannot be listed");
    }
    if let Some(p) = privileges
        .into_iter()
        .find(|p| !READ_ONLY_PRIVILEGES.contains(p))
    {
        return refused(&format!("READ, but it needs {p}"));
    }
    if user_code_ran {
        return refused("READ, but it runs user code at analysis");
    }
    Ok(())
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

    /// From now on only [`READ_ONLY_PRIVILEGES`] are allowed.
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

#[cfg(test)]
mod tests {
    use krishiv_session::{Privilege, StatementClass};

    use super::allows_parts;

    /// The analyze layer refuses `Monitor` and `Usage` (others' `system.*` rows, sending code the server
    /// runs), and anything that is not a complete read; it allows `Select` and `Browse`.
    #[test]
    fn mcp_refuses_monitor_and_usage() {
        let read = StatementClass::Read;
        assert!(allows_parts(read, false, [Privilege::Select, Privilege::Browse], false).is_ok());
        for p in [
            Privilege::Monitor,
            Privilege::Usage,
            Privilege::Modify,
            Privilege::Manage,
        ] {
            let e = allows_parts(read, false, [Privilege::Select, p], false).unwrap_err();
            assert_eq!(e.error_class().name(), "KRISHIV_MCP_STATEMENT_NOT_ALLOWED");
            assert!(e.to_string().contains(&p.to_string()), "{e}");
        }
        assert!(allows_parts(read, true, [], false).is_err());
        assert!(allows_parts(read, false, [], true).is_err());
        let e = allows_parts(StatementClass::Write, false, [], false).unwrap_err();
        assert!(e.to_string().contains("WRITE"), "{e}");
    }
}
