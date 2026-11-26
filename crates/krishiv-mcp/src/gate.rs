// SPDX-License-Identifier: Apache-2.0
//! The read-only gate (ADR 0025, design `P2-T095` section 2.3, `P3-T063-analyze-api.md` §3.4), four
//! layers:
//! 1. the statement must parse as exactly one statement of class `Read` ([`check_read`]: sync and
//!    session-free, before the call queues);
//! 2. its analysis, in the call's session, must be a complete read ([`check_analysis`]): class `READ`,
//!    not `partial`, and needing no privilege outside [`READ_ONLY_PRIVILEGES`]. It also refuses an
//!    analysis that reports user code ran (`USER_CODE_RAN_AT_ANALYSIS`, a Python UDTF's `analyze`), but
//!    only after that analysis: the check is post hoc. Today no user code can run there only because
//!    nothing sets that warning yet (PR-48b) and an MCP session has no Python UDF runtime and no way
//!    to register a Python function (`CREATE FUNCTION` is not a read, so layer 1 refuses it);
//!    `tools::tests::child_sessions_have_no_python_udf_runtime` pins the second fact. When PR-48b lands,
//!    the refusal has to move before the analysis (follow-up, design §3.4);
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

    /// The policy asked now: `inner` until [`Self::lock`], then the read-only one over it.
    fn active(&self) -> &dyn AccessPolicy {
        if self.locked.load(Ordering::Acquire) {
            &self.read_only
        } else {
            &*self.inner
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
        self.active().check(principal, securable, privilege)
    }

    // The rest is forwarded to the policy `check` asks, as the trait asks of a wrapper (G6
    // review n4): without it a column-level or remote `inner` is asked table-level questions,
    // one at a time, and its refusals are counted under the default label.
    fn column_level(&self) -> bool {
        self.active().column_level()
    }

    fn denied(&self, principal: &Principal, securable: &Securable, privilege: Privilege) {
        self.active().denied(principal, securable, privilege);
    }

    fn prefetch<'a>(
        &'a self,
        principal: &'a Principal,
        items: &'a [(Securable, Privilege)],
    ) -> BoxFuture<'a, ()> {
        self.active().prefetch(principal, items)
    }

    fn decides(&self, securable: &Securable) -> bool {
        self.active().decides(securable)
    }

    fn refusal_label(&self) -> Option<&'static str> {
        self.active().refusal_label()
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

    /// The gate parses with the default profile's grammar (Spark 4.3): a `TIME(9)` cast is a read,
    /// not a parse error (R3 review finding 12).
    #[test]
    fn the_gate_reads_the_default_profiles_grammar() {
        assert!(super::check_read("SELECT CAST(c AS TIME(9)) FROM t").is_ok());
        assert!(super::check_read("SELECT CAST(c AS TIMESTAMP_NTZ(6)) FROM t").is_ok());
    }

    /// The analyze layer refuses `Monitor` and `Usage` (others' `system.*` rows, sending code the server
    /// runs), and anything that is not a complete read; it allows `Select` and `Browse`. A unit test: no
    /// SQL statement asks `Monitor` or `Usage` at `authorize()` today (see the integration test
    /// `monitor_and_usage_through_the_tool_path`).
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

    /// G6 review n4: the lock forwards the trait's other methods, before and after `lock`.
    #[test]
    fn the_setup_lock_forwards_to_the_policy_it_asks() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use futures::future::BoxFuture;
        use krishiv_common::error::KrishivError;
        use krishiv_plan::analyzer::{
            AccessDecision, AccessPolicy, PolicyVersion, Principal, PrincipalKind, Securable,
        };

        use super::SetupLock;

        #[derive(Default)]
        struct Remote {
            prefetched: AtomicUsize,
            told: AtomicUsize,
        }
        impl AccessPolicy for Remote {
            fn version(&self) -> PolicyVersion {
                PolicyVersion(3)
            }
            fn check<'a>(
                &'a self,
                _: &'a Principal,
                _: &'a Securable,
                _: Privilege,
            ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
                Box::pin(std::future::ready(Ok(AccessDecision::Allow)))
            }
            fn column_level(&self) -> bool {
                true
            }
            fn denied(&self, _: &Principal, _: &Securable, _: Privilege) {
                self.told.fetch_add(1, Ordering::Relaxed);
            }
            fn prefetch<'a>(
                &'a self,
                _: &'a Principal,
                _: &'a [(Securable, Privilege)],
            ) -> BoxFuture<'a, ()> {
                self.prefetched.fetch_add(1, Ordering::Relaxed);
                Box::pin(std::future::ready(()))
            }
            fn decides(&self, securable: &Securable) -> bool {
                !matches!(securable, Securable::System(_))
            }
            fn refusal_label(&self) -> Option<&'static str> {
                Some("external")
            }
        }

        let remote = Arc::new(Remote::default());
        let lock = SetupLock::new(Arc::clone(&remote) as Arc<dyn AccessPolicy>);
        let ana = Principal::new("ana", PrincipalKind::User);
        let system = Securable::System("cluster".to_owned());
        for locked in [false, true] {
            if locked {
                lock.lock();
            }
            assert!(lock.column_level(), "locked: {locked}");
            assert!(!lock.decides(&system), "locked: {locked}");
            assert_eq!(lock.refusal_label(), Some("external"));
            futures::executor::block_on(lock.prefetch(&ana, &[]));
            lock.denied(&ana, &system, Privilege::Select);
        }
        assert_eq!(remote.prefetched.load(Ordering::Relaxed), 2);
        assert_eq!(remote.told.load(Ordering::Relaxed), 2);
    }
}
