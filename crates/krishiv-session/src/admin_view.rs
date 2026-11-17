// SPDX-License-Identifier: Apache-2.0
//! What an administrator's API shows of a runtime's sessions and queries (P3-T066 §5.3): the rows
//! of `system.sessions` and `system.queries` as plain data, and who owns each, so the server can
//! apply ADR 0046's own-rows rule and the `Operate` / `Monitor` privileges before it answers.
//! Nothing here asks the access policy; the caller has.

use krishiv_plan::analyzer::Principal;

use crate::runtime::EngineRuntime;
use crate::sessions::{SessionRow, owner_key};
use crate::system::QueryRow;

/// A session as `system.sessions` shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionInfo {
    /// The session id.
    pub session_id: String,
    /// The principal the session acts for.
    pub principal: String,
    /// The service principal acting for it (trusted impersonation), if any.
    pub delegated_by: Option<String>,
    /// The client type it announced, if any (client-asserted).
    pub client_type: Option<String>,
    /// `connect`, `flight_sql` or `embedded`.
    pub transport: String,
    /// `open`, `closed`, `expired` or `killed`.
    pub state: String,
    /// When it was created, in milliseconds since the Unix epoch.
    pub created_ms: i64,
    /// When it was last used.
    pub last_active_ms: i64,
    /// When it ended, if it did.
    pub closed_ms: Option<i64>,
    /// Why it ended (`release_session`, `idle_timeout`, `admin_kill`, `drain`), if it did.
    pub close_reason: Option<String>,
    /// The session's tags.
    pub tags: Vec<(String, String)>,
    /// Queries it runs now.
    pub queries_running: i64,
    /// Queries it has started.
    pub queries_started: i64,
    owner: String,
}

impl SessionInfo {
    /// Whether `principal` owns the session (the own-rows rule of `system.sessions`).
    #[must_use]
    pub fn is_owned_by(&self, principal: &Principal) -> bool {
        self.owner == owner_key(principal)
    }

    fn of(r: SessionRow) -> Self {
        Self {
            session_id: r.session_id,
            principal: r.principal,
            delegated_by: r.delegated_by,
            client_type: r.client_type,
            transport: r.transport.to_owned(),
            state: r.state.to_owned(),
            created_ms: r.created_ms,
            last_active_ms: r.last_active_ms,
            closed_ms: r.closed_ms,
            close_reason: r.close_reason.map(str::to_owned),
            tags: r.tags,
            queries_running: r.queries_running,
            queries_started: r.queries_started,
            owner: r.owner,
        }
    }
}

/// A query as `system.queries` shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryInfo {
    /// The query id (the operation id, or a minted UUIDv7).
    pub id: String,
    /// The session that ran it.
    pub session_id: String,
    /// The principal it ran for.
    pub principal: String,
    /// `queued`, `running`, `finished`, `failed`, `cancelled` or `denied`.
    pub state: String,
    /// `READ`, `WRITE`, `DDL`, `SESSION` or `ADMIN`, when classified.
    pub statement_class: Option<String>,
    /// The statement text, cut and masked by the session's text policy.
    pub text: Option<String>,
    /// When it was submitted, in milliseconds since the Unix epoch.
    pub submitted_ms: i64,
    /// When admission let it run.
    pub started_ms: Option<i64>,
    /// When it ended.
    pub ended_ms: Option<i64>,
    /// The error class of a failed query.
    pub error_class: Option<String>,
    /// The resource group it ran in, if a controller placed it.
    pub resource_group: Option<String>,
    /// The tags it carried (the session's and the request's).
    pub tags: Vec<(String, String)>,
    owner: String,
}

impl QueryInfo {
    /// Whether `principal` owns the query.
    #[must_use]
    pub fn is_owned_by(&self, principal: &Principal) -> bool {
        self.owner == owner_key(principal)
    }

    fn of(r: QueryRow) -> Self {
        Self {
            id: r.id,
            session_id: r.session_id,
            principal: r.principal,
            state: r.state.to_owned(),
            statement_class: r.statement_class.map(str::to_owned),
            text: r.text,
            submitted_ms: r.submitted_ms,
            started_ms: r.started_ms,
            ended_ms: r.ended_ms,
            error_class: r.error_class,
            resource_group: r.resource_group,
            tags: r.tags,
            owner: r.owner,
        }
    }
}

impl EngineRuntime {
    /// Every session the runtime remembers (live ones by creation time, then the ended ones),
    /// whoever owns it. The caller filters by [`SessionInfo::is_owned_by`] or holds `Monitor`.
    #[must_use]
    pub fn session_infos(&self) -> Vec<SessionInfo> {
        self.sessions
            .snapshot(|| self.running_counts())
            .into_iter()
            .map(SessionInfo::of)
            .collect()
    }

    /// Every query in the runtime's history (running and queued ones included), oldest first,
    /// whoever owns it.
    #[must_use]
    pub fn query_infos(&self) -> Vec<QueryInfo> {
        self.history
            .snapshot(None)
            .into_iter()
            .map(QueryInfo::of)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use krishiv_plan::analyzer::{Principal, PrincipalKind};

    use crate::Session;

    #[test]
    fn a_session_and_its_query_show_with_their_owner() {
        let rt = crate::EngineRuntime::builder().build().unwrap();
        let alice = Principal::new("alice", PrincipalKind::User);
        let s = Session::builder()
            .runtime(Arc::clone(&rt))
            .principal(alice.clone())
            .build()
            .unwrap();
        block_on(async { s.sql("SELECT 1").await.unwrap().collect().await.unwrap() });
        let sessions = rt.session_infos();
        let mine = sessions
            .iter()
            .find(|i| i.principal == "alice")
            .expect("alice's session is listed");
        assert!(mine.is_owned_by(&alice));
        assert!(!mine.is_owned_by(&Principal::new("bob", PrincipalKind::User)));
        assert!(mine.queries_started >= 1);
        let queries = rt.query_infos();
        assert!(!queries.is_empty());
        let q = &queries[0];
        assert!(q.is_owned_by(&alice));
        assert!(!q.is_owned_by(&Principal::new("bob", PrincipalKind::User)));
    }
}
