// SPDX-License-Identifier: Apache-2.0
//! The narrow transport an Iceberg REST catalog client gives the connector so its tables commit
//! (design note `P3-T026-T027-iceberg-scan-and-commit.md` §7.2): the connector builds the spec's
//! `CommitTableRequest` and maps each answer onto the commit loop's outcomes; the transport only
//! sends bytes (`POST /v1/{prefix}/namespaces/{ns}/tables/{table}`) and loads the table.
//!
//! No `iceberg` type crosses this boundary: bodies are JSON bytes. Attach a transport with
//! [`IcebergTable::with_rest_commit`](crate::IcebergTable::with_rest_commit).

use std::time::Duration;

use async_trait::async_trait;
use krishiv_common::error::KrishivError;

/// One answer of the catalog to a commit request, after the transport's own token refresh (one
/// resend after a 401, before which the request was not applied).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransportReply {
    /// The HTTP status.
    pub status: u16,
    /// `Retry-After`, in seconds, when the answer had one.
    pub retry_after: Option<Duration>,
    /// The body (a `LoadTableResult` on 200, an `ErrorModel` otherwise), within the
    /// transport's response cap.
    pub body: Vec<u8>,
}

impl TransportReply {
    /// An answer with `status` and `body`, no `Retry-After`.
    #[must_use]
    pub fn new(status: u16, body: Vec<u8>) -> Self {
        Self {
            status,
            retry_after: None,
            body,
        }
    }

    /// The same answer with `Retry-After` set.
    #[must_use]
    pub fn with_retry_after(mut self, wait: Duration) -> Self {
        self.retry_after = Some(wait);
        self
    }
}

/// A commit request that got no answer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransportFailure {
    /// Whether any of the request may have reached the catalog: `false` only when it provably
    /// never left (refused by the egress policy, no token, no connection), so the catalog cannot
    /// have applied it; `true` for a timeout, a reset or an unreadable answer.
    pub sent: bool,
    /// What happened, for messages and the log (never a token or a server's text).
    pub reason: String,
}

impl TransportFailure {
    /// A failure; `sent` as [`Self::sent`].
    #[must_use]
    pub fn new(sent: bool, reason: impl Into<String>) -> Self {
        Self {
            sent,
            reason: reason.into(),
        }
    }
}

/// The commit endpoint of one table of an Iceberg REST catalog, as the catalog client sends it
/// (auth, egress policy, `X-Iceberg-Access-Delegation`, caps: the client's).
#[async_trait]
pub trait RestCommitTransport: Send + Sync {
    /// The table's name, for messages (`catalog.ns.table`).
    fn name(&self) -> String;

    /// Sends `body` (a `CommitTableRequest`, JSON) once; never retried blind: only a 401 may be
    /// resent once after a token refresh.
    ///
    /// # Errors
    /// No answer arrived ([`TransportFailure::sent`] says whether it may have been applied).
    async fn post_commit(&self, body: Vec<u8>) -> Result<TransportReply, TransportFailure>;

    /// The table's `loadTable` answer (a `LoadTableResult`, JSON), every snapshot when
    /// `all_snapshots` (`snapshots=all`), else `snapshots=refs`. Never served from a cache.
    ///
    /// # Errors
    /// The load failed.
    async fn load(&self, all_snapshots: bool) -> Result<Vec<u8>, KrishivError>;
}
