// SPDX-License-Identifier: Apache-2.0
//! The production [`CommitTable`] of an Iceberg REST catalog (design note §7.2, PR-36c part 2):
//! [`RestCommitTable`] builds the spec's `CommitTableRequest` from iceberg-rust's serde types
//! (`identifier`, `requirements`, `updates`) and sends it through a [`RestCommitTransport`], then
//! maps the answer onto [`CommitOutcome`] exactly as its doc table says (Java's
//! `ErrorHandlers.tableCommitHandler`, apache-iceberg 1.12.0):
//!
//! | Answer | Outcome |
//! |---|---|
//! | 200 | `Committed`, the `metadata` of the `LoadTableResult` (one that does not parse: `Unknown`) |
//! | 400 or 409 "Snapshot already exists for" (duplicate snapshot id) | `Unknown`: probably committed |
//! | 400 or 409 stale (sequence number, `first-row-id` or `timestamp-ms` behind; Java's "Validation failed, please retry") | `Conflict { stale: true }` |
//! | other 409 (`CommitFailedException`) | `Conflict { stale: false }` |
//! | 500, 502, 503, 504; no answer after the request may have left | `Unknown` (a `Retry-After` named in the reason) |
//! | no answer and the request provably never left (egress refusal, no token, no connection) | `Failed` |
//! | any other 2xx (201, 202, 204, ...): the request was accepted | `Unknown`: it may have landed |
//! | 401, 403 (the transport has refreshed the token once), 404, 422, any other status | `Failed` |
//!
//! A `Failed` answer is `KRISHIV_ICEBERG_COMMIT_FAILED` naming the status (never the server's
//! message); the loop decides that a failure answered to a resend is still unknown. The server's
//! message is read only to tell stale and duplicate answers apart, then logged, cut short.
//!
//! `Retry-After` is kept in the reason only: the loop's probe waits follow
//! `commit.status-check.*` (a follow-up may let a 503's `Retry-After` set the next wait).

use std::sync::Arc;

use async_trait::async_trait;
use iceberg::spec::TableMetadata;
use iceberg::{TableIdent, TableRequirement, TableUpdate};
use krishiv_common::error::{InternalError, KrishivError, KrishivIcebergCommitFailed};
use serde_json::Value;

use super::append::WRITE_ID_PROP;
use super::commit::{CommitOutcome, CommitTable, REMOVED};
use crate::rest_commit::{RestCommitTransport, TransportFailure, TransportReply};

/// The common prefix of Java's "Snapshot already exists for id: N" and iceberg-rust's
/// "Snapshot already exists for: 'N'" (design note §7.2).
const DUPLICATE: &str = "Snapshot already exists for";

/// The texts of a request behind the table (design note §7.2: iceberg-rust's three refusals in
/// `table_metadata_builder.rs`, Java's `TableMetadata.java` texts and the 1.12 server's 409
/// "Validation failed, please retry").
const STALE: [&str; 4] = [
    "older than last sequence number",
    "first-row-id is behind",
    "Invalid snapshot timestamp",
    "Validation failed, please retry",
];

/// Longest server message a log line carries.
const LOGGED_CHARS: usize = 300;

/// The commit endpoint of one REST catalog table (module docs).
pub(crate) struct RestCommitTable {
    transport: Arc<dyn RestCommitTransport>,
    ident: TableIdent,
}

impl RestCommitTable {
    /// The table `ident` committed through `transport`.
    pub(crate) fn new(transport: Arc<dyn RestCommitTransport>, ident: TableIdent) -> Self {
        Self { transport, ident }
    }
}

/// The `CommitTableRequest` body of one request.
pub(crate) fn request_body(
    ident: &TableIdent,
    requirements: &[TableRequirement],
    updates: &[TableUpdate],
) -> Result<Vec<u8>, KrishivError> {
    let body = serde_json::json!({
        "identifier": {
            "namespace": ident.namespace().as_ref(),
            "name": ident.name(),
        },
        "requirements": requirements,
        "updates": updates,
    });
    serde_json::to_vec(&body).map_err(|e| {
        InternalError {
            message: format!("an Iceberg commit request that does not serialize: {e}"),
        }
        .into()
    })
}

/// The table metadata of a `LoadTableResult` body.
pub(crate) fn load_result_metadata(body: &[u8]) -> Result<TableMetadata, String> {
    let value: Value = serde_json::from_slice(body).map_err(|e| format!("not JSON: {e}"))?;
    let metadata = value
        .get("metadata")
        .cloned()
        .ok_or_else(|| "no `metadata` member".to_owned())?;
    serde_json::from_value(metadata).map_err(|e| format!("metadata that does not parse: {e}"))
}

/// The `ErrorModel` of a body: its `type` and `message`, when present.
fn error_model(body: &[u8]) -> (Option<String>, Option<String>) {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return (None, None);
    };
    let error = value.get("error").unwrap_or(&value);
    let text = |k: &str| error.get(k).and_then(Value::as_str).map(str::to_owned);
    (text("type"), text("message"))
}

fn shortened(text: &str) -> String {
    text.chars().take(LOGGED_CHARS).collect()
}

/// The snapshot id and write key a request carries (its `add-snapshot`).
fn request_ids(updates: &[TableUpdate]) -> (String, String) {
    updates
        .iter()
        .find_map(|u| match u {
            TableUpdate::AddSnapshot { snapshot } => Some((
                snapshot.snapshot_id().to_string(),
                snapshot
                    .summary()
                    .additional_properties
                    .get(WRITE_ID_PROP)
                    .cloned()
                    .unwrap_or_default(),
            )),
            _ => None,
        })
        .unwrap_or_default()
}

/// What one request came to, from the transport's answer (module docs).
pub(crate) fn outcome(
    table: &str,
    updates: &[TableUpdate],
    answer: Result<TransportReply, TransportFailure>,
) -> CommitOutcome {
    let failed = |reason: String| {
        let (snapshot_id, write_id) = request_ids(updates);
        CommitOutcome::Failed(
            KrishivIcebergCommitFailed {
                table: table.to_owned(),
                snapshot_id,
                write_id,
                attempts: "1".to_owned(),
                reason,
                files: REMOVED.to_owned(),
            }
            .into(),
        )
    };
    let reply = match answer {
        Ok(reply) => reply,
        Err(f) if f.sent => {
            return CommitOutcome::Unknown(format!(
                "no answer after the request was sent ({})",
                f.reason
            ));
        }
        Err(f) => return failed(format!("the request was never sent ({})", f.reason)),
    };
    let status = reply.status;
    if status == 200 {
        return match load_result_metadata(&reply.body) {
            Ok(m) => CommitOutcome::Committed(Box::new(m)),
            Err(e) => CommitOutcome::Unknown(format!("a 200 answer whose {e}")),
        };
    }
    // A 2xx other than a parsable 200 (a 202 or 204 from a proxy or an odd catalog) says the
    // request was accepted: the snapshot may be in the table, so it is never a sure failure.
    if (200..300).contains(&status) {
        return CommitOutcome::Unknown(format!(
            "an HTTP {status} answer that is not the commit's result"
        ));
    }
    let (kind, message) = error_model(&reply.body);
    let message = message.unwrap_or_default();
    if !message.is_empty() {
        tracing::warn!(
            target: "krishiv::catalog",
            table,
            status,
            kind = kind.as_deref().unwrap_or(""),
            message = shortened(&message).as_str(),
            "an Iceberg REST catalog refused a commit request"
        );
    }
    let kind_text = kind.map(|k| format!(" ({k})")).unwrap_or_default();
    match status {
        400 | 409 if message.contains(DUPLICATE) => CommitOutcome::Unknown(format!(
            "HTTP {status}: the snapshot id is already in the table (probably committed)"
        )),
        400 | 409 if STALE.iter().any(|s| message.contains(s)) => {
            CommitOutcome::Conflict { stale: true }
        }
        409 => CommitOutcome::Conflict { stale: false },
        500 | 502 | 503 | 504 => {
            let wait = reply
                .retry_after
                .map(|w| format!(", Retry-After {} s", w.as_secs()))
                .unwrap_or_default();
            CommitOutcome::Unknown(format!("HTTP {status}{kind_text}{wait}"))
        }
        _ => failed(format!(
            "the catalog refused the request with HTTP {status}{kind_text}"
        )),
    }
}

#[async_trait]
impl CommitTable for RestCommitTable {
    fn name(&self) -> String {
        self.transport.name()
    }

    async fn commit(
        &self,
        requirements: Vec<TableRequirement>,
        updates: Vec<TableUpdate>,
    ) -> CommitOutcome {
        let body = match request_body(&self.ident, &requirements, &updates) {
            Ok(b) => b,
            Err(e) => return CommitOutcome::Failed(e),
        };
        let answer = self.transport.post_commit(body).await;
        outcome(&self.transport.name(), &updates, answer)
    }

    async fn load(&self, all_snapshots: bool) -> Result<TableMetadata, KrishivError> {
        let body = self.transport.load(all_snapshots).await?;
        load_result_metadata(&body).map_err(|e| {
            InternalError {
                message: format!(
                    "the Iceberg REST catalog's table {} answered with {e}",
                    self.transport.name()
                ),
            }
            .into()
        })
    }
}
