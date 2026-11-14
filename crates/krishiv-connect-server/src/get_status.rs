// SPDX-License-Identifier: Apache-2.0
//! `GetStatus` (design P3-T011a §3.12): the states of a session's
//! operations.
//!
//! **Unverified against Spark**: Spark's `SparkConnectGetStatusHandler`
//! (v4.3.0) was not read; everything below follows design P3-T011a §3.12,
//! not Spark's source. Verified only: Spark keeps ended operations in
//! `SessionHolder.inactiveOperations`, `expireAfterAccess` 30 minutes, keyed
//! by the operation id string (`SessionHolder.scala:108-115, 208-263`), and
//! the request, response and state enum of `base.proto:1228-1300`.
//!
//! `operation_status` absent: no statuses. Present without ids: every
//! operation the session knows, the live ones and those that ended within
//! the window [`Operations`](crate::operation::Operations) keeps (30 minutes
//! after the last lookup). Present with ids: one status per distinct id, in
//! the order first given, ids deduplicated by their canonical UUID form (the
//! one lookups use; Spark keys by the raw string, so it may answer `UNKNOWN`
//! for a non-canonical spelling: unverified), each echoed as first given; an
//! id the session does not know (or that is not a UUID) is `UNKNOWN`.
//! `UNSPECIFIED` is never sent. Request and operation extensions are
//! ignored (Krishiv has no `spark.connect.extensions.getStatus.classes`
//! plugins, `Connect.scala` v4.3.0 line 266; what Spark does with them is
//! unverified).

use std::collections::HashSet;

use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::get_status_response::OperationStatus;
use krishiv_connect::proto::v4_3::get_status_response::operation_status::OperationState as Wire;

use krishiv_session::QueryId;

use crate::manager::SessionEntry;
use crate::operation::OperationState;

/// The wire value of `state`.
fn wire(state: Option<OperationState>) -> Wire {
    match state {
        None => Wire::Unknown,
        Some(OperationState::Running) => Wire::Running,
        Some(OperationState::Terminating) => Wire::Terminating,
        Some(OperationState::Succeeded) => Wire::Succeeded,
        Some(OperationState::Failed) => Wire::Failed,
        Some(OperationState::Cancelled) => Wire::Cancelled,
    }
}

fn status(operation_id: String, state: Option<OperationState>) -> OperationStatus {
    OperationStatus {
        operation_id,
        state: wire(state).into(),
        extensions: Vec::new(),
    }
}

/// `GetStatus` on the session `entry`.
#[must_use]
pub fn get_status(entry: &SessionEntry, request: &pb::GetStatusRequest) -> pb::GetStatusResponse {
    let operations = entry.operations();
    let operation_statuses = match &request.operation_status {
        None => Vec::new(),
        Some(asked) if asked.operation_ids.is_empty() => operations
            .known()
            .into_iter()
            .map(|(id, state)| status(id.as_str().to_owned(), Some(state)))
            .collect(),
        Some(asked) => {
            let mut seen = HashSet::new();
            asked
                .operation_ids
                .iter()
                .filter(|id| {
                    let key = QueryId::parse(id)
                        .map_or_else(|_| (*id).clone(), |q| q.as_str().to_owned());
                    seen.insert(key)
                })
                .map(|id| status(id.clone(), operations.state_of(id)))
                .collect()
        }
    };
    pb::GetStatusResponse {
        session_id: request.session_id.clone(),
        server_side_session_id: entry.server_side_id().to_owned(),
        operation_statuses,
        extensions: Vec::new(),
    }
}
