// SPDX-License-Identifier: Apache-2.0
//! `GetStatus` (design P3-T011a §3.12): the states of a session's
//! operations.
//!
//! `operation_status` absent: no statuses. Present without ids: every
//! operation the session knows, the live ones and those that ended within
//! the window [`Operations`](crate::operation::Operations) keeps (30 minutes
//! after the last lookup). Present with ids: one status per distinct id, in
//! the order first given; an id the session does not know (or that is not a
//! UUID) is `UNKNOWN`. `UNSPECIFIED` is never sent. Request extensions are
//! ignored (Spark hands them to `spark.connect.extensions.getStatus.classes`
//! plugins; Krishiv has none).

use std::collections::HashSet;

use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::get_status_response::OperationStatus;
use krishiv_connect::proto::v4_3::get_status_response::operation_status::OperationState as Wire;

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
                .filter(|id| seen.insert(id.as_str()))
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
