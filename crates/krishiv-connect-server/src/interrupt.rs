// SPDX-License-Identifier: Apache-2.0
//! `Interrupt` (design P3-T011a §3.7; Spark's
//! `SparkConnectInterruptHandler`, v4.3.0 lines 31-73, and
//! `SessionHolder.interruptAll` / `interruptOperation`, lines 272-325), in
//! the caller's session only (threat model T-Z5):
//!
//! - `ALL`: every live operation of the session is cancelled;
//! - `OPERATION_ID`: the operation of that id, if live (any text: an unknown
//!   or malformed id interrupts nothing), and it leaves the session's map at
//!   once, as Spark's `closeOperation`;
//! - `TAG`: `UNSUPPORTED_FEATURE.INTERRUPT_TYPE` until P3-T064 (PR-38b)
//!   gives operations their tags' cancellation.
//!
//! `interrupted_ids` lists only operations this call interrupted (not one
//! cancelled before or one that had sent its last response). A cancelled
//! operation's stream ends with `OPERATION_CANCELED`; its engine query is
//! cancelled through the session (`Session::cancel`, one per reported id)
//! so it stops at its next batch boundary and returns its memory.

use krishiv_common::error::{
    InvalidParameterValueInterruptTypeOperationIdRequiresId,
    InvalidParameterValueInterruptTypeTagRequiresTag, KrishivError,
    UnsupportedFeatureInterruptType,
};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::interrupt_request::{Interrupt, InterruptType};

use crate::manager::SessionEntry;

/// Answers `request` on the session `entry`.
///
/// # Errors
/// `INVALID_PARAMETER_VALUE.INTERRUPT_TYPE_TAG_REQUIRES_TAG` /
/// `..._OPERATION_ID_REQUIRES_ID` for a missing tag or id;
/// `UNSUPPORTED_FEATURE.INTERRUPT_TYPE` for `TAG` (until P3-T064) and for an
/// unspecified or unknown type.
pub fn interrupt(
    entry: &SessionEntry,
    request: &pb::InterruptRequest,
) -> Result<pb::InterruptResponse, KrishivError> {
    let kind = InterruptType::try_from(request.interrupt_type);
    let interrupted = match kind {
        Ok(InterruptType::All) => {
            // Only the operations it reports (Spark's
            // `SessionHolder.interruptAll`): one registered after the
            // snapshot keeps running.
            let mut ids = Vec::new();
            for op in entry.operations().all() {
                if op.cancel() {
                    entry.session().cancel(op.id());
                    ids.push(op.id().as_str().to_owned());
                }
            }
            ids
        }
        Ok(InterruptType::Tag) => {
            if !matches!(request.interrupt, Some(Interrupt::OperationTag(_))) {
                return Err(InvalidParameterValueInterruptTypeTagRequiresTag {
                    parameter: "operation_tag".to_owned(),
                    function_name: "interrupt".to_owned(),
                }
                .into());
            }
            return Err(unsupported("INTERRUPT_TYPE_TAG"));
        }
        Ok(InterruptType::OperationId) => {
            let Some(Interrupt::OperationId(id)) = &request.interrupt else {
                return Err(InvalidParameterValueInterruptTypeOperationIdRequiresId {
                    parameter: "operation_id".to_owned(),
                    function_name: "interrupt".to_owned(),
                }
                .into());
            };
            match entry.operations().get(id) {
                Some(op) if op.cancel() => {
                    entry.session().cancel(op.id());
                    entry.operations().remove(&op);
                    vec![id.clone()]
                }
                _ => Vec::new(),
            }
        }
        Ok(InterruptType::Unspecified) => return Err(unsupported("INTERRUPT_TYPE_UNSPECIFIED")),
        Err(_) => return Err(unsupported("UNRECOGNIZED")),
    };
    Ok(pb::InterruptResponse {
        session_id: request.session_id.clone(),
        server_side_session_id: entry.server_side_id().to_owned(),
        interrupted_ids: interrupted,
    })
}

/// Spark's `other.toString` of the enum value.
fn unsupported(kind: &str) -> KrishivError {
    UnsupportedFeatureInterruptType {
        interrupt_type: kind.to_owned(),
    }
    .into()
}
