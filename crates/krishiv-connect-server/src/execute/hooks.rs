// SPDX-License-Identifier: Apache-2.0
//! Test-only fault points of `ExecutePlan`, keyed by operation id so tests
//! running in one process do not see each other's faults: a panic inside
//! the encoder, a plan preparation that never ends on its own (slow
//! analysis), and a finish landing just after an abandon recorded the
//! operation. Compiled only for this crate's unit tests; production builds
//! have no hook.

use std::sync::{Mutex, PoisonError};

/// A fault injected into one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fault {
    /// The encoder panics after the schema response.
    PanicInEncoder,
    /// The plan's preparation waits until it is dropped.
    HoldPreparation,
    /// The query finishes right after maintenance recorded the abandoned
    /// operation's final state (a finish racing the abandon).
    FinishAfterAbandonRecord,
}

/// (operation id, fault, whether the fault point was reached).
static FAULTS: Mutex<Vec<(String, Fault, bool)>> = Mutex::new(Vec::new());

fn fault_of(id: &str) -> Option<Fault> {
    let mut faults = FAULTS.lock().unwrap_or_else(PoisonError::into_inner);
    faults
        .iter_mut()
        .find(|(i, ..)| i == id)
        .map(|(_, f, reached)| {
            *reached = true;
            *f
        })
}

/// Injects `fault` into operation `id`.
pub(crate) fn inject(id: &str, fault: Fault) {
    FAULTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push((id.to_owned(), fault, false));
}

/// Whether operation `id` reached its fault point.
pub(crate) fn reached(id: &str) -> bool {
    FAULTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .any(|(i, _, r)| i == id && *r)
}

/// The fault point of plan preparation.
pub(crate) async fn preparing(id: &str) {
    if fault_of(id) == Some(Fault::HoldPreparation) {
        std::future::pending::<()>().await;
    }
}

/// The fault point inside the encoder.
pub(crate) fn encoding(id: &str) {
    if fault_of(id) == Some(Fault::PanicInEncoder) {
        panic!("injected encoder panic of operation {id}");
    }
}

/// The fault point after maintenance removed (and recorded) an abandoned
/// operation, before the lock is released.
pub(crate) fn abandon_recorded(op: &crate::operation::Operation) {
    if fault_of(op.id().as_str()) == Some(Fault::FinishAfterAbandonRecord) {
        op.finish();
    }
}
