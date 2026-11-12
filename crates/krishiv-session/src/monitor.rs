// SPDX-License-Identifier: Apache-2.0
//! The `MONITOR` decision behind the per-principal `system` tables.

use std::sync::{Arc, OnceLock};

use krishiv_plan::analyzer::{AccessDecision, AccessPolicy, Principal, Privilege, Securable};

/// Whether one query's principal holds `MONITOR` on the `system` securable,
/// which shows every principal's rows in `system.queries` and
/// `system.operators` (ADR 0036 update 2026-11-05). Asked once per query,
/// when analysis loads one of those tables; until then, and when the policy
/// says anything but a plain allow, the principal sees its own rows only.
pub(crate) struct Monitor {
    principal: Arc<Principal>,
    policy: Arc<dyn AccessPolicy>,
    decided: OnceLock<bool>,
}

impl std::fmt::Debug for Monitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Monitor")
            .field("decided", &self.decided.get())
            .finish_non_exhaustive()
    }
}

impl Monitor {
    pub(crate) fn new(principal: Arc<Principal>, policy: Arc<dyn AccessPolicy>) -> Self {
        Self {
            principal,
            policy,
            decided: OnceLock::new(),
        }
    }

    /// Asks the policy, once.
    pub(crate) async fn decide(&self) {
        if self.decided.get().is_some() {
            return;
        }
        let securable = Securable::System(crate::system::SYSTEM.to_owned());
        let allowed = matches!(
            self.policy
                .check(&self.principal, &securable, Privilege::Monitor)
                .await,
            Ok(AccessDecision::Allow)
        );
        let _ = self.decided.set(allowed);
    }

    /// Whether every principal's rows show.
    pub(crate) fn sees_all(&self) -> bool {
        self.decided.get().copied().unwrap_or(false)
    }
}
