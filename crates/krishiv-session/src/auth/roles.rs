// SPDX-License-Identifier: Apache-2.0
//! [`RolePolicy`]: the ADR 0046 role matrix as an [`AccessPolicy`] provider (ADR 0046 §1, §3).
//!
//! A session of an authenticated server principal gets `PathPrefixPolicy(RolePolicy(role))`
//! (design §3.6). There is no role check anywhere else: every surface asks the one
//! `authorize` path, so a denial has one error class, one log record and one counter.
//!
//! What each role may do, by privilege (the securable decides only where the ADR says so):
//!
//! | Privilege | viewer | operator | admin |
//! |---|---|---|---|
//! | `Browse` (metadata) | yes | yes | yes |
//! | `Select` on `system.queries`, `system.sessions` (own rows; others' need `Monitor`), `capabilities`, `metrics` | yes | yes | yes |
//! | `Select` on the other `system.*` tables (`memory`, `operators`, `caches`) | no | yes | yes |
//! | `Select` on anything else (tables, views, functions, paths) | no (T062 grants) | yes | yes |
//! | `Usage` on a secret or connection | no | yes | yes |
//! | `Usage` on `System("udf")` (client code) | `udf_min_role` | `udf_min_role` | yes |
//! | `Monitor` on `System("metrics")` | yes | yes | yes |
//! | `Monitor` elsewhere (others' rows in `system.*`) | no | yes | yes |
//! | `Operate` (others' queries and sessions) | no | yes | yes |
//! | `Modify`, `Create`, `Drop` | no | no | yes |
//! | `Manage` (cluster-scope changes) | no | no | yes |
//! | `Impersonate` | never part of a role (§5) | | |
//!
//! A principal with no role is denied everything (§2).

use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_common::principal::Principal;
use krishiv_plan::analyzer::{AccessDecision, AccessPolicy, PolicyVersion, Privilege, Securable};

use super::Role;

/// The fixed `System(..)` names (ADR 0046 §4).
pub(crate) const SYSTEM_UDF: &str = "udf";
/// `/metrics` scrapes.
pub(crate) const SYSTEM_METRICS: &str = "metrics";
/// The reserved `system` catalog.
const SYSTEM_CATALOG: &str = "system";
/// The `system` tables a viewer reads: `queries` and `sessions` show only the principal's own
/// rows without `Monitor`, `capabilities` and `metrics` hold no principal's data (ADR 0046 §3,
/// §3c). The others (`memory`, `operators`, `caches`) name other principals' work: operator.
const VIEWER_SYSTEM_TABLES: [&str; 4] = ["queries", "sessions", "capabilities", "metrics"];

/// The ADR 0046 role matrix for one session's principal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RolePolicy {
    role: Option<Role>,
    udf_min_role: Role,
    version: PolicyVersion,
}

impl RolePolicy {
    /// The policy of a principal with `role` (`None`: no role, everything denied). Client
    /// code needs `udf_min_role` or above (`krishiv.auth.udf.minRole`; `Admin` until the
    /// T014b sandbox merges, ADR 0046 N1). `version` changes whenever the role mapping is
    /// reloaded, so plan-cache keys cannot carry an old decision.
    #[must_use]
    pub fn new(role: Option<Role>, udf_min_role: Role, version: PolicyVersion) -> Self {
        Self {
            role,
            udf_min_role,
            version,
        }
    }

    /// The role, if any.
    #[must_use]
    pub fn role(&self) -> Option<Role> {
        self.role
    }

    /// The least role that holds `privilege` on `securable`; `None` when no role does.
    #[must_use]
    pub fn required(&self, securable: &Securable, privilege: Privilege) -> Option<Role> {
        required(securable, privilege, self.udf_min_role)
    }

    /// Whether this policy's role holds `privilege` on `securable`.
    #[must_use]
    pub fn allows(&self, securable: &Securable, privilege: Privilege) -> bool {
        match (self.role, self.required(securable, privilege)) {
            (Some(have), Some(need)) => have >= need,
            _ => false,
        }
    }
}

/// The matrix (see the module docs).
fn required(securable: &Securable, privilege: Privilege, udf_min_role: Role) -> Option<Role> {
    let system = |name: &str| matches!(securable, Securable::System(n) if n == name);
    match privilege {
        Privilege::Browse => Some(Role::Viewer),
        Privilege::Select => match securable {
            Securable::Table(t)
                if t.catalog.eq_ignore_ascii_case(SYSTEM_CATALOG)
                    && VIEWER_SYSTEM_TABLES
                        .iter()
                        .any(|n| t.name.eq_ignore_ascii_case(n)) =>
            {
                Some(Role::Viewer)
            }
            _ => Some(Role::Operator),
        },
        Privilege::Usage if system(SYSTEM_UDF) => Some(udf_min_role.max(Role::Viewer)),
        Privilege::Monitor if system(SYSTEM_METRICS) => Some(Role::Viewer),
        // Secrets and connections; others' rows in `system.*`; others' running work.
        Privilege::Usage | Privilege::Monitor | Privilege::Operate => Some(Role::Operator),
        Privilege::Modify | Privilege::Create | Privilege::Drop | Privilege::Manage => {
            Some(Role::Admin)
        }
        // `Impersonate` is never implied by a role (§5; granted per service principal by
        // T061), and a privilege added later is denied until the matrix names it.
        _ => None,
    }
}

impl AccessPolicy for RolePolicy {
    fn version(&self) -> PolicyVersion {
        self.version
    }

    fn check<'a>(
        &'a self,
        _principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = if self.allows(securable, privilege) {
            AccessDecision::Allow
        } else {
            AccessDecision::Deny {
                reason: match self.role {
                    Some(role) => format!("role {role} does not hold {privilege}"),
                    None => "the principal has no role".to_owned(),
                },
            }
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}
