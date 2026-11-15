// SPDX-License-Identifier: Apache-2.0
//! [`SessionGuard`]: the admin listener's [`Guard`] when authentication is configured (P3-T002
//! design §3.1; ADR 0046 §3, §3c, §8).
//!
//! The presented token goes to the server's one authenticator (`ServerAuth`, the same one the
//! Spark Connect listener uses); the authenticated role is then checked against the ADR 0046
//! matrix through [`RolePolicy`], so there is no second role table here. An authentication
//! failure is logged and counted by the authenticator; a denial writes one `krishiv::audit`
//! record (principal, role, surface, securable, privilege; never a reason).

use std::sync::Arc;

use futures::future::BoxFuture;
use krishiv_session::auth::{Authenticator, Credentials, Role, RolePolicy, Surface};
use krishiv_session::{PolicyVersion, Privilege, Securable};

use crate::guard::{Access, Guard, GuardRequest, GuardSurface, Refusal};

/// Authenticates through an [`Authenticator`] and checks the role.
#[derive(Clone)]
pub struct SessionGuard {
    auth: Arc<dyn Authenticator>,
}

impl std::fmt::Debug for SessionGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionGuard").finish_non_exhaustive()
    }
}

impl SessionGuard {
    /// A guard over `auth` (normally the server's `ServerAuth`).
    #[must_use]
    pub fn new(auth: Arc<dyn Authenticator>) -> Self {
        Self { auth }
    }
}

/// The securable and privilege an [`Access`] asks for; `None` for any authenticated caller.
fn requirement(access: Access) -> Option<(Securable, Privilege)> {
    let system = |name: &str| Securable::System(name.to_owned());
    match access {
        Access::Authenticated => None,
        Access::Metrics => Some((system("metrics"), Privilege::Monitor)),
        Access::Monitor => Some((system("system"), Privilege::Monitor)),
        Access::Operate => Some((system("queries"), Privilege::Operate)),
    }
}

fn surface(s: GuardSurface) -> Surface {
    match s {
        GuardSurface::Metrics => Surface::Metrics,
        GuardSurface::Ui => Surface::Ui,
        GuardSurface::Admin => Surface::Admin,
    }
}

impl Guard for SessionGuard {
    fn check<'a>(&'a self, request: GuardRequest<'a>) -> BoxFuture<'a, Result<(), Refusal>> {
        Box::pin(async move {
            let surface = surface(request.surface);
            let Some(peer) = request.peer else {
                tracing::error!(
                    target: "krishiv::auth",
                    surface = surface.as_str(),
                    "no peer address recorded for an authenticated route; refusing"
                );
                return Err(Refusal::Misconfigured);
            };
            let mut credentials = Credentials::new(peer, surface);
            credentials.malformed = request.malformed;
            if let Some(token) = request.token {
                credentials = credentials.with_bearer(token);
            }
            let authenticated = match self.auth.authenticate(&credentials).await {
                Ok(a) => a,
                Err(f) if f.is_retryable() => return Err(Refusal::Unavailable),
                Err(_) => return Err(Refusal::Unauthenticated),
            };
            let Some((securable, privilege)) = requirement(request.access) else {
                return Ok(());
            };
            // Client code is not reachable from these routes, so the UDF role does not matter;
            // the strictest value keeps it from ever widening a decision.
            let policy = RolePolicy::new(authenticated.role, Role::Admin, PolicyVersion(0));
            if policy.allows(&securable, privilege) {
                return Ok(());
            }
            tracing::warn!(
                target: "krishiv::audit",
                principal = %authenticated.principal.name,
                role = authenticated.role.map_or("none", Role::as_str),
                surface = surface.as_str(),
                securable = %securable,
                privilege = %privilege,
                error_class = "KRISHIV_INSUFFICIENT_PERMISSIONS",
                "access denied"
            );
            Err(Refusal::Denied)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use krishiv_session::auth::{AuthFailure, Authenticated, Method};
    use krishiv_session::{Principal, PrincipalKind};

    use super::*;

    /// Accepts `viewer`, `operator` and `admin` as tokens naming their role; `none` maps to no
    /// role; `slow` is unavailable; anything else is invalid.
    struct Fake;

    impl Authenticator for Fake {
        fn authenticate<'a>(
            &'a self,
            c: &'a Credentials<'a>,
        ) -> BoxFuture<'a, Result<Authenticated, AuthFailure>> {
            let result = match c.bearer {
                None => Err(AuthFailure::Missing),
                Some("slow") => Err(AuthFailure::Unavailable),
                Some("none") => Ok(Authenticated::new(
                    Principal::new("n", PrincipalKind::User),
                    None,
                    Method::Token,
                )),
                Some(t) => Role::parse(t)
                    .map(|r| {
                        Authenticated::new(
                            Principal::new(t, PrincipalKind::User),
                            Some(r),
                            Method::Token,
                        )
                    })
                    .ok_or(AuthFailure::Invalid),
            };
            Box::pin(std::future::ready(result))
        }
    }

    async fn check(token: Option<&str>, access: Access) -> Result<(), Refusal> {
        let peer: SocketAddr = ([127, 0, 0, 1], 9).into();
        let guard = SessionGuard::new(Arc::new(Fake));
        let request = GuardRequest::new(GuardSurface::Ui, access)
            .with_token(token)
            .with_peer(Some(peer));
        guard.check(request).await
    }

    #[tokio::test]
    async fn roles_meet_the_matrix() {
        use Access::{Authenticated as Any, Metrics, Monitor, Operate};
        for (token, access, expected) in [
            ("viewer", Any, Ok(())),
            ("viewer", Metrics, Ok(())),
            ("viewer", Monitor, Err(Refusal::Denied)),
            ("viewer", Operate, Err(Refusal::Denied)),
            ("operator", Metrics, Ok(())),
            ("operator", Monitor, Ok(())),
            ("operator", Operate, Ok(())),
            ("admin", Operate, Ok(())),
            ("none", Any, Ok(())),
            ("none", Metrics, Err(Refusal::Denied)),
        ] {
            assert_eq!(
                check(Some(token), access).await,
                expected,
                "{token} {access:?}"
            );
        }
    }

    #[tokio::test]
    async fn failures_map_to_refusals() {
        assert_eq!(
            check(None, Access::Metrics).await,
            Err(Refusal::Unauthenticated)
        );
        assert_eq!(
            check(Some("bogus"), Access::Metrics).await,
            Err(Refusal::Unauthenticated)
        );
        assert_eq!(
            check(Some("slow"), Access::Metrics).await,
            Err(Refusal::Unavailable)
        );
    }

    #[tokio::test]
    async fn no_peer_fails_closed() {
        let guard = SessionGuard::new(Arc::new(Fake));
        let request =
            GuardRequest::new(GuardSurface::Metrics, Access::Metrics).with_token(Some("admin"));
        assert_eq!(guard.check(request).await, Err(Refusal::Misconfigured));
    }
}
