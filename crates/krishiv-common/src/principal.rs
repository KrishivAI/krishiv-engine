// SPDX-License-Identifier: Apache-2.0
//! Who runs a query: [`Principal`], shared by the planner's access checks, the I/O layer's secret
//! resolver and block-cache scopes (E12, E15), so it lives below all of them.

/// `parts` quoted as Spark's `toSQLId` does: `` `a`.`b` ``.
fn quoted(parts: &[String]) -> String {
    let quote = |p: &String| format!("`{}`", p.replace('`', "``"));
    parts.iter().map(quote).collect::<Vec<_>>().join(".")
}

/// Who runs a query. Build it with [`Principal::new`]; more attributes
/// arrive with authentication (P2-T136).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Principal {
    /// The user or service name.
    pub name: String,
    /// Group memberships.
    pub groups: Vec<String>,
    /// The tenant, in multi-tenant servers.
    pub tenant: Option<String>,
    /// User or service principal.
    pub kind: PrincipalKind,
    /// The service principal that acts for this one (trusted impersonation, P3-T061 §4.2 rule
    /// 4), `None` when the principal authenticated itself. Authorization checks the principal
    /// itself; the audit and `system.queries` / `system.sessions` show both. It never enters
    /// [`Self::id`]: what a query may read is the end user's, whoever asked for it.
    pub delegated_by: Option<String>,
    /// The reader this principal acts for while it reads the query of a `SQL SECURITY DEFINER`
    /// view as the view's definer (P3-T062 §2.4), or `scheduler` for a materialized view refresh
    /// run as its owner. `None` for a principal that acts for itself. Such a principal holds
    /// only what is granted to its own name: never the session's admin role, never the groups or
    /// the objects of the statement that runs it. It enters [`Self::id`], so a plan analyzed
    /// for it is never reused for the definer itself.
    pub definer_for: Option<String>,
}

/// The kind of a principal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PrincipalKind {
    /// A person.
    User,
    /// A service principal.
    Service,
}

impl Principal {
    /// A principal of `kind` named `name`, without groups or tenant.
    #[must_use]
    pub fn new(name: impl Into<String>, kind: PrincipalKind) -> Self {
        Self {
            name: name.into(),
            groups: Vec::new(),
            tenant: None,
            kind,
            delegated_by: None,
            definer_for: None,
        }
    }

    /// The principal `name` reading a `SQL SECURITY DEFINER` view's query for `reader` (see
    /// [`Self::definer_for`]): a user principal without groups or tenant, so it holds only what
    /// is granted to `name` itself.
    #[must_use]
    pub fn definer(name: impl Into<String>, reader: impl Into<String>) -> Self {
        Self {
            definer_for: Some(reader.into()),
            ..Self::new(name, PrincipalKind::User)
        }
    }

    /// This principal, acting through the service principal `delegator` (trusted
    /// impersonation).
    #[must_use]
    pub fn with_delegator(mut self, delegator: impl Into<String>) -> Self {
        self.delegated_by = Some(delegator.into());
        self
    }

    /// The principal of a context nobody has set one on: it has no name, and
    /// [`Self::is_unset`] says so. Nothing may be resolved or authorized for it
    /// (an I/O context that was never told who it acts for fails closed).
    #[must_use]
    pub fn unset() -> Self {
        Self::new("", PrincipalKind::User)
    }

    /// Whether this is [`Self::unset`] (or any principal without a name).
    #[must_use]
    pub fn is_unset(&self) -> bool {
        self.name.is_empty()
    }

    /// The embedded default: the local OS user.
    #[must_use]
    pub fn local_user() -> Self {
        let name = ["USER", "USERNAME"]
            .iter()
            .find_map(|k| std::env::var(k).ok())
            .unwrap_or_else(|| "unknown".to_owned());
        Self::new(name, PrincipalKind::User)
    }

    /// The identity that enters the context fingerprint: kind, tenant,
    /// name, the sorted groups (a policy may decide by group) and the reader a definer acts for.
    #[must_use]
    pub fn id(&self) -> String {
        let kind = match self.kind {
            PrincipalKind::User => "user",
            PrincipalKind::Service => "service",
        };
        let mut groups = self.groups.clone();
        groups.sort();
        groups.dedup();
        // A missing tenant is `-`, never a quoted empty name: `Some("")` is a tenant of its own.
        let tenant = self
            .tenant
            .as_ref()
            .map_or_else(|| "-".to_owned(), |t| quoted(std::slice::from_ref(t)));
        format!(
            "{kind}:{tenant}/{}{}{}",
            quoted(std::slice::from_ref(&self.name)),
            if groups.is_empty() {
                String::new()
            } else {
                format!("[{}]", quoted(&groups))
            },
            self.definer_for
                .as_ref()
                .map_or_else(String::new, |by| format!(
                    "<{}",
                    quoted(std::slice::from_ref(by))
                ))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_tenant_and_an_empty_tenant_have_different_ids() {
        let none = Principal::new("a", PrincipalKind::User);
        let mut empty = none.clone();
        empty.tenant = Some(String::new());
        assert_eq!(none.id(), "user:-/`a`");
        assert_eq!(empty.id(), "user:``/`a`");
        assert_ne!(none.id(), empty.id());
    }

    #[test]
    fn ids_do_not_collide_across_separators() {
        let mut a = Principal::new("x/`y`", PrincipalKind::User);
        a.tenant = Some("t".to_owned());
        let mut b = Principal::new("y", PrincipalKind::User);
        b.tenant = Some("t/`x".to_owned());
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn a_definer_principal_has_its_own_id() {
        let owner = Principal::new("ana", PrincipalKind::User);
        let derived = Principal::definer("ana", "bob");
        assert_eq!(derived.definer_for.as_deref(), Some("bob"));
        assert_eq!(derived.delegated_by, None);
        assert_eq!(owner.id(), "user:-/`ana`");
        assert_eq!(derived.id(), "user:-/`ana`<`bob`");
    }

    #[test]
    fn the_delegator_is_kept_but_not_in_the_id() {
        let alice = Principal::new("alice", PrincipalKind::User);
        let via = alice.clone().with_delegator("portal");
        assert_eq!(via.delegated_by.as_deref(), Some("portal"));
        assert_eq!(alice.delegated_by, None);
        assert_eq!(via.id(), alice.id());
        assert_ne!(via, alice);
    }

    #[test]
    fn the_unset_principal_is_recognised() {
        assert!(Principal::unset().is_unset());
        assert!(!Principal::local_user().is_unset());
        assert!(!Principal::new("a", PrincipalKind::Service).is_unset());
    }
}
