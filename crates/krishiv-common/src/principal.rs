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
        }
    }

    /// The embedded default: the local OS user.
    #[must_use]
    pub fn local_user() -> Self {
        let name = ["USER", "USERNAME"]
            .iter()
            .find_map(|k| std::env::var(k).ok())
            .unwrap_or_else(|| "unknown".to_owned());
        Self {
            name,
            groups: Vec::new(),
            tenant: None,
            kind: PrincipalKind::User,
        }
    }

    /// The identity that enters the context fingerprint: kind, tenant,
    /// name and the sorted groups (a policy may decide by group).
    #[must_use]
    pub fn id(&self) -> String {
        let kind = match self.kind {
            PrincipalKind::User => "user",
            PrincipalKind::Service => "service",
        };
        let mut groups = self.groups.clone();
        groups.sort();
        groups.dedup();
        format!(
            "{kind}:{}/{}{}",
            quoted(std::slice::from_ref(
                &self.tenant.clone().unwrap_or_default()
            )),
            quoted(std::slice::from_ref(&self.name)),
            if groups.is_empty() {
                String::new()
            } else {
                format!("[{}]", quoted(&groups))
            }
        )
    }
}
