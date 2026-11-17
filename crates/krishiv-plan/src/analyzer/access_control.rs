// SPDX-License-Identifier: Apache-2.0
//! The access-control commands (P3-T062, design note §2.2): `GRANT`, `REVOKE`, `SHOW GRANTS` and
//! the role statements, as the front ends hand them to the session. A Krishiv extension: open
//! Spark has no grammar for them (its parser refuses `GRANT` as an unsupported Hive command), so
//! there is no Spark golden; the syntax follows the Unity Catalog form.
//!
//! Names are as written (unresolved): the session locates them the way analysis locates a read
//! ([`super::table_ident`], [`super::namespace_parts`]) before it builds a [`super::Securable`].

use std::fmt;

use krishiv_common::secret::OptionValue;

use super::catalog::Privilege;

/// Who a grant is for: `user:<name>`, `group:<name>` or `role:<name>`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Grantee {
    /// A user or service principal, by its authenticated name.
    User(String),
    /// A group the authenticator asserts for a principal.
    Group(String),
    /// A role created with `CREATE ROLE`.
    Role(String),
}

impl Grantee {
    /// `user`, `group` or `role`.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::User(_) => "user",
            Self::Group(_) => "group",
            Self::Role(_) => "role",
        }
    }

    /// The name, without the kind.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::User(n) | Self::Group(n) | Self::Role(n) => n,
        }
    }
}

impl fmt::Display for Grantee {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind(), self.name())
    }
}

/// What a grant is on, as written (`<securable>` of note §2.2).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GrantObject {
    /// `CATALOG c`.
    Catalog(String),
    /// `SCHEMA | NAMESPACE | DATABASE n`: the name parts as written.
    Namespace(Vec<String>),
    /// `[TABLE] t`.
    Table(Vec<String>),
    /// `VIEW v`.
    View(Vec<String>),
    /// `FUNCTION f`.
    Function(Vec<String>),
    /// `VOLUME | PATH 'url'`: credentials in the URL stay typed (never printed).
    Path(OptionValue),
    /// `CONNECTION c`.
    Connection(String),
    /// `SECRET SCOPE s`.
    SecretScope(String),
}

impl GrantObject {
    /// The object kind as `SHOW GRANTS` prints it.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Catalog(_) => "CATALOG",
            Self::Namespace(_) => "NAMESPACE",
            Self::Table(_) => "TABLE",
            Self::View(_) => "VIEW",
            Self::Function(_) => "FUNCTION",
            Self::Path(_) => "PATH",
            Self::Connection(_) => "CONNECTION",
            Self::SecretScope(_) => "SECRET SCOPE",
        }
    }

    /// Whether column privileges (`SELECT (a, b)`) may name it.
    #[must_use]
    pub const fn has_columns(&self) -> bool {
        matches!(self, Self::Table(_) | Self::View(_))
    }
}

/// One privilege of a `GRANT` / `REVOKE` list, with the columns of `SELECT (a, b)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantPrivilege {
    /// The privilege.
    pub privilege: Privilege,
    /// The columns it is limited to; empty for the whole object.
    pub columns: Vec<String>,
}

/// The privileges of a `GRANT` / `REVOKE`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GrantPrivileges {
    /// `ALL [PRIVILEGES]`: every privilege [`GRANTABLE`] lists.
    All,
    /// A list, each once.
    List(Vec<GrantPrivilege>),
}

/// The privileges `GRANT` accepts by name (note §2.1), in the order `ALL PRIVILEGES` expands to.
/// The `System(..)` privileges (`OPERATE`, `MANAGE`, `IMPERSONATE`) are the role matrix's and
/// never granted (ADR 0046 §7).
pub const GRANTABLE: [Privilege; 7] = [
    Privilege::Select,
    Privilege::Modify,
    Privilege::Create,
    Privilege::Drop,
    Privilege::Monitor,
    Privilege::Usage,
    Privilege::Browse,
];

/// The privilege named `word` (any case) in a `GRANT` list.
#[must_use]
pub fn grantable_privilege(word: &str) -> Option<Privilege> {
    GRANTABLE
        .into_iter()
        .find(|p| p.to_string().eq_ignore_ascii_case(word))
}

/// An access-control command.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AccessCommand {
    /// `GRANT privs ON object TO grantees [WITH GRANT OPTION]`.
    Grant {
        /// The privileges.
        privileges: GrantPrivileges,
        /// The object.
        object: GrantObject,
        /// The grantees, at least one.
        grantees: Vec<Grantee>,
        /// `WITH GRANT OPTION`.
        with_grant_option: bool,
    },
    /// `REVOKE [GRANT OPTION FOR] privs ON object FROM grantees`.
    Revoke {
        /// `GRANT OPTION FOR`: only the right to grant is taken back.
        grant_option_only: bool,
        /// The privileges.
        privileges: GrantPrivileges,
        /// The object.
        object: GrantObject,
        /// The grantees, at least one.
        grantees: Vec<Grantee>,
    },
    /// `SHOW GRANTS [ON object] [TO grantee]`.
    ShowGrants {
        /// Only grants on this object.
        object: Option<GrantObject>,
        /// Only this grantee's grants; none: the caller's effective grants.
        grantee: Option<Grantee>,
    },
    /// `CREATE ROLE r`.
    CreateRole {
        /// The role.
        name: String,
    },
    /// `DROP ROLE r`.
    DropRole {
        /// The role.
        name: String,
    },
    /// `GRANT ROLE r TO grantees`.
    GrantRole {
        /// The role.
        role: String,
        /// The members added.
        grantees: Vec<Grantee>,
    },
    /// `REVOKE ROLE r FROM grantees`.
    RevokeRole {
        /// The role.
        role: String,
        /// The members removed.
        grantees: Vec<Grantee>,
    },
}

impl AccessCommand {
    /// The statement's name in errors and audit records (`GRANT`, `SHOW GRANTS`, ...).
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Grant { .. } => "GRANT",
            Self::Revoke { .. } => "REVOKE",
            Self::ShowGrants { .. } => "SHOW GRANTS",
            Self::CreateRole { .. } => "CREATE ROLE",
            Self::DropRole { .. } => "DROP ROLE",
            Self::GrantRole { .. } => "GRANT ROLE",
            Self::RevokeRole { .. } => "REVOKE ROLE",
        }
    }

    /// Whether the command only reads (`SHOW GRANTS`).
    #[must_use]
    pub const fn is_read(&self) -> bool {
        matches!(self, Self::ShowGrants { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_privileges() {
        assert_eq!(Grantee::Role("r".to_owned()).to_string(), "role:r");
        assert_eq!(grantable_privilege("select"), Some(Privilege::Select));
        assert_eq!(grantable_privilege("Browse"), Some(Privilege::Browse));
        // The role matrix's privileges are never granted.
        for word in ["MANAGE", "OPERATE", "IMPERSONATE", "ALL", ""] {
            assert_eq!(grantable_privilege(word), None, "{word}");
        }
        assert!(GrantObject::Table(vec!["t".to_owned()]).has_columns());
        assert!(!GrantObject::Catalog("c".to_owned()).has_columns());
        let path = GrantObject::Path(OptionValue::from_user("path", "s3://u:pw@b/p"));
        assert!(!format!("{path:?}").contains("pw"), "{path:?}");
    }
}
