// SPDX-License-Identifier: Apache-2.0
//! The grants model (design P3-T062 §2.1): normalised object keys, grant rows, roles and owners, and
//! the decision over them. Pure data: no I/O, no locks.

use std::collections::{BTreeMap, BTreeSet};

use krishiv_plan::analyzer::access_control::Grantee;
use krishiv_plan::analyzer::{Principal, Privilege, Securable};

/// What a key names. Tables and views are one kind: analysis asks about a name as a table before
/// it knows it is a view, so a grant on either covers the name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum KeyKind {
    /// A catalog (one part) or a namespace (the catalog, then the namespace parts).
    Namespace,
    /// A table or view: the catalog, the namespace parts, the name.
    Relation,
    /// A column: the relation's parts, then the column.
    Column,
    /// A catalog function, its name parts as the read names it.
    Function,
    /// A path prefix: one part, the normalised URL without credentials (see [`path_key`]).
    Path,
    /// A named connection.
    Connection,
    /// A secret scope.
    Secret,
}

impl KeyKind {
    /// The kind as `SHOW GRANTS` and the store file write it.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Namespace => "NAMESPACE",
            Self::Relation => "TABLE",
            Self::Column => "COLUMN",
            Self::Function => "FUNCTION",
            Self::Path => "PATH",
            Self::Connection => "CONNECTION",
            Self::Secret => "SECRET SCOPE",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        [
            Self::Namespace,
            Self::Relation,
            Self::Column,
            Self::Function,
            Self::Path,
            Self::Connection,
            Self::Secret,
        ]
        .into_iter()
        .find(|k| k.name() == name)
    }
}

/// A normalised securable: names folded to lower case (Spark's catalogs are case-insensitive;
/// see the report's amendment on case-sensitive analysis), paths normalised.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ObjectKey {
    pub(crate) kind: KeyKind,
    pub(crate) parts: Vec<String>,
}

fn fold(parts: &[String]) -> Vec<String> {
    parts.iter().map(|p| p.to_lowercase()).collect()
}

impl ObjectKey {
    pub(crate) fn new(kind: KeyKind, parts: Vec<String>) -> Self {
        Self { kind, parts }
    }

    /// The key of a data securable; `None` for `System(..)` (the role matrix's), for a path that
    /// does not normalise, and for any securable added later (denied until it has a key).
    pub(crate) fn of(securable: &Securable) -> Option<Self> {
        let relation = |t: &krishiv_plan::analyzer::TableIdent| {
            let mut parts = vec![t.catalog.clone()];
            parts.extend(t.namespace.iter().cloned());
            parts.push(t.name.clone());
            fold(&parts)
        };
        Some(match securable {
            Securable::Table(t) | Securable::View(t) => Self::new(KeyKind::Relation, relation(t)),
            Securable::Column { table, column } => {
                let mut parts = relation(table);
                parts.push(column.to_lowercase());
                Self::new(KeyKind::Column, parts)
            }
            Securable::Function(n) if !n.is_empty() => Self::new(KeyKind::Function, fold(n)),
            Securable::Namespace(n) if !n.is_empty() => Self::new(KeyKind::Namespace, fold(n)),
            Securable::Path(p) => Self::new(KeyKind::Path, vec![path_key(&p.to_string())?]),
            Securable::Connection(c) => Self::new(KeyKind::Connection, vec![c.to_lowercase()]),
            Securable::Secret { scope } => Self::new(KeyKind::Secret, vec![scope.clone()]),
            _ => return None,
        })
    }

    /// The keys whose grants cover this one: itself, then its parents up to the catalog (a column's
    /// relation, a relation's or function's namespaces). Paths are matched by prefix instead
    /// ([`Self::covered_by`]).
    pub(crate) fn lineage(&self) -> Vec<Self> {
        let mut out = vec![self.clone()];
        let namespaces = match self.kind {
            KeyKind::Column => {
                let relation = self.parts[..self.parts.len().saturating_sub(1)].to_vec();
                out.push(Self::new(KeyKind::Relation, relation));
                self.parts.len().saturating_sub(2)
            }
            KeyKind::Relation | KeyKind::Function | KeyKind::Namespace => {
                self.parts.len().saturating_sub(1)
            }
            KeyKind::Path | KeyKind::Connection | KeyKind::Secret => 0,
        };
        for n in (1..=namespaces).rev() {
            out.push(Self::new(KeyKind::Namespace, self.parts[..n].to_vec()));
        }
        out
    }

    /// Whether a grant on `grant` covers this key: one of its [lineage](Self::lineage), or for a
    /// path a prefix of whole segments.
    pub(crate) fn covered_by(&self, grant: &Self) -> bool {
        if self.kind == KeyKind::Path {
            return grant.kind == KeyKind::Path
                && match (self.parts.first(), grant.parts.first()) {
                    (Some(me), Some(prefix)) => segment_prefix(prefix, me),
                    _ => false,
                };
        }
        self.lineage().contains(grant)
    }

    /// The key as `SHOW GRANTS` prints it (parts joined by `.`; a path as is).
    pub(crate) fn display(&self) -> String {
        self.parts.join(".")
    }
}

/// Whether `prefix` is `path` or one of its parents, by whole `/` segments.
fn segment_prefix(prefix: &str, path: &str) -> bool {
    let prefix = prefix.trim_end_matches('/');
    path == prefix
        || path
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// The grant key of a path: scheme and host in lower case, the userinfo, query and fragment
/// dropped (a credential never becomes a key), `.` and `..` resolved and repeated `/` folded; a
/// bare absolute path and `file:/a` / `file:///a` are `file:///a`. `None` for a path that
/// climbs above its root, a relative path, or one with a redacted credential in its host or path.
pub(crate) fn path_key(text: &str) -> Option<String> {
    let (scheme, rest) = match text.split_once(':') {
        Some((s, r))
            if !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c))
                && !s.contains('/') =>
        {
            (s.to_ascii_lowercase(), r)
        }
        _ if text.starts_with('/') => ("file".to_owned(), text),
        _ => return None,
    };
    let rest = rest.split(['?', '#']).next().unwrap_or_default();
    let (authority, path) = if let Some(after) = rest.strip_prefix("//") {
        match after.split_once('/') {
            Some((a, p)) => (a, p),
            None => (after, ""),
        }
    } else if scheme == "file" {
        ("", rest)
    } else {
        return None;
    };
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host)
        .to_ascii_lowercase();
    if scheme != "file" && host.is_empty() {
        return None;
    }
    // A credential the redacted form hides anywhere but the userinfo and query: never a key.
    if host.contains(krishiv_common::secret::REDACTED)
        || path.contains(krishiv_common::secret::REDACTED)
    {
        return None;
    }
    let mut segments: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                segments.pop()?;
            }
            s => segments.push(s),
        }
    }
    Some(format!("{scheme}://{host}/{}", segments.join("/")))
}

/// One grant row (note §2.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GrantRow {
    pub(crate) grantee: Grantee,
    pub(crate) privilege: Privilege,
    pub(crate) object: ObjectKey,
    pub(crate) with_grant_option: bool,
    pub(crate) granted_by: String,
    /// Milliseconds since the Unix epoch.
    pub(crate) granted_at: i64,
}

/// A role: its creator (who may drop it and change its members) and its members.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct RoleEntry {
    pub(crate) owner: String,
    pub(crate) members: BTreeSet<Grantee>,
}

/// Everything the grants provider decides from, at one epoch. Never edited in place: a change
/// builds the next state and the store swaps it in whole.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct GrantState {
    /// Bumped by every change, in the same swap (the policy version).
    pub(crate) epoch: u64,
    pub(crate) grants: Vec<GrantRow>,
    pub(crate) roles: BTreeMap<String, RoleEntry>,
    /// Object owners (their creators, Q4), by user name: an implicit grant of every privilege
    /// with the grant option.
    pub(crate) owners: BTreeMap<ObjectKey, String>,
}

impl GrantState {
    /// The principal's effective grantees: the user, its groups, and every role any of them is a
    /// member of, transitively.
    pub(crate) fn effective(&self, principal: &Principal) -> BTreeSet<Grantee> {
        let mut out: BTreeSet<Grantee> = BTreeSet::new();
        out.insert(Grantee::User(principal.name.clone()));
        out.extend(principal.groups.iter().cloned().map(Grantee::Group));
        // Each pass adds at least one role or stops; roles are finite.
        loop {
            let before = out.len();
            for (name, role) in &self.roles {
                if role.members.iter().any(|m| out.contains(m)) {
                    out.insert(Grantee::Role(name.clone()));
                }
            }
            if out.len() == before {
                return out;
            }
        }
    }

    /// Whether `principal` owns `key` or one of its parents.
    pub(crate) fn owns(&self, principal: &Principal, key: &ObjectKey) -> bool {
        self.owners
            .iter()
            .any(|(k, owner)| *owner == principal.name && key.covered_by(k))
    }

    /// The rows of `grantees` that cover `key`.
    fn covering<'a>(
        &'a self,
        grantees: &'a BTreeSet<Grantee>,
        key: &'a ObjectKey,
    ) -> impl Iterator<Item = &'a GrantRow> + 'a {
        self.grants
            .iter()
            .filter(move |g| grantees.contains(&g.grantee) && key.covered_by(&g.object))
    }

    /// Whether `principal` holds `privilege` on `key` (note §2.1): an owner holds everything; a
    /// grant on the key or a parent counts; `Browse` is held through any grant that covers the
    /// key, or any column grant of a relation. No `DENY` (Q2).
    pub(crate) fn allows(
        &self,
        principal: &Principal,
        key: &ObjectKey,
        privilege: Privilege,
    ) -> bool {
        if self.owns(principal, key) {
            return true;
        }
        let grantees = self.effective(principal);
        if privilege == Privilege::Browse {
            return self.covering(&grantees, key).next().is_some()
                || (key.kind == KeyKind::Relation
                    && self.grants.iter().any(|g| {
                        grantees.contains(&g.grantee)
                            && g.object.kind == KeyKind::Column
                            && g.object.parts[..g.object.parts.len().saturating_sub(1)]
                                == key.parts[..]
                    }));
        }
        self.covering(&grantees, key)
            .any(|g| g.privilege == privilege)
    }

    /// Whether `principal` may grant `privilege` on `key` to others: it owns the key (or a
    /// parent), or holds the privilege there with the grant option.
    pub(crate) fn may_grant(
        &self,
        principal: &Principal,
        key: &ObjectKey,
        privilege: Privilege,
    ) -> bool {
        if self.owns(principal, key) {
            return true;
        }
        let grantees = self.effective(principal);
        self.covering(&grantees, key)
            .any(|g| g.privilege == privilege && g.with_grant_option)
    }

    /// Whether making `member` a member of `role` closes a cycle (the role would contain itself).
    pub(crate) fn would_cycle(&self, role: &str, member: &Grantee) -> bool {
        let Grantee::Role(start) = member else {
            return false;
        };
        // Is `start` reachable from `role` by "is a member of" edges? Then `role` already sits
        // inside `start`, and adding `start` to `role` loops.
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut todo: Vec<&str> = vec![role];
        while let Some(r) = todo.pop() {
            if r == start {
                return true;
            }
            if !seen.insert(r) {
                continue;
            }
            for (name, entry) in &self.roles {
                if entry.members.contains(&Grantee::Role(r.to_owned())) {
                    todo.push(name);
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_plan::analyzer::{PrincipalKind, TableIdent};

    fn table(c: &str, ns: &str, t: &str) -> ObjectKey {
        ObjectKey::of(&Securable::Table(TableIdent {
            catalog: c.to_owned(),
            namespace: vec![ns.to_owned()],
            name: t.to_owned(),
        }))
        .unwrap()
    }

    #[test]
    fn paths_normalise_without_credentials() {
        for (input, key) in [
            ("/data/a/../b//c/", Some("file:///data/b/c")),
            ("file:/data/x", Some("file:///data/x")),
            ("file:///data/x", Some("file:///data/x")),
            ("S3://Bucket/P/q?versionId=1#f", Some("s3://bucket/P/q")),
            ("s3://key:secret@bucket/p", Some("s3://bucket/p")),
            ("/../etc", None),
            ("relative/x", None),
            ("s3:///nohost", None),
        ] {
            assert_eq!(path_key(input).as_deref(), key, "{input}");
        }
        let r = krishiv_common::secret::REDACTED;
        assert_eq!(
            path_key(&format!("s3://u:{r}@b/p?sig={r}")).as_deref(),
            Some("s3://b/p")
        );
        assert_eq!(path_key(&format!("s3://b/{r}/p")), None);
        assert!(segment_prefix("s3://b/p", "s3://b/p/q"));
        assert!(segment_prefix("s3://b/p/", "s3://b/p"));
        assert!(!segment_prefix("s3://b/p", "s3://b/pq"));
    }

    #[test]
    fn lineage_runs_to_the_catalog() {
        let key = table("C", "S", "T");
        assert_eq!(key.parts, ["c", "s", "t"]);
        let names: Vec<(KeyKind, String)> = key
            .lineage()
            .into_iter()
            .map(|k| (k.kind, k.display()))
            .collect();
        assert_eq!(
            names,
            [
                (KeyKind::Relation, "c.s.t".to_owned()),
                (KeyKind::Namespace, "c.s".to_owned()),
                (KeyKind::Namespace, "c".to_owned()),
            ]
        );
    }

    #[test]
    fn roles_and_cycles() {
        let mut state = GrantState::default();
        state.roles.insert(
            "a".to_owned(),
            RoleEntry {
                owner: "admin".to_owned(),
                members: [Grantee::Group("eng".to_owned())].into(),
            },
        );
        state.roles.insert(
            "b".to_owned(),
            RoleEntry {
                owner: "admin".to_owned(),
                members: [Grantee::Role("a".to_owned())].into(),
            },
        );
        let mut p = Principal::new("ana", PrincipalKind::User);
        p.groups.push("eng".to_owned());
        let e = state.effective(&p);
        assert!(
            e.contains(&Grantee::Role("a".to_owned()))
                && e.contains(&Grantee::Role("b".to_owned()))
        );
        assert!(state.would_cycle("a", &Grantee::Role("b".to_owned())));
        assert!(state.would_cycle("a", &Grantee::Role("a".to_owned())));
        assert!(!state.would_cycle("b", &Grantee::Role("c".to_owned())));
        assert!(!state.would_cycle("a", &Grantee::User("b".to_owned())));
    }
}
