// SPDX-License-Identifier: Apache-2.0
//! The analyzer's principals, securables and privileges as `krishiv.access.v1` messages. The
//! analyzer's enums are non-exhaustive: a kind this build has no wire form for is sent as
//! unspecified (no securable kind, `PRIVILEGE_UNSPECIFIED`), which a service must deny, and a
//! test lists every kind this build knows.

use krishiv_access_proto as pb;
use krishiv_plan::analyzer::TableIdent;
use krishiv_session::{Principal, PrincipalKind, Privilege, Securable};

pub(super) fn principal(p: &Principal) -> pb::Principal {
    pb::Principal {
        name: p.name.clone(),
        groups: p.groups.clone(),
        tenant: p.tenant.clone().unwrap_or_default(),
        delegated_by: p.delegated_by.clone().unwrap_or_default(),
        kind: match p.kind {
            PrincipalKind::User => pb::PrincipalKind::User,
            PrincipalKind::Service => pb::PrincipalKind::Service,
            // A kind this build does not name reads as unspecified, which a service must deny.
            _ => pb::PrincipalKind::Unspecified,
        }
        .into(),
        definer_for: p.definer_for.clone().unwrap_or_default(),
    }
}

fn object(t: &TableIdent) -> pb::ObjectName {
    pb::ObjectName {
        catalog: t.catalog.clone(),
        namespace: t.namespace.clone(),
        name: t.name.clone(),
    }
}

fn parts(parts: &[String]) -> pb::NameParts {
    pb::NameParts {
        parts: parts.to_vec(),
    }
}

pub(super) fn securable(s: &Securable) -> pb::Securable {
    use pb::securable::Kind;
    let kind = match s {
        Securable::Table(t) => Kind::Table(object(t)),
        Securable::View(t) => Kind::View(object(t)),
        Securable::Column { table, column } => Kind::Column(pb::Column {
            table: Some(object(table)),
            column: column.clone(),
        }),
        Securable::Function(name) => Kind::Function(parts(name)),
        Securable::Path(path) => Kind::Path(credential_free(&path.to_string())),
        Securable::Namespace(name) => Kind::Namespace(parts(name)),
        Securable::System(name) => Kind::System(name.clone()),
        Securable::Secret { scope } => Kind::SecretScope(scope.clone()),
        Securable::Connection(name) => Kind::Connection(name.clone()),
        _ => return pb::Securable { kind: None },
    };
    pb::Securable { kind: Some(kind) }
}

/// Whether this build has a wire form for the principal's kind: one it has none for is denied
/// without asking (review n3).
pub(super) fn known_principal(p: &Principal) -> bool {
    matches!(p.kind, PrincipalKind::User | PrincipalKind::Service)
}

/// Whether this build has a wire form for both: a pair it has none for is denied without asking.
pub(super) fn known(s: &Securable, p: Privilege) -> bool {
    securable(s).kind.is_some() && privilege(p) != pb::Privilege::Unspecified
}

pub(super) fn privilege(p: Privilege) -> pb::Privilege {
    match p {
        Privilege::Select => pb::Privilege::Select,
        Privilege::Modify => pb::Privilege::Modify,
        Privilege::Create => pb::Privilege::Create,
        Privilege::Drop => pb::Privilege::Drop,
        Privilege::Monitor => pb::Privilege::Monitor,
        Privilege::Usage => pb::Privilege::Usage,
        Privilege::Browse => pb::Privilege::Browse,
        Privilege::Operate => pb::Privilege::Operate,
        Privilege::Manage => pb::Privilege::Manage,
        Privilege::Impersonate => pb::Privilege::Impersonate,
        _ => pb::Privilege::Unspecified,
    }
}

/// A location without its credentials. The text is the path's display form, in which a typed
/// secret is already redacted (ADR 0038). User info is removed from any URL; the query and
/// fragment only from an `http(s)` URL, where they are a request's (a signed URL's signature) and
/// not part of the object's name. A local path, a `file:` path and an object-store key are sent
/// whole: `?` and `#` are characters of their names there, and the service must decide the
/// object that is read (review M4).
fn is_scheme(text: &str) -> bool {
    let mut chars = text.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

pub(super) fn credential_free(text: &str) -> String {
    // A URL only when what comes before `://` is a scheme (RFC 3986: a letter, then letters,
    // digits, `+`, `-`, `.`); `/data/x://u@h/f` is a local path and goes as written (review D5).
    let Some((scheme, rest)) = text
        .split_once("://")
        .filter(|(scheme, _)| is_scheme(scheme))
    else {
        return text.to_owned();
    };
    let rest = if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") {
        rest.split(['?', '#']).next().unwrap_or_default()
    } else {
        rest
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => rest.split_at(i),
        None => (rest, ""),
    };
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    format!("{scheme}://{host}{path}")
}
