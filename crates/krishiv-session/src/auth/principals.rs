// SPDX-License-Identifier: Apache-2.0
//! The rights of named principals (design P3-T061 §3, amendment A2).
//!
//! Credentials stay where T002 put them (`krishiv.auth.tokens.*`, certificates, OIDC); what a
//! principal may do beyond its role is set by its name, whatever method authenticated it:
//!
//! ```text
//! krishiv.auth.principals.<name>.privileges  = impersonate          (closed list; default none)
//! krishiv.auth.principals.<name>.impersonates = alice, team-*, group:analysts
//! krishiv.auth.principals.<name>.expires      = 2027-03-01           (UTC date; refused from then on)
//! ```
//!
//! `impersonates` lists the end users the principal may act for: a name, a glob with `*`, or
//! `group:<g>` for any member of a group. An empty list with `impersonate` granted means no one
//! (explicit, §4.2 rule 2); `impersonates` without `impersonate` is refused, so a list that
//! would do nothing never looks like it works.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use krishiv_common::error::KrishivError;

use super::secret_ref::invalid;

/// A right a principal holds by name, beyond its role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum PrincipalPrivilege {
    /// Act for another principal (trusted impersonation, ADR 0046 §5).
    Impersonate,
}

impl PrincipalPrivilege {
    /// The configuration spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Impersonate => "impersonate",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        [Self::Impersonate]
            .into_iter()
            .find(|p| p.as_str().eq_ignore_ascii_case(s))
    }
}

/// One entry of an `impersonates` list.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImpersonationTarget {
    /// An end-user name; `*` matches any run of characters.
    Name(String),
    /// Any member of the group.
    Group(String),
}

impl ImpersonationTarget {
    /// Whether the end user `name` with `groups` matches.
    #[must_use]
    pub fn matches(&self, name: &str, groups: &[String]) -> bool {
        match self {
            Self::Name(pattern) => glob_matches(pattern, name),
            Self::Group(g) => groups.iter().any(|x| x == g),
        }
    }
}

/// What one named principal may do beyond its role.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PrincipalRights {
    /// The privileges granted by name.
    pub privileges: Vec<PrincipalPrivilege>,
    /// Whom it may impersonate (meaningful with [`PrincipalPrivilege::Impersonate`]).
    pub impersonates: Vec<ImpersonationTarget>,
    /// When its credentials stop working (the start of a UTC day), if set.
    pub expires: Option<SystemTime>,
}

impl PrincipalRights {
    /// Whether it holds `privilege`.
    #[must_use]
    pub fn has(&self, privilege: PrincipalPrivilege) -> bool {
        self.privileges.contains(&privilege)
    }

    /// Whether it may act for the end user `name` with `groups`: it holds `impersonate` and an
    /// entry of `impersonates` matches.
    #[must_use]
    pub fn may_impersonate(&self, name: &str, groups: &[String]) -> bool {
        self.has(PrincipalPrivilege::Impersonate)
            && self.impersonates.iter().any(|t| t.matches(name, groups))
    }

    /// Whether its credentials have expired at `now`.
    #[must_use]
    pub fn expired_at(&self, now: SystemTime) -> bool {
        self.expires.is_some_and(|e| now >= e)
    }
}

/// `krishiv.auth.principals.*`: the rights of each named principal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrincipalTable {
    by_name: BTreeMap<String, PrincipalRights>,
}

impl PrincipalTable {
    /// The rights of `name`, if it has an entry.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&PrincipalRights> {
        self.by_name.get(name)
    }

    /// Whether `name` holds `privilege`.
    #[must_use]
    pub fn has(&self, name: &str, privilege: PrincipalPrivilege) -> bool {
        self.get(name).is_some_and(|r| r.has(privilege))
    }

    /// The named entries, by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &PrincipalRights)> {
        self.by_name.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Whether no principal has an entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Applies `krishiv.auth.principals.<rest>` = `raw`; `rest` is the part after
    /// `principals.`.
    pub(super) fn apply(&mut self, key: &str, rest: &str, raw: &str) -> Result<(), KrishivError> {
        let Some((name, field)) = rest.rsplit_once('.') else {
            return Err(invalid(key, key, "not a known authentication setting"));
        };
        if name.is_empty() || name.contains('.') {
            return Err(invalid(
                key,
                name,
                "a principal name is one non-empty segment",
            ));
        }
        let entry = self.by_name.entry(name.to_owned()).or_default();
        match field {
            "privileges" => entry.privileges = parse_privileges(key, raw)?,
            "impersonates" => entry.impersonates = parse_targets(key, raw)?,
            "expires" => entry.expires = Some(parse_date(key, raw)?),
            _ => return Err(invalid(key, key, "not a known authentication setting")),
        }
        Ok(())
    }

    /// Refuses an `impersonates` list on a principal without `impersonate`.
    pub(super) fn validate(&self) -> Result<(), KrishivError> {
        for (name, rights) in &self.by_name {
            if !rights.impersonates.is_empty() && !rights.has(PrincipalPrivilege::Impersonate) {
                return Err(invalid(
                    &format!("krishiv.auth.principals.{name}.impersonates"),
                    "",
                    "needs `impersonate` in krishiv.auth.principals.<name>.privileges",
                ));
            }
        }
        Ok(())
    }
}

fn items(raw: &str) -> impl Iterator<Item = &str> {
    raw.split(',').map(str::trim).filter(|s| !s.is_empty())
}

fn parse_privileges(key: &str, raw: &str) -> Result<Vec<PrincipalPrivilege>, KrishivError> {
    let mut out = Vec::new();
    for item in items(raw) {
        let p = PrincipalPrivilege::parse(item)
            .ok_or_else(|| invalid(key, item, "expected a list of: impersonate"))?;
        if !out.contains(&p) {
            out.push(p);
        }
    }
    Ok(out)
}

fn parse_targets(key: &str, raw: &str) -> Result<Vec<ImpersonationTarget>, KrishivError> {
    items(raw)
        .map(|item| match item.strip_prefix("group:") {
            Some(g) if g.trim().is_empty() => Err(invalid(key, item, "a group name is missing")),
            Some(g) => Ok(ImpersonationTarget::Group(g.trim().to_owned())),
            None => Ok(ImpersonationTarget::Name(item.to_owned())),
        })
        .collect()
}

/// `YYYY-MM-DD`, as the start of that day in UTC.
fn parse_date(key: &str, raw: &str) -> Result<SystemTime, KrishivError> {
    let bad = || invalid(key, raw, "expected a UTC date YYYY-MM-DD");
    let s = raw.trim();
    let mut parts = s.split('-');
    let (Some(y), Some(m), Some(d), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(bad());
    };
    if y.len() != 4 || m.len() != 2 || d.len() != 2 {
        return Err(bad());
    }
    let digits = |p: &str| -> Option<u32> {
        p.bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| p.parse().ok())
            .flatten()
    };
    let (Some(y), Some(m), Some(d)) = (digits(y), digits(m), digits(d)) else {
        return Err(bad());
    };
    let days = days_from_civil(y, m, d).ok_or_else(bad)?;
    if y < 1970 {
        return Err(bad());
    }
    let secs = u64::try_from(days)
        .map_err(|_| bad())?
        .saturating_mul(86_400);
    Ok(UNIX_EPOCH + Duration::from_secs(secs))
}

/// Days since 1970-01-01 of a proleptic Gregorian date, `None` for a date that does not exist.
fn days_from_civil(y: u32, m: u32, d: u32) -> Option<i64> {
    let leap = (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400);
    let month_days = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return None,
    };
    if d == 0 || d > month_days {
        return None;
    }
    // Howard Hinnant's `days_from_civil`.
    let y = i64::from(y) - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

/// `pattern` with `*` as any run of characters (none included), matched against all of `text`.
fn glob_matches(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    // Iterative wildcard match with one backtrack point: linear in practice, no recursion.
    let (mut pi, mut ti) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        if pi < p.len() && p[pi] == '*' {
            star = Some((pi, ti));
            pi += 1;
        } else if pi < p.len() && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if let Some((sp, st)) = star {
            pi = sp + 1;
            ti = st + 1;
            star = Some((sp, st + 1));
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(settings: &[(&str, &str)]) -> Result<PrincipalTable, KrishivError> {
        let mut t = PrincipalTable::default();
        for (k, v) in settings {
            let rest = k.strip_prefix("krishiv.auth.principals.").unwrap_or(k);
            t.apply(k, rest, v)?;
        }
        t.validate()?;
        Ok(t)
    }

    #[test]
    fn globs_match_whole_names() {
        assert!(glob_matches("alice", "alice"));
        assert!(!glob_matches("alice", "alice2"));
        assert!(glob_matches("team-*", "team-a"));
        assert!(glob_matches("team-*", "team-"));
        assert!(!glob_matches("team-*", "xteam-a"));
        assert!(glob_matches("*", ""));
        assert!(glob_matches("a*b*c", "aXbYbc"));
        assert!(!glob_matches("a*b*c", "aXbYbd"));
        assert!(glob_matches("*@corp", "bob@corp"));
        assert!(!glob_matches("", "a"));
    }

    #[test]
    fn dates_are_the_start_of_the_utc_day() {
        let t = parse_date("k", "1970-01-02").expect("valid");
        assert_eq!(t, UNIX_EPOCH + Duration::from_secs(86_400));
        let t = parse_date("k", "2024-03-01").expect("valid");
        assert_eq!(t, UNIX_EPOCH + Duration::from_secs(1_709_251_200));
        assert!(parse_date("k", "2023-02-29").is_err());
        assert!(parse_date("k", "2024-02-29").is_ok());
        for bad in [
            "2024-1-01",
            "24-01-01",
            "2024-13-01",
            "2024-01-00",
            "2024/01/01",
            "1969-12-31",
            "+202-01-01",
            "2024-01-01-01",
            "",
        ] {
            assert!(parse_date("k", bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn rights_parse_and_answer() {
        let t = table(&[
            ("krishiv.auth.principals.portal.privileges", "Impersonate"),
            (
                "krishiv.auth.principals.portal.impersonates",
                "alice, team-*, group:analysts",
            ),
            ("krishiv.auth.principals.portal.expires", "2099-01-01"),
        ])
        .expect("valid");
        let r = t.get("portal").expect("entry");
        assert!(t.has("portal", PrincipalPrivilege::Impersonate));
        assert!(!t.has("other", PrincipalPrivilege::Impersonate));
        assert!(r.may_impersonate("alice", &[]));
        assert!(r.may_impersonate("team-x", &[]));
        assert!(r.may_impersonate("carol", &["analysts".to_owned()]));
        assert!(!r.may_impersonate("carol", &["devs".to_owned()]));
        assert!(!r.expired_at(SystemTime::now()));
        assert!(r.expired_at(UNIX_EPOCH + Duration::from_secs(4_070_908_800)));
    }

    #[test]
    fn an_empty_list_with_the_privilege_means_no_one() {
        let t = table(&[("krishiv.auth.principals.p.privileges", "impersonate")]).expect("valid");
        assert!(!t.get("p").expect("entry").may_impersonate("alice", &[]));
    }

    #[test]
    fn a_list_without_the_privilege_is_refused() {
        let err = table(&[("krishiv.auth.principals.p.impersonates", "alice")])
            .expect_err("refused")
            .to_string();
        assert!(err.contains("needs `impersonate`"), "{err}");
    }

    #[test]
    fn bad_keys_and_values_are_refused() {
        for (k, v) in [
            ("krishiv.auth.principals.p.privileges", "impersonate, root"),
            ("krishiv.auth.principals.p.role", "admin"),
            ("krishiv.auth.principals.p", "x"),
            ("krishiv.auth.principals..privileges", "impersonate"),
            ("krishiv.auth.principals.a.b.privileges", "impersonate"),
            ("krishiv.auth.principals.p.impersonates", "group:"),
            ("krishiv.auth.principals.p.expires", "tomorrow"),
        ] {
            assert!(table(&[(k, v)]).is_err(), "{k} = {v}");
        }
    }
}
