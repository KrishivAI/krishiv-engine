// SPDX-License-Identifier: Apache-2.0
//! The denial counter of a server (`krishiv_auth_denied_total{surface, privilege}`, ADR 0046
//! §8): one count per access check refused with `KRISHIV_INSUFFICIENT_PERMISSIONS`, by the
//! listener the caller came in on and the privilege asked for. A server shares one
//! [`DeniedCounts`] between its listeners' guards and its sessions
//! ([`crate::Session::count_denials`]).

use std::sync::atomic::{AtomicU64, Ordering};

use krishiv_plan::analyzer::Privilege;

use super::Surface;

/// Every privilege, in counter order (a closed set: the label stays bounded).
const PRIVILEGES: [Privilege; 10] = [
    Privilege::Select,
    Privilege::Modify,
    Privilege::Create,
    Privilege::Drop,
    Privilege::Monitor,
    Privilege::Usage,
    Privilege::Browse,
    Privilege::Operate,
    Privilege::Manage,
    Privilege::Impersonate,
];

/// One denial counter.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeniedCount {
    /// The listener.
    pub surface: Surface,
    /// The privilege asked for.
    pub privilege: Privilege,
    /// Denials so far.
    pub count: u64,
}

/// The denial counters of one server.
#[derive(Debug)]
pub struct DeniedCounts {
    /// Indexed by `surface * PRIVILEGES.len() + privilege`.
    counts: [AtomicU64; Surface::ALL.len() * PRIVILEGES.len()],
}

impl Default for DeniedCounts {
    fn default() -> Self {
        Self {
            counts: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl DeniedCounts {
    /// Counters at zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Counts one denial of `privilege` on `surface`.
    pub fn record(&self, surface: Surface, privilege: Privilege) {
        if let Some(i) = index(surface, privilege) {
            self.counts[i].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Counts one denial on `surface` of the privilege named as `Privilege`'s `Display` writes it
    /// (the `privilege` parameter of `KRISHIV_INSUFFICIENT_PERMISSIONS`); an unknown name counts
    /// nothing.
    pub fn record_named(&self, surface: Surface, privilege: &str) {
        if let Some(p) = PRIVILEGES.iter().find(|p| p.to_string() == privilege) {
            self.record(surface, *p);
        }
    }

    /// Every counter, by surface then privilege.
    #[must_use]
    pub fn counts(&self) -> Vec<DeniedCount> {
        let mut out = Vec::with_capacity(self.counts.len());
        for surface in Surface::ALL {
            for privilege in PRIVILEGES {
                if let Some(i) = index(surface, privilege) {
                    out.push(DeniedCount {
                        surface,
                        privilege,
                        count: self.counts[i].load(Ordering::Relaxed),
                    });
                }
            }
        }
        out
    }
}

fn index(surface: Surface, privilege: Privilege) -> Option<usize> {
    let s = Surface::ALL.iter().position(|x| *x == surface)?;
    let p = PRIVILEGES.iter().position(|x| *x == privilege)?;
    Some(s * PRIVILEGES.len() + p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_surface_and_privilege_counts_on_its_own() {
        let d = DeniedCounts::new();
        d.record(Surface::Connect, Privilege::Modify);
        d.record(Surface::Connect, Privilege::Modify);
        d.record_named(Surface::Ui, "OPERATE");
        d.record_named(Surface::Ui, "NOT_A_PRIVILEGE");
        let counts = d.counts();
        assert_eq!(counts.len(), Surface::ALL.len() * PRIVILEGES.len());
        let get = |s, p| {
            counts
                .iter()
                .find(|c| c.surface == s && c.privilege == p)
                .map(|c| c.count)
        };
        assert_eq!(get(Surface::Connect, Privilege::Modify), Some(2));
        assert_eq!(get(Surface::Ui, Privilege::Operate), Some(1));
        assert_eq!(get(Surface::Admin, Privilege::Modify), Some(0));
        assert_eq!(counts.iter().map(|c| c.count).sum::<u64>(), 3);
    }

    #[test]
    fn every_privilege_has_a_counter() {
        // A new `Privilege` variant must be added to `PRIVILEGES` (and the metric's label).
        for p in PRIVILEGES {
            assert!(index(Surface::Connect, p).is_some());
        }
        let names: Vec<String> = PRIVILEGES.iter().map(ToString::to_string).collect();
        assert_eq!(
            names,
            [
                "SELECT",
                "MODIFY",
                "CREATE",
                "DROP",
                "MONITOR",
                "USAGE",
                "BROWSE",
                "OPERATE",
                "MANAGE",
                "IMPERSONATE"
            ]
        );
    }
}
