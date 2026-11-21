// SPDX-License-Identifier: Apache-2.0
//! [`IntersectionPolicy`]: several access-policy providers as one (ADR 0049 §1, owner Q6).
//!
//! Every member that [decides](AccessPolicy::decides) a securable is asked, in order, and access
//! is allowed only when each of them allows; a securable no member decides is refused. A denial
//! or an error from any member ends the check (fail closed), so adding a provider only ever
//! narrows access. `AllowWith` from a member is passed on, and the analyzer refuses it until row
//! filters and column masks exist (P5-T045).
//!
//! Reads per column: the list [decides columns](AccessPolicy::column_level) when any member does.
//! A member that decides whole tables only is then asked as it would be alone: a column read is
//! asked as `Select` on the column's table or view (the kind the list last saw that table asked
//! as, else a table), once per table and privilege within one analysis stage.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::{
    AccessDecision, AccessPolicy, PolicyVersion, Principal, Privilege, Securable, TableIdent,
};

/// The most table kinds and refusals the list remembers; past it, the memory is cleared (a
/// forgotten kind is asked as a table, a forgotten refusal is told to every deciding member).
const REMEMBERED: usize = 4096;

/// The most refusals waiting for [`AccessPolicy::denied`]; the oldest goes first.
const REFUSALS: usize = 64;

/// Several providers, all of which must allow (ADR 0049 §1).
pub struct IntersectionPolicy {
    members: Vec<Arc<dyn AccessPolicy>>,
    column_level: bool,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Whether a table name was last asked about as a view.
    views: HashMap<TableIdent, bool>,
    /// The member that refused a (securable, privilege), until [`AccessPolicy::denied`] is told
    /// (the newest last; `Securable` is not `Hash`, and the analyzer reports a refusal at once).
    refused_by: Vec<(Securable, Privilege, usize)>,
    /// The [`AccessPolicy::refusal_label`] of the member behind the latest refusal.
    last_label: Option<&'static str>,
}

impl std::fmt::Debug for IntersectionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntersectionPolicy")
            .field("members", &self.members.len())
            .field("column_level", &self.column_level)
            .finish_non_exhaustive()
    }
}

impl IntersectionPolicy {
    /// The intersection of `members`, asked in this order. An empty list refuses everything.
    #[must_use]
    pub fn new(members: Vec<Arc<dyn AccessPolicy>>) -> Self {
        let column_level = members.iter().any(|m| m.column_level());
        Self {
            members,
            column_level,
            state: Mutex::new(State::default()),
        }
    }

    /// The members, in order.
    #[must_use]
    pub fn members(&self) -> &[Arc<dyn AccessPolicy>] {
        &self.members
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        // A poisoned lock only means a panic elsewhere held it; the maps stay usable.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The securable `member` is asked about for `securable`: unchanged, except a column
    /// read for a member that decides whole tables, which is asked about the column's relation.
    fn question(&self, member: &dyn AccessPolicy, securable: &Securable) -> Securable {
        if let Securable::Column { table, .. } = securable
            && !member.column_level()
        {
            let view = self.state().views.get(table).copied().unwrap_or(false);
            return if view {
                Securable::View(table.clone())
            } else {
                Securable::Table(table.clone())
            };
        }
        securable.clone()
    }

    fn note_kind(&self, securable: &Securable) {
        let (table, view) = match securable {
            Securable::Table(t) => (t, false),
            Securable::View(t) => (t, true),
            _ => return,
        };
        let mut state = self.state();
        if state.views.len() >= REMEMBERED && !state.views.contains_key(table) {
            state.views.clear();
        }
        state.views.insert(table.clone(), view);
    }

    fn note_refusal(&self, securable: &Securable, privilege: Privilege, member: usize) {
        let label = self.members.get(member).and_then(|m| m.refusal_label());
        let mut state = self.state();
        state.last_label = label;
        if state.refused_by.len() >= REFUSALS {
            state.refused_by.remove(0);
        }
        state
            .refused_by
            .push((securable.clone(), privilege, member));
    }

    fn take_refusal(&self, securable: &Securable, privilege: Privilege) -> Option<usize> {
        let mut state = self.state();
        let i = state
            .refused_by
            .iter()
            .rposition(|(s, p, _)| s == securable && *p == privilege)?;
        Some(state.refused_by.remove(i).2)
    }
}

/// A version that changes when any member's does (ADR 0049 §1): FNV-1a over the members'.
fn combined_version(members: &[Arc<dyn AccessPolicy>]) -> PolicyVersion {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for m in members {
        for byte in m.version().0.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(PRIME);
        }
    }
    PolicyVersion(h)
}

impl AccessPolicy for IntersectionPolicy {
    fn version(&self) -> PolicyVersion {
        combined_version(&self.members)
    }

    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        Box::pin(async move {
            self.note_kind(securable);
            let mut decided = false;
            let mut with = None;
            for (i, member) in self.members.iter().enumerate() {
                if !member.decides(securable) {
                    continue;
                }
                decided = true;
                let asked = self.question(&**member, securable);
                match member.check(principal, &asked, privilege).await {
                    Ok(AccessDecision::Allow) => {}
                    Ok(AccessDecision::AllowWith(policy)) => {
                        with.get_or_insert(policy);
                    }
                    Ok(deny @ AccessDecision::Deny { .. }) => {
                        self.note_refusal(securable, privilege, i);
                        return Ok(deny);
                    }
                    Err(e) => {
                        self.note_refusal(securable, privilege, i);
                        return Err(e);
                    }
                    // A decision this build does not know is a refusal (fail closed).
                    Ok(_) => {
                        self.note_refusal(securable, privilege, i);
                        return Ok(AccessDecision::Deny {
                            reason: "an access provider gave an unknown decision".to_owned(),
                        });
                    }
                }
            }
            if !decided {
                self.state().last_label = None;
                return Ok(AccessDecision::Deny {
                    reason: "no access provider decides this securable".to_owned(),
                });
            }
            Ok(with.map_or(AccessDecision::Allow, AccessDecision::AllowWith))
        })
    }

    fn column_level(&self) -> bool {
        self.column_level
    }

    fn denied(&self, principal: &Principal, securable: &Securable, privilege: Privilege) {
        match self
            .take_refusal(securable, privilege)
            .and_then(|i| self.members.get(i))
        {
            Some(member) => {
                let asked = self.question(&**member, securable);
                member.denied(principal, &asked, privilege);
            }
            // Refused for want of a deciding member, or forgotten: told to every decider.
            None => {
                for member in self.members.iter().filter(|m| m.decides(securable)) {
                    let asked = self.question(&**member, securable);
                    member.denied(principal, &asked, privilege);
                }
            }
        }
    }

    fn prefetch<'a>(
        &'a self,
        principal: &'a Principal,
        items: &'a [(Securable, Privilege)],
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            for item in items {
                self.note_kind(&item.0);
            }
            for member in &self.members {
                // Keyed by the debug form (`Securable` is not `Hash`): two paths that differ only
                // in a redacted secret are one hint, which only costs that hint.
                let mut seen = HashSet::new();
                let mut asked: Vec<(Securable, Privilege)> = Vec::new();
                for (securable, privilege) in items {
                    if !member.decides(securable) {
                        continue;
                    }
                    let item = (self.question(&**member, securable), *privilege);
                    if seen.insert(format!("{item:?}")) {
                        asked.push(item);
                    }
                }
                if !asked.is_empty() {
                    member.prefetch(principal, &asked).await;
                }
            }
        })
    }

    fn decides(&self, securable: &Securable) -> bool {
        self.members.iter().any(|m| m.decides(securable))
    }

    fn refusal_label(&self) -> Option<&'static str> {
        self.state().last_label
    }
}

/// `inner` deciding only the server's own objects (`System(..)`): the role matrix in a provider
/// list that names neither `roles` nor `grants` (ADR 0049 §1, ADR 0046 §7), so the list's data
/// securables are left to its other members.
pub struct SystemOnlyPolicy {
    inner: Arc<dyn AccessPolicy>,
}

impl std::fmt::Debug for SystemOnlyPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemOnlyPolicy").finish_non_exhaustive()
    }
}

impl SystemOnlyPolicy {
    /// `inner`, for `System(..)` securables only.
    #[must_use]
    pub fn new(inner: Arc<dyn AccessPolicy>) -> Self {
        Self { inner }
    }
}

impl AccessPolicy for SystemOnlyPolicy {
    fn version(&self) -> PolicyVersion {
        self.inner.version()
    }

    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        if !self.decides(securable) {
            return Box::pin(std::future::ready(Ok(AccessDecision::Deny {
                reason: "not a server object".to_owned(),
            })));
        }
        self.inner.check(principal, securable, privilege)
    }

    fn denied(&self, principal: &Principal, securable: &Securable, privilege: Privilege) {
        self.inner.denied(principal, securable, privilege);
    }

    fn decides(&self, securable: &Securable) -> bool {
        matches!(securable, Securable::System(_))
    }
}

#[cfg(test)]
mod tests;
