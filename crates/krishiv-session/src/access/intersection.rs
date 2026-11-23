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
//! asked as `Select` on the column's relation, as every kind (table, view) the list has seen that
//! name asked as. Kinds only accumulate, so a later hint never turns a view's question into a
//! table's; a column whose relation was never asked about (or was forgotten) is refused. Such a
//! member is asked each relation question once per analysis stage (review m3): an allowed answer
//! serves the stage's other columns of that relation, for the same principal and member version,
//! until the next [`AccessPolicy::prefetch`] (a new stage), or for 2 s at most.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::{
    AccessDecision, AccessPolicy, PolicyVersion, Principal, Privilege, Securable, TableIdent,
};

/// The most table kinds and refusals the list remembers; past it, the memory is cleared (a
/// column of a forgotten relation is refused, a forgotten refusal is told to every deciding
/// member). The memory is per caller session, so a session that has named more than 4,096
/// relations can get a spurious 42501 for a query checked while the clear happens: its column
/// pass finds its relation forgotten. That fails closed; running the query again passes.
const REMEMBERED: usize = 4096;

/// A name asked about as a table.
const AS_TABLE: u8 = 1;
/// A name asked about as a view.
const AS_VIEW: u8 = 2;

/// The most refusals waiting for [`AccessPolicy::denied`]; the oldest goes first.
const REFUSALS: usize = 64;

/// How long an allowed relation answer serves a stage's other columns (review m3): the
/// `external` provider's prefetched answers live as long.
const STAGE_ANSWER_LIFETIME: Duration = Duration::from_secs(2);

/// The most relation answers one stage keeps; past it, the oldest is asked again.
const STAGE_ANSWERS: usize = 1024;

/// Several providers, all of which must allow (ADR 0049 §1).
pub struct IntersectionPolicy {
    members: Vec<Arc<dyn AccessPolicy>>,
    column_level: bool,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// The kinds a table name was asked about as ([`AS_TABLE`] | [`AS_VIEW`]).
    views: HashMap<TableIdent, u8>,
    /// The member that refused a (securable, privilege), until [`AccessPolicy::denied`] is told
    /// (the newest last; `Securable` is not `Hash`, and the analyzer reports a refusal at once).
    refused_by: Vec<(Securable, Privilege, usize)>,
    /// The [`AccessPolicy::refusal_label`] of the member behind the latest refusal.
    last_label: Option<&'static str>,
    /// The relation questions a table-level member allowed in this stage (review m3), keyed by
    /// member, principal, question (debug form: `Securable` is not `Hash`) and privilege, with
    /// the member's version and when it answered. Cleared by every prefetch.
    stage_answers: HashMap<StageKey, (PolicyVersion, Instant)>,
}

type StageKey = (usize, Principal, String, Privilege);

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

    /// The securables `member` is asked about for `securable`: itself, except a column read for
    /// a member that decides whole tables, which is asked about the column's relation as each
    /// kind seen. `None`: the relation's kind is unknown, a refusal.
    fn question(&self, member: &dyn AccessPolicy, securable: &Securable) -> Option<Vec<Securable>> {
        if let Securable::Column { table, .. } = securable
            && !member.column_level()
        {
            let kinds = self.state().views.get(table).copied()?;
            let mut asked = Vec::with_capacity(2);
            if kinds & AS_TABLE != 0 {
                asked.push(Securable::Table(table.clone()));
            }
            if kinds & AS_VIEW != 0 {
                asked.push(Securable::View(table.clone()));
            }
            return Some(asked);
        }
        Some(vec![securable.clone()])
    }

    /// Whether `member` has a say on `securable`: it decides it, or (a column read it would be
    /// asked as its relation) it decides the relation.
    fn asks(member: &dyn AccessPolicy, securable: &Securable) -> bool {
        if member.decides(securable) {
            return true;
        }
        match securable {
            Securable::Column { table, .. } if !member.column_level() => {
                member.decides(&Securable::Table(table.clone()))
                    || member.decides(&Securable::View(table.clone()))
            }
            _ => false,
        }
    }

    /// What `member` says about `securable`: every question it is asked must be allowed.
    async fn ask(
        &self,
        index: usize,
        member: &dyn AccessPolicy,
        principal: &Principal,
        securable: &Securable,
        privilege: Privilege,
    ) -> Result<AccessDecision, KrishivError> {
        let Some(questions) = self.question(member, securable) else {
            return Ok(AccessDecision::Deny {
                reason: "the column's relation was not checked before its columns".to_owned(),
            });
        };
        // A column asked as its relation: the stage's earlier allowed answer serves it.
        let translated = matches!(securable, Securable::Column { .. }) && !member.column_level();
        let mut with = None;
        for asked in &questions {
            let key =
                translated.then(|| (index, principal.clone(), format!("{asked:?}"), privilege));
            if let Some(key) = &key
                && self.answered(key, member.version())
            {
                continue;
            }
            match member.check(principal, asked, privilege).await? {
                AccessDecision::Allow => {
                    if let Some(key) = key {
                        self.note_answer(key, member.version());
                    }
                }
                AccessDecision::AllowWith(policy) => {
                    with.get_or_insert(policy);
                }
                deny @ AccessDecision::Deny { .. } => return Ok(deny),
                // A decision this build does not know is a refusal (fail closed).
                _ => {
                    return Ok(AccessDecision::Deny {
                        reason: "an access provider gave an unknown decision".to_owned(),
                    });
                }
            }
        }
        Ok(with.map_or(AccessDecision::Allow, AccessDecision::AllowWith))
    }

    /// Whether `key` was allowed in this stage, at `version`, within [`STAGE_ANSWER_LIFETIME`].
    fn answered(&self, key: &StageKey, version: PolicyVersion) -> bool {
        self.state()
            .stage_answers
            .get(key)
            .is_some_and(|(v, at)| *v == version && at.elapsed() < STAGE_ANSWER_LIFETIME)
    }

    fn note_answer(&self, key: StageKey, version: PolicyVersion) {
        let mut state = self.state();
        if state.stage_answers.len() >= STAGE_ANSWERS
            && let Some(oldest) = state
                .stage_answers
                .iter()
                .min_by_key(|(_, (_, at))| *at)
                .map(|(k, _)| k.clone())
        {
            state.stage_answers.remove(&oldest);
        }
        state.stage_answers.insert(key, (version, Instant::now()));
    }

    fn note_kind(&self, securable: &Securable) {
        let (table, kind) = match securable {
            Securable::Table(t) => (t, AS_TABLE),
            Securable::View(t) => (t, AS_VIEW),
            _ => return,
        };
        let mut state = self.state();
        if state.views.len() >= REMEMBERED && !state.views.contains_key(table) {
            state.views.clear();
        }
        *state.views.entry(table.clone()).or_default() |= kind;
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

    /// Tells `member` of a refusal of `securable`, as the questions it was asked (the column
    /// itself when its relation's kind is not known).
    fn tell(
        &self,
        member: &dyn AccessPolicy,
        principal: &Principal,
        securable: &Securable,
        privilege: Privilege,
    ) {
        let questions = self
            .question(member, securable)
            .unwrap_or_else(|| vec![securable.clone()]);
        for asked in &questions {
            member.denied(principal, asked, privilege);
        }
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
                if !Self::asks(&**member, securable) {
                    continue;
                }
                decided = true;
                match self
                    .ask(i, &**member, principal, securable, privilege)
                    .await
                {
                    Ok(AccessDecision::Allow) => {}
                    Ok(AccessDecision::AllowWith(policy)) => {
                        with.get_or_insert(policy);
                    }
                    Ok(deny) => {
                        self.note_refusal(securable, privilege, i);
                        return Ok(deny);
                    }
                    Err(e) => {
                        self.note_refusal(securable, privilege, i);
                        return Err(e);
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
            Some(member) => self.tell(&**member, principal, securable, privilege),
            // Refused for want of a deciding member, or forgotten: told to every decider.
            None => {
                for member in self.members.iter().filter(|m| Self::asks(&***m, securable)) {
                    self.tell(&**member, principal, securable, privilege);
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
            // A new stage: the previous stage's relation answers serve none of its columns.
            self.state().stage_answers.clear();
            for item in items {
                self.note_kind(&item.0);
            }
            for member in &self.members {
                // Keyed by the debug form (`Securable` is not `Hash`): two paths that differ only
                // in a redacted secret are one hint, which only costs that hint.
                let mut seen = HashSet::new();
                let mut asked: Vec<(Securable, Privilege)> = Vec::new();
                for (securable, privilege) in items {
                    if !Self::asks(&**member, securable) {
                        continue;
                    }
                    for question in self.question(&**member, securable).unwrap_or_default() {
                        let item = (question, *privilege);
                        if seen.insert(format!("{item:?}")) {
                            asked.push(item);
                        }
                    }
                }
                // Every stage reaches every member, an empty one too: a member's stage state (the
                // `external` provider's answers) is replaced by each call, so a skipped call would
                // leave the previous stage's answers in place for this one.
                member.prefetch(principal, &asked).await;
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
