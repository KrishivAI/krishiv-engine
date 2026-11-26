// SPDX-License-Identifier: Apache-2.0
//! [`OwnRefusals`]: the refusals a wrapping policy makes itself, kept apart from those of the
//! provider it wraps (G6 review of 7f535d472). A wrapper that refuses on its own rule
//! (`PathPrefixPolicy`'s confinement, `ReadOnlyPolicy`'s read-only rule) must not tell the inner
//! provider of that refusal through [`AccessPolicy::denied`] (it never decided it, so it would
//! count or audit a denial it did not make), and its [`AccessPolicy::refusal_label`] is not the
//! inner provider's but the server's own rule, `roles`.
//!
//! [`AccessPolicy::denied`]: krishiv_plan::analyzer::AccessPolicy::denied
//! [`AccessPolicy::refusal_label`]: krishiv_plan::analyzer::AccessPolicy::refusal_label

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use krishiv_plan::analyzer::{Privilege, Securable};

/// The most own refusals kept for `denied`; past it, the oldest goes (and is then told to the
/// inner provider, which only costs that provider one count).
const KEPT: usize = 64;

/// A wrapper's own refusals. Clones share one list (a cloned policy is the same policy).
#[derive(Debug, Default, Clone)]
pub(crate) struct OwnRefusals {
    state: Arc<Mutex<State>>,
}

#[derive(Debug, Default)]
struct State {
    /// Refused by the wrapper and not yet told (the newest last).
    waiting: Vec<(Securable, Privilege)>,
    /// Whether the wrapper's latest answer was its own refusal. Per policy, not per query, as
    /// `IntersectionPolicy`'s label is: two concurrent queries of one session may swap labels.
    latest: bool,
}

impl OwnRefusals {
    fn state(&self) -> MutexGuard<'_, State> {
        // A poisoned lock only means a panic elsewhere held it; the list stays usable.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The wrapper refused `securable` for `privilege` on its own rule.
    pub(crate) fn refused(&self, securable: &Securable, privilege: Privilege) {
        let mut state = self.state();
        state.latest = true;
        if state.waiting.len() >= KEPT {
            state.waiting.remove(0);
        }
        state.waiting.push((securable.clone(), privilege));
    }

    /// The wrapper handed a question to the inner provider.
    pub(crate) fn passed_on(&self) {
        self.state().latest = false;
    }

    /// Whether a refusal being told was the wrapper's own (and is now taken).
    pub(crate) fn take(&self, securable: &Securable, privilege: Privilege) -> bool {
        let mut state = self.state();
        let Some(i) = state
            .waiting
            .iter()
            .rposition(|(s, p)| s == securable && *p == privilege)
        else {
            return false;
        };
        state.waiting.remove(i);
        true
    }

    /// Whether the wrapper's latest answer was its own refusal.
    pub(crate) fn latest(&self) -> bool {
        self.state().latest
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use krishiv_common::error::KrishivError;
    use krishiv_plan::analyzer::{
        AccessDecision, AccessPolicy, BoxFuture, PolicyVersion, Principal, PrincipalKind,
        TableIdent,
    };

    use super::*;
    use crate::access::ReadOnlyPolicy;

    /// A provider that refuses, names itself `external` and counts what it is told.
    #[derive(Default)]
    struct Labelled {
        told: AtomicUsize,
    }

    impl AccessPolicy for Labelled {
        fn version(&self) -> PolicyVersion {
            PolicyVersion(1)
        }

        fn check<'a>(
            &'a self,
            _: &'a Principal,
            _: &'a Securable,
            _: Privilege,
        ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
            Box::pin(std::future::ready(Ok(AccessDecision::Deny {
                reason: "no".to_owned(),
            })))
        }

        fn denied(&self, _: &Principal, _: &Securable, _: Privilege) {
            self.told.fetch_add(1, Ordering::Relaxed);
        }

        fn refusal_label(&self) -> Option<&'static str> {
            Some("external")
        }
    }

    /// Review nit of 7f535d472: a read-only refusal is the wrapper's own, as a confinement
    /// refusal is `PathPrefixPolicy`'s; a refusal of the provider behind it is still told.
    #[test]
    fn a_read_only_refusal_is_not_the_inner_providers() {
        let inner = Arc::new(Labelled::default());
        let policy = ReadOnlyPolicy::new(Arc::clone(&inner) as _);
        let ana = Principal::new("ana", PrincipalKind::User);
        let t = Securable::Table(TableIdent {
            catalog: "spark_catalog".to_owned(),
            namespace: vec!["default".to_owned()],
            name: "t".to_owned(),
        });
        let d = futures::executor::block_on(policy.check(&ana, &t, Privilege::Modify));
        assert!(matches!(d, Ok(AccessDecision::Deny { .. })), "{d:?}");
        assert_eq!(policy.refusal_label(), None);
        policy.denied(&ana, &t, Privilege::Modify);
        assert_eq!(inner.told.load(Ordering::Relaxed), 0);

        let d = futures::executor::block_on(policy.check(&ana, &t, Privilege::Select));
        assert!(matches!(d, Ok(AccessDecision::Deny { .. })), "{d:?}");
        assert_eq!(policy.refusal_label(), Some("external"));
        policy.denied(&ana, &t, Privilege::Select);
        assert_eq!(inner.told.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn the_oldest_own_refusal_goes_past_the_bound() {
        let own = OwnRefusals::default();
        let path = |i: usize| Securable::System(format!("s{i}"));
        for i in 0..=KEPT {
            own.refused(&path(i), Privilege::Select);
        }
        assert!(!own.take(&path(0), Privilege::Select));
        assert!(own.take(&path(KEPT), Privilege::Select));
        assert!(!own.take(&path(KEPT), Privilege::Select));
        assert!(own.latest());
        own.passed_on();
        assert!(!own.latest());
    }
}
