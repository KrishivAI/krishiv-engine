// SPDX-License-Identifier: Apache-2.0
//! The reference a [`PlanNode::SecurityBarrier`](crate::PlanNode::SecurityBarrier) carries: the
//! guarded table, the principal its policy was resolved for and the policy version (design
//! `P3-E14-security-barrier.md` §3).

use krishiv_common::PrincipalKey;

use crate::analyzer::Securable;

/// Which policy a barrier applies: equal references from one plan apply the same filter and masks.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PolicyRef {
    /// The table the policy guards.
    pub securable: Securable,
    /// The principal the policy was resolved for (two principals may share policy text and
    /// version, as definer-rights views will, and never share a barrier).
    pub principal: PrincipalKey,
    /// The `AccessPolicy` version the query was authorized against (`CacheKey` carries it too).
    pub version: u64,
}

impl PolicyRef {
    /// A reference to the policy on `securable` for `principal` at `version`.
    #[must_use]
    pub fn new(securable: Securable, principal: PrincipalKey, version: u64) -> Self {
        Self {
            securable,
            principal,
            version,
        }
    }
}

/// By hand: a path securable holds an `OptionValue`, which has no `Hash` (it may carry a
/// secret). The securable is hashed by its `Display` form, a function of its value, so equal
/// references hash equally.
impl std::hash::Hash for PolicyRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.securable.to_string().hash(state);
        self.principal.hash(state);
        self.version.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv_common::{Principal, PrincipalKey, PrincipalKind};
    use krishiv_types::DataType;

    use super::PolicyRef;
    use crate::analyzer::{Securable, TableIdent};
    use crate::{
        BoundColumn, LogicalPlan, Plan, PlanNode, ReadSource, Scan, normalized_subtree_hash,
    };

    fn scan() -> LogicalPlan {
        LogicalPlan::new(PlanNode::Scan(Scan {
            source: ReadSource::Table {
                name: vec!["t".to_owned()],
                options: Default::default(),
            },
            output: vec![BoundColumn {
                id: crate::AttrId(1),
                name: "k".to_owned(),
                data_type: DataType::Integer,
                nullable: true,
            }],
            version: None,
        }))
    }

    fn policy(user: &str, version: u64) -> PolicyRef {
        PolicyRef::new(
            Securable::Table(TableIdent {
                catalog: "spark_catalog".to_owned(),
                namespace: vec!["default".to_owned()],
                name: "t".to_owned(),
            }),
            PrincipalKey::of(&Principal::new(user, PrincipalKind::User)),
            version,
        )
    }

    fn barrier(user: &str, version: u64) -> Arc<LogicalPlan> {
        Arc::new(scan().security_barrier(policy(user, version)))
    }

    #[test]
    fn a_barrier_is_equal_by_its_policy_and_input() {
        assert!(*barrier("ana", 1) == *barrier("ana", 1));
        assert!(*barrier("ana", 1) != *barrier("bob", 1), "principal");
        assert!(*barrier("ana", 1) != *barrier("ana", 2), "version");
        assert!(*barrier("ana", 1) != scan(), "never its unguarded input");
        let hash = |p: &Arc<LogicalPlan>| Plan::try_new(Arc::clone(p)).unwrap().structural_hash();
        assert_eq!(hash(&barrier("ana", 1)), hash(&barrier("ana", 1)));
        assert_ne!(hash(&barrier("ana", 1)), hash(&barrier("bob", 1)));
        assert_ne!(hash(&barrier("ana", 1)), hash(&barrier("ana", 2)));
        assert_ne!(
            normalized_subtree_hash(&barrier("ana", 1)),
            normalized_subtree_hash(&barrier("bob", 1))
        );
        assert_ne!(
            normalized_subtree_hash(&barrier("ana", 1)),
            normalized_subtree_hash(&Arc::new(scan()))
        );
    }

    #[test]
    fn equal_references_hash_equally() {
        use std::hash::{BuildHasher, RandomState};
        let state = RandomState::new();
        let h = |r: &PolicyRef| state.hash_one(r);
        assert_eq!(h(&policy("ana", 1)), h(&policy("ana", 1)));
        assert_ne!(h(&policy("ana", 1)), h(&policy("bob", 1)));
        assert_ne!(h(&policy("ana", 1)), h(&policy("ana", 2)));
    }

    #[test]
    fn a_barrier_keeps_its_input_output_and_prints_its_table() {
        let b = barrier("ana", 1);
        assert_eq!(b.resolved_output(), scan().resolved_output());
        assert!(b.is_resolved());
        let text = format!("{}", Plan::try_new(b).unwrap());
        assert!(
            text.contains("SecurityBarrier [table `spark_catalog`.`default`.`t`]"),
            "{text}"
        );
    }
}
