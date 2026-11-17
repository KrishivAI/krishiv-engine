// SPDX-License-Identifier: Apache-2.0
//! `CachedRemoteRelation` (design P3-T083 §3.4): a DataFrame the server holds under an id, sent
//! back by the client (the result of a `checkpoint_command`).
//!
//! The translator cannot resolve the id (it is a pure function of the plan), so the relation is a
//! read of a table named [`cached_remote_table`]`(id)` and every id met is recorded in the conf's
//! [`CachedRemoteIds`]. The host looks each one up in its session, raises
//! `CONNECT_INVALID_PLAN.DATAFRAME_NOT_FOUND` for one it does not hold and replaces the reads by
//! the frames' plans (`Session::from_plan_over`). The namespace is reserved: a table of that
//! namespace is never a catalog table.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, PoisonError};

/// The namespace of the table that stands for a cached remote relation.
pub const CACHED_REMOTE_NAMESPACE: &str = "__krishiv_cached_remote";

/// The table name a `CachedRemoteRelation` with `relation_id` translates to.
#[must_use]
pub fn cached_remote_table(relation_id: &str) -> Vec<String> {
    vec![CACHED_REMOTE_NAMESPACE.to_owned(), relation_id.to_owned()]
}

/// The ids of the cached remote relations a translation met.
#[derive(Clone, Default)]
pub struct CachedRemoteIds(Arc<Mutex<BTreeSet<String>>>);

impl CachedRemoteIds {
    pub(super) fn record(&self, id: &str) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id.to_owned());
    }

    /// The ids recorded so far, in order, each once; none are left behind.
    #[must_use]
    pub fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner))
            .into_iter()
            .collect()
    }
}

impl std::fmt::Debug for CachedRemoteIds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CachedRemoteIds")
    }
}

impl PartialEq for CachedRemoteIds {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for CachedRemoteIds {}

#[cfg(test)]
mod tests {
    use krishiv_plan::{PlanNode, ReadSource};

    use super::super::testing::{boxed, relation};
    use super::super::{TranslateConf, translate_command, translate_relation};
    use super::*;
    use crate::proto::v4_3 as pb;
    use crate::proto::v4_3::relation::RelType;
    use crate::translate::{CheckpointSpec, Command};

    fn remote(id: &str) -> pb::Relation {
        relation(RelType::CachedRemoteRelation(pb::CachedRemoteRelation {
            relation_id: id.to_owned(),
        }))
    }

    #[test]
    fn a_cached_remote_relation_reads_the_reserved_table_and_records_its_id() {
        let conf = TranslateConf::default();
        let join = relation(RelType::Join(Box::new(pb::Join {
            left: boxed(remote("b")),
            right: boxed(remote("a")),
            join_type: pb::join::JoinType::Cross as i32,
            ..pb::Join::default()
        })));
        let plan = translate_relation(&join, &conf).expect("translates");
        let mut names = Vec::new();
        fn walk(p: &krishiv_plan::LogicalPlan, out: &mut Vec<Vec<String>>) {
            if let PlanNode::Read(ReadSource::Table { name, .. }) = &p.node {
                out.push(name.clone());
            }
            p.for_each_child(&mut |c| walk(c, out));
        }
        walk(plan.root(), &mut names);
        assert_eq!(
            names,
            vec![cached_remote_table("b"), cached_remote_table("a")]
        );
        // Each id once, in order; the recorder is drained.
        assert_eq!(conf.cached_remote.take(), vec!["a", "b"]);
        assert!(conf.cached_remote.take().is_empty());
    }

    #[test]
    fn the_commands_are_read_with_their_flags() {
        let checkpoint = pb::Command {
            command_type: Some(pb::command::CommandType::CheckpointCommand(Box::new(
                pb::CheckpointCommand {
                    relation: None,
                    local: true,
                    eager: false,
                    storage_level: Some(pb::StorageLevel {
                        use_disk: true,
                        replication: 2,
                        ..pb::StorageLevel::default()
                    }),
                },
            ))),
        };
        let Command::Checkpoint(CheckpointSpec {
            local,
            eager,
            storage_level,
        }) = translate_command(&checkpoint).expect("a checkpoint")
        else {
            panic!("not a checkpoint");
        };
        assert!(local && !eager);
        assert_eq!(storage_level.map(|l| l.replication), Some(2));
        let remove = pb::Command {
            command_type: Some(pb::command::CommandType::RemoveCachedRemoteRelationCommand(
                Box::new(pb::RemoveCachedRemoteRelationCommand {
                    relation: Some(pb::CachedRemoteRelation {
                        relation_id: "x".to_owned(),
                    }),
                }),
            )),
        };
        assert_eq!(
            translate_command(&remove).expect("a removal"),
            Command::RemoveCachedRemoteRelation {
                relation_id: "x".to_owned()
            }
        );
    }
}
