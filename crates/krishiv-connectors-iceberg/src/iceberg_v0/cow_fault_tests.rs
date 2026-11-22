// SPDX-License-Identifier: Apache-2.0
//! Fault injection on the overwrite path (`filter.rs` through `commit.rs`,
//! review 36b2a medium): `commit_tests`' scripted catalog, the write under
//! test removing the table's first file (`init.parquet`) and adding its own.
//! An unknown answer then a rebase that finds the removal missing (nothing
//! deleted), an unknown answer then the landed commit, and a known failure
//! (the rewritten and merged manifests deleted, the base's kept); a resend
//! of the same attempt after a still-unknown answer, and a rebase after an
//! unknown answer whose re-filter succeeds (round 2).

use std::collections::BTreeSet;
use std::sync::Arc;

use iceberg::spec::{Operation, TableMetadata};
use krishiv_common::error::ErrorClass;

use super::append::{WriteIds, prepare_attempt, write_new_manifests};
use super::append_tests::MAX;
use super::commit::CommitClock as _;
use super::commit_tests::{Case, Fault, Foreign, INIT, MINE, OURS, When, case, case_with, exists};
use super::filter::Removals;
use super::scan::read_manifest_files;

/// The removal of `name` alone.
fn removal(c: &Case, name: &str) -> Removals {
    Removals {
        data_files: BTreeSet::from([c.env.at(name)]),
        delete_files: BTreeSet::new(),
        ..Removals::default()
    }
}

/// Another writer's `DELETE` of every row of `name`, as snapshot `id`.
fn foreign_removal(c: &Case, name: &str, id: i64) -> Foreign {
    let io = c.env.reg.file_io();
    let clock = Arc::clone(&c.clock);
    let removals = removal(c, name);
    Box::new(move |m: TableMetadata| {
        Box::pin(async move {
            let uuid = format!("uuid-{id}");
            let mut new = write_new_manifests(&io, &m, Vec::new(), id, &uuid, &mut Vec::new())
                .await
                .unwrap();
            new.removals = Some(removals);
            let ts = clock.now_ms().max(m.last_updated_ms() + 1);
            let ids = WriteIds {
                branch: "main",
                write_id: &uuid,
                query_id: None,
                idempotency_key: None,
            };
            let a = prepare_attempt(&io, &m, &mut new, ids, ts, MAX)
                .await
                .unwrap();
            (a.requirements, a.updates)
        })
    })
}

/// The manifests of the snapshot the write read.
fn base_manifests(c: &Case) -> Vec<String> {
    let io = c.env.reg.file_io();
    let s = c.base.current_snapshot().unwrap();
    c.env
        .rt
        .block_on(read_manifest_files(
            &io,
            s.manifest_list(),
            c.base.format_version() as u8,
            MAX,
        ))
        .unwrap()
        .into_iter()
        .map(|m| m.manifest_path)
        .collect()
}

/// A 503 before the table (an unknown answer), then another writer removes
/// the file this write removes: the probe sees the branch moved, the rebase
/// finds the removal missing, a conflict; a request had an unknown answer,
/// so nothing is deleted, the rewritten manifest of the first attempt
/// included.
#[test]
fn an_unknown_answer_then_a_missing_removal_keeps_every_file() {
    let mut c = case();
    c.new.removals = Some(removal(&c, "init.parquet"));
    c.cat.script(&[Fault::ServerErrorBeforeApply]);
    c.cat.on(
        When::BeforeLoad(1),
        foreign_removal(&c, "init.parquet", 970),
    );
    let e = c.run().unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{e}"
    );
    assert!(
        e.message().contains("Missing required files to delete"),
        "{e}"
    );
    assert!(e.message().contains("files were kept"), "{e}");
    assert_eq!(c.cat.requests(), 1);
    assert!(!c.new.merged.is_empty());
    for path in &c.new.merged {
        assert!(exists(path), "{path}");
    }
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    c.check(&[]);
}

/// A lost answer to a request the table applied: the probe finds the
/// snapshot, an `overwrite` removing the file; the rewritten manifest is the
/// one the committed list names.
#[test]
fn an_unknown_answer_then_a_landed_overwrite_commits() {
    let mut c = case();
    c.new.removals = Some(removal(&c, "init.parquet"));
    c.cat.script(&[Fault::LostResponse]);
    let m = c.run().unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    let s = m.snapshot_by_id(OURS).unwrap();
    assert_eq!(s.summary().operation, Operation::Overwrite);
    assert_eq!(
        s.summary()
            .additional_properties
            .get("deleted-data-files")
            .map(String::as_str),
        Some("1")
    );
    assert_eq!(c.cat.requests(), 1);
    assert!(exists(&c.list(1)) && exists(&c.manifest()) && exists(&c.data()));
    let io = c.env.reg.file_io();
    let listed: Vec<String> = c
        .env
        .rt
        .block_on(read_manifest_files(&io, s.manifest_list(), 2, MAX))
        .unwrap()
        .into_iter()
        .map(|m| m.manifest_path)
        .collect();
    assert!(!c.new.merged.is_empty());
    for path in &c.new.merged {
        assert!(exists(path), "{path}");
        assert!(listed.contains(path), "{path} not in {listed:?}");
    }
    c.check(&[MINE]);
}

/// A 401 to the first request is a known failure: the rewritten manifest,
/// the merged one (`min-count-to-merge` 2 merges the new manifest with the
/// rewrite), the new manifest, the list and the data file are deleted; the
/// base's manifests are kept and the table reads as before.
#[test]
fn a_known_failure_deletes_the_rewritten_and_merged_manifests_only() {
    let mut c = case_with(2, &[("commit.manifest.min-count-to-merge", "2")]);
    c.new.removals = Some(removal(&c, "init.parquet"));
    c.cat.script(&[Fault::Unauthorized401]);
    let base = base_manifests(&c);
    let e = c.run().unwrap_err();
    assert!(e.to_string().contains("401"), "{e}");
    assert_eq!(c.cat.requests(), 1);
    // The filter's rewrite and the merge's output.
    assert_eq!(c.new.merged.len(), 2, "{:?}", c.new.merged);
    for path in &c.new.merged {
        assert!(!exists(path), "{path}");
        assert!(!base.contains(path), "{path}");
    }
    assert!(!exists(&c.list(1)) && !exists(&c.manifest()) && !exists(&c.data()));
    assert!(!base.is_empty());
    for path in &base {
        assert!(exists(path), "{path}");
    }
    c.check(&[INIT]);
}

/// Review 36b2a round 2: a 503 before the table, and every probe finds the
/// branch still at the base (still unknown): the loop resends the **same**
/// prepared attempt (one manifest list, one rewritten manifest, no second
/// filter run), which lands.
#[test]
fn a_still_unknown_answer_resends_the_same_overwrite_attempt() {
    let mut c = case();
    c.new.removals = Some(removal(&c, "init.parquet"));
    c.cat.script(&[Fault::ServerErrorBeforeApply]);
    let m = c.run().unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(c.cat.requests(), 2);
    // The filter ran once: one rewritten manifest, one list.
    assert_eq!(c.new.merged.len(), 1, "{:?}", c.new.merged);
    assert!(exists(&c.list(1)) && !exists(&c.list(2)));
    let s = m.snapshot_by_id(OURS).unwrap();
    assert_eq!(s.manifest_list(), c.list(1));
    assert!(exists(&c.new.merged[0]));
    c.check(&[MINE]);
}

/// Review 36b2a round 2: a 503, then another writer appends; the probe sees
/// the branch moved and the rebase re-applies the removal to the new base
/// (a second rewritten manifest), which commits: the removed file is gone,
/// the foreign file kept, the first attempt's rewrite cleaned up.
#[test]
fn an_unknown_answer_then_a_rebase_refilters_and_commits() {
    let mut c = case();
    c.new.removals = Some(removal(&c, "init.parquet"));
    c.cat.script(&[Fault::ServerErrorBeforeApply]);
    c.cat
        .on(When::BeforeLoad(1), c.foreign("other.parquet", 980, "main"));
    let m = c.run().unwrap();
    assert_eq!(m.current_snapshot_id(), Some(OURS));
    assert_eq!(c.cat.requests(), 2);
    assert_eq!(c.new.merged.len(), 2, "{:?}", c.new.merged);
    let s = m.snapshot_by_id(OURS).unwrap();
    assert_eq!(s.manifest_list(), c.list(2));
    assert_eq!(
        s.summary()
            .additional_properties
            .get("deleted-data-files")
            .map(String::as_str),
        Some("1")
    );
    // The first attempt's rewrite is in no list; the second's is listed.
    let io = c.env.reg.file_io();
    let listed: Vec<String> = c
        .env
        .rt
        .block_on(read_manifest_files(&io, s.manifest_list(), 2, MAX))
        .unwrap()
        .into_iter()
        .map(|m| m.manifest_path)
        .collect();
    assert!(!listed.contains(&c.new.merged[0]));
    assert!(listed.contains(&c.new.merged[1]));
    assert!(!exists(&c.new.merged[0]) && exists(&c.new.merged[1]));
    c.check(&[MINE, "980|1|other.parquet"]);
}
