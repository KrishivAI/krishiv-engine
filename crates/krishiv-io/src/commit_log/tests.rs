// SPDX-License-Identifier: Apache-2.0
//! The commit log over an in-memory store.

use std::sync::Arc;

use krishiv_common::error::ErrorClass;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

use super::*;
use crate::io::{IoConfig, IoScheduler};

struct Setup {
    io: IoContext,
    store: StoreId,
    inner: Arc<InMemory>,
}

fn setup() -> Setup {
    let inner = Arc::new(InMemory::new());
    let sched = IoScheduler::new(IoConfig::default());
    let store = StoreId::new("memory://log");
    sched.register_store(store.clone(), Arc::clone(&inner) as Arc<dyn ObjectStore>);
    Setup {
        io: sched.context(),
        store,
        inner,
    }
}

fn log(s: &Setup) -> CommitLog<'_> {
    CommitLog::new(
        &s.io,
        s.store.clone(),
        Path::from("t"),
        "memory://log/t".to_owned(),
    )
}

fn file(name: &str) -> AddedFile {
    AddedFile {
        path: name.to_owned(),
        size: 10,
        rows: 2,
        partition: vec![
            ("year".to_owned(), Some("2020".to_owned())),
            ("r".to_owned(), None),
        ],
    }
}

fn entry(write: &str, mode: CommitMode, add: &[&str], remove: &[&str]) -> LogEntry {
    LogEntry {
        query_id: "q".to_owned(),
        write_id: write.to_owned(),
        job_uuid: "job".to_owned(),
        mode,
        add: add.iter().map(|n| file(n)).collect(),
        remove: remove.iter().map(|r| (*r).to_owned()).collect(),
    }
}

fn names(s: &Snapshot) -> Vec<&str> {
    s.files.keys().map(String::as_str).collect()
}

#[tokio::test]
async fn an_empty_directory_has_no_log() {
    let s = setup();
    assert!(!log(&s).exists().await.unwrap());
    assert_eq!(log(&s).snapshot().await.unwrap(), Snapshot::default());
}

#[tokio::test]
async fn commits_replay_in_order_with_removals() {
    let s = setup();
    let l = log(&s);
    assert_eq!(
        l.commit(None, &entry("a", CommitMode::Append, &["f1", "f2"], &[]))
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        l.commit(
            Some(0),
            &entry("b", CommitMode::Overwrite, &["f3"], &["f1", "f2"])
        )
        .await
        .unwrap(),
        1
    );
    assert!(l.exists().await.unwrap());
    let snap = l.snapshot().await.unwrap();
    assert_eq!(snap.version, Some(1));
    assert_eq!(names(&snap), ["f3"]);
    assert_eq!(snap.files["f3"], file("f3"));
}

#[tokio::test]
async fn an_append_that_loses_the_race_takes_the_next_version() {
    let s = setup();
    let l = log(&s);
    l.commit(None, &entry("a", CommitMode::Append, &["f1"], &[]))
        .await
        .unwrap();
    // Both writers read version 0.
    let v1 = l
        .commit(Some(0), &entry("b", CommitMode::Append, &["f2"], &[]))
        .await
        .unwrap();
    let v2 = l
        .commit(Some(0), &entry("c", CommitMode::Append, &["f3"], &[]))
        .await
        .unwrap();
    assert_eq!((v1, v2), (1, 2));
    assert_eq!(names(&l.snapshot().await.unwrap()), ["f1", "f2", "f3"]);
}

#[tokio::test]
async fn an_overwrite_conflicts_with_a_later_commit() {
    let s = setup();
    let l = log(&s);
    l.commit(None, &entry("a", CommitMode::Append, &["f1"], &[]))
        .await
        .unwrap();
    l.commit(Some(0), &entry("b", CommitMode::Append, &["f2"], &[]))
        .await
        .unwrap();
    let e = l
        .commit(
            Some(0),
            &entry("c", CommitMode::Overwrite, &["f3"], &["f1"]),
        )
        .await
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivConcurrentCommit);
    assert_eq!(names(&l.snapshot().await.unwrap()), ["f1", "f2"]);
}

#[tokio::test]
async fn a_checkpoint_bounds_replay_and_equals_the_full_replay() {
    let s = setup();
    let l = log(&s);
    let mut base = None;
    for i in 0..(CHECKPOINT_EVERY + 5) {
        let v = l
            .commit(
                base,
                &entry(
                    &format!("w{i}"),
                    CommitMode::Append,
                    &[&format!("f{i:03}")],
                    &[],
                ),
            )
            .await
            .unwrap();
        base = Some(v);
    }
    assert!(
        s.inner
            .head(&Path::from(format!(
                "t/_krishiv_log/{:020}.checkpoint.json",
                CHECKPOINT_EVERY - 1
            )))
            .await
            .is_ok()
    );
    let with_checkpoint = l.snapshot().await.unwrap();
    assert_eq!(
        with_checkpoint.files.len(),
        usize::try_from(CHECKPOINT_EVERY + 5).unwrap()
    );
    // Dropping the entries the checkpoint covers does not change the result.
    for v in 0..CHECKPOINT_EVERY {
        s.inner
            .delete(&Path::from(format!("t/_krishiv_log/{v:020}.json")))
            .await
            .unwrap();
    }
    assert_eq!(l.snapshot().await.unwrap(), with_checkpoint);
}

#[tokio::test]
async fn a_gap_is_corruption() {
    let s = setup();
    let l = log(&s);
    l.commit(None, &entry("a", CommitMode::Append, &["f1"], &[]))
        .await
        .unwrap();
    l.commit(Some(0), &entry("b", CommitMode::Append, &["f2"], &[]))
        .await
        .unwrap();
    s.inner
        .delete(&Path::from(format!("t/_krishiv_log/{:020}.json", 0)))
        .await
        .unwrap();
    let e = l.snapshot().await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivCommitLogCorrupt);
}

#[tokio::test]
async fn an_entry_from_a_newer_format_is_refused() {
    let s = setup();
    s.inner
        .put(
            &Path::from("t/_krishiv_log/00000000000000000000.json"),
            br#"{"format":2}"#.to_vec().into(),
        )
        .await
        .unwrap();
    let e = log(&s).snapshot().await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivCommitLogCorrupt);
}

#[test]
fn entries_round_trip() {
    let e = entry("w", CommitMode::Overwrite, &["a/b.parquet"], &["old"]);
    assert_eq!(decode_entry(&encode_entry(&e)).unwrap(), e);
}

#[tokio::test]
async fn the_probe_passes_on_a_store_with_put_if_absent_and_leaves_no_log() {
    let s = setup();
    let l = log(&s);
    assert!(l.supports_put_if_absent("p1").await.unwrap());
    assert!(!l.exists().await.unwrap());
    assert_eq!(l.snapshot().await.unwrap(), Snapshot::default());
}
