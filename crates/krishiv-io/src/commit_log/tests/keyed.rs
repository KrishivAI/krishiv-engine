// SPDX-License-Identifier: Apache-2.0
//! Idempotency keys in the commit log (E34).

use super::*;

fn keyed(write: &str, key: &str) -> LogEntry {
    LogEntry {
        idempotency_key: Some(key.to_owned()),
        ..entry(write, CommitMode::Append, &[write], &[])
    }
}

#[tokio::test]
async fn a_second_commit_with_the_same_key_writes_nothing_and_names_the_first() {
    let s = setup();
    let l = log(&s);
    assert_eq!(
        l.commit_keyed(None, &keyed("a", "k1")).await.unwrap(),
        Keyed::Committed(0)
    );
    // A retry of the statement: a new write id and files, the same key.
    assert_eq!(
        l.commit_keyed(Some(0), &keyed("b", "k1")).await.unwrap(),
        Keyed::Replayed(0)
    );
    // Another key appends.
    assert_eq!(
        l.commit_keyed(Some(0), &keyed("c", "k2")).await.unwrap(),
        Keyed::Committed(1)
    );
    let snap = l.snapshot().await.unwrap();
    assert_eq!(snap.files.keys().collect::<Vec<_>>(), ["a", "c"]);
    assert_eq!(snap.idempotency_keys.get("k1"), Some(&0));
}

#[tokio::test]
async fn a_key_survives_a_checkpoint() {
    let s = setup();
    let l = log(&s);
    l.commit_keyed(None, &keyed("first", "keep")).await.unwrap();
    for i in 1..CHECKPOINT_EVERY {
        l.commit(
            Some(i - 1),
            &entry(&format!("w{i}"), CommitMode::Append, &[], &[]),
        )
        .await
        .unwrap();
    }
    let (_, checkpoints) = l.versions().await.unwrap();
    assert!(checkpoints.is_some(), "a checkpoint was written");
    assert_eq!(
        l.commit_keyed(Some(CHECKPOINT_EVERY - 1), &keyed("again", "keep"))
            .await
            .unwrap(),
        Keyed::Replayed(0)
    );
}

#[tokio::test]
async fn an_entry_without_a_key_commits_as_before() {
    let s = setup();
    let l = log(&s);
    let e = entry("a", CommitMode::Append, &["a"], &[]);
    assert_eq!(l.commit_keyed(None, &e).await.unwrap(), Keyed::Committed(0));
    assert_eq!(
        l.commit_keyed(Some(0), &e).await.unwrap(),
        Keyed::Committed(1)
    );
}

/// A retry that checked the key before a concurrent retry committed it finds the key when its
/// slot is taken, instead of appending at the next version.
#[tokio::test]
async fn an_append_that_loses_its_slot_looks_for_its_key_again() {
    let s = setup();
    let l = log(&s);
    l.commit_keyed(None, &keyed("a", "k")).await.unwrap();
    // The stale retry: its base is the empty log, so version 0 is taken.
    assert_eq!(
        l.commit_inner(None, &keyed("b", "k"), true).await.unwrap(),
        Keyed::Replayed(0)
    );
    let snap = l.snapshot().await.unwrap();
    assert_eq!(snap.files.keys().collect::<Vec<_>>(), ["a"]);
}
