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

/// A store that, at the end of its `n`th listing, writes `entry` at `path` (the listing itself
/// does not see it): another writer's commit landing just after a reader looked.
#[derive(Debug)]
struct CommitAfterList {
    inner: Arc<InMemory>,
    n: u32,
    lists: std::sync::Mutex<u32>,
    path: Path,
    entry: bytes::Bytes,
}

impl std::fmt::Display for CommitAfterList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CommitAfterList")
    }
}

#[async_trait::async_trait]
impl ObjectStore for CommitAfterList {
    async fn put_opts(
        &self,
        location: &Path,
        payload: object_store::PutPayload,
        opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures::stream::BoxStream<'static, object_store::Result<Path>>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>> {
        use futures::StreamExt;
        let listing = self.inner.list(prefix);
        let hit = {
            let mut lists = self.lists.lock().unwrap();
            *lists += 1;
            *lists == self.n
        };
        if !hit {
            return listing;
        }
        let (inner, path, entry) = (
            Arc::clone(&self.inner),
            self.path.clone(),
            self.entry.clone(),
        );
        let commit = futures::stream::once(async move {
            inner.put(&path, entry.into()).await.unwrap();
        })
        .filter_map(|()| std::future::ready(None));
        listing.chain(commit).boxed()
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

/// An append that lost its slot to an unrelated commit checks for its key; a concurrent retry of
/// the same key commits right after that check. The next slot comes from the checked snapshot, so
/// the append loses that slot too and finds the key, instead of skipping past it and committing
/// the key twice.
#[tokio::test]
async fn a_same_key_commit_after_the_check_is_not_skipped_over() {
    let inner = Arc::new(InMemory::new());
    let winner = keyed("a", "k");
    let store = Arc::new(CommitAfterList {
        inner: Arc::clone(&inner),
        // 1: commit_keyed's first check; 2: the check after losing version 0.
        n: 2,
        lists: std::sync::Mutex::new(0),
        path: Path::from("t/_krishiv_log/00000000000000000001.json"),
        entry: encode_entry(&winner).into(),
    });
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("memory://race");
    sched.register_store(id.clone(), store as Arc<dyn ObjectStore>);
    let io = sched.context();
    let l = CommitLog::new(&io, id, Path::from("t"), "memory://race/t".to_owned());
    // The unrelated commit at version 0, written beneath the log (no listing).
    inner
        .put(
            &Path::from("t/_krishiv_log/00000000000000000000.json"),
            encode_entry(&entry("x", CommitMode::Append, &["x"], &[])).into(),
        )
        .await
        .unwrap();
    assert_eq!(
        l.commit_keyed(None, &keyed("b", "k")).await.unwrap(),
        Keyed::Replayed(1)
    );
    let snap = l.snapshot().await.unwrap();
    assert_eq!(snap.version, Some(1));
    assert_eq!(snap.files.keys().collect::<Vec<_>>(), ["a", "x"]);
}

/// A put that landed but reported an error (its answer lost, the read-back failed too) finds its
/// own key afterwards: that is this attempt's commit, not a replay of an earlier one whose cleanup
/// would delete the files the log now lists.
#[tokio::test(start_paused = true)]
async fn an_own_landed_commit_found_after_an_error_is_committed_not_replayed() {
    use crate::io::FaultStore;
    let inner = Arc::new(InMemory::new());
    let e = keyed("w", "k");
    // The entry lands just after commit_keyed's first check (listing 1)...
    let hook = Arc::new(CommitAfterList {
        inner: Arc::clone(&inner),
        n: 1,
        lists: std::sync::Mutex::new(0),
        path: Path::from("t/_krishiv_log/00000000000000000000.json"),
        entry: encode_entry(&e).into(),
    });
    let fault = FaultStore::new(hook as Arc<dyn ObjectStore>);
    let sched = IoScheduler::new(IoConfig {
        max_retries: 0,
        ..IoConfig::default()
    });
    let id = StoreId::new("memory://own");
    sched.register_store(id.clone(), Arc::clone(&fault) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let l = CommitLog::new(&io, id, Path::from("t"), "memory://own/t".to_owned());
    // ...and the put's answer is lost with no retry left and a failed read-back: an error.
    fault.fail_next_puts(1);
    fault.fail_next_gets(1, None);
    assert_eq!(l.commit_keyed(None, &e).await.unwrap(), Keyed::Committed(0));
    assert_eq!(l.snapshot().await.unwrap().version, Some(0));
}
