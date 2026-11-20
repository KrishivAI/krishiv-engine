// SPDX-License-Identifier: Apache-2.0
//! Faults on the merge-on-read commit path (`row_delta.rs` through
//! `commit.rs`, design note §7.4): `sink_tests`' catalog behind [`Faulty`],
//! which answers chosen requests as unknown or refused and can let another
//! writer in before a load. A known failure deletes the statement's Puffin,
//! position delete and data files; an unknown answer keeps them whether the
//! write then lands or ends in a conflict; an aborted task leaves nothing.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use iceberg::spec::{Operation, TableMetadata};
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::{ErrorClass, InternalError, KrishivError};
use krishiv_exec::planner::RowLevelOperation;
use krishiv_memory::{ConsumerHandle, MemoryConfig, MemoryPool, TreePool};

use super::commit::{CommitOutcome, CommitTable};
use super::cow_tests::{
    add_deletes, append, create, dvs_in_one_puffin, position_delete_scoped, summary,
};
use super::mor_tests::{Mor, Task, files_ending, only_data_file};
use super::provider_tests::{Env, env};
use super::sink_tests::{Catalog, expected, parquet_files, range, read_back};
use super::table::IcebergTable;
use crate::plan::VersionSpec;

/// How one request is answered.
#[derive(Debug, Clone, Copy)]
enum Fault {
    /// A 503 before the table saw it.
    UnknownBeforeApply,
    /// Applied, the answer lost.
    LostResponse,
    /// A 401.
    Refused,
}

type Hook = Box<dyn FnOnce() + Send>;

/// [`Catalog`] with scripted answers and a hook run before the first load.
struct Faulty {
    inner: Arc<Catalog>,
    script: Mutex<VecDeque<Fault>>,
    before_load: Mutex<Option<Hook>>,
}

#[async_trait]
impl CommitTable for Faulty {
    fn name(&self) -> String {
        self.inner.name()
    }

    async fn commit(
        &self,
        requirements: Vec<TableRequirement>,
        updates: Vec<TableUpdate>,
    ) -> CommitOutcome {
        let fault = self.script.lock().unwrap().pop_front();
        match fault {
            None => self.inner.commit(requirements, updates).await,
            Some(Fault::UnknownBeforeApply) => CommitOutcome::Unknown("503".to_owned()),
            Some(Fault::LostResponse) => {
                let out = self.inner.commit(requirements, updates).await;
                assert!(matches!(out, CommitOutcome::Committed(_)), "{out:?}");
                CommitOutcome::Unknown("a lost response".to_owned())
            }
            Some(Fault::Refused) => CommitOutcome::Failed(
                InternalError {
                    message: "401 Unauthorized".to_owned(),
                }
                .into(),
            ),
        }
    }

    async fn load(&self, all: bool) -> Result<TableMetadata, KrishivError> {
        let hook = self.before_load.lock().unwrap().take();
        if let Some(h) = hook {
            h();
        }
        self.inner.load(all).await
    }
}

/// Status checks and retries without real waits.
const FAST: &[(&str, &str)] = &[
    ("commit.status-check.min-wait-ms", "1"),
    ("commit.status-check.max-wait-ms", "1"),
    ("commit.retry.min-wait-ms", "1"),
    ("commit.retry.max-wait-ms", "1"),
];

/// A statement over `catalog` whose commits go through `faults`.
fn faulty<'a>(
    env: &'a Env,
    catalog: &Arc<Catalog>,
    faults: &[Fault],
    op: RowLevelOperation,
) -> (Mor<'a>, Arc<Faulty>) {
    let f = Arc::new(Faulty {
        inner: Arc::clone(catalog),
        script: Mutex::new(faults.iter().copied().collect()),
        before_load: Mutex::new(None),
    });
    let table = env
        .rt
        .block_on(IcebergTable::load(&env.reg, &catalog.location(), "db", "t"))
        .unwrap()
        .with_committer(Arc::clone(&f) as Arc<dyn CommitTable>);
    let p = env.provider(&table, &VersionSpec::Current).unwrap();
    (Mor::over(env, p, op), f)
}

/// A v3 table of rows 0..10 and 10..20 (two data files).
fn table(env: &Env) -> Arc<Catalog> {
    let t = create(env, 3, false, FAST);
    let catalog = Catalog::new(&t);
    append(env, &catalog, range(0, 10));
    append(env, &catalog, range(10, 20));
    catalog
}

/// A 401 to the first request: a known failure; the statement's Puffin file
/// and data files are deleted, the table reads as before.
#[test]
fn a_known_failure_deletes_the_dvs_and_data_files() {
    let env = env();
    let catalog = table(&env);
    let data_before = parquet_files(&env, false);
    let (mor, _) = faulty(&env, &catalog, &[Fault::Refused], RowLevelOperation::Update);
    let err = mor.update("id IN (2, 12)", "'u'").unwrap_err();
    assert!(err.to_string().contains("401"), "{err}");
    assert!(files_ending(&env, ".puffin").is_empty());
    assert_eq!(parquet_files(&env, true), data_before);
    assert_eq!(read_back(&env, &catalog), expected(0..20));
}

/// A 503 before the table saw the request, then another writer adds a DV
/// for the same data file: the rebase's validation conflicts, and since an
/// answer was unknown nothing is deleted (§7.4).
#[test]
fn an_unknown_answer_then_a_concurrent_dv_keeps_the_files() {
    let env = env();
    let t = create(&env, 3, false, FAST);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    // The data file holding `id = 3`, by identity: paths carry a random
    // attempt id, so their order is random.
    let target = only_data_file(&env, &catalog);
    append(&env, &catalog, range(10, 20));
    let (mor, f) = faulty(
        &env,
        &catalog,
        &[Fault::UnknownBeforeApply],
        RowLevelOperation::Delete,
    );
    let ids = mor.ids("id = 3");
    // The other writer's DV, staged now and made visible before the probe.
    let before = catalog.state();
    add_deletes(&env, &catalog, |files| {
        assert!(files.contains(&target), "{files:?}");
        dvs_in_one_puffin(&env, "foreign.puffin", std::slice::from_ref(&target), &[0])
    });
    let after = catalog.state();
    catalog.restore(before);
    let cat = Arc::clone(&catalog);
    *f.before_load.lock().unwrap() = Some(Box::new(move || cat.restore(after)));
    let err = mor
        .run(vec![Task {
            deletes: ids,
            ..Task::default()
        }])
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivIcebergCommitConflict,
        "{err}"
    );
    assert!(
        err.to_string().contains("Found concurrently added DV for"),
        "{err}"
    );
    assert!(err.to_string().contains("files were kept"), "{err}");
    assert_eq!(files_ending(&env, "-deletes.puffin").len(), 1);
}

/// A lost answer to a request the table applied: the probe finds the
/// snapshot; the Puffin file and the new data file are kept and read.
#[test]
fn an_unknown_answer_then_the_landed_row_delta_commits() {
    let env = env();
    let catalog = table(&env);
    let (mor, _) = faulty(
        &env,
        &catalog,
        &[Fault::LostResponse],
        RowLevelOperation::Update,
    );
    let m = mor.update("id = 12", "'u'").unwrap();
    assert_eq!(summary(&m).0, Operation::Overwrite);
    let mut want = expected(0..20);
    want[12] = "12|0|u".to_owned();
    assert_eq!(read_back(&env, &catalog), want);
    assert_eq!(files_ending(&env, "-deletes.puffin").len(), 1);
    assert_eq!(catalog.metadata().snapshots().count(), 3);
}

/// A v2 statement refused with a known failure deletes its position delete
/// files.
#[test]
fn a_known_failure_deletes_the_position_delete_files() {
    let env = env();
    let t = create(&env, 2, false, FAST);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    let (mor, _) = faulty(&env, &catalog, &[Fault::Refused], RowLevelOperation::Delete);
    let err = mor.delete("id = 4").unwrap_err();
    assert!(err.to_string().contains("401"), "{err}");
    assert!(files_ending(&env, "-deletes.parquet").is_empty());
    assert_eq!(read_back(&env, &catalog), expected(0..10));
}

/// A task aborted after it wrote rows leaves no data file; a task dropped
/// before `finish` leaves none either.
#[test]
fn an_aborted_task_leaves_nothing() {
    let env = env();
    let t = create(&env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 3));
    let before = parquet_files(&env, false);
    let mor = Mor::open(&env, &catalog, RowLevelOperation::Merge);
    let w = mor.writer();
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    env.rt.block_on(async {
        let mut task = w.task(0, ConsumerHandle::register(Arc::clone(&pool), "a", None));
        task.insert(&mor.new_rows(5, 7)).await.unwrap();
        task.abort().await;
        let mut task = w.task(1, ConsumerHandle::register(Arc::clone(&pool), "b", None));
        task.insert(&mor.new_rows(8, 9)).await.unwrap();
        drop(task);
    });
    assert_eq!(parquet_files(&env, true), before);
    assert_eq!(catalog.metadata().snapshots().count(), 1);
}

/// Every file under the test's directory.
fn all_files(env: &Env) -> std::collections::BTreeSet<String> {
    files_ending(env, "").into_iter().collect()
}

/// Waits (up to 5 s) for background deletes to bring the directory back to
/// `before` and keep it there for 100 ms (a write in flight may land late);
/// returns what is left over.
fn settle(env: &Env, before: &std::collections::BTreeSet<String>) -> Vec<String> {
    let extra = || {
        all_files(env)
            .difference(before)
            .cloned()
            .collect::<Vec<_>>()
    };
    let mut quiet = 0;
    let mut left = extra();
    for _ in 0..1_000 {
        left = extra();
        quiet = if left.is_empty() { quiet + 1 } else { 0 };
        if quiet >= 20 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    // A cleanup that took a file of the baseline with the leftovers is no cleanup.
    let lost: Vec<_> = before.difference(&all_files(env)).cloned().collect();
    assert!(lost.is_empty(), "baseline files deleted: {lost:?}");
    left
}

/// Polls a `DELETE` task's `finish` over `ids` until it is ready (its
/// output), or after `polls` polls (`Some`), or (`None`) once at least
/// `files` files beyond `before` exist; then drops it.
fn drive(
    env: &Env,
    w: &Arc<super::row_delta::MergeOnReadWrite>,
    ids: &[super::mor_tests::RowId],
    stop: Option<usize>,
    files: usize,
    before: &std::collections::BTreeSet<String>,
) -> Option<super::row_delta::TaskOutput> {
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let mut task = w.task(0, ConsumerHandle::register(pool, "t", None));
    for (f, p) in ids {
        task.delete(f, *p).unwrap();
    }
    env.rt.block_on(async {
        let mut finish = std::pin::pin!(task.finish());
        let mut n = 0;
        loop {
            if let std::task::Poll::Ready(out) = futures::poll!(finish.as_mut()) {
                return Some(out.unwrap());
            }
            n += 1;
            let done = match stop {
                Some(max) => n >= max,
                None => all_files(env).difference(before).count() >= files,
            };
            if done {
                return None;
            }
            tokio::task::yield_now().await;
        }
    })
}

/// Review 36c1 M3: a `DELETE` task's `finish` over two data files (v2: two
/// position delete files; v3: one Puffin file; each absorbing an old
/// file-scoped delete of the first file, so it reads before it writes),
/// dropped after a growing number of polls, yielding in between, leaves no file
/// wherever it stopped: `finish`'s guard deletes the delete files written,
/// and a `put` dropped in flight deletes its file once the store is done.
/// Then, deterministically, a drop once two new files exist (v2: the first
/// file written, the second's upload begun). Once `finish` completes, the
/// output never committed is aborted (`MergeOnReadWrite::abort`) and leaves
/// no file either. (A task's data writer dropped while it closes a file is
/// `write.rs`'s documented limit, so new rows are tested through
/// `an_aborted_statement_deletes_its_finished_tasks_files`.)
#[test]
fn a_dropped_finish_and_an_uncommitted_output_leave_no_file() {
    for version in [2_u8, 3] {
        let env = env();
        let t = create(&env, version, false, &[]);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 10));
        let first = only_data_file(&env, &catalog);
        append(&env, &catalog, range(10, 20));
        add_deletes(&env, &catalog, |_| {
            if version == 3 {
                dvs_in_one_puffin(&env, "old.puffin", std::slice::from_ref(&first), &[0])
            } else {
                vec![position_delete_scoped(&env, "old.parquet", &first, 0, true)]
            }
        });
        let before = all_files(&env);
        let mor = Mor::open(&env, &catalog, RowLevelOperation::Delete);
        let ids = mor.ids("id IN (2, 3, 12)");
        let w = mor.writer();
        let none = Vec::<String>::new();
        let mut polls = 1;
        loop {
            let out = drive(&env, &w, &ids, Some(polls), 0, &before);
            if let Some(out) = out {
                assert_eq!(out.files.len(), if version == 3 { 1 } else { 2 });
                assert!(all_files(&env).len() > before.len());
                env.rt.block_on(w.abort(vec![out]));
                assert_eq!(settle(&env, &before), none, "v{version}");
                break;
            }
            assert_eq!(
                settle(&env, &before),
                none,
                "v{version} after {polls} polls"
            );
            // Every early stop, then sparser (a put spins many polls).
            polls += 1 + polls / 8;
        }
        let mut caught = version == 3;
        for _ in 0..if caught { 0 } else { 100 } {
            match drive(&env, &w, &ids, None, 2, &before) {
                Some(out) => env.rt.block_on(w.abort(vec![out])),
                None => caught = true,
            }
            assert_eq!(settle(&env, &before), none, "v{version}");
            if caught {
                break;
            }
        }
        assert!(caught, "v{version}: no drop caught two files");
        assert_eq!(catalog.metadata().snapshots().count(), 3);
    }
}

/// Review 36c1 M3: two tasks, the first finished, the second aborted, then
/// the statement aborted before its commit (Java's
/// `PositionDeltaBatchWrite.abort`): no file is left, for DVs and v2
/// position delete files alike.
#[test]
fn an_aborted_statement_deletes_its_finished_tasks_files() {
    for version in [2_u8, 3] {
        let env = env();
        let t = create(&env, version, false, &[]);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 10));
        let before = all_files(&env);
        let mor = Mor::open(&env, &catalog, RowLevelOperation::Merge);
        let ids = mor.ids("id = 4");
        let rows = mor.updated("id = 4", "'m'");
        let w = mor.writer();
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        env.rt.block_on(async {
            let mut first = w.task(0, ConsumerHandle::register(Arc::clone(&pool), "a", None));
            for (f, p) in &ids {
                first.delete(f, *p).unwrap();
            }
            for b in &rows {
                first.insert(b).await.unwrap();
            }
            let out = first.finish().await.unwrap();
            assert_eq!(out.files.len(), 1, "v{version}");
            assert_eq!(out.data.len(), 1, "v{version}");
            let mut second = w.task(1, ConsumerHandle::register(Arc::clone(&pool), "b", None));
            second.insert(&mor.new_rows(30, 32)).await.unwrap();
            second.abort().await;
            w.abort(vec![out]).await;
        });
        assert_eq!(settle(&env, &before), Vec::<String>::new(), "v{version}");
        assert_eq!(read_back(&env, &catalog), expected(0..10));
    }
}
