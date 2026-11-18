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
use super::cow_tests::{add_deletes, append, create, dvs_in_one_puffin, summary};
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
    mor.delete("id = 4").unwrap_err();
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
