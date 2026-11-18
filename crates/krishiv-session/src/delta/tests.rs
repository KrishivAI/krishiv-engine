// SPDX-License-Identifier: Apache-2.0
//! One snapshot per table root per query while a commit lands between two scans' opens; the
//! next query sees the commit; another credential misses the runtime's cache.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::secret::OptionValue;
use krishiv_plan::ReadSource;

use super::{DELTA, DeltaInputs, DeltaReads};
use crate::session::Session;

#[expect(clippy::unwrap_used, reason = "test helper")]
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// `session-appends` without its last commit (kept beside the table), and a function landing it.
#[expect(clippy::unwrap_used, reason = "test helper")]
fn two_commits(tmp: &Path) -> (PathBuf, impl Fn()) {
    let dir = tmp.join("t");
    copy_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/delta-4.4.1/session-appends"),
        &dir,
    );
    let last = |ext: &str| dir.join(format!("_delta_log/00000000000000000002.{ext}"));
    for ext in ["json", "crc"] {
        std::fs::rename(last(ext), dir.with_extension(ext)).unwrap();
    }
    let land = {
        let dir = dir.clone();
        move || {
            for ext in ["crc", "json"] {
                std::fs::rename(
                    dir.with_extension(ext),
                    dir.join(format!("_delta_log/00000000000000000002.{ext}")),
                )
                .unwrap();
            }
        }
    };
    (dir, land)
}

fn read(path: &str) -> ReadSource {
    ReadSource::files(
        DELTA,
        vec![OptionValue::from_user("path", path)],
        std::collections::BTreeMap::new(),
    )
}

#[test]
fn two_scans_of_a_query_pin_one_version_while_a_commit_lands() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, land) = two_commits(tmp.path());
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let cache = Arc::clone(&s.state.runtime.delta);
    let query = s.query_env();
    let first = block_on(query.catalog.delta.open(&read(&path), None)).unwrap();
    assert_eq!(first.version(), 1);
    land();
    // The second scan spells the root another way: the same table, the same version.
    let second = block_on(query.catalog.delta.open(&read(&format!("{path}/")), None)).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(second.version(), 1);
    assert_eq!(cache.misses(), 1);
    // The next query starts from the cached snapshot and sees the commit.
    let next = s.query_env();
    let newer = block_on(next.catalog.delta.open(&read(&path), None)).unwrap();
    assert_eq!(newer.version(), 2);
    assert_eq!((cache.hits(), cache.misses()), (1, 1));
    // A query under another credential identity never reads that entry.
    let other = DeltaReads::new(DeltaInputs {
        io: s.state.io.context(),
        paths: Arc::new(krishiv_exec::files::LocalPaths::new(
            krishiv_io::StoreId::new(crate::session::LOCAL_STORE),
            "/",
        )),
        runtime: s.state.runtime.executor.io_handle().clone(),
        config: Arc::new(krishiv_common::config::Config::new()),
        cache: Arc::clone(&cache),
        credential: "someone-else".to_owned(),
    });
    let theirs = block_on(other.open(&read(&path), None)).unwrap();
    assert_eq!(theirs.version(), 2);
    assert_eq!((cache.hits(), cache.misses()), (1, 2));
}

/// A query's reads dropped (its planning future dropped) end the planning-time Delta work: an
/// engine they made refuses the next kernel call.
#[test]
fn dropped_reads_cancel_their_kernel_calls() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _land) = two_commits(tmp.path());
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let query = s.query_env();
    let (store, root) = query.catalog.delta.root_of(&read(&path)).unwrap();
    let engine = query.catalog.delta.engine(store, root).unwrap();
    assert!(block_on(engine.delta_blocking(|_| ())).is_ok());
    drop(query);
    let e = block_on(engine.delta_blocking(|_| ())).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::OperationCanceled,
        "{e}"
    );
}

/// An interrupt during analysis reaches the planning-time kernel calls of reads built inside the
/// action, even while something (a frame) still holds the reads: their drop is not needed.
#[test]
fn an_interrupted_action_cancels_its_reads_kernel_calls() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _land) = two_commits(tmp.path());
    let path = dir.to_string_lossy().into_owned();
    let s = Arc::new(Session::new().unwrap());
    let actions = Arc::new(crate::action::Actions::default());
    let guard = actions.begin(1, Some("op"));
    let (tx, rx) = std::sync::mpsc::channel();
    let running = {
        let s = Arc::clone(&s);
        std::thread::spawn(move || {
            block_on(guard.run(async move {
                let query = s.query_env();
                let reads = Arc::clone(&query.catalog.delta);
                let (store, root) = reads.root_of(&read(&path))?;
                let engine = reads.engine(store, root)?;
                tx.send((reads, engine)).ok();
                std::future::pending::<Result<(), krishiv_common::error::KrishivError>>().await
            }))
        })
    };
    let (held, engine) = rx.recv().unwrap();
    assert!(block_on(engine.delta_blocking(|_| ())).is_ok());
    actions.cancel_where(crate::cancel::CancelReason::User, &|_, _| true);
    assert!(
        running.join().unwrap().is_err(),
        "the action ends with the cancel"
    );
    // The watcher runs on the I/O runtime: give it a moment, never forever.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let e = loop {
        match block_on(engine.delta_blocking(|_| ())) {
            Err(e) => break e,
            Ok(()) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Ok(()) => panic!("the interrupt never reached the planning engine"),
        }
    };
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::OperationCanceled,
        "{e}"
    );
    drop(held);
}

/// Reads built inside an action start their interrupt watcher only at the first planning
/// kernel call: an env that opens no Delta table spawns nothing (review 940176301 N1).
#[test]
fn the_interrupt_watcher_starts_at_the_first_kernel_call() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _land) = two_commits(tmp.path());
    let path = dir.to_string_lossy().into_owned();
    let s = Arc::new(Session::new().unwrap());
    let outside = s.query_env();
    assert!(
        !outside.catalog.delta.watcher_pending(),
        "no watcher outside an action"
    );
    let actions = Arc::new(crate::action::Actions::default());
    let guard = actions.begin(1, Some("op"));
    let r: Result<(), krishiv_common::error::KrishivError> = block_on(guard.run(async move {
        let unused = s.query_env();
        assert!(
            unused.catalog.delta.watcher_pending(),
            "not started by building the env"
        );
        drop(unused);
        let query = s.query_env();
        let reads = Arc::clone(&query.catalog.delta);
        let (store, root) = reads.root_of(&read(&path))?;
        assert!(reads.watcher_pending());
        let _engine = reads.engine(store.clone(), root.clone())?;
        assert!(!reads.watcher_pending(), "started by the first kernel call");
        let _again = reads.engine(store, root)?;
        Ok(())
    }));
    assert!(r.is_ok(), "{r:?}");
}

/// An action interrupted before its reads' first kernel call: that call's engine is cancelled
/// already, with no wait for the watcher task (review of a77e1ba21, N1/N2).
#[test]
fn an_interrupt_before_the_first_kernel_call_cancels_it() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _land) = two_commits(tmp.path());
    let path = dir.to_string_lossy().into_owned();
    let s = Arc::new(Session::new().unwrap());
    let actions = Arc::new(crate::action::Actions::default());
    let guard = actions.begin(1, Some("op"));
    let (built_tx, built_rx) = std::sync::mpsc::channel();
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let running = std::thread::spawn(move || {
        block_on(guard.run(async move {
            let query = s.query_env();
            let reads = Arc::clone(&query.catalog.delta);
            built_tx.send(()).ok();
            go_rx.recv().ok();
            let (store, root) = reads.root_of(&read(&path))?;
            // Built after the interrupt, within the same poll of the action's work (nothing
            // above awaits), so the action itself has not seen the cancel yet.
            let engine = reads.engine(store, root)?;
            // On this thread, at once: the token fired inside `engine`, not on the watcher task.
            Ok((engine, reads.cancelled()))
        }))
    });
    built_rx.recv().unwrap();
    actions.cancel_where(crate::cancel::CancelReason::User, &|_, _| true);
    go_tx.send(()).unwrap();
    let (engine, cancelled) = running
        .join()
        .unwrap()
        .expect("the work ends in its first poll");
    assert!(cancelled, "the token fired before engine() returned");
    // At once: no sleep, no retry; the watcher task need not have run.
    let e = block_on(engine.delta_blocking(|_| ()))
        .expect_err("the interrupt reaches the first kernel call");
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::OperationCanceled,
        "{e}"
    );
}

/// `krishiv.delta.snapshotCache.maxBytes` reaches the runtime's cache: under a bound that holds
/// a snapshot but not its listing, the table is cached without its files (re-review N6).
#[test]
fn the_runtime_cache_takes_its_byte_bound_from_the_config() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _land) = two_commits(tmp.path());
    let path = dir.to_string_lossy().into_owned();
    let bound = krishiv_connectors_delta::table::cache::ENTRY_BYTES.to_string();
    let rt = crate::runtime::EngineRuntime::builder()
        .config("krishiv.delta.snapshotCache.maxBytes", &bound)
        .build()
        .unwrap();
    let s = Session::builder().runtime(rt).build().unwrap();
    let cache = Arc::clone(&s.state.runtime.delta);
    let query = s.query_env();
    let reads = &query.catalog.delta;
    let table = block_on(reads.open(&read(&path), None)).unwrap();
    assert_eq!(table.version(), 1);
    let (store, root) = reads.root_of(&read(&path)).unwrap();
    let key = krishiv_connectors_delta::table::CacheKey {
        root: root.as_str().to_owned(),
        store,
        credential: reads.credential.clone(),
    };
    assert_eq!(cache.cached_version(&key), Some(1));
    assert!(!cache.keeps_listing(&key));
}

/// Two reads of one root with different `fs.*` options reach it through different stores (the
/// options' hash in the store id), so the query-local key of each is its own (review: the key is
/// the store and the root).
#[test]
fn reads_of_one_root_through_different_stores_have_their_own_keys() {
    use krishiv_io::StoreId;
    let root = url::Url::parse("s3://bucket/t/").unwrap();
    let plain = StoreId::new("s3://bucket");
    let other = StoreId::new("s3://bucket#00000000000000ab");
    let key = super::reads::table_key;
    assert_ne!(key(&plain, &root), key(&other, &root));
    assert_eq!(key(&other, &root), key(&other.clone(), &root.clone()));
    let elsewhere = url::Url::parse("s3://bucket/u/").unwrap();
    assert_ne!(key(&plain, &root), key(&plain, &elsewhere));
}

#[test]
fn a_declared_schema_on_a_delta_read_is_refused_by_name() {
    use krishiv_types::{DataType, StructField, StructType};
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _land) = two_commits(tmp.path());
    let s = Session::new().unwrap();
    let source = read(&dir.to_string_lossy()).with_schema(StructType::new([StructField::new(
        "id",
        DataType::Long,
        true,
    )]));
    let df = crate::dataframe::DataFrame::new(
        s,
        Arc::new(krishiv_plan::LogicalPlan::new(
            krishiv_plan::PlanNode::Read(source),
        )),
    );
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(e.to_string().contains("user-specified schema"), "{e}");
}
