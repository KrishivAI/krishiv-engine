// SPDX-License-Identifier: Apache-2.0
//! Tests of `expire_snapshots` (`maintenance/expire.rs`, PR-36e): the retention rule over real
//! tables, the metadata commit and the cleanup of the files only the removed snapshots reached.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64};

use krishiv_common::error::ErrorClass;
use krishiv_exec::planner::RowLevelOperation;

use super::commit_tests::TestClock;
use super::cow_tests::{Dml, append, create};
use super::maintenance::expire::{Defaults, ExpireArgs, expire, plan};
use super::provider_tests::{Env, env};
use super::sink_tests::{Catalog, expected, range, read_back};

const FAR: i64 = i64::MAX / 4;

fn clock() -> TestClock {
    TestClock {
        now: AtomicI64::new(FAR),
        waits: Mutex::new(Vec::new()),
        max_jitter: AtomicBool::new(false),
    }
}

/// A v2 table with four appends and a copy-on-write delete: five snapshots, the first data file
/// replaced by the delete.
fn five_snapshots(env: &Env) -> std::sync::Arc<Catalog> {
    let t = create(env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    for i in 0..4 {
        append(env, &catalog, range(i * 10, i * 10 + 10));
    }
    Dml::open(env, &catalog, RowLevelOperation::Delete)
        .delete("id < 5")
        .unwrap();
    assert_eq!(catalog.metadata().snapshots().count(), 5);
    catalog
}

fn args(retain_last: Option<i64>) -> ExpireArgs {
    ExpireArgs {
        older_than_ms: Some(FAR),
        retain_last,
        ..ExpireArgs::default()
    }
}

#[test]
fn the_plan_keeps_the_newest_snapshots_it_is_told_to() {
    let env = env();
    let catalog = five_snapshots(&env);
    let meta = catalog.metadata();
    let defaults = Defaults::of(&meta, FAR);
    let p = plan(&meta, &defaults, &args(Some(1))).unwrap();
    assert_eq!(p.snapshots.len(), 4);
    assert!(!p.snapshots.contains(&meta.current_snapshot_id().unwrap()));
    let p = plan(&meta, &defaults, &args(Some(3))).unwrap();
    assert_eq!(p.snapshots.len(), 2);
    // Nothing is older than the cutoff: nothing is removed.
    let young = ExpireArgs {
        older_than_ms: Some(0),
        ..ExpireArgs::default()
    };
    assert!(plan(&meta, &defaults, &young).unwrap().is_empty());
}

#[test]
fn an_explicit_snapshot_a_ref_names_is_refused() {
    let env = env();
    let catalog = five_snapshots(&env);
    let meta = catalog.metadata();
    let head = meta.current_snapshot_id().unwrap();
    let explicit = ExpireArgs {
        snapshot_ids: vec![head, 42],
        ..ExpireArgs::default()
    };
    let e = plan(&meta, &Defaults::of(&meta, FAR), &explicit).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivIcebergProcedureArgument);
    assert!(e.to_string().contains("Still referenced by refs"), "{e}");
}

#[test]
fn retain_last_below_one_and_a_disabled_gc_are_refused() {
    let env = env();
    let catalog = five_snapshots(&env);
    let e = env
        .rt
        .block_on(expire(
            &*catalog,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(0)),
        ))
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivIcebergProcedureArgument);
    assert!(e.to_string().contains("must be at least 1"), "{e}");

    let env = super::provider_tests::env();
    let t = create(&env, 2, false, &[("gc.enabled", "false")]);
    let gc_off = Catalog::new(&t);
    let e = env
        .rt
        .block_on(expire(
            &*gc_off,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(1)),
        ))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergProcedureGcDisabled
    );
}

#[test]
fn expire_commits_and_deletes_only_what_no_kept_snapshot_reaches() {
    let env = env();
    let catalog = five_snapshots(&env);
    let before = catalog.metadata();
    let done = env
        .rt
        .block_on(expire(
            &*catalog,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(1)),
        ))
        .unwrap();
    let after = catalog.metadata();
    assert_eq!(after.snapshots().count(), 1);
    assert_eq!(after.current_snapshot_id(), before.current_snapshot_id());
    let d = done.deleted;
    // Each removed snapshot's manifest list; the first data file the delete replaced.
    assert_eq!(d.manifest_lists, 4);
    assert_eq!(d.data_files, 1);
    // Appends merge their manifests, so fewer manifests than snapshots were reached.
    assert!(d.manifests >= 1, "{d:?}");
    // The table still reads the same rows.
    assert_eq!(read_back(&env, &catalog), expected(5..40));
    // A second run finds nothing to remove and sends no request.
    let again = env
        .rt
        .block_on(expire(
            &*catalog,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(1)),
        ))
        .unwrap();
    assert_eq!(
        again.deleted,
        super::maintenance::expire::Deleted::default()
    );
}

#[test]
fn retain_last_two_keeps_two_snapshots() {
    let env = env();
    let catalog = five_snapshots(&env);
    let done = env
        .rt
        .block_on(expire(
            &*catalog,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(2)),
        ))
        .unwrap();
    assert!(done.deleted.manifest_lists >= 3);
    assert_eq!(catalog.metadata().snapshots().count(), 2);
}
