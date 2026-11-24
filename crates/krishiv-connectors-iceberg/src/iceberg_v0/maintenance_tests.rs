// SPDX-License-Identifier: Apache-2.0
//! Tests of `expire_snapshots` (`maintenance/expire.rs`, PR-36e): the retention rule over real
//! tables, the metadata commit and the cleanup of the files only the removed snapshots reached.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64};

use async_trait::async_trait;
use iceberg::spec::TableMetadata;
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::planner::RowLevelOperation;

use super::commit::{CommitOutcome, CommitTable};
use super::commit_tests::TestClock;
use super::cow_tests::{Dml, append, create};
use super::maintenance::expire::{Defaults, ExpireArgs, expire, plan};
use super::provider_tests::{Env, env};
use super::sink_tests::{Catalog, expected, range, read_back};
use iceberg::spec::{
    SnapshotReference, SnapshotRetention, TableMetadataBuilder, Transform, UnboundPartitionSpec,
};

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

    // Java's `Boolean.parseBoolean`: anything but `true` (any case) turns GC off.
    for value in ["false", "FALSE", "no"] {
        let env = super::provider_tests::env();
        let t = create(&env, 2, false, &[("gc.enabled", value)]);
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
            ErrorClass::KrishivIcebergProcedureGcDisabled,
            "{value}"
        );
    }
}

/// A request that removes snapshots asserts every ref as planned, `main` included, so a
/// concurrent rollback onto a removed snapshot conflicts.
#[test]
fn a_snapshot_removal_asserts_every_ref() {
    let env = env();
    let catalog = five_snapshots(&env);
    let meta = catalog.metadata();
    let newest = chain(&meta)[0];
    let meta = edit(meta, |b| {
        b.set_ref(
            "t1",
            SnapshotReference::new(
                newest,
                SnapshotRetention::Tag {
                    max_ref_age_ms: None,
                },
            ),
        )
        .unwrap()
    });
    let p = plan(&meta, &Defaults::of(&meta, FAR), &args(Some(1))).unwrap();
    assert!(!p.snapshots.is_empty());
    let (requirements, _) = super::maintenance::expire::request(&meta, &p).unwrap();
    let mut asserted: Vec<(String, Option<i64>)> = requirements
        .iter()
        .filter_map(|r| match r {
            TableRequirement::RefSnapshotIdMatch { r#ref, snapshot_id } => {
                Some((r#ref.clone(), *snapshot_id))
            }
            _ => None,
        })
        .collect();
    asserted.sort();
    assert_eq!(
        asserted,
        [
            ("main".to_owned(), meta.current_snapshot_id()),
            ("t1".to_owned(), Some(newest)),
        ]
    );
}

/// A table renamed away and replaced at its name during the retries is another table: the expire
/// stops, sends nothing more and deletes nothing.
#[test]
fn a_replaced_table_stops_the_expire() {
    struct Replaced {
        first: std::sync::Arc<Catalog>,
        other: std::sync::Arc<Catalog>,
        loads: std::sync::atomic::AtomicUsize,
        commits: std::sync::atomic::AtomicUsize,
        unknown: bool,
        /// The commit lands in `first`; the reload after it finds `other`.
        land: bool,
    }
    #[async_trait]
    impl CommitTable for Replaced {
        fn name(&self) -> String {
            self.first.name()
        }
        async fn commit(&self, r: Vec<TableRequirement>, u: Vec<TableUpdate>) -> CommitOutcome {
            self.commits
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.land {
                self.first.commit(r, u).await
            } else if self.unknown {
                CommitOutcome::Unknown("503 service unavailable".to_owned())
            } else {
                CommitOutcome::Conflict { stale: false }
            }
        }
        async fn load(&self, all: bool) -> Result<TableMetadata, KrishivError> {
            self.load_located(all).await.map(|(m, _)| m)
        }
        async fn load_located(
            &self,
            all: bool,
        ) -> Result<(TableMetadata, Option<String>), KrishivError> {
            if self.loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                self.first.load_located(all).await
            } else {
                self.other.load_located(all).await
            }
        }
    }
    // A conflict's reload, and an unknown outcome's probe: the replacement is never planned on
    // nor taken for the outcome.
    // And the reload after a commit that landed: no file is deleted from what it finds.
    for (unknown, land) in [(false, false), (true, false), (false, true)] {
        let env = env();
        let first = five_snapshots(&env);
        let other_env = super::provider_tests::env();
        let other = five_snapshots(&other_env);
        assert_ne!(first.metadata().uuid(), other.metadata().uuid());
        let table = Replaced {
            first,
            other,
            loads: std::sync::atomic::AtomicUsize::new(0),
            commits: std::sync::atomic::AtomicUsize::new(0),
            unknown,
            land,
        };
        let e = env
            .rt
            .block_on(expire(
                &table,
                &env.reg.file_io(),
                &clock(),
                1 << 30,
                &args(Some(1)),
            ))
            .unwrap_err();
        let want = if unknown {
            ErrorClass::KrishivIcebergExpireStateUnknown
        } else {
            ErrorClass::KrishivCatalogTableUnsupported
        };
        assert_eq!(e.error_class(), want, "{e}");
        assert!(e.to_string().contains("replaced"), "{e}");
        assert_eq!(table.commits.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
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
    // Five snapshots, two kept: exactly the three removed snapshots' manifest lists go.
    assert_eq!(done.deleted.manifest_lists, 3, "{:?}", done.deleted);
    assert_eq!(catalog.metadata().snapshots().count(), 2);
}

/// The snapshot ids from the current one back to the first, newest first.
fn chain(meta: &TableMetadata) -> Vec<i64> {
    let mut out = Vec::new();
    let mut at = meta.current_snapshot_id();
    while let Some(id) = at {
        out.push(id);
        at = meta.snapshot_by_id(id).and_then(|s| s.parent_snapshot_id());
    }
    out
}

fn edit(
    meta: TableMetadata,
    f: impl FnOnce(TableMetadataBuilder) -> TableMetadataBuilder,
) -> TableMetadata {
    f(TableMetadataBuilder::new_from_metadata(meta, None))
        .build()
        .unwrap()
        .metadata
}

/// `main` at the oldest snapshot, a tag at the third: the tag's ancestors are not referenced by
/// it (Java's `unreferencedSnapshotsToRetain`), so the young second snapshot stays.
#[test]
fn a_tag_references_only_its_head() {
    let env = env();
    let catalog = five_snapshots(&env);
    let meta = catalog.metadata();
    let ids = chain(&meta);
    let (first, second, third) = (ids[4], ids[3], ids[2]);
    let meta = edit(meta, |b| {
        b.set_ref(
            "main",
            SnapshotReference::new(
                first,
                SnapshotRetention::Branch {
                    min_snapshots_to_keep: None,
                    max_snapshot_age_ms: None,
                    max_ref_age_ms: None,
                },
            ),
        )
        .unwrap()
        .set_ref(
            "t1",
            SnapshotReference::new(
                third,
                SnapshotRetention::Tag {
                    max_ref_age_ms: None,
                },
            ),
        )
        .unwrap()
    });
    let young = ExpireArgs {
        older_than_ms: meta.snapshot_by_id(second).map(|s| s.timestamp_ms()),
        ..ExpireArgs::default()
    };
    let p = plan(&meta, &Defaults::of(&meta, FAR), &young).unwrap();
    assert!(!p.snapshots.contains(&second), "{p:?}");
    assert!(p.snapshots.is_empty(), "{p:?}");
}

/// Explicit ids add to the age rule's removals (Java's `RemoveSnapshots` applies both).
#[test]
fn explicit_ids_add_to_the_age_rule() {
    let env = env();
    let catalog = five_snapshots(&env);
    let meta = catalog.metadata();
    let ids = chain(&meta);
    // `retain_last = 2` keeps the two newest by age; the explicit id removes the second newest
    // too, so the result differs from the age rule alone.
    let explicit = ExpireArgs {
        snapshot_ids: vec![ids[1]],
        ..args(Some(2))
    };
    let alone = plan(&meta, &Defaults::of(&meta, FAR), &args(Some(2))).unwrap();
    assert!(!alone.snapshots.contains(&ids[1]), "{alone:?}");
    let p = plan(&meta, &Defaults::of(&meta, FAR), &explicit).unwrap();
    let mut want = ids[1..].to_vec();
    want.sort_unstable();
    assert_eq!(p.snapshots, want);
}

/// A retention property of `i64::MIN` saturates instead of overflowing.
#[test]
fn an_extreme_retention_property_does_not_overflow() {
    let env = env();
    let catalog = five_snapshots(&env);
    let meta = edit(catalog.metadata(), |b| {
        b.set_properties(std::collections::HashMap::from([(
            "history.expire.max-snapshot-age-ms".to_owned(),
            i64::MIN.to_string(),
        )]))
        .unwrap()
    });
    let p = plan(
        &meta,
        &Defaults::of(&meta, FAR),
        &ExpireArgs {
            retain_last: Some(1),
            ..ExpireArgs::default()
        },
    )
    .unwrap();
    // The cutoff saturates at `i64::MAX`: every unretained snapshot is old enough.
    assert_eq!(p.snapshots.len(), 4);
}

/// The five snapshots' table with a spec 1 (identity on `x`) added, made the default when
/// `default` holds.
fn with_spec_one(env: &Env, default: bool) -> std::sync::Arc<Catalog> {
    let catalog = five_snapshots(env);
    let spec = UnboundPartitionSpec::builder()
        .add_partition_field(2, "x_identity", Transform::Identity)
        .unwrap()
        .build();
    let (meta, _) = catalog.state();
    let meta = edit(meta, |b| {
        let b = b.add_partition_spec(spec).unwrap();
        if default {
            b.set_default_partition_spec(-1).unwrap()
        } else {
            b
        }
    });
    // A metadata file of its own: an expire plans over the file its answer names.
    make_current(&catalog, meta, "spec-one");
    catalog
}

fn cleaning(retain_last: Option<i64>, older_than_ms: i64) -> ExpireArgs {
    ExpireArgs {
        older_than_ms: Some(older_than_ms),
        retain_last,
        clean_expired_metadata: true,
        ..ExpireArgs::default()
    }
}

/// The kept snapshot's manifests were written with spec 0, no longer the default: spec 0 stays.
#[test]
fn a_spec_a_kept_manifest_names_is_kept() {
    let env = env();
    let catalog = with_spec_one(&env, true);
    env.rt
        .block_on(expire(
            &*catalog,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &cleaning(Some(1), FAR),
        ))
        .unwrap();
    let after = catalog.metadata();
    assert_eq!(after.snapshots().count(), 1);
    assert!(after.partition_spec_by_id(0).is_some());
    assert!(after.partition_spec_by_id(1).is_some());
}

/// Spec 1 is neither the default nor named by a manifest: it is removed, alone.
#[test]
fn an_unreached_spec_is_removed() {
    let env = env();
    let catalog = with_spec_one(&env, false);
    env.rt
        .block_on(expire(
            &*catalog,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &cleaning(None, 0),
        ))
        .unwrap();
    let after = catalog.metadata();
    assert_eq!(after.snapshots().count(), 5);
    assert!(after.partition_spec_by_id(1).is_none());
}

/// [`Catalog`] answering every commit with one outcome and applying none; with `fail_reload`,
/// every load after the first fails.
struct Refusing {
    inner: std::sync::Arc<Catalog>,
    unknown: bool,
    fail_reload: bool,
    loads: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl CommitTable for Refusing {
    fn name(&self) -> String {
        self.inner.name()
    }

    async fn commit(&self, _: Vec<TableRequirement>, _: Vec<TableUpdate>) -> CommitOutcome {
        if self.unknown {
            CommitOutcome::Unknown("503 service unavailable".to_owned())
        } else {
            CommitOutcome::Conflict { stale: false }
        }
    }

    async fn load(&self, all: bool) -> Result<TableMetadata, KrishivError> {
        self.load_located(all).await.map(|(m, _)| m)
    }

    async fn load_located(
        &self,
        all: bool,
    ) -> Result<(TableMetadata, Option<String>), KrishivError> {
        let n = self.loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.fail_reload && n > 0 {
            return Err(krishiv_common::error::InternalError {
                message: "the catalog is unreachable".to_owned(),
            }
            .into());
        }
        self.inner.load_located(all).await
    }
}

fn refusing(catalog: std::sync::Arc<Catalog>, unknown: bool, fail_reload: bool) -> Refusing {
    Refusing {
        inner: catalog,
        unknown,
        fail_reload,
        loads: std::sync::atomic::AtomicUsize::new(0),
    }
}

fn run_expire(env: &Env, table: &Refusing, a: &ExpireArgs) -> KrishivError {
    env.rt
        .block_on(expire(table, &env.reg.file_io(), &clock(), 1 << 30, a))
        .unwrap_err()
}

fn expire_against(unknown: bool) -> KrishivError {
    let env = env();
    let table = refusing(five_snapshots(&env), unknown, false);
    run_expire(&env, &table, &args(Some(1)))
}

/// An unknown answer whose check cannot reload the table stays unknown, naming both causes.
#[test]
fn an_unknown_answer_whose_reload_fails_is_expire_state_unknown() {
    let env = env();
    let table = refusing(five_snapshots(&env), true, true);
    let e = run_expire(&env, &table, &args(Some(1)));
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergExpireStateUnknown
    );
    assert!(e.to_string().contains("could not be reloaded"), "{e}");
}

/// A request removing only a spec, answered unknown and not applied, is not taken as landed.
#[test]
fn an_unknown_spec_only_removal_is_expire_state_unknown() {
    let env = env();
    let table = refusing(with_spec_one(&env, false), true, false);
    let e = run_expire(&env, &table, &cleaning(None, 0));
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergExpireStateUnknown
    );
}

#[test]
fn commits_that_always_conflict_end_in_krishiv_iceberg_expire_failed() {
    let e = expire_against(false);
    assert_eq!(e.error_class(), ErrorClass::KrishivIcebergExpireFailed);
}

#[test]
fn an_unknown_answer_with_the_snapshots_still_there_is_expire_state_unknown() {
    let e = expire_against(true);
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergExpireStateUnknown
    );
}

/// Writes `meta` as a new metadata file of `catalog` and makes it the current one.
fn make_current(catalog: &Catalog, meta: TableMetadata, name: &str) {
    let path = format!("{}/metadata/{name}.metadata.json", meta.location());
    std::fs::write(
        super::sink_tests::local_path(&path),
        serde_json::to_vec(&meta).unwrap(),
    )
    .unwrap();
    catalog.restore((meta, path));
}

/// A catalog that answers with the refs' snapshots only (a server ignoring `snapshots=all`),
/// while the metadata file it names holds more.
struct RefsOnly {
    inner: std::sync::Arc<Catalog>,
}

#[async_trait]
impl CommitTable for RefsOnly {
    fn name(&self) -> String {
        "lake.db.t".to_owned()
    }

    async fn commit(&self, r: Vec<TableRequirement>, u: Vec<TableUpdate>) -> CommitOutcome {
        self.inner.commit(r, u).await
    }

    async fn load(&self, all: bool) -> Result<TableMetadata, KrishivError> {
        self.load_located(all).await.map(|(m, _)| m)
    }

    async fn load_located(
        &self,
        all: bool,
    ) -> Result<(TableMetadata, Option<String>), KrishivError> {
        let (m, location) = self.inner.load_located(all).await?;
        let reached = chain(&m);
        let unreferenced: Vec<i64> = m
            .snapshots()
            .map(|s| s.snapshot_id())
            .filter(|id| !reached.contains(id))
            .collect();
        let m = edit(m, |b| b.remove_snapshots(&unreferenced));
        Ok((m, location))
    }
}

/// Design §4: `main` rolled back to the third snapshot leaves the two newer ones unreferenced,
/// as a staged WAP snapshot is; a catalog that leaves them out of its answer is refused before
/// any request or delete, since their files would look unreachable.
#[test]
fn a_snapshot_list_that_differs_from_its_metadata_file_is_refused() {
    let env = env();
    let catalog = five_snapshots(&env);
    let meta = catalog.metadata();
    let ids = chain(&meta);
    let rolled = edit(meta, |b| {
        b.set_ref(
            "main",
            SnapshotReference::new(
                ids[2],
                SnapshotRetention::Branch {
                    min_snapshots_to_keep: None,
                    max_snapshot_age_ms: None,
                    max_ref_age_ms: None,
                },
            ),
        )
        .unwrap()
    });
    make_current(&catalog, rolled, "v900-rolled");
    let files_before = super::cow_tests::live(&env, &catalog.metadata());
    let table = RefsOnly {
        inner: std::sync::Arc::clone(&catalog),
    };
    let e = env
        .rt
        .block_on(expire(
            &table,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(1)),
        ))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogTableUnsupported,
        "{e}"
    );
    assert!(
        e.to_string().contains("differs from its metadata file"),
        "{e}"
    );
    assert_eq!(catalog.metadata().snapshots().count(), 5);
    assert_eq!(
        super::cow_tests::live(&env, &catalog.metadata()),
        files_before
    );
    // The same table through a catalog that answers in full expires as usual.
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
    assert!(done.deleted.manifest_lists >= 1, "{:?}", done.deleted);
}

/// A catalog that applies each request and then answers that the outcome is unknown.
struct LandsUnknown {
    inner: std::sync::Arc<Catalog>,
}

#[async_trait]
impl CommitTable for LandsUnknown {
    fn name(&self) -> String {
        self.inner.name()
    }

    async fn commit(&self, r: Vec<TableRequirement>, u: Vec<TableUpdate>) -> CommitOutcome {
        match self.inner.commit(r, u).await {
            CommitOutcome::Committed(_) => CommitOutcome::Unknown("502 bad gateway".to_owned()),
            other => other,
        }
    }

    async fn load(&self, all: bool) -> Result<TableMetadata, KrishivError> {
        self.inner.load(all).await
    }

    async fn load_located(
        &self,
        all: bool,
    ) -> Result<(TableMetadata, Option<String>), KrishivError> {
        self.inner.load_located(all).await
    }
}

/// An unknown answer to a request that landed: the probe finds it, and the cleanup runs
/// against the probe's reload as after a plain commit.
#[test]
fn an_unknown_answer_that_landed_cleans_up_against_the_reload() {
    let env = env();
    let catalog = five_snapshots(&env);
    let table = LandsUnknown {
        inner: std::sync::Arc::clone(&catalog),
    };
    let done = env
        .rt
        .block_on(expire(
            &table,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(1)),
        ))
        .unwrap();
    assert_eq!(catalog.metadata().snapshots().count(), 1);
    assert_eq!(done.deleted.manifest_lists, 4);
    assert_eq!(done.deleted.data_files, 1);
    assert_eq!(read_back(&env, &catalog), expected(5..40));
}

/// A catalog whose first request meets a foreign commit (a tag on `tagged`) and conflicts;
/// later requests go through.
struct ForeignTagFirst {
    inner: std::sync::Arc<Catalog>,
    tagged: i64,
    done: AtomicBool,
}

#[async_trait]
impl CommitTable for ForeignTagFirst {
    fn name(&self) -> String {
        self.inner.name()
    }

    async fn commit(&self, r: Vec<TableRequirement>, u: Vec<TableUpdate>) -> CommitOutcome {
        if !self.done.swap(true, std::sync::atomic::Ordering::SeqCst) {
            let tag = TableUpdate::SetSnapshotRef {
                ref_name: "keep".to_owned(),
                reference: SnapshotReference::new(
                    self.tagged,
                    SnapshotRetention::Tag {
                        max_ref_age_ms: None,
                    },
                ),
            };
            assert!(matches!(
                self.inner.commit(Vec::new(), vec![tag]).await,
                CommitOutcome::Committed(_)
            ));
            return CommitOutcome::Conflict { stale: false };
        }
        self.inner.commit(r, u).await
    }

    async fn load(&self, all: bool) -> Result<TableMetadata, KrishivError> {
        self.inner.load(all).await
    }

    async fn load_located(
        &self,
        all: bool,
    ) -> Result<(TableMetadata, Option<String>), KrishivError> {
        self.inner.load_located(all).await
    }
}

/// A conflict after a foreign commit that changes the plan: the next attempt plans on the
/// reload, so the snapshot the foreign tag now keeps stays.
#[test]
fn a_conflict_replans_on_the_reload() {
    let env = env();
    let catalog = five_snapshots(&env);
    let tagged = chain(&catalog.metadata())[3];
    let table = ForeignTagFirst {
        inner: std::sync::Arc::clone(&catalog),
        tagged,
        done: AtomicBool::new(false),
    };
    let done = env
        .rt
        .block_on(expire(
            &table,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(1)),
        ))
        .unwrap();
    let after = catalog.metadata();
    assert_eq!(after.snapshots().count(), 2);
    assert!(after.snapshot_by_id(tagged).is_some());
    assert_eq!(done.deleted.manifest_lists, 3);
}

/// Design §4.1: a Puffin file holding the DVs of two data files is deleted only when no live
/// entry of a kept snapshot names its path. After a delete rewrote one of the two files, the
/// kept snapshot still names the Puffin file for the other: expiring every older snapshot
/// keeps it, and the table reads the same rows.
#[test]
fn a_shared_puffin_file_stays_while_a_kept_snapshot_names_it() {
    use super::cow_tests::{add_deletes, dvs_in_one_puffin, live};
    let env = env();
    let t = create(&env, 3, false, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    append(&env, &catalog, range(10, 20));
    add_deletes(&env, &catalog, |files| {
        dvs_in_one_puffin(&env, "shared.puffin", files, &[0, 1])
    });
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .delete("id = 5")
        .unwrap();
    let rows = read_back(&env, &catalog);
    assert_eq!(live(&env, &catalog.metadata()).1, ["shared.puffin"]);
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
    assert_eq!(catalog.metadata().snapshots().count(), 1);
    assert_eq!(done.deleted.position_delete_files, 0, "{:?}", done.deleted);
    // The rewritten data file went with the old snapshots.
    assert_eq!(done.deleted.data_files, 1, "{:?}", done.deleted);
    let shared = env.at("shared.puffin");
    assert!(std::path::Path::new(super::sink_tests::local_path(&shared)).exists());
    assert_eq!(read_back(&env, &catalog), rows);
}

/// [`Catalog`] whose first commit turns GC off (a foreign property change) and conflicts.
struct GcTurnedOff {
    inner: std::sync::Arc<Catalog>,
    done: AtomicBool,
}

#[async_trait]
impl CommitTable for GcTurnedOff {
    fn name(&self) -> String {
        self.inner.name()
    }

    async fn commit(&self, r: Vec<TableRequirement>, u: Vec<TableUpdate>) -> CommitOutcome {
        if !self.done.swap(true, std::sync::atomic::Ordering::SeqCst) {
            let off = TableUpdate::SetProperties {
                updates: [("gc.enabled".to_owned(), "false".to_owned())].into(),
            };
            assert!(matches!(
                self.inner.commit(Vec::new(), vec![off]).await,
                CommitOutcome::Committed(_)
            ));
            return CommitOutcome::Conflict { stale: false };
        }
        self.inner.commit(r, u).await
    }

    async fn load(&self, all: bool) -> Result<TableMetadata, KrishivError> {
        self.inner.load(all).await
    }

    async fn load_located(
        &self,
        all: bool,
    ) -> Result<(TableMetadata, Option<String>), KrishivError> {
        self.inner.load_located(all).await
    }
}

/// A conflict whose reload has GC off ends the expire: no second request, nothing deleted.
#[test]
fn gc_turned_off_during_the_retries_stops_the_expire() {
    let env = env();
    let catalog = five_snapshots(&env);
    let before = catalog.metadata().snapshots().count();
    let table = GcTurnedOff {
        inner: std::sync::Arc::clone(&catalog),
        done: AtomicBool::new(false),
    };
    let e = env
        .rt
        .block_on(expire(
            &table,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(1)),
        ))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergProcedureGcDisabled,
        "{e}"
    );
    assert_eq!(catalog.metadata().snapshots().count(), before);
}

/// [`Catalog`] whose answers leave out the table's properties while the metadata file it names
/// has them: the file decides.
struct PropertiesDropped {
    inner: std::sync::Arc<Catalog>,
}

#[async_trait]
impl CommitTable for PropertiesDropped {
    fn name(&self) -> String {
        self.inner.name()
    }

    async fn commit(&self, r: Vec<TableRequirement>, u: Vec<TableUpdate>) -> CommitOutcome {
        self.inner.commit(r, u).await
    }

    async fn load(&self, all: bool) -> Result<TableMetadata, KrishivError> {
        self.load_located(all).await.map(|(m, _)| m)
    }

    async fn load_located(
        &self,
        all: bool,
    ) -> Result<(TableMetadata, Option<String>), KrishivError> {
        let (meta, location) = self.inner.load_located(all).await?;
        let keys: Vec<String> = meta.properties().keys().cloned().collect();
        let meta = edit(meta, |b| b.remove_properties(&keys).unwrap());
        Ok((meta, location))
    }
}

/// The answer says nothing of `gc.enabled`, the checked metadata file turns it off: refused.
#[test]
fn the_checked_metadata_file_decides_over_the_answer() {
    let env = super::provider_tests::env();
    let t = create(&env, 2, false, &[("gc.enabled", "false")]);
    let table = PropertiesDropped {
        inner: Catalog::new(&t),
    };
    let e = env
        .rt
        .block_on(expire(
            &table,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(1)),
        ))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergProcedureGcDisabled,
        "{e}"
    );
}

/// [`Catalog`] whose first load answers an older state of the table (the start), so the expire's
/// first request conflicts and its committed base differs from the start.
struct StaleStart {
    inner: std::sync::Arc<Catalog>,
    start: (TableMetadata, String),
    loads: std::sync::atomic::AtomicUsize,
    /// The first commit is applied but answered as a conflict (it landed unnoticed).
    land_first: bool,
    commits: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl CommitTable for StaleStart {
    fn name(&self) -> String {
        self.inner.name()
    }

    async fn commit(&self, r: Vec<TableRequirement>, u: Vec<TableUpdate>) -> CommitOutcome {
        let n = self
            .commits
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let out = self.inner.commit(r, u).await;
        if self.land_first && n == 0 {
            assert!(matches!(out, CommitOutcome::Committed(_)));
            return CommitOutcome::Conflict { stale: false };
        }
        out
    }

    async fn load(&self, all: bool) -> Result<TableMetadata, KrishivError> {
        self.load_located(all).await.map(|(m, _)| m)
    }

    async fn load_located(
        &self,
        all: bool,
    ) -> Result<(TableMetadata, Option<String>), KrishivError> {
        if self.loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            let (meta, location) = self.start.clone();
            return Ok((meta, Some(location)));
        }
        self.inner.load_located(all).await
    }
}

/// Five snapshots and the state after the first two (its metadata file still on disk).
fn five_with_an_older_state(env: &Env) -> (std::sync::Arc<Catalog>, (TableMetadata, String)) {
    let t = create(env, 2, false, &[]);
    let catalog = Catalog::new(&t);
    append(env, &catalog, range(0, 10));
    append(env, &catalog, range(10, 20));
    let older = catalog.state();
    append(env, &catalog, range(20, 30));
    append(env, &catalog, range(30, 40));
    Dml::open(env, &catalog, RowLevelOperation::Delete)
        .delete("id < 5")
        .unwrap();
    assert_eq!(catalog.metadata().snapshots().count(), 5);
    (catalog, older)
}

/// The cleanup reads both bases. Stale start: the start had two snapshots, the committed base
/// five, and the four removed snapshots' manifest lists all go (the start alone would give two).
/// Landed unnoticed: the first request lands but is answered as a conflict, the replan is empty,
/// and the cleanup still runs against the start. Either way the kept snapshot's manifest list stays.
#[test]
fn the_cleanup_reads_the_start_and_the_committed_base() {
    for land_first in [false, true] {
        let env = env();
        let (catalog, older) = five_with_an_older_state(&env);
        let start = if land_first { catalog.state() } else { older };
        let table = StaleStart {
            inner: std::sync::Arc::clone(&catalog),
            start,
            loads: std::sync::atomic::AtomicUsize::new(0),
            land_first,
            commits: std::sync::atomic::AtomicUsize::new(0),
        };
        let done = env
            .rt
            .block_on(expire(
                &table,
                &env.reg.file_io(),
                &clock(),
                1 << 30,
                &args(Some(1)),
            ))
            .unwrap();
        let after = catalog.metadata();
        assert_eq!(after.snapshots().count(), 1, "{land_first}");
        assert_eq!(
            done.deleted.manifest_lists, 4,
            "{land_first}: {:?}",
            done.deleted
        );
        let kept = after.current_snapshot().unwrap().manifest_list().to_owned();
        assert!(
            std::path::Path::new(super::sink_tests::local_path(&kept)).exists(),
            "{land_first}: {kept}"
        );
    }
}

/// The answer's uuid is not its metadata file's: refused before anything is planned.
#[test]
fn an_answer_whose_uuid_is_not_its_files_is_refused() {
    struct OtherUuid {
        inner: std::sync::Arc<Catalog>,
        other: TableMetadata,
    }
    #[async_trait]
    impl CommitTable for OtherUuid {
        fn name(&self) -> String {
            self.inner.name()
        }
        async fn commit(&self, r: Vec<TableRequirement>, u: Vec<TableUpdate>) -> CommitOutcome {
            self.inner.commit(r, u).await
        }
        async fn load(&self, all: bool) -> Result<TableMetadata, KrishivError> {
            self.load_located(all).await.map(|(m, _)| m)
        }
        async fn load_located(
            &self,
            all: bool,
        ) -> Result<(TableMetadata, Option<String>), KrishivError> {
            let (meta, location) = self.inner.load_located(all).await?;
            let meta = edit(meta, |b| b.assign_uuid(self.other.uuid()));
            Ok((meta, location))
        }
    }
    let env = env();
    let catalog = five_snapshots(&env);
    let other_env = super::provider_tests::env();
    let table = OtherUuid {
        inner: std::sync::Arc::clone(&catalog),
        other: five_snapshots(&other_env).metadata(),
    };
    let e = env
        .rt
        .block_on(expire(
            &table,
            &env.reg.file_io(),
            &clock(),
            1 << 30,
            &args(Some(1)),
        ))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivCatalogTableUnsupported,
        "{e}"
    );
    assert!(
        e.to_string().contains("differs from its metadata file"),
        "{e}"
    );
    assert_eq!(catalog.metadata().snapshots().count(), 5);
}
