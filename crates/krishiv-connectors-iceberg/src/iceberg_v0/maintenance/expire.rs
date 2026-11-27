// SPDX-License-Identifier: Apache-2.0
//! `expire_snapshots` (design P3-T027 PR-36e §4.1): the retention rule of Iceberg's
//! `RemoveSnapshots` ([`plan`], pure over the table's metadata), the metadata-only commit with
//! its retries ([`expire`]), and the cleanup of the files only the removed snapshots reached.

use std::collections::{HashMap, HashSet};

use iceberg::spec::{SnapshotRetention, TableMetadata};
use iceberg::{TableRequirement, TableUpdate};
use krishiv_common::error::{
    KrishivCatalogTableUnsupported, KrishivError, KrishivIcebergExpireFailed,
    KrishivIcebergExpireStateUnknown, KrishivIcebergProcedureArgument,
    KrishivIcebergProcedureGcDisabled,
};

use super::super::commit::{CommitClock, CommitOutcome, CommitTable};
use super::super::scan::{parse_manifest, read_capped, read_manifest_files, snapshot_refs};
use crate::plan::entry::{ContentKind, EntryStatus};

const DAY_MS: i64 = 86_400_000;
/// Java's `TableProperties.MAX_SNAPSHOT_AGE_MS_DEFAULT`: 5 days.
const MAX_SNAPSHOT_AGE_MS: i64 = 5 * DAY_MS;
/// Java's `TableProperties.MIN_SNAPSHOTS_TO_KEEP_DEFAULT`.
const MIN_SNAPSHOTS_TO_KEEP: i64 = 1;
/// Java's `DELETE_NUM_RETRIES`.
const DELETE_RETRIES: usize = 3;

/// The arguments of one `expire_snapshots` call.
#[derive(Debug, Clone, Default)]
pub(crate) struct ExpireArgs {
    /// `older_than`, in milliseconds since the epoch.
    pub(crate) older_than_ms: Option<i64>,
    /// `retain_last`.
    pub(crate) retain_last: Option<i64>,
    /// `snapshot_ids`.
    pub(crate) snapshot_ids: Vec<i64>,
    /// `clean_expired_metadata`.
    pub(crate) clean_expired_metadata: bool,
    /// `max_concurrent_deletes`: how many file deletes run at once (one when absent, at most
    /// [`MAX_CONCURRENT_DELETES`]).
    pub(crate) max_concurrent_deletes: Option<usize>,
}

/// What an expire removes from the table's metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Snapshots, in id order.
    pub(crate) snapshots: Vec<i64>,
    /// Refs dropped (a ref to a missing snapshot, or one past its max ref age), in name order.
    pub(crate) refs: Vec<String>,
    /// Partition specs no kept snapshot reaches (`clean_expired_metadata`; filled by
    /// [`with_obsolete_specs`] from the kept snapshots' manifest lists).
    pub(crate) specs: Vec<i32>,
    /// Schemas no kept snapshot reaches (`clean_expired_metadata`).
    pub(crate) schemas: Vec<i32>,
}

impl Plan {
    /// Whether nothing is removed: no request is sent.
    pub(crate) fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
            && self.refs.is_empty()
            && self.specs.is_empty()
            && self.schemas.is_empty()
    }
}

fn argument(message: String) -> KrishivError {
    KrishivIcebergProcedureArgument { message }.into()
}

fn prop(meta: &TableMetadata, key: &str) -> Option<i64> {
    meta.properties().get(key).and_then(|v| v.parse().ok())
}

/// The checks of the procedure's own arguments and of `gc.enabled` (Iceberg's `Preconditions`).
///
/// # Errors
/// `KRISHIV_ICEBERG_PROCEDURE_ARGUMENT`, `KRISHIV_ICEBERG_PROCEDURE_GC_DISABLED`.
pub(crate) fn check(meta: &TableMetadata, args: &ExpireArgs) -> Result<(), KrishivError> {
    if let Some(n) = args.retain_last
        && n < 1
    {
        return Err(argument(format!(
            "Number of snapshots to retain must be at least 1, cannot be: {n}"
        )));
    }
    // Java's `Boolean.parseBoolean`: present, only `true` in any case enables GC; absent, it is on.
    if meta
        .properties()
        .get("gc.enabled")
        .is_some_and(|v| !v.eq_ignore_ascii_case("true"))
    {
        return Err(KrishivIcebergProcedureGcDisabled {
            message:
                "Cannot expire snapshots: GC is disabled (deleting files may corrupt other tables)"
                    .to_owned(),
        }
        .into());
    }
    Ok(())
}

/// The snapshots, refs and schemas an expire at `now_ms` removes from `meta` (the table as
/// loaded with every snapshot): Iceberg's `RemoveSnapshots.internalApply`. The retention
/// defaults are those of the first load (`defaults`). The specs need the kept snapshots'
/// manifest lists, so [`with_obsolete_specs`] adds them.
///
/// Time arithmetic saturates: a retention property is a catalog's untrusted `i64`.
///
/// # Errors
/// `KRISHIV_ICEBERG_PROCEDURE_ARGUMENT` for an explicit snapshot a kept ref still names.
pub(crate) fn plan(
    meta: &TableMetadata,
    defaults: &Defaults,
    args: &ExpireArgs,
) -> Result<Plan, KrishivError> {
    let now = defaults.now_ms;
    let older_than = args
        .older_than_ms
        .unwrap_or_else(|| now.saturating_sub(defaults.max_snapshot_age_ms));
    let min_keep = args.retain_last.unwrap_or(defaults.min_snapshots_to_keep);
    let refs = snapshot_refs(meta)?;
    let ts = |id: i64| meta.snapshot_by_id(id).map(|s| s.timestamp_ms());

    // Refs: `main` always; another kept while young enough; one to a missing snapshot dropped.
    let mut kept_refs: Vec<(&String, &iceberg::spec::SnapshotReference)> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    for (name, r) in &refs {
        // `main` is kept whatever it points at (`computeRetainedRefs` skips it first).
        if name == "main" {
            kept_refs.push((name, r));
            continue;
        }
        let Some(t) = ts(r.snapshot_id) else {
            dropped.push(name.clone());
            continue;
        };
        let (own_max_ref_age, _) = match &r.retention {
            SnapshotRetention::Branch { max_ref_age_ms, .. }
            | SnapshotRetention::Tag { max_ref_age_ms } => (*max_ref_age_ms, ()),
        };
        let max_ref_age = own_max_ref_age.unwrap_or(defaults.max_ref_age_ms);
        if now.saturating_sub(t) > max_ref_age {
            dropped.push(name.clone());
        } else {
            kept_refs.push((name, r));
        }
    }
    dropped.sort();

    let mut retained: HashSet<i64> = HashSet::new();
    for (_, r) in &kept_refs {
        retained.insert(r.snapshot_id);
        if let SnapshotRetention::Branch {
            min_snapshots_to_keep,
            max_snapshot_age_ms,
            ..
        } = &r.retention
        {
            let keep_min = min_snapshots_to_keep.map_or(min_keep, i64::from);
            let expire_before = max_snapshot_age_ms.map_or(older_than, |a| now.saturating_sub(a));
            let mut kept = 0_i64;
            let mut at = Some(r.snapshot_id);
            while let Some(id) = at {
                let Some(snapshot) = meta.snapshot_by_id(id) else {
                    break;
                };
                if kept < keep_min || snapshot.timestamp_ms() >= expire_before {
                    retained.insert(id);
                    kept += 1;
                    at = snapshot.parent_snapshot_id();
                } else {
                    break;
                }
            }
        }
    }
    // Explicit ids first (Java's `idsToRemove`): one a kept ref names is refused.
    let mut remove: Vec<i64> = Vec::new();
    for &id in &args.snapshot_ids {
        if meta.snapshot_by_id(id).is_none() {
            continue;
        }
        let mut named: Vec<&str> = kept_refs
            .iter()
            .filter(|(_, r)| r.snapshot_id == id)
            .map(|(n, _)| n.as_str())
            .collect();
        if !named.is_empty() {
            named.sort_unstable();
            return Err(argument(format!(
                "Cannot expire {id}. Still referenced by refs: [{}]",
                named.join(", ")
            )));
        }
        remove.push(id);
    }
    // Then the age rule, which explicit ids do not switch off (`unreferencedSnapshotsToRetain`):
    // a branch references its ancestors, a tag only its head; an unreferenced snapshot stays
    // while young; a referenced one in no retained set is expired.
    let referenced = referenced(meta, &kept_refs);
    for s in meta.snapshots() {
        let id = s.snapshot_id();
        if retained.contains(&id) {
            continue;
        }
        if !referenced.contains(&id) && s.timestamp_ms() >= older_than {
            continue;
        }
        remove.push(id);
    }
    remove.sort_unstable();
    remove.dedup();

    let mut plan = Plan {
        snapshots: remove,
        refs: dropped,
        specs: Vec::new(),
        schemas: Vec::new(),
    };
    if args.clean_expired_metadata {
        let removed: HashSet<i64> = plan.snapshots.iter().copied().collect();
        let mut schemas: HashSet<i32> = HashSet::from([meta.current_schema_id()]);
        for s in meta.snapshots() {
            if removed.contains(&s.snapshot_id()) {
                continue;
            }
            if let Some(id) = s.schema_id() {
                schemas.insert(id);
            }
        }
        for schema in meta.schemas_iter() {
            if !schemas.contains(&schema.schema_id()) {
                plan.schemas.push(schema.schema_id());
            }
        }
        plan.schemas.sort_unstable();
    }
    Ok(plan)
}

/// The snapshots the kept refs reference: a branch its head and every ancestor, a tag its head
/// only (Java's `unreferencedSnapshotsToRetain`).
fn referenced(
    meta: &TableMetadata,
    kept_refs: &[(&String, &iceberg::spec::SnapshotReference)],
) -> HashSet<i64> {
    let mut seen = HashSet::new();
    for (_, r) in kept_refs {
        if !matches!(r.retention, SnapshotRetention::Branch { .. }) {
            seen.insert(r.snapshot_id);
            continue;
        }
        let mut at = Some(r.snapshot_id);
        while let Some(id) = at {
            if !seen.insert(id) {
                break;
            }
            at = meta.snapshot_by_id(id).and_then(|s| s.parent_snapshot_id());
        }
    }
    seen
}

/// With `clean_expired_metadata`, adds to `plan` the partition specs neither the default spec
/// nor any manifest of a kept snapshot names (Java reads every kept snapshot's manifests' spec
/// ids before it removes a spec). Run on every attempt's base.
///
/// # Errors
/// A manifest list that cannot be read: the expire fails before any request.
pub(crate) async fn with_obsolete_specs(
    io: &iceberg::io::FileIO,
    meta: &TableMetadata,
    args: &ExpireArgs,
    mut plan: Plan,
    max_bytes: u64,
) -> Result<Plan, KrishivError> {
    if !args.clean_expired_metadata {
        return Ok(plan);
    }
    let removed: HashSet<i64> = plan.snapshots.iter().copied().collect();
    let mut reached: HashSet<i32> = HashSet::from([meta.default_partition_spec_id()]);
    let version = format_version(meta);
    for s in meta.snapshots() {
        if removed.contains(&s.snapshot_id()) {
            continue;
        }
        for m in read_manifest_files(io, s.manifest_list(), version, max_bytes).await? {
            reached.insert(m.partition_spec_id);
        }
    }
    plan.specs = meta
        .partition_specs_iter()
        .map(|spec| spec.spec_id())
        .filter(|id| !reached.contains(id))
        .collect();
    plan.specs.sort_unstable();
    Ok(plan)
}

fn format_version(meta: &TableMetadata) -> u8 {
    match meta.format_version() {
        iceberg::spec::FormatVersion::V1 => 1,
        iceberg::spec::FormatVersion::V2 => 2,
        iceberg::spec::FormatVersion::V3 => 3,
    }
}

/// The retention settings an expire reads once, from the table as loaded at its start
/// (Java's `RemoveSnapshots` constructor).
#[derive(Debug, Clone)]
pub(crate) struct Defaults {
    /// The clock when the procedure began.
    pub(crate) now_ms: i64,
    /// `history.expire.max-snapshot-age-ms`.
    pub(crate) max_snapshot_age_ms: i64,
    /// `history.expire.min-snapshots-to-keep`.
    pub(crate) min_snapshots_to_keep: i64,
    /// `history.expire.max-ref-age-ms`.
    pub(crate) max_ref_age_ms: i64,
}

impl Defaults {
    /// The settings of `meta` at `now_ms`.
    pub(crate) fn of(meta: &TableMetadata, now_ms: i64) -> Self {
        Self {
            now_ms,
            max_snapshot_age_ms: prop(meta, "history.expire.max-snapshot-age-ms")
                .unwrap_or(MAX_SNAPSHOT_AGE_MS),
            min_snapshots_to_keep: prop(meta, "history.expire.min-snapshots-to-keep")
                .unwrap_or(MIN_SNAPSHOTS_TO_KEEP),
            max_ref_age_ms: prop(meta, "history.expire.max-ref-age-ms").unwrap_or(i64::MAX),
        }
    }
}

/// The requirements and updates of `plan` against `meta`: `assert-table-uuid`; with a snapshot
/// removal every ref as planned, `main` included (stricter than Java's `UpdateRequirements`,
/// which asserts none: a concurrent append to `main`, or any ref moved, conflicts and the expire
/// plans again on the new base, so a busy table can use up its retries, crate CLAUDE.md known
/// limits); with only a spec or schema removal the default spec, the current schema and every
/// branch but `main` (Java's, so an append to `main` never conflicts there).
pub(crate) fn request(
    meta: &TableMetadata,
    plan: &Plan,
) -> Result<(Vec<TableRequirement>, Vec<TableUpdate>), KrishivError> {
    let mut requirements = vec![TableRequirement::UuidMatch { uuid: meta.uuid() }];
    let mut updates: Vec<TableUpdate> = plan
        .refs
        .iter()
        .map(|r| TableUpdate::RemoveSnapshotRef {
            ref_name: r.clone(),
        })
        .collect();
    if !plan.snapshots.is_empty() {
        updates.push(TableUpdate::RemoveSnapshots {
            snapshot_ids: plan.snapshots.clone(),
        });
    }
    if !plan.specs.is_empty() {
        updates.push(TableUpdate::RemovePartitionSpecs {
            spec_ids: plan.specs.clone(),
        });
    }
    if !plan.schemas.is_empty() {
        updates.push(TableUpdate::RemoveSchemas {
            schema_ids: plan.schemas.clone(),
        });
    }
    if !plan.snapshots.is_empty() {
        // Every ref as planned, `main` included: a concurrent rollback or a ref moved onto a
        // snapshot this removes conflicts instead of having its ref dropped by the server (Iceberg
        // 1.12.0's `UpdateRequirements` adds none for `RemoveSnapshots`; a stricter check). A ref
        // created after planning under a new name is not covered: the REST requirements cannot
        // name it.
        let mut all: Vec<(String, i64)> = snapshot_refs(meta)?
            .into_iter()
            .map(|(n, r)| (n, r.snapshot_id))
            .collect();
        all.sort();
        for (r#ref, snapshot_id) in all {
            requirements.push(TableRequirement::RefSnapshotIdMatch {
                r#ref,
                snapshot_id: Some(snapshot_id),
            });
        }
    }
    if !plan.specs.is_empty() || !plan.schemas.is_empty() {
        requirements.push(TableRequirement::DefaultSpecIdMatch {
            default_spec_id: meta.default_partition_spec_id(),
        });
        requirements.push(TableRequirement::CurrentSchemaIdMatch {
            current_schema_id: meta.current_schema_id(),
        });
        let mut branches: Vec<(String, i64)> = snapshot_refs(meta)?
            .into_iter()
            .filter(|(n, r)| n != "main" && matches!(r.retention, SnapshotRetention::Branch { .. }))
            .map(|(n, r)| (n, r.snapshot_id))
            .collect();
        branches.sort();
        // With snapshots removed, every ref is asserted above already.
        if !plan.snapshots.is_empty() {
            branches.clear();
        }
        for (r#ref, snapshot_id) in branches {
            requirements.push(TableRequirement::RefSnapshotIdMatch {
                r#ref,
                snapshot_id: Some(snapshot_id),
            });
        }
    }
    Ok((requirements, updates))
}

/// Counts of the files an expire deleted: Iceberg's result row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Deleted {
    pub(crate) data_files: i64,
    pub(crate) position_delete_files: i64,
    pub(crate) equality_delete_files: i64,
    pub(crate) manifests: i64,
    pub(crate) manifest_lists: i64,
    pub(crate) statistics_files: i64,
}

/// What a file is, for the counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Kind {
    Data,
    PositionDeletes,
    EqualityDeletes,
    Manifest,
    ManifestList,
    Statistics,
}

/// The decoded manifests of one walk, by manifest path.
pub(crate) type ManifestCache = HashMap<String, std::sync::Arc<[(String, Kind)]>>;

/// Every file a snapshot reaches, by path: its manifest list, its manifests, the live entries
/// of those (`ADDED`/`EXISTING`), and the statistics files naming it.
pub(crate) async fn reachable(
    io: &iceberg::io::FileIO,
    meta: &TableMetadata,
    snapshot_id: i64,
    max_bytes: u64,
    manifests: &mut ManifestCache,
    out: &mut HashMap<String, Kind>,
) -> Result<(), KrishivError> {
    let Some(snapshot) = meta.snapshot_by_id(snapshot_id) else {
        return Ok(());
    };
    let list = snapshot.manifest_list().to_owned();
    out.insert(list.clone(), Kind::ManifestList);
    let version = format_version(meta);
    for m in read_manifest_files(io, &list, version, max_bytes).await? {
        out.insert(m.manifest_path.clone(), Kind::Manifest);
        let entries = if let Some(hit) = manifests.get(&m.manifest_path) {
            std::sync::Arc::clone(hit)
        } else {
            let bytes = read_capped(
                io,
                "manifest",
                &m.manifest_path,
                u64::try_from(m.manifest_length).ok(),
                max_bytes,
            )
            .await?;
            let parsed = parse_manifest(&bytes, m.partition_spec_id)?;
            let entries: Vec<(String, Kind)> = parsed
                .into_iter()
                .filter(|e| e.status != EntryStatus::Deleted)
                .map(|e| {
                    (
                        e.file.path.clone(),
                        match e.file.content {
                            ContentKind::Data => Kind::Data,
                            ContentKind::PositionDeletes => Kind::PositionDeletes,
                            ContentKind::EqualityDeletes => Kind::EqualityDeletes,
                        },
                    )
                })
                .collect();
            let entries: std::sync::Arc<[(String, Kind)]> = entries.into();
            manifests.insert(m.manifest_path.clone(), std::sync::Arc::clone(&entries));
            entries
        };
        for (path, kind) in entries.iter() {
            out.insert(path.clone(), *kind);
        }
    }
    for s in meta.statistics_iter() {
        if s.snapshot_id == snapshot_id {
            out.insert(s.statistics_path.clone(), Kind::Statistics);
        }
    }
    for s in meta.partition_statistics_iter() {
        if s.snapshot_id == snapshot_id {
            out.insert(s.statistics_path.clone(), Kind::Statistics);
        }
    }
    Ok(())
}

/// Deletes the files only the snapshots of the `bases` that `after` lacks reached (the Spark
/// action's `findExpiredSnapshotIds(originalMetadata, updatedMetadata)`): candidates are
/// everything reachable from those snapshots, in each base that has them, minus everything
/// reachable from any snapshot of `after`, the checked reload after the commit, by path. The
/// bases are the table as loaded when the procedure began (design §4.1: an earlier attempt that
/// landed unnoticed removed snapshots only it had) and the base the committed request was planned
/// on (snapshots concurrent writers added after the start and this request removed). Up to `concurrency` deletes run at once, each retried three times; one that
/// fails is logged and not counted, and never fails the expire.
///
/// The path sets are held in memory, uncharged, as Java's `ReachableFileCleanup` holds them
/// (crate CLAUDE.md, known limits).
///
/// # Errors
/// A manifest list or manifest that cannot be read: nothing is deleted then.
pub(crate) async fn cleanup(
    io: &iceberg::io::FileIO,
    bases: &[&TableMetadata],
    after: &TableMetadata,
    max_bytes: u64,
    concurrency: usize,
) -> Result<Deleted, KrishivError> {
    use futures::StreamExt as _;
    let kept: HashSet<i64> = after.snapshots().map(|s| s.snapshot_id()).collect();
    let mut cache = HashMap::new();
    let mut candidates: HashMap<String, Kind> = HashMap::new();
    let mut seen: HashSet<i64> = HashSet::new();
    for base in bases {
        let removed: Vec<i64> = base
            .snapshots()
            .map(|s| s.snapshot_id())
            .filter(|id| !kept.contains(id) && seen.insert(*id))
            .collect();
        for id in removed {
            reachable(io, base, id, max_bytes, &mut cache, &mut candidates).await?;
        }
    }
    // Nothing to delete (a spec- or schema-only commit): the kept snapshots are not read, so an
    // unreadable manifest of theirs fails nothing.
    if candidates.is_empty() {
        return Ok(Deleted::default());
    }
    let mut valid: HashMap<String, Kind> = HashMap::new();
    for id in &kept {
        reachable(io, after, *id, max_bytes, &mut cache, &mut valid).await?;
    }
    let mut counts = Deleted::default();
    let mut doomed: Vec<(String, Kind)> = candidates
        .into_iter()
        .filter(|(p, _)| !valid.contains_key(p))
        .collect();
    doomed.sort();
    let done: Vec<(Kind, bool)> = futures::stream::iter(doomed)
        .map(|(path, kind)| async move { (kind, delete_with_retries(io, &path).await) })
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await;
    for (kind, deleted) in done {
        if deleted {
            match kind {
                Kind::Data => counts.data_files += 1,
                Kind::PositionDeletes => counts.position_delete_files += 1,
                Kind::EqualityDeletes => counts.equality_delete_files += 1,
                Kind::Manifest => counts.manifests += 1,
                Kind::ManifestList => counts.manifest_lists += 1,
                Kind::Statistics => counts.statistics_files += 1,
            }
        }
    }
    Ok(counts)
}

/// One file deleted, retried; `false` when it is gone already or every try failed.
async fn delete_with_retries(io: &iceberg::io::FileIO, path: &str) -> bool {
    for attempt in 1..=DELETE_RETRIES {
        if let Ok(false) = io.exists(path).await {
            return false;
        }
        match io.delete(path).await {
            Ok(()) => return true,
            Err(e) => tracing::warn!(path, attempt, error = %e, "an expired file was not deleted"),
        }
    }
    false
}

/// What an expire came to.
#[derive(Debug)]
pub(crate) struct Expired {
    /// The files deleted.
    pub(crate) deleted: Deleted,
}

/// Whether `plan`, made on `base`, landed in `now`: everything it removes is absent and every
/// snapshot of `base` it keeps is still there (design §6.1).
fn landed(base: &TableMetadata, now: &TableMetadata, plan: &Plan) -> bool {
    let Ok(refs) = snapshot_refs(now) else {
        return false;
    };
    let removed: HashSet<i64> = plan.snapshots.iter().copied().collect();
    base.snapshots()
        .map(|s| s.snapshot_id())
        .filter(|id| !removed.contains(id))
        .all(|id| now.snapshot_by_id(id).is_some())
        && plan
            .snapshots
            .iter()
            .all(|id| now.snapshot_by_id(*id).is_none())
        && plan.refs.iter().all(|r| !refs.contains_key(r))
        && plan
            .specs
            .iter()
            .all(|id| now.partition_spec_by_id(*id).is_none())
        && plan
            .schemas
            .iter()
            .all(|id| now.schema_by_id(*id).is_none())
}

/// Runs `expire_snapshots` over `table`: loads every snapshot, plans, commits the metadata change
/// with the retries below, then deletes what only the removed snapshots reached.
///
/// Every load (the start, each attempt's refresh, the unknown outcome's probe and the reload
/// after the commit) is checked against the metadata file it names ([`checked_load`]), so the
/// retention and the file sets are computed over every snapshot the table has.
///
/// A conflict reloads the table and plans again. An unknown outcome (a request that may have
/// landed) is looked up in the table: every planned snapshot, ref, spec and schema gone means it
/// landed; otherwise, or when that load fails, the error is
/// `KRISHIV_ICEBERG_EXPIRE_STATE_UNKNOWN` and nothing is deleted or recomputed (design §6.1).
///
/// # Errors
/// The argument and gc checks; `KRISHIV_ICEBERG_EXPIRE_FAILED` when no request applied in the
/// budget; `KRISHIV_ICEBERG_EXPIRE_STATE_UNKNOWN`; a failed load or read.
pub(crate) async fn expire(
    table: &dyn CommitTable,
    io: &iceberg::io::FileIO,
    clock: &dyn CommitClock,
    max_metadata_bytes: u64,
    args: &ExpireArgs,
) -> Result<Expired, KrishivError> {
    let start = checked_load(table, io, max_metadata_bytes).await?;
    check(&start, args)?;
    // Every later load must be this table: one renamed away and replaced at the name is another.
    let same = |m: TableMetadata| -> Result<TableMetadata, KrishivError> {
        if m.uuid() == start.uuid() {
            Ok(m)
        } else {
            Err(replaced(&table.name(), &start, &m))
        }
    };
    // The argument may lower the width, never raise it past the engine's cap.
    let concurrency = args
        .max_concurrent_deletes
        .unwrap_or(1)
        .clamp(1, MAX_CONCURRENT_DELETES);
    let defaults = Defaults::of(&start, clock.now_ms());
    // Catalog properties are untrusted: bounded before any arithmetic (an hour per wait at most).
    let retries = prop(&start, "commit.retry.num-retries")
        .unwrap_or(4)
        .clamp(0, 100);
    let min_wait = prop(&start, "commit.retry.min-wait-ms")
        .unwrap_or(100)
        .clamp(1, MAX_WAIT_MS);
    let max_wait = prop(&start, "commit.retry.max-wait-ms")
        .unwrap_or(60_000)
        .clamp(min_wait, MAX_WAIT_MS);
    let name = table.name();
    let mut meta = start.clone();
    let mut last;
    let mut attempts = 0_i64;
    let mut planned: Plan;
    loop {
        // A reload after a conflict may have turned GC off.
        check(&meta, args)?;
        planned = plan(&meta, &defaults, args)?;
        planned = with_obsolete_specs(io, &meta, args, planned, max_metadata_bytes).await?;
        if planned.is_empty() {
            // After a conflict an earlier attempt may have landed unnoticed: its removals are in
            // `meta`, a checked reload of this table, so their files are cleaned as Java's action
            // diffs the start against the refreshed table. On the first attempt `meta` is the
            // start and nothing is read.
            let deleted = cleanup(io, &[&start], &meta, max_metadata_bytes, concurrency).await?;
            return Ok(Expired { deleted });
        }
        let (requirements, updates) = request(&meta, &planned)?;
        attempts += 1;
        match table.commit(requirements, updates).await {
            CommitOutcome::Committed(_) => {
                // The valid side is a checked reload, never the commit's answer as sent. If it
                // fails, the snapshots are already removed: the error stands (Java fails the
                // action there too), and the log says only the file cleanup was skipped (a re-run
                // finds nothing to expire; the files are orphans for `remove_orphan_files`).
                let after = checked_load(table, io, max_metadata_bytes)
                    .await
                    .and_then(same)
                    .inspect_err(|e| {
                        tracing::warn!(
                            table = %name,
                            snapshots = ?planned.snapshots,
                            error = %e,
                            "expire_snapshots committed; its file cleanup was skipped"
                        );
                    })?;
                let deleted = cleanup(
                    io,
                    &[&start, &meta],
                    &after,
                    max_metadata_bytes,
                    concurrency,
                )
                .await?;
                return Ok(Expired { deleted });
            }
            CommitOutcome::Conflict { .. } => last = "a conflicting commit".to_owned(),
            CommitOutcome::Unknown(reason) => {
                // It may have landed: it did when the planned snapshots are gone.
                let unknown = |reason: String| -> KrishivError {
                    KrishivIcebergExpireStateUnknown {
                        snapshot_ids: format!("{:?}", planned.snapshots),
                        table: name.clone(),
                        reason,
                    }
                    .into()
                };
                let now = match checked_load(table, io, max_metadata_bytes)
                    .await
                    .and_then(same)
                {
                    Ok(now) => now,
                    Err(e) => {
                        return Err(unknown(format!(
                            "{reason}; the table could not be reloaded to check: {e}"
                        )));
                    }
                };
                if !landed(&meta, &now, &planned) {
                    return Err(unknown(reason));
                }
                let deleted =
                    cleanup(io, &[&start, &meta], &now, max_metadata_bytes, concurrency).await?;
                return Ok(Expired { deleted });
            }
            CommitOutcome::Failed(e) => return Err(e),
        }
        if attempts > retries {
            break;
        }
        let wait = min_wait
            .saturating_mul(1_i64 << (attempts - 1).clamp(0, 20))
            .min(max_wait);
        let wait = wait.saturating_add(
            i64::try_from(clock.jitter_ms(u64::try_from(wait / 10 + 1).unwrap_or(1))).unwrap_or(0),
        );
        clock
            .sleep(std::time::Duration::from_millis(
                u64::try_from(wait).unwrap_or(0),
            ))
            .await;
        meta = same(checked_load(table, io, max_metadata_bytes).await?)?;
    }
    Err(KrishivIcebergExpireFailed {
        snapshot_ids: format!("{:?}", planned.snapshots),
        table: name,
        attempts: attempts.to_string(),
        reason: last,
    }
    .into())
}

/// The table with every snapshot, checked against the metadata file its answer names (design
/// §4, review 2 F2): a catalog that left a snapshot out (a server ignoring `snapshots=all`, a
/// staged WAP snapshot) would have its files deleted. The file's snapshot ids must equal the
/// answer's, both ways.
///
/// # Errors
/// `KRISHIV_CATALOG_TABLE_UNSUPPORTED` for an answer without `metadata-location`, a metadata
/// file that cannot be read or parsed, or a snapshot list that differs from it; the load's own
/// failure.
pub(crate) async fn checked_load(
    table: &dyn CommitTable,
    io: &iceberg::io::FileIO,
    max_bytes: u64,
) -> Result<TableMetadata, KrishivError> {
    checked_load_located(table, io, max_bytes)
        .await
        .map(|(meta, _)| meta)
}

/// [`checked_load`], with the location of the metadata file it checked against.
///
/// # Errors
/// As [`checked_load`].
pub(crate) async fn checked_load_located(
    table: &dyn CommitTable,
    io: &iceberg::io::FileIO,
    max_bytes: u64,
) -> Result<(TableMetadata, String), KrishivError> {
    let (meta, location) = table.load_located(true).await?;
    let name = table.name();
    let refused = |location: &str, detail: &str| -> KrishivError {
        tracing::warn!(table = %name, location, detail, "an expire refused the table's snapshot list");
        let (catalog, relation) = name.split_once('.').unwrap_or(("", name.as_str()));
        KrishivCatalogTableUnsupported {
            catalog: catalog.to_owned(),
            relation_name: relation.to_owned(),
            feature: format!("a snapshot list that differs from its metadata file `{location}`"),
        }
        .into()
    };
    let Some(location) = location else {
        return Err(refused("(none)", "the answer names no metadata-location"));
    };
    let bytes = read_capped(io, "metadata file", &location, None, max_bytes)
        .await
        .map_err(|e| refused(&location, &e.to_string()))?;
    let file: TableMetadata = serde_json::from_slice(&bytes)
        .map_err(|e| refused(&location, &format!("does not parse: {e}")))?;
    let ids =
        |m: &TableMetadata| -> HashSet<i64> { m.snapshots().map(|s| s.snapshot_id()).collect() };
    if ids(&file) != ids(&meta) {
        return Err(refused(&location, "the snapshot ids differ"));
    }
    if file.uuid() != meta.uuid() {
        return Err(refused(&location, "the table uuids differ"));
    }
    // The file decides: its refs, manifest lists and properties are what retention and cleanup
    // read, never the answer's copy of them.
    Ok((file, location))
}

/// The most file deletes an expire runs at once, whatever `max_concurrent_deletes` asks.
pub(crate) const MAX_CONCURRENT_DELETES: usize = 64;

/// The longest wait between commit attempts, whatever the table's properties say.
const MAX_WAIT_MS: i64 = 3_600_000;

/// A later load found another table at the name (its uuid is not the one the expire started on):
/// nothing more is sent or deleted.
fn replaced(name: &str, start: &TableMetadata, now: &TableMetadata) -> KrishivError {
    tracing::warn!(table = %name, started = %start.uuid(), now = %now.uuid(), "an expire's table was replaced");
    let (catalog, relation) = name.split_once('.').unwrap_or(("", name));
    KrishivCatalogTableUnsupported {
        catalog: catalog.to_owned(),
        relation_name: relation.to_owned(),
        feature: format!(
            "a table replaced during expire_snapshots (uuid {} became {})",
            start.uuid(),
            now.uuid()
        ),
    }
    .into()
}
