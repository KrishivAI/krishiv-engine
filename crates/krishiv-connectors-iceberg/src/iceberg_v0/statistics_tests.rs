// SPDX-License-Identifier: Apache-2.0
//! Statistics on write (P3-T024, PR-54b; `statistics_file`): the sketched columns and the switches; appends
//! through the sink (`sink_tests`' catalog) whose snapshots carry a Puffin statistics file read back through
//! PR-54a's reader (`ndv_by_field`, `read_sketch`) and the provider's distinct counts; and the commit loop's
//! simulation (`commit_tests`' `TestCatalog`): rebases onto foreign and sketched parents, a resend, a refusal of
//! the update, a failed Puffin write, and the statistics files every outcome keeps or deletes.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::Int64Array;
use datasketches::theta::CompactThetaSketch;
use iceberg::spec::{NestedField, PrimitiveType, Schema, StatisticsFile, TableMetadata, Type};
use krishiv_common::config::Config;
use krishiv_exec::planner::TableProvider;

use super::commit::CommitTable;
use super::commit_tests::{
    Case, Fault, Foreign, INIT, MINE, OUR_UUID, OURS, TestClock, When, case_on, exists,
    foreign_full,
};
use super::provider::IcebergTableProvider;
use super::provider_tests::{Env, env, ixn_schema};
use super::puffin_stats::{ndv_by_field, read_sketch};
use super::sink_tests::{Catalog, PerDriver, create, input_schema, range, request, run};
use super::statistics_file::{
    AttemptSnapshot, MAX_SKETCHED_COLUMNS, STATISTICS_PROPERTY, WriteSketches, attempt_statistics,
    enabled, sketch_columns, statistics_skipped_total,
};
use super::table::IcebergTable;
use crate::plan::VersionSpec;
use crate::plan::theta_value::{new_sketch, update_column};
use crate::plan::types::IcebergType;

/// Three standard errors of a `k` = 4,096 sketch, relative.
const RSE3: f64 = 3.0 / 64.0;

fn within_3_rse(estimate: u64, truth: u64) -> bool {
    (estimate as f64 - truth as f64).abs() <= RSE3 * truth as f64
}

/// The top-level primitive columns but `uuid`, in order, at most 100.
#[test]
fn sketched_columns_are_top_level_primitives_but_uuid_capped() {
    let mut fields = vec![
        NestedField::optional(1, "u", Type::Primitive(PrimitiveType::Uuid)).into(),
        NestedField::optional(
            2,
            "s",
            Type::Struct(iceberg::spec::StructType::new(vec![
                NestedField::optional(3, "a", Type::Primitive(PrimitiveType::Int)).into(),
            ])),
        )
        .into(),
    ];
    for i in 0..105 {
        fields.push(
            NestedField::optional(
                10 + i,
                format!("c{i}"),
                Type::Primitive(PrimitiveType::Long),
            )
            .into(),
        );
    }
    let schema = Schema::builder().with_fields(fields).build().unwrap();
    let cols = sketch_columns(&schema);
    assert_eq!(cols.len(), MAX_SKETCHED_COLUMNS);
    assert_eq!((cols[0].index, cols[0].field_id), (2, 10));
    assert_eq!(cols[99].field_id, 109);
    assert!(cols.iter().all(|c| c.ty == IcebergType::Long));
}

#[test]
fn the_switch_and_the_table_property() {
    let props = |v: &str| HashMap::from([(STATISTICS_PROPERTY.to_owned(), v.to_owned())]);
    assert!(enabled(true, &HashMap::new()));
    assert!(enabled(true, &props("TRUE")));
    assert!(!enabled(false, &HashMap::new()));
    assert!(!enabled(true, &props("false")));
    // Java's `parseBoolean`: anything but `true` is off.
    assert!(!enabled(true, &props("yes")));
}

/// The provider of the table at `location` committing through `catalog`, under `config`.
fn provider_with(
    env: &Env,
    location: &str,
    catalog: &Arc<Catalog>,
    config: &Config,
) -> IcebergTableProvider {
    let table = env
        .rt
        .block_on(IcebergTable::load(&env.reg, location, "db", "t"))
        .unwrap()
        .with_committer(Arc::clone(catalog) as Arc<dyn CommitTable>);
    env.rt
        .block_on(IcebergTableProvider::open(
            &table,
            &VersionSpec::Current,
            config,
            4,
        ))
        .unwrap()
}

/// Appends `ranges` (one driver each) into the table of `catalog`; the pool is empty after.
fn append(env: &Env, catalog: &Arc<Catalog>, config: &Config, ranges: &[(i64, i64)]) {
    let p = provider_with(env, &catalog.location(), catalog, config);
    let sink = p.writer(&request(None)).unwrap();
    let input = Arc::new(PerDriver {
        schema: input_schema(),
        batches: ranges.iter().map(|(a, b)| range(*a, *b)).collect(),
        stall: false,
    });
    let (out, pool, _runner) = run(env, sink, input, None);
    out.unwrap();
    assert_eq!(pool.reserved(), 0, "the sketches' memory is released");
}

/// The statistics file of the current snapshot of `m`.
fn current_file(m: &TableMetadata) -> Option<&StatisticsFile> {
    m.statistics_for_snapshot(m.current_snapshot_id()?)
}

/// `ndv` of each field of the current snapshot, read back through PR-54a's reader.
fn current_ndv(m: &TableMetadata) -> HashMap<i32, u64> {
    ndv_by_field(m, m.current_snapshot_id().unwrap())
}

/// Checks the file's sizes and that every blob reads back with the estimate its `ndv` names.
fn check_file(env: &Env, file: &StatisticsFile) {
    let path = super::sink_tests::local_path(&file.statistics_path);
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() as i64, file.file_size_in_bytes);
    let footer = bytes.len() - usize::try_from(file.file_footer_size_in_bytes).unwrap();
    assert_eq!(
        &bytes[footer..footer + 4],
        b"PFA1",
        "the footer starts with the magic"
    );
    for b in &file.blob_metadata {
        let s = env
            .rt
            .block_on(read_sketch(&env.reg.file_io(), file, b.fields[0]))
            .unwrap()
            .unwrap();
        let ndv: f64 = b.properties["ndv"].parse().unwrap();
        assert_eq!(ndv, s.estimate().trunc());
    }
}

/// A CTAS (an empty table, two drivers) gets one blob per column with `ndv` within 3 RSE of the truth, read
/// back by the provider as distinct counts; an append of overlapping keys unions with it; the first snapshot
/// keeps its own file.
#[test]
fn a_ctas_has_statistics_and_an_append_unions_them() {
    let env = env();
    let t = create(&env, ixn_schema(), true, &[]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, &on(), &[(0, 10_000), (10_000, 20_000)]);
    let m = catalog.metadata();
    let file = current_file(&m).unwrap().clone();
    assert_eq!(file.blob_metadata.len(), 3);
    check_file(&env, &file);
    let ndv = current_ndv(&m);
    assert!(within_3_rse(ndv[&1], 20_000), "{ndv:?}");
    assert_eq!(ndv[&2], 3);
    assert!(within_3_rse(ndv[&3], 20_000), "{ndv:?}");
    let first = m.current_snapshot_id().unwrap();
    assert!(
        file.statistics_path
            .contains(&format!("/metadata/{first}-")),
        "{}",
        file.statistics_path
    );
    // The provider reads them as distinct counts (never above the rows).
    let p = provider_with(&env, &catalog.location(), &catalog, &on());
    let stats = p.statistics().unwrap();
    let distinct = stats.columns["id"].distinct.unwrap();
    assert!(within_3_rse(distinct, 20_000), "{distinct}");

    append(&env, &catalog, &on(), &[(10_000, 30_000)]);
    let m = catalog.metadata();
    let ndv = current_ndv(&m);
    assert!(within_3_rse(ndv[&1], 30_000), "{ndv:?}");
    assert_eq!(ndv[&2], 3);
    check_file(&env, current_file(&m).unwrap());
    // Every snapshot keeps its statistics (Java's retention).
    assert_eq!(m.statistics_iter().count(), 2);
    assert!(m.statistics_for_snapshot(first).is_some());
}

/// A config with statistics on (the default is off until U3's write-cost number, review of 2026-11-17).
fn on() -> Config {
    let mut c = Config::new();
    c.set("krishiv.iceberg.write.statistics.enabled", "true")
        .unwrap();
    c
}

/// The switch off, then on: the second snapshot's parent has no statistics, so it gets none (rule 2), and
/// neither does a table that opts out by property.
#[test]
fn no_statistics_on_an_unsketched_parent_or_when_switched_off() {
    let env = env();
    let t = create(&env, ixn_schema(), false, &[]);
    let catalog = Catalog::new(&t);
    let mut off = on();
    off.set("krishiv.iceberg.write.statistics.enabled", "false")
        .unwrap();
    append(&env, &catalog, &off, &[(0, 100)]);
    assert!(current_file(&catalog.metadata()).is_none());
    append(&env, &catalog, &on(), &[(100, 200)]);
    let m = catalog.metadata();
    assert_eq!(m.snapshots().count(), 2);
    assert_eq!(m.statistics_iter().count(), 0);

    let env = super::provider_tests::env();
    let t = create(&env, ixn_schema(), false, &[(STATISTICS_PROPERTY, "false")]);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, &on(), &[(0, 100)]);
    assert_eq!(catalog.metadata().statistics_iter().count(), 0);
    let stats_files = std::fs::read_dir(env.dir.path().join("db/t/metadata"))
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|x| x == "stats")
        })
        .count();
    assert_eq!(stats_files, 0);
}

/// An exact (under 4,096 values) sketch of the longs `values`.
fn sketch(values: impl Iterator<Item = i64>) -> CompactThetaSketch {
    let mut s = new_sketch().unwrap();
    let a = Int64Array::from(values.collect::<Vec<_>>());
    assert!(update_column(&mut s, &a, &IcebergType::Long));
    s.compact(true)
}

/// Sketches of `id` (field 1) for `values`.
pub(super) fn sketches(values: std::ops::Range<i64>) -> WriteSketches {
    WriteSketches::of(vec![(1, IcebergType::Long, sketch(values))])
}

/// A foreign append of one row (`id`) in partition 1 as snapshot `id`, with statistics of `values` when given.
fn foreign_sketched(
    env: &Env,
    clock: &Arc<TestClock>,
    m: &TableMetadata,
    name: &str,
    id: i64,
    values: Option<std::ops::Range<i64>>,
) -> Foreign {
    foreign_full(env, clock, m, name, id, "main", None, values)
}

/// `commit_tests`' case (row `1|1|init` committed, the write of `10|2|ours` pending) with the initial snapshot
/// sketched over `init` (when given) and the write's sketch over `ours`.
fn case(init: Option<std::ops::Range<i64>>, ours: std::ops::Range<i64>) -> Case {
    match init {
        Some(init) => case_on(2, &[], Some((init, ours))),
        None => {
            let mut c = case_on(2, &[], None);
            c.new.statistics = Some(sketches(ours));
            c
        }
    }
}

impl Case {
    /// Attempt `n`'s statistics file.
    fn stats(&self, n: u32) -> String {
        format!(
            "{}/metadata/{OURS}-{OUR_UUID}-{n}.stats",
            self.base.location()
        )
    }
}

/// The `ndv` of `id` in the statistics of our snapshot in `m`, if any.
fn our_ndv(m: &TableMetadata) -> Option<u64> {
    ndv_by_field(m, OURS).get(&1).copied()
}

/// One request: the union of the parent's sketch and ours, in our attempt's file, named by `set-statistics`.
#[test]
fn statistics_ride_in_the_append_request() {
    let mut c = case(Some(0..100), 50..120);
    let m = c.run().unwrap();
    assert_eq!(our_ndv(&m), Some(120));
    let f = m.statistics_for_snapshot(OURS).unwrap();
    assert_eq!(f.statistics_path, c.stats(1));
    assert_eq!(f.blob_metadata[0].sequence_number, 2);
    check_file(&c.env, f);
    c.check(&[INIT, MINE]);
}

/// A parent without statistics: nothing written, no update (rule 2).
#[test]
fn an_unsketched_parent_gets_no_statistics() {
    let mut c = case(None, 0..10);
    let m = c.run().unwrap();
    assert!(m.statistics_for_snapshot(OURS).is_none());
    assert!(!exists(&c.stats(1)));
    c.check(&[INIT, MINE]);
}

/// A rebase onto a foreign snapshot without statistics drops the update; attempt 1's file is deleted.
#[test]
fn a_rebase_onto_a_foreign_snapshot_drops_the_update() {
    let mut c = case(Some(0..100), 50..120);
    let f = foreign_sketched(&c.env, &c.clock, &c.base, "f.parquet", 900, None);
    c.cat.on(When::BeforeRequest(1), f);
    let m = c.run().unwrap();
    assert_eq!(
        m.snapshot_by_id(OURS).unwrap().parent_snapshot_id(),
        Some(900)
    );
    assert!(m.statistics_for_snapshot(OURS).is_none());
    assert!(!exists(&c.stats(1)) && !exists(&c.stats(2)));
    c.check(&[INIT, MINE, "900|1|f.parquet"]);
}

/// A rebase onto a sketched foreign snapshot unions with its blob, in attempt 2's file; attempt 1's is deleted.
#[test]
fn a_rebase_onto_a_sketched_parent_unions_again() {
    let mut c = case(Some(0..100), 50..120);
    let f = foreign_sketched(&c.env, &c.clock, &c.base, "f.parquet", 900, Some(0..300));
    c.cat.on(When::BeforeRequest(1), f);
    let m = c.run().unwrap();
    assert_eq!(our_ndv(&m), Some(300));
    assert_eq!(
        m.statistics_for_snapshot(OURS).unwrap().statistics_path,
        c.stats(2)
    );
    assert!(!exists(&c.stats(1)) && exists(&c.stats(2)));
    // The foreign writer's own file stays.
    assert!(m.statistics_for_snapshot(900).is_some());
}

/// An unknown outcome resends the same attempt: the same statistics file, and no other.
#[test]
fn a_resend_reuses_the_attempts_statistics_file() {
    let mut c = case(Some(0..100), 50..120);
    c.cat.script(&[Fault::ServerErrorBeforeApply]);
    let m = c.run().unwrap();
    assert_eq!(c.cat.requests(), 2);
    assert_eq!(
        m.statistics_for_snapshot(OURS).unwrap().statistics_path,
        c.stats(1)
    );
    assert!(!exists(&c.stats(2)));
}

/// A refusal of the request carrying `set-statistics` is resent at once without it (counted); data commits.
#[test]
fn a_refused_statistics_update_is_dropped_and_the_data_commits() {
    let before = statistics_skipped_total();
    let mut c = case(Some(0..100), 50..120);
    c.cat.script(&[Fault::Unauthorized401]);
    let m = c.run().unwrap();
    assert_eq!(c.cat.requests(), 2);
    assert!(c.waits().is_empty(), "the resend does not wait");
    assert!(m.statistics_for_snapshot(OURS).is_none());
    assert!(!exists(&c.stats(1)), "the abandoned file is deleted");
    assert!(statistics_skipped_total() > before);
    c.check(&[INIT, MINE]);
}

/// Refused twice: the write fails as before (a known failure), and every file it wrote is deleted, the
/// statistics file included.
#[test]
fn a_failed_write_deletes_its_statistics_files() {
    let mut c = case(Some(0..100), 50..120);
    c.cat
        .script(&[Fault::Unauthorized401, Fault::Unauthorized401]);
    let e = c.run().unwrap_err();
    assert!(e.to_string().contains("401"), "{e}");
    assert!(!exists(&c.stats(1)) && !exists(&c.list(1)) && !exists(&c.data()));
}

/// A Puffin write that fails (a directory where the file goes) drops the statistics; the data commits.
#[test]
fn a_failed_puffin_write_still_commits_the_data() {
    let mut c = case(Some(0..100), 50..120);
    std::fs::create_dir_all(super::sink_tests::local_path(&c.stats(1))).unwrap();
    let m = c.run().unwrap();
    assert!(m.statistics_for_snapshot(OURS).is_none());
    c.check(&[INIT, MINE]);
}

/// A parent's statistics file that cannot be read drops the statistics; the data commits.
#[test]
fn an_unreadable_parent_file_still_commits_the_data() {
    let mut c = case(Some(0..100), 50..120);
    let parent = c
        .base
        .statistics_for_snapshot(1)
        .unwrap()
        .statistics_path
        .clone();
    std::fs::remove_file(super::sink_tests::local_path(&parent)).unwrap();
    let m = c.run().unwrap();
    assert!(m.statistics_for_snapshot(OURS).is_none());
    assert!(!exists(&c.stats(1)));
    c.check(&[INIT, MINE]);
}

/// A task's sketches are charged to its reservation and released with them; a refused reservation gives no
/// sketches (counted), and a write with one such task gets no statistics at all (rule 2: never a partial union).
#[test]
fn sketch_memory_is_charged_and_a_refusal_drops_the_statistics() {
    use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};

    use super::statistics_file::{SKETCH_BYTES, TaskSketches, union_tasks};

    let columns: Arc<[_]> = sketch_columns(&ixn_schema()).into();
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
        limit: Some(4 * SKETCH_BYTES),
        ..MemoryConfig::default()
    });
    let reservation = || Reservation::new(Arc::clone(&pool), pool.register("sketches", None));
    let mut task = TaskSketches::new(Arc::clone(&columns), reservation()).unwrap();
    assert_eq!(pool.reserved(), 3 * SKETCH_BYTES);
    assert!(format!("{task:?}").contains("TaskSketches"));
    assert!(task.update(&range(0, 10)));
    let before = statistics_skipped_total();
    let refused = TaskSketches::new(Arc::clone(&columns), reservation());
    assert!(refused.is_none());
    // Counted once for the write, where the union finds the task without sketches.
    assert!(union_tasks(&columns, vec![Some(task), refused], reservation()).is_none());
    assert!(statistics_skipped_total() > before);
    assert_eq!(pool.reserved(), 0);
    // No task (no rows): empty sketches of every column.
    let empty = union_tasks(&columns, Vec::new(), reservation()).unwrap();
    assert_eq!(empty.columns.len(), 3);
    assert!(format!("{empty:?}").contains("WriteSketches"));
    assert!(empty.columns.iter().all(|(_, _, s)| s.is_empty()));
}

/// S2 of the review: the request carrying statistics lands but its answer is lost; the probe finds it, and the
/// file the committed metadata names is kept.
#[test]
fn a_landed_request_with_a_lost_answer_keeps_its_statistics_file() {
    let mut c = case(Some(0..100), 50..120);
    c.cat.script(&[Fault::LostResponse]);
    let m = c.run().unwrap();
    assert_eq!(c.cat.requests(), 1);
    let f = m.statistics_for_snapshot(OURS).unwrap();
    assert_eq!(f.statistics_path, c.stats(1));
    assert!(exists(&c.stats(1)));
    check_file(&c.env, f);
    c.check(&[INIT, MINE]);
}

/// S2 of the review: a failure answered to a resend after an unknown answer says nothing about the first
/// request, so it does not take the resend without statistics: the same attempt, statistics and all, is resent
/// after the probe.
#[test]
fn a_failure_after_an_unknown_answer_keeps_the_statistics_update() {
    let mut c = case(Some(0..100), 50..120);
    c.cat
        .script(&[Fault::ServerErrorBeforeApply, Fault::Unauthorized401]);
    let m = c.run().unwrap();
    assert_eq!(c.cat.requests(), 3);
    assert_eq!(our_ndv(&m), Some(120));
    assert_eq!(
        m.statistics_for_snapshot(OURS).unwrap().statistics_path,
        c.stats(1)
    );
    c.check(&[INIT, MINE]);
}

/// S1 of the review (review N6): a column whose type is not the sketch's, in the current schema or in the
/// parent's, is left out and never unioned; the same call with the types in line writes the blob.
#[test]
fn a_column_of_another_type_is_left_out() {
    use iceberg::spec::TableMetadataBuilder;

    let c = case(Some(0..100), 50..120);
    let io = c.env.reg.file_io();
    let with_id = |ty: PrimitiveType| {
        let mut fields: Vec<_> = ixn_schema().as_struct().fields().to_vec();
        fields[0] = NestedField::optional(1, "id", Type::Primitive(ty)).into();
        Schema::builder().with_fields(fields).build().unwrap()
    };
    let evolved = |ty: PrimitiveType| {
        TableMetadataBuilder::new_from_metadata(c.base.clone(), None)
            .add_current_schema(with_id(ty))
            .unwrap()
            .build()
            .unwrap()
            .metadata
    };
    let field_ids = |m: &TableMetadata, ty: IcebergType, path: &str| {
        let parent = m.snapshot_by_id(1).unwrap();
        let at = AttemptSnapshot {
            metadata: m,
            parent: Some(parent.as_ref()),
            snapshot_id: OURS,
            sequence_number: 2,
        };
        let s = WriteSketches::of(vec![(1, ty, sketch(50..120))]);
        let path = format!("{}/metadata/{path}.stats", m.location());
        c.env
            .rt
            .block_on(attempt_statistics(&io, at, &s, &path))
            .unwrap()
            .map(|f| {
                f.blob_metadata
                    .iter()
                    .map(|b| b.fields[0])
                    .collect::<Vec<_>>()
            })
    };
    // In line: the parent's and the current schema say long.
    assert_eq!(field_ids(&c.base, IcebergType::Long, "same"), Some(vec![1]));
    // The sketch was built under `long`, the current schema says `int` (changed between open and rebase).
    assert_eq!(
        field_ids(&evolved(PrimitiveType::Int), IcebergType::Long, "current"),
        None
    );
    // The current schema matches the sketch (`int`), the parent's blob was written under `long`.
    assert_eq!(
        field_ids(&evolved(PrimitiveType::Int), IcebergType::Int, "parent"),
        None
    );
}

/// A copy-on-write rewrite (DELETE, UPDATE, MERGE) sketches nothing even with the switch on: unioning the
/// parent's blob with the rewritten rows would keep counting the deleted keys (review of 2026-11-19).
#[test]
fn a_copy_on_write_rewrite_sketches_nothing() {
    use super::filter::Removals;
    use super::sink::{IcebergWriteSink, Overwrite};
    use super::validate::Validation;

    let env = env();
    let t = create(&env, ixn_schema(), true, &[]);
    let catalog = Catalog::new(&t);
    let p = provider_with(&env, &catalog.location(), &catalog, &on());
    let target = Arc::clone(p.write_target().unwrap());
    let append = IcebergWriteSink::try_new(Arc::clone(&target), &request(None)).unwrap();
    assert!(append.sketches_columns());
    let overwrite = Overwrite {
        removals: Removals::default(),
        validation: Validation::none(),
        lineage: false,
    };
    let rewrite =
        IcebergWriteSink::try_rewrite(target, &input_schema(), "db.t", None, overwrite).unwrap();
    assert!(!rewrite.sketches_columns());
}
