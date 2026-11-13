// SPDX-License-Identifier: Apache-2.0
//! Tests of view expansion, cycles and temp-table pinning.

/// The scans of the streaming pipeline of `root`.
fn streaming_scans(root: &PhysNode) -> Vec<&krishiv_exec::physical::ScanNode> {
    super::streaming_leaves(root).0
}

use super::*;
use arrow_array::Array as _;
use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::Field;
use arrow_schema::Schema;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_types::DataType;

use crate::session::Session;

#[test]
fn work_runs_on_the_given_runtime_and_a_drop_aborts_it() {
    let io = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("test-io")
        .enable_all()
        .build()
        .unwrap();
    let name = futures::executor::block_on(run_on(io.handle(), async {
        Ok(std::thread::current().name().map(str::to_owned))
    }))
    .unwrap();
    assert_eq!(name.as_deref(), Some("test-io"));
    // Dropping the waiting future aborts a task that never finishes.
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let mut waiting = Box::pin(run_on(io.handle(), async move {
        let _held = tx;
        std::future::pending::<Result<(), KrishivError>>().await
    }));
    let waker = futures::task::noop_waker();
    let _ = waiting
        .as_mut()
        .poll(&mut std::task::Context::from_waker(&waker));
    drop(waiting);
    assert_eq!(
        rx.recv_timeout(std::time::Duration::from_secs(5)),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected),
        "the aborted task must drop its sender"
    );
}

fn files(col: &str) -> TempTable {
    TempTable::Files {
        known: false,
        permanent: false,
        partitions: Some(Default::default()),
        source: ReadSource::Files {
            format: "parquet".to_owned(),
            paths: vec![],
            options: Default::default(),
        },
        columns: vec![TableColumn::new(col.to_owned(), DataType::Integer, false)],
    }
}

fn first_column(t: &TempTable) -> &str {
    match t {
        TempTable::Files { columns, .. }
        | TempTable::Memory { columns, .. }
        | TempTable::View { columns, .. } => &columns[0].name,
    }
}

#[test]
fn expanding_a_cycle_is_an_internal_error() {
    // `CREATE TEMPORARY VIEW` rejects recursive views itself; the
    // expansion guard stays for anything else that closes a cycle.
    let session = SessionCatalog::default();
    let view = |name: &str| TempTable::View {
        plan: Arc::new(LogicalPlan::table([name])),
        columns: Vec::new(),
        pins: Vec::new(),
        text: None,
    };
    session.put("a", view("b"));
    session.put("b", view("a"));
    let plan = Arc::new(LogicalPlan::table(["a"]));
    let e = expand_temp_views(&session, &plan, false).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(e.message().contains("`a`"), "{e}");
}

#[test]
fn every_spelling_binds_the_table_loaded_first() {
    let session = SessionCatalog::default();
    session.put("people", files("a"));
    let loaded = Mutex::new(HashMap::new());
    let form = NameForm {
        case_sensitive: false,
        qualified: false,
        temporary_only: false,
    };
    let t = resolve(&loaded, &session, "people", form).unwrap();
    loaded.lock().unwrap().insert("people".to_owned(), t);
    // Another thread replaces the table.
    session.put("people", files("b"));
    let again = resolve(&loaded, &session, "PEOPLE", form).unwrap();
    assert_eq!(first_column(&again), "a");
    // A case-sensitive spelling that is not registered does not resolve
    // through the loaded entry.
    let cs = NameForm {
        case_sensitive: true,
        ..form
    };
    assert!(resolve(&loaded, &session, "PEOPLE", cs).is_none());
}

/// A session with the memory table `t(id)` holding `values`.
fn int_session(values: Vec<i32>) -> Session {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(values))],
    )
    .unwrap();
    session.register_batches("t", schema, vec![batch]).unwrap();
    session
}

/// The `id` column of `sql`, ordered.
fn select_ids(session: &Session, sql: &str) -> Vec<i32> {
    let df = block_on(session.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let mut out = Vec::new();
    for batch in &batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        out.extend((0..batch.num_rows()).map(|i| ids.value(i)));
    }
    out.sort_unstable();
    out
}

#[test]
fn view_creation_stores_pins_and_view_together() {
    // The pins and the owning view insert in one critical section, so
    // one epoch bump: no concurrent `put` or `drop_table` can collect
    // a new pin before its view arrives.
    let s = int_session(vec![1]);
    let before = s.state.catalog.epoch();
    block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
    assert_eq!(s.state.catalog.epoch(), before + 1);
    assert_eq!(s.state.catalog.entry_count(), 3);
    assert_eq!(select_ids(&s, "SELECT id FROM v"), vec![1]);
}

#[test]
fn reregistering_a_snapshot_view_does_not_grow_the_catalog() {
    let s = int_session(vec![1]);
    block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
    // The table, its pin and the view: replacing the view must not add
    // to that.
    assert_eq!(s.state.catalog.entry_count(), 3);
    for _ in 0..100 {
        block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
    }
    assert_eq!(s.state.catalog.entry_count(), 3);
    assert_eq!(select_ids(&s, "SELECT id FROM v"), vec![1]);
}

#[test]
fn dropping_a_view_removes_the_pins_it_owned() {
    let s = int_session(vec![1]);
    block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
    assert_eq!(s.state.catalog.entry_count(), 3);
    block_on(s.sql("DROP VIEW v")).unwrap();
    assert_eq!(s.state.catalog.entry_count(), 1);
}

#[test]
fn a_pin_shared_by_two_views_survives_dropping_one() {
    let s = int_session(vec![1]);
    block_on(s.table("t").create_or_replace_temp_view("v1")).unwrap();
    // A view over a view reuses the pin instead of stacking a new one.
    block_on(s.table("v1").create_or_replace_temp_view("v2")).unwrap();
    assert_eq!(s.state.catalog.entry_count(), 4);
    assert!(s.drop_temp_view("v1"));
    assert_eq!(s.state.catalog.entry_count(), 3);
    // The surviving view still reads the snapshot, however `t` changes.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![99]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    assert_eq!(select_ids(&s, "SELECT id FROM v2"), vec![1]);
}

#[test]
fn hidden_names_resolve_as_missing_tables() {
    let s = int_session(vec![1]);
    block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
    // Exactly as a missing table: same class, state and message shape.
    let missing = block_on(s.sql("SELECT * FROM nope")).unwrap_err();
    assert_eq!(missing.error_class(), ErrorClass::TableOrViewNotFound);
    let e = block_on(s.sql("SELECT * FROM `__krishiv_snapshot_0`")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    assert_eq!(e.sql_state(), missing.sql_state());
    assert_eq!(
        e.message().replace("`__krishiv_snapshot_0`", "`nope`"),
        missing.message()
    );
    // Any letter case, quoted or not.
    let e = block_on(s.sql("SELECT * FROM __KRISHIV_SNAPSHOT_0")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    // With read options too, while a pin of that name exists: the
    // options do not reach the pinned snapshot.
    let missing_options = block_on(s.sql("SELECT * FROM nope WITH ('a' = 'b')")).unwrap_err();
    assert_eq!(
        missing_options.error_class(),
        ErrorClass::TableOrViewNotFound
    );
    let e = block_on(s.sql("SELECT * FROM __krishiv_snapshot_0 WITH ('a' = 'b')")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    assert_eq!(e.sql_state(), missing_options.sql_state());
    assert_eq!(
        e.message().replace("`__krishiv_snapshot_0`", "`nope`"),
        missing_options.message()
    );
}

#[test]
fn view_built_from_itself_reads_the_old_definition() {
    let s = int_session(vec![1]);
    block_on(s.table("t").create_or_replace_temp_view("t")).unwrap();
    assert_eq!(select_ids(&s, "SELECT id FROM t"), vec![1]);
    // Rebuilding the self-view reuses the pin instead of orphaning one.
    block_on(s.table("t").create_or_replace_temp_view("t")).unwrap();
    assert_eq!(select_ids(&s, "SELECT id FROM t"), vec![1]);
    assert_eq!(s.state.catalog.entry_count(), 2);
}

#[test]
fn hidden_names_are_rejected_for_new_views_and_tables() {
    let s = int_session(vec![1]);
    let e = block_on(
        s.table("t")
            .create_or_replace_temp_view("__krishiv_snapshot_7"),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivReservedTempViewName);
    assert_eq!(e.error_class().name(), "KRISHIV_RESERVED_TEMP_VIEW_NAME");
    assert_eq!(e.sql_state(), Some("42602"));
    assert!(e.message().contains("`__krishiv_snapshot_`"), "{e}");
    let e =
        block_on(s.sql("CREATE TEMP VIEW `__krishiv_snapshot_7` AS SELECT id FROM t")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivReservedTempViewName);
    let e = block_on(s.sql("CREATE OR REPLACE TEMP VIEW V__KRISHIV_SNAPSHOT_ AS SELECT id FROM t"));
    // A name that merely contains the prefix is fine.
    assert!(e.is_ok(), "{e:?}");
    block_on(s.sql("DROP VIEW V__KRISHIV_SNAPSHOT_")).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    let e = s
        .register_batches("__KRISHIV_SNAPSHOT_7", schema, vec![batch])
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivReservedTempViewName);
}

/// Writes `rows` `id` values to `path` as Parquet with row groups of at
/// most `row_group` rows.
fn write_parquet(path: &std::path::Path, rows: i32, row_group: usize) {
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(row_group))
        .build();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    let mut at = 0;
    while at < rows {
        let take = (rows - at).min(50);
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(
                (at..at + take).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        at += take;
    }
    writer.close().unwrap();
}

/// The `id` column of `batches`, sorted.
fn ids_of(batches: &[RecordBatch]) -> Vec<i32> {
    let mut out = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        out.extend((0..batch.num_rows()).map(|i| ids.value(i)));
    }
    out.sort_unstable();
    out
}

#[test]
fn task_specs_deal_file_scans_deterministically_and_cover_everything() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 500, 100);
    write_parquet(&dir.path().join("b.parquet"), 300, 100);
    let session = Session::builder()
        // Small splits: every file is cut into several byte ranges.
        .config("spark.sql.files.maxPartitionBytes", "1024")
        .build()
        .unwrap();
    let df = session.read().parquet([dir.path().to_str().unwrap()]);
    let analysis = rt.block_on(df.analysis()).unwrap();
    let (_, physical) = rt
        .block_on(session.physical(&analysis.analyzed, &analysis.env))
        .unwrap();
    let catalog = &analysis.env.catalog;
    // The same listing and config give the same assignment, so a
    // retried task reads the same splits.
    let first = catalog.task_specs(&physical, 3, 3).unwrap();
    let second = catalog.task_specs(&physical, 3, 3).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
    // Every task's partitions are contiguous, and together the tasks
    // cover every planned partition exactly once.
    let mut indexes = Vec::new();
    for task in &first {
        assert_eq!(task.scans.len(), 1, "one scan: {task:?}");
        let splits = task.scans.values().next().unwrap();
        let krishiv_exec::physical::ScanSplits::Files { files, partitions } = splits else {
            panic!("file splits: {splits:?}");
        };
        assert!(!files.is_empty());
        let mut last: Option<usize> = None;
        for part in partitions {
            if let Some(previous) = last {
                assert_eq!(part.index, previous + 1, "contiguous: {partitions:?}");
            }
            last = Some(part.index);
            indexes.push(part.index);
        }
    }
    indexes.sort_unstable();
    assert!(!indexes.is_empty(), "several splits");
    assert_eq!(
        indexes,
        (0..indexes.len()).collect::<Vec<_>>(),
        "every partition exactly once"
    );
    // The union of 1..=4 tasks is the single run, with no duplicates.
    let whole = ids_of(&rt.block_on(df.collect()).unwrap());
    assert_eq!(whole.len(), 800);
    for tasks in 1..=4_usize {
        let specs = catalog
            .task_specs(&physical, tasks, u64::try_from(tasks).unwrap())
            .unwrap();
        assert_eq!(specs.len(), tasks);
        let mut union = Vec::new();
        for task in &specs {
            union.extend(ids_of(
                &rt.block_on(session.run_task(&physical, &analysis.env, task))
                    .unwrap(),
            ));
        }
        union.sort_unstable();
        assert_eq!(union, whole, "{tasks} tasks");
    }
}

#[test]
fn task_specs_deal_a_view_over_files() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 500, 100);
    write_parquet(&dir.path().join("b.parquet"), 300, 100);
    let session = Session::builder()
        // Small splits: every file is cut into several byte ranges.
        .config("spark.sql.files.maxPartitionBytes", "1024")
        .build()
        .unwrap();
    // A view over a files read keeps listing them: the scan names the
    // view, and its splits come from the opened listing.
    let files = session.read().parquet([dir.path().to_str().unwrap()]);
    rt.block_on(files.create_or_replace_temp_view("pv"))
        .unwrap();
    let df = block_on(session.sql("SELECT id FROM pv")).unwrap();
    let analysis = rt.block_on(df.analysis()).unwrap();
    assert!(analysis.env.catalog.loaded_file_views());
    let (_, physical) = rt
        .block_on(session.physical(&analysis.analyzed, &analysis.env))
        .unwrap();
    let catalog = &analysis.env.catalog;
    // The same listing and config give the same assignment, so a
    // retried task reads the same splits.
    let first = catalog.task_specs(&physical, 3, 3).unwrap();
    let second = catalog.task_specs(&physical, 3, 3).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
    // The union of 1..=3 tasks is the single run, with no duplicates:
    // every task reads exactly its splits, never the whole view.
    let whole = ids_of(&rt.block_on(df.collect()).unwrap());
    assert_eq!(whole.len(), 800);
    for tasks in 1..=3_usize {
        let specs = catalog
            .task_specs(&physical, tasks, u64::try_from(tasks).unwrap())
            .unwrap();
        assert_eq!(specs.len(), tasks);
        let mut union = Vec::new();
        for task in &specs {
            assert_eq!(task.scans.len(), 1, "one scan: {task:?}");
            union.extend(ids_of(
                &rt.block_on(session.run_task(&physical, &analysis.env, task))
                    .unwrap(),
            ));
        }
        union.sort_unstable();
        assert_eq!(union, whole, "{tasks} tasks");
    }
}

#[test]
fn task_specs_deal_a_reference_with_options_to_a_table_over_files() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 500, 100);
    write_parquet(&dir.path().join("b.parquet"), 300, 100);
    let session = Session::builder()
        .config("spark.sql.files.maxPartitionBytes", "1024")
        .build()
        .unwrap();
    block_on(session.sql(&format!(
        "CREATE TABLE pt (id INT) USING parquet LOCATION '{}'",
        dir.path().display()
    )))
    .unwrap();
    // The reference's options name another listing than the bare table's: its splits come
    // from the listing opened for the reference.
    let df = block_on(session.sql("SELECT id FROM pt WITH ('mergeSchema' = 'false')")).unwrap();
    let analysis = rt.block_on(df.analysis()).unwrap();
    let (_, physical) = rt
        .block_on(session.physical(&analysis.analyzed, &analysis.env))
        .unwrap();
    let specs = analysis.env.catalog.task_specs(&physical, 2, 2).unwrap();
    assert_eq!(specs.len(), 2);
    let mut union = Vec::new();
    for task in &specs {
        union.extend(ids_of(
            &rt.block_on(session.run_task(&physical, &analysis.env, task))
                .unwrap(),
        ));
    }
    union.sort_unstable();
    assert_eq!(union, ids_of(&rt.block_on(df.collect()).unwrap()));
    assert_eq!(union.len(), 800);
}

/// A streaming `Range` has no split assignment: one task is fine,
/// several would each emit the whole range, so they are refused.
#[test]
fn task_specs_refuse_a_range_across_several_tasks() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let session = Session::new().unwrap();
    let df = session.range(0, 10, 1, None);
    let analysis = rt.block_on(df.analysis()).unwrap();
    let (_, physical) = rt
        .block_on(session.physical(&analysis.analyzed, &analysis.env))
        .unwrap();
    assert_eq!(
        analysis
            .env
            .catalog
            .task_specs(&physical, 1, 1)
            .unwrap()
            .len(),
        1
    );
    let e = analysis
        .env
        .catalog
        .task_specs(&physical, 2, 2)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn a_file_changed_between_listing_and_reading_fails() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_parquet(&dir.path().join("a.parquet"), 200, 100);
    let session = Session::builder()
        .config("spark.sql.files.maxPartitionBytes", "4096")
        .build()
        .unwrap();
    let df = session.read().parquet([dir.path().to_str().unwrap()]);
    let analysis = rt.block_on(df.analysis()).unwrap();
    let (_, physical) = rt
        .block_on(session.physical(&analysis.analyzed, &analysis.env))
        .unwrap();
    let specs = analysis.env.catalog.task_specs(&physical, 1, 1).unwrap();
    assert_eq!(specs.len(), 1);
    // New bytes under an old name: the listed version is stale (the
    // size differs too, so the failure does not depend on timestamp
    // granularity).
    write_parquet(&dir.path().join("a.parquet"), 10, 100);
    let e = rt
        .block_on(session.run_task(&physical, &analysis.env, &specs[0]))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivFileChangedDuringRead,
        "{e}"
    );
}

/// The `id` values of `sql` over `session`, sorted.
fn table_ids(session: &Session, sql: &str) -> Vec<i32> {
    let df = block_on(session.sql(sql)).unwrap();
    ids_of(&block_on(df.collect()).unwrap())
}

#[test]
fn task_specs_deal_memory_tables_and_release_them() {
    use krishiv_exec::physical::TaskSpec;
    let session = Session::new().unwrap();
    let base = session.memory_reserved();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batches_of = |values: Vec<i32>| {
        values
            .chunks(3)
            .map(|chunk| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int32Array::from(chunk.to_vec()))],
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let values: Vec<i32> = (0..14).collect();
    session
        .register_batches("t", Arc::clone(&schema), batches_of(values.clone()))
        .unwrap();
    // The same data under another name shares its content hash.
    session
        .register_batches("u", Arc::clone(&schema), batches_of(values))
        .unwrap();
    session
        .register_batches("v", Arc::clone(&schema), batches_of(vec![99]))
        .unwrap();
    let hash_of = |name: &str| {
        let entry = session.state.catalog.get(name, false).unwrap();
        let TempTable::Memory { table, .. } = &*entry else {
            panic!("a memory table: {name}");
        };
        table.relation_lease().unwrap().hash()
    };
    assert_eq!(hash_of("t"), hash_of("u"), "same data, same hash");
    assert_ne!(hash_of("t"), hash_of("v"), "different data, different hash");
    assert_eq!(session.relations().len(), 2, "t and u share one entry");
    assert!(
        session.relations().reserved() > 0,
        "the entries hold store bytes while leased"
    );
    // Holding tables never consumes the query pool's budget (tight
    // spill tests register large tables): the store has its own pool.
    assert_eq!(
        session.memory_reserved(),
        base,
        "registration leaves the session pool alone"
    );
    // The union of 1..=4 tasks is the table, with no duplicates; the
    // same listing gives the same assignment.
    let df = session.table("t");
    let analysis = block_on(df.analysis()).unwrap();
    let (_, physical) = block_on(session.physical(&analysis.analyzed, &analysis.env)).unwrap();
    let whole = table_ids(&session, "SELECT id FROM t");
    assert_eq!(whole, (0..14).collect::<Vec<_>>());
    for tasks in 1..=4_usize {
        let catalog = &analysis.env.catalog;
        let specs = catalog
            .task_specs(&physical, tasks, u64::try_from(tasks).unwrap())
            .unwrap();
        assert_eq!(
            specs,
            catalog
                .task_specs(&physical, tasks, u64::try_from(tasks).unwrap())
                .unwrap(),
            "deterministic"
        );
        assert_eq!(specs.len(), tasks);
        let mut union = Vec::new();
        for task in &specs {
            assert_eq!(task.scans.len(), 1, "one scan: {task:?}");
            union.extend(ids_of(
                &block_on(session.run_task(&physical, &analysis.env, task)).unwrap(),
            ));
        }
        union.sort_unstable();
        assert_eq!(union, whole, "{tasks} tasks");
    }
    // A scan without an entry reads the whole table, exactly as before.
    let single =
        ids_of(&block_on(session.run_task(&physical, &analysis.env, &TaskSpec::whole())).unwrap());
    assert_eq!(single, whole);
    // The test's analysis holds a table lease through its loaded
    // tables: release it before the tables are dropped.
    drop(analysis);
    drop(df);
    // Dropping the tables releases their entries and their bytes.
    assert!(session.drop_temp_view("t"));
    assert_eq!(session.relations().len(), 2, "u still holds the entry");
    assert!(session.drop_temp_view("u"));
    assert_eq!(session.relations().len(), 1);
    assert!(session.drop_temp_view("v"));
    assert!(session.relations().is_empty());
    assert_eq!(
        session.relations().reserved(),
        0,
        "the last lease returns the bytes"
    );
    assert_eq!(
        session.memory_reserved(),
        base,
        "queries and drops leave the session pool as it was"
    );
}

#[test]
fn task_specs_splits_only_the_streaming_pipeline() {
    // Broadcast builds and scalar subqueries read whole: tasks join
    // probe shares against the full build and evaluate the subquery
    // over its full input, so their union is the single run.
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batches_of = |values: Vec<i32>| {
        values
            .chunks(3)
            .map(|chunk| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int32Array::from(chunk.to_vec()))],
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    session
        .register_batches("t", Arc::clone(&schema), batches_of((0..14).collect()))
        .unwrap();
    session
        .register_batches("u", Arc::clone(&schema), batches_of((0..14).collect()))
        .unwrap();
    session
        .register_batches("v", Arc::clone(&schema), batches_of(vec![5]))
        .unwrap();
    let sql = "SELECT t.id FROM t JOIN u ON t.id = u.id WHERE t.id > (SELECT max(id) FROM v)";
    let whole = table_ids(&session, sql);
    assert_eq!(whole, (6..14).collect::<Vec<_>>());
    let df = block_on(session.sql(sql)).unwrap();
    let analysis = block_on(df.analysis()).unwrap();
    let (_, physical) = block_on(session.physical(&analysis.analyzed, &analysis.env)).unwrap();
    // A shared plan runs in one task until it lowers to an exchange.
    let shared = physical
        .clone()
        .with_shared(vec![krishiv_exec::physical::SharedPlan {
            id: krishiv_exec::physical::SharedId(0),
            plan: PhysNode::OneRow,
            explain: krishiv_exec::planner::ExplainNode::leaf("OneRowRelation", ""),
            deterministic: true,
            ordered: false,
        }])
        .unwrap();
    let e = analysis.env.catalog.task_specs(&shared, 2, 2).unwrap_err();
    assert!(e.to_string().contains("a shared plan across"), "{e}");
    assert_eq!(
        analysis
            .env
            .catalog
            .task_specs(&shared, 1, 1)
            .unwrap()
            .len(),
        1
    );
    for tasks in [2_usize, 3_usize] {
        let specs = analysis
            .env
            .catalog
            .task_specs(&physical, tasks, u64::try_from(tasks).unwrap())
            .unwrap();
        assert_eq!(specs.len(), tasks);
        // Only the streaming scan is dealt ranges; builds and the
        // subquery's scan read whole (no entry).
        let streaming = streaming_scans(physical.root());
        assert_eq!(streaming.len(), 1, "one streaming scan");
        for task in &specs {
            assert_eq!(task.scans.len(), 1, "probe only: {task:?}");
            assert!(
                task.scans.contains_key(&streaming[0].id),
                "the streaming scan: {task:?}"
            );
        }
        let mut union = Vec::new();
        for task in &specs {
            union.extend(ids_of(
                &block_on(session.run_task(&physical, &analysis.env, task)).unwrap(),
            ));
        }
        union.sort_unstable();
        assert_eq!(union, whole, "{tasks} tasks");
    }
    drop(analysis);
    drop(df);
}

#[test]
fn task_specs_rejects_no_tasks() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    session.register_batches("t", schema, vec![batch]).unwrap();
    let df = session.table("t");
    let analysis = block_on(df.analysis()).unwrap();
    let (_, physical) = block_on(session.physical(&analysis.analyzed, &analysis.env)).unwrap();
    let e = analysis
        .env
        .catalog
        .task_specs(&physical, 0, 0)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    // The analysis holds the old table through its loaded tables.
    drop(analysis);
    drop(df);
    // Replacing the table releases the old entry: one entry either way.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![2]))],
    )
    .unwrap();
    session.register_batches("t", schema, vec![batch]).unwrap();
    assert_eq!(session.relations().len(), 1);
    assert_eq!(table_ids(&session, "SELECT id FROM t"), [2]);
}

/// A scan plan over `source`: what `task_specs` deals splits for.
fn split_scan_plan(source: ReadSource) -> PhysicalPlan {
    use krishiv_exec::RowStageSettings;
    use krishiv_exec::physical::{ExprEnv, ScanNode};
    use krishiv_exec::planner::ExplainNode;
    use krishiv_io::ColumnPath;
    use krishiv_plan::{AttrId, BoundColumn};

    let scan = Scan {
        source,
        output: vec![BoundColumn {
            id: AttrId(1),
            name: "id".to_owned(),
            data_type: DataType::Integer,
            nullable: true,
        }],
        version: None,
    };
    PhysicalPlan::new(
        PhysNode::Scan(ScanNode {
            id: ScanId(1),
            scan,
            projection: vec![ColumnPath::new("id")],
            filter: None,
            runtime_filters: Vec::new(),
            row_stage: RowStageSettings::default(),
            row_id: false,
        }),
        Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            true,
        )])),
        ExplainNode::leaf("Scan", ""),
        ExprEnv {
            ansi: true,
            legacy_statistical_aggregate: false,
            zone: None,
            zone_name: None,
            query_start_micros: 0,
            user: Arc::from("u"),
            map_key_dedup_last_win: false,
            legacy_java_charsets: false,
            legacy_coding_error_action: false,
            legacy_time_parser: false,
            binary_output_style: None,
        },
    )
}

/// A table scan source over `name`.
fn table_source(name: &str) -> ReadSource {
    ReadSource::Table {
        name: vec![
            "spark_catalog".to_owned(),
            "default".to_owned(),
            name.to_owned(),
        ],
        options: BTreeMap::new(),
    }
}

#[test]
fn task_specs_rejects_what_it_cannot_split() {
    let session = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();
    session.register_batches("t", schema, vec![batch]).unwrap();
    block_on(session.sql("CREATE TEMPORARY VIEW w AS SELECT 1 AS id")).unwrap();
    // The analysis references the view (inlined afterwards), so the
    // catalog holds its entry while no scan names it.
    let df = block_on(session.sql("SELECT id FROM w")).unwrap();
    let analysis = block_on(df.analysis()).unwrap();
    let catalog = &analysis.env.catalog;
    // The view's own entry, as an analysis that kept it would hold
    // it: a view reaching split planning was not inlined.
    let view = session.state.catalog.get("w", false).unwrap();
    catalog.loaded.lock().unwrap().insert("w".to_owned(), view);

    // A scan of a table the analysis never loaded.
    let e = catalog
        .task_specs(&split_scan_plan(table_source("nope")), 2, 2)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert!(e.to_string().contains("was not loaded"), "{e}");

    // A scan of a temporary view: views inline during analysis, so a
    // view reaching split planning is a coordinator bug.
    let e = catalog
        .task_specs(&split_scan_plan(table_source("w")), 2, 2)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert!(e.to_string().contains("was not inlined"), "{e}");

    // A scan of files that were never opened.
    let e = catalog
        .task_specs(
            &split_scan_plan(ReadSource::Files {
                format: "parquet".to_owned(),
                paths: Vec::new(),
                options: BTreeMap::new(),
            }),
            2,
            2,
        )
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert!(e.to_string().contains("not opened"), "{e}");

    // More tasks than a `u32` holds: rejected before anything is dealt.
    let e = catalog
        .task_specs(
            &split_scan_plan(table_source("t")),
            usize::try_from(u32::MAX).unwrap() + 1,
            1,
        )
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    drop(analysis);
    drop(df);
}

/// A copied catalog leaves out the script results no copied view pins: the
/// clone has no frame that reads them, and a shared entry would keep the
/// result alive in both catalogs once its frame is gone.
#[test]
fn a_copied_catalog_leaves_out_script_results() {
    let s = int_session(vec![1]);
    let base = s.state.catalog.entry_count();
    let df = block_on(s.sql("BEGIN SELECT id FROM range(5); END")).unwrap();
    assert_eq!(s.state.catalog.entry_count(), base + 1);
    let c = s.clone_session().unwrap();
    assert_eq!(c.state.catalog.entry_count(), base);
    // The source's frame still reads its result.
    let rows: usize = block_on(df.collect())
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 5);
    // Once the frame goes, the next change of the source's catalog drops it.
    drop(df);
    block_on(s.sql("CREATE TEMP VIEW gc_v AS SELECT 1 AS a")).unwrap();
    assert_eq!(s.state.catalog.entry_count(), base + 1);
    block_on(c.sql("CREATE TEMP VIEW gc_v AS SELECT 1 AS a")).unwrap();
    assert_eq!(c.state.catalog.entry_count(), base + 1);
}

/// A script result a copied view pins is kept, under a handle of the
/// clone's own: once each session drops its view, each collects its entry
/// (a shared handle would keep both alive while the other session lives).
#[test]
fn a_pinned_script_result_is_collected_in_each_session_after_its_view_goes() {
    let s = int_session(vec![1]);
    let base = s.state.catalog.entry_count();
    let df = block_on(s.sql("BEGIN SELECT id FROM range(5); END")).unwrap();
    block_on(df.create_or_replace_temp_view("pinned_v")).unwrap();
    drop(df);
    // The view and the result it pins.
    assert_eq!(s.state.catalog.entry_count(), base + 2);
    let c = s.clone_session().unwrap();
    assert_eq!(c.state.catalog.entry_count(), base + 2);
    let rows: usize = block_on(block_on(c.sql("SELECT * FROM pinned_v")).unwrap().collect())
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 5);
    for session in [&s, &c] {
        block_on(session.sql("DROP VIEW pinned_v")).unwrap();
        block_on(session.sql("CREATE TEMP VIEW gc_v AS SELECT 1 AS a")).unwrap();
        assert_eq!(session.state.catalog.entry_count(), base + 1);
    }
}

/// A clone shares the relation store its copied memory tables' leases live
/// in: the batch splits of a copied table resolve through the clone's
/// store, and its tasks read the table.
#[test]
fn task_specs_of_a_copied_memory_table_resolve_in_the_clone() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let s = int_session(vec![1, 2, 3]);
    let c = s.clone_session().unwrap();
    let df = block_on(c.sql("SELECT id FROM t")).unwrap();
    let analysis = rt.block_on(df.analysis()).unwrap();
    let (_, physical) = rt
        .block_on(c.physical(&analysis.analyzed, &analysis.env))
        .unwrap();
    let specs = analysis.env.catalog.task_specs(&physical, 1, 1).unwrap();
    let mut batches = 0;
    for task in &specs {
        for splits in task.scans.values() {
            if let krishiv_exec::physical::ScanSplits::Batches { relation, range } = splits {
                c.relations()
                    .resolve(&relation.hash, range.clone())
                    .unwrap();
                batches += 1;
            }
        }
    }
    assert!(batches > 0, "a batch split: {specs:?}");
    let mut union = Vec::new();
    for task in &specs {
        union.extend(ids_of(
            &rt.block_on(c.run_task(&physical, &analysis.env, task))
                .unwrap(),
        ));
    }
    union.sort_unstable();
    assert_eq!(union, vec![1, 2, 3]);
}
