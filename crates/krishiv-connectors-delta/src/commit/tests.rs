// SPDX-License-Identifier: Apache-2.0
//! The commit loop and the conflict checker on a copy of the oracle table `partitioned`
//! (Spark 4.2.0 + delta-spark 4.4.1, protocol (1, 2), partitioned by `p`, `n`, `d`, `ts`):
//! `delta_conflict_matrix` (one case per rule), `delta_commit_response_lost`,
//! `delta_rebuild_at_checked_version`, the attempt bound and the commitInfo Krishiv writes.
//!
//! Commits add rows for files that are never written: the log is all the checker reads.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::RecordBatch;
use delta_kernel::expressions::{Expression, MapData, Predicate, Scalar};
use delta_kernel::schema::{DataType, MapType};
use delta_kernel::{Snapshot, SnapshotRef};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::CancellationToken;
use krishiv_io::{IoConfig, IoScheduler, StoreId};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use serde_json::{Value, json};
use url::Url;

use super::run::{delay, remove_actions};
use super::winner::FileAction;
use super::*;
use crate::engine::eval::create_many;
use crate::engine::pool::LogPool;
use crate::engine::{DeltaEngine, DeltaLimits, KrishivEngineData};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/delta-4.4.1/partitioned")
}

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

struct Env {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    rt: tokio::runtime::Runtime,
    engine: DeltaEngine,
}

impl Env {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        copy_dir(&fixture(), &dir);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let sched = IoScheduler::new(IoConfig::default());
        let id = StoreId::new("file");
        sched.register_whole_listing_store(
            id.clone(),
            Arc::new(krishiv_io::SettledLocalFileSystem::new()),
        );
        let process: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        sched.attach_memory(&process);
        let pool: &'static LogPool = Box::leak(Box::new(LogPool::new(2)));
        let engine = DeltaEngine::new(
            sched.context().with_memory(Arc::clone(&process)),
            id,
            Url::from_directory_path(&dir).unwrap(),
            rt.handle().clone(),
            DeltaLimits::default(),
            Arc::new(CancellationToken::new()),
            pool,
        )
        .unwrap();
        Self {
            _tmp: tmp,
            dir,
            rt,
            engine,
        }
    }

    fn snapshot(&self) -> SnapshotRef {
        let root = self.engine.table_root().as_str().to_owned();
        self.rt
            .block_on(
                self.engine
                    .delta_blocking(move |e| Snapshot::builder_for(&root).build(e).unwrap()),
            )
            .unwrap()
    }

    fn commit(
        &self,
        snapshot: &SnapshotRef,
        staged: StagedCommit,
    ) -> Result<Committed, KrishivError> {
        self.commit_with(snapshot, staged, fast())
    }

    fn commit_with(
        &self,
        snapshot: &SnapshotRef,
        staged: StagedCommit,
        limits: RetryLimits,
    ) -> Result<Committed, KrishivError> {
        self.rt
            .block_on(commit(&self.engine, Arc::clone(snapshot), staged, limits))
    }

    /// The scan metadata rows of `snapshot`, with only the file whose path starts with `prefix`
    /// selected.
    fn removes(&self, snapshot: &SnapshotRef, prefix: &str) -> StagedRemoves {
        let snapshot = Arc::clone(snapshot);
        let prefix = prefix.to_owned();
        self.rt
            .block_on(self.engine.delta_blocking(move |e| {
                let scan = snapshot.scan_builder().build().unwrap();
                let mut batches = Vec::new();
                for m in scan.scan_metadata(e).unwrap() {
                    let (data, selection) = m.unwrap().scan_files.into_parts();
                    let batch = KrishivEngineData::from_engine_data(data)
                        .unwrap()
                        .into_batch();
                    batches.push((batch, selection));
                }
                let (batch, selection) = batches.into_iter().next().unwrap();
                // Only selected rows (the live adds) have a path.
                let selection: Vec<bool> = (0..selection.len())
                    .map(|i| {
                        selection[i] && {
                            let mut one = vec![false; batch.num_rows()];
                            one[i] = true;
                            let staged = StagedRemoves {
                                scan_rows: batch.clone(),
                                selection: one,
                            };
                            remove_actions(&staged).unwrap()[0]
                                .path
                                .starts_with(&prefix)
                        }
                    })
                    .collect();
                assert_eq!(selection.iter().filter(|s| **s).count(), 1, "{prefix}");
                StagedRemoves {
                    scan_rows: batch,
                    selection,
                }
            }))
            .unwrap()
    }

    fn log(&self, version: u64) -> Vec<Value> {
        std::fs::read_to_string(self.dir.join(format!("_delta_log/{version:020}.json")))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn write_commit(&self, version: u64, actions: &[Value]) {
        let text: String = actions.iter().map(|a| format!("{a}\n")).collect();
        std::fs::write(
            self.dir.join(format!("_delta_log/{version:020}.json")),
            text,
        )
        .unwrap();
    }

    fn latest(&self) -> u64 {
        std::fs::read_dir(self.dir.join("_delta_log"))
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().into_string().unwrap();
                n.strip_suffix(".json")?.parse::<u64>().ok()
            })
            .max()
            .unwrap()
    }
}

/// Short waits, few attempts.
fn fast() -> RetryLimits {
    RetryLimits {
        max_attempts: 10,
        timeout: std::time::Duration::from_secs(60),
        backoff_min: std::time::Duration::from_millis(1),
        backoff_max: std::time::Duration::from_millis(4),
    }
}

/// The kernel's `add_files_schema` (`path`, `partitionValues`, `size`, `modificationTime`,
/// `stats`).
fn add_schema() -> delta_kernel::schema::StructType {
    use delta_kernel::schema::StructField;
    let stats = DataType::try_struct_type([
        StructField::nullable("numRecords", DataType::LONG),
        StructField::nullable("nullCount", DataType::try_struct_type([]).unwrap()),
        StructField::nullable("minValues", DataType::try_struct_type([]).unwrap()),
        StructField::nullable("maxValues", DataType::try_struct_type([]).unwrap()),
        StructField::nullable("tightBounds", DataType::BOOLEAN),
    ])
    .unwrap();
    delta_kernel::schema::StructType::new_unchecked([
        StructField::not_null("path", DataType::STRING),
        StructField::not_null(
            "partitionValues",
            MapType::new(DataType::STRING, DataType::STRING, true),
        ),
        StructField::not_null("size", DataType::LONG),
        StructField::not_null("modificationTime", DataType::LONG),
        StructField::nullable("stats", stats),
    ])
}

/// One added file in partition `p` (the other partition columns NULL).
fn add(path: &str, p: Option<&str>) -> RecordBatch {
    let schema = add_schema();
    let values = MapData::try_new(
        MapType::new(DataType::STRING, DataType::STRING, true),
        [
            ("p", p.map(str::to_owned)),
            ("n", None),
            ("d", None),
            ("ts", None),
        ]
        .into_iter()
        .map(|(k, v)| {
            (
                Scalar::String(k.to_owned()),
                v.map_or(Scalar::Null(DataType::STRING), Scalar::String),
            )
        }),
    )
    .unwrap();
    let stats = schema.field("stats").unwrap().data_type().clone();
    create_many(
        &schema,
        &[vec![
            Scalar::String(path.to_owned()),
            Scalar::Map(values),
            Scalar::Long(10),
            Scalar::Long(1),
            Scalar::Null(stats),
        ]],
    )
    .unwrap()
}

fn staged(write_id: &str, adds: Vec<RecordBatch>, read_set: ReadSet) -> StagedCommit {
    StagedCommit {
        table: "delta.`t`".to_owned(),
        write_id: write_id.to_owned(),
        operation: CommitOperation::Write,
        operation_parameters: vec![("mode".to_owned(), Some("Append".to_owned()))],
        operation_metrics: Vec::new(),
        adds,
        removes: Vec::new(),
        read_set,
    }
}

fn whole_table() -> ReadSet {
    ReadSet {
        whole_table: true,
        ..ReadSet::default()
    }
}

/// A read of partition `p = value`.
fn reads_partition(value: &str) -> ReadSet {
    ReadSet {
        predicates: vec![Arc::new(Predicate::eq(
            Expression::column(["p"]),
            Expression::literal(value),
        ))],
        ..ReadSet::default()
    }
}

fn class(r: Result<Committed, KrishivError>) -> (ErrorClass, String) {
    let e = r.unwrap_err();
    (e.error_class(), e.to_string())
}

#[test]
fn a_blind_append_commits_and_records_krishivs_commit_info() {
    let env = Env::new();
    let s = env.snapshot();
    let c = env
        .commit(
            &s,
            staged(
                "w1",
                vec![add("x/1.parquet", Some("x"))],
                ReadSet::default(),
            ),
        )
        .unwrap();
    assert_eq!(
        (c.version, c.attempts, c.found_by_landed_check),
        (2, 1, false)
    );
    let log = env.log(2);
    let info = log.iter().find_map(|a| a.get("commitInfo")).unwrap();
    assert_eq!(info["tags"][WRITE_ID_TAG], "w1");
    assert_eq!(info["isolationLevel"], "Serializable");
    assert_eq!(info["readVersion"], 1);
    assert_eq!(info["isBlindAppend"], true);
    assert_eq!(info["operation"], "WRITE");
    assert_eq!(info["operationParameters"]["mode"], "Append");
    assert!(log.iter().any(|a| a["add"]["path"] == "x/1.parquet"));
}

#[test]
fn two_blind_appends_from_one_snapshot_both_commit() {
    let env = Env::new();
    let s = env.snapshot();
    env.commit(
        &s,
        staged("a", vec![add("a.parquet", Some("x"))], ReadSet::default()),
    )
    .unwrap();
    let b = env
        .commit(
            &s,
            staged("b", vec![add("b.parquet", Some("x"))], ReadSet::default()),
        )
        .unwrap();
    assert_eq!((b.version, b.attempts), (3, 2));
    assert_eq!(env.latest(), 3);
}

/// One case per rule of section 5.3 (`delta_conflict_matrix`).
#[test]
fn delta_conflict_matrix() {
    // 6. Append: a whole-table reader conflicts with any add, with the add's partition.
    let env = Env::new();
    let s = env.snapshot();
    env.commit(
        &s,
        staged("a", vec![add("a.parquet", Some("x"))], ReadSet::default()),
    )
    .unwrap();
    let (c, m) = class(env.commit(
        &s,
        staged("b", vec![add("b.parquet", Some("y"))], whole_table()),
    ));
    assert_eq!(c, ErrorClass::DeltaConcurrentAppendWithPartitionHint, "{m}");
    assert!(
        m.contains("a concurrent WRITE added data to table delta.`t` committed at version 2"),
        "{m}"
    );
    assert!(
        m.contains("in the partition [p=x, n=null, d=null, ts=null]"),
        "{m}"
    );
    assert!(m.contains(conflict::DOC_LINK), "{m}");

    // 6. Append: a reader of p = 'x' conflicts with an add in x, not with one in z.
    let env = Env::new();
    let s = env.snapshot();
    env.commit(
        &s,
        staged("a", vec![add("a.parquet", Some("z"))], ReadSet::default()),
    )
    .unwrap();
    let ok = env
        .commit(
            &s,
            staged("b", vec![add("b.parquet", Some("x"))], reads_partition("x")),
        )
        .unwrap();
    assert_eq!(ok.version, 3);
    let (c, _) = class(env.commit(
        &s,
        staged("c", vec![add("c.parquet", Some("q"))], reads_partition("x")),
    ));
    assert_eq!(c, ErrorClass::DeltaConcurrentAppendWithPartitionHint);

    // 7. Delete-read: a winner removes a file the transaction read.
    let env = Env::new();
    let s = env.snapshot();
    let removed = env.removes(&s, "p=a%20b/");
    let read = remove_actions(&removed).unwrap();
    let mut a = staged("a", Vec::new(), whole_table());
    a.operation = CommitOperation::Delete;
    a.removes = vec![removed];
    env.commit(&s, a).unwrap();
    let reader = ReadSet {
        files: read,
        predicates: reads_partition("a b").predicates,
        ..ReadSet::default()
    };
    let (c, m) = class(env.commit(&s, staged("b", vec![add("b.parquet", Some("a b"))], reader)));
    assert_eq!(
        c,
        ErrorClass::DeltaConcurrentDeleteReadWithPartitionHint,
        "{m}"
    );
    assert!(
        m.contains(
            "a concurrent DELETE deleted data from table delta.`t` (committed at version 2)"
        ),
        "{m}"
    );
    assert!(
        m.contains("[p=a b, n=1, d=2024-01-01, ts=2024-01-01T12:30:45.123456Z]"),
        "{m}"
    );

    // 7. Delete-read: any winner remove after a whole-table read, without a hint.
    let (c, _) = class(env.commit(&s, staged("c", Vec::new(), whole_table())));
    assert_eq!(c, ErrorClass::DeltaConcurrentDeleteReadWithoutHint);

    // 8. Delete-delete: both remove one file.
    let env = Env::new();
    let s = env.snapshot();
    let mut a = staged("a", Vec::new(), ReadSet::default());
    a.operation = CommitOperation::Delete;
    a.removes = vec![env.removes(&s, "p=a%20b/")];
    env.commit(&s, a.clone()).unwrap();
    a.write_id = "b".to_owned();
    let (c, m) = class(env.commit(&s, a));
    assert_eq!(
        c,
        ErrorClass::DeltaConcurrentDeleteDeleteWithPartitionHint,
        "{m}"
    );
}

#[test]
fn metadata_protocol_and_app_id_winners() {
    let commit_info = json!({"commitInfo": {"operation": "SET TBLPROPERTIES", "timestamp": 1}});
    // The table's own metaData action, as a winner would write it again.
    let metadata = std::fs::read_to_string(fixture().join("_delta_log/00000000000000000000.json"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|a| a.get("metaData").is_some())
        .unwrap();

    // 2. Metadata: any winner metaData conflicts, with the winner's commitInfo.
    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(2, &[commit_info.clone(), metadata.clone()]);
    let (c, m) = class(env.commit(
        &s,
        staged("b", vec![add("b.parquet", Some("x"))], ReadSet::default()),
    ));
    assert_eq!(c, ErrorClass::DeltaMetadataChanged, "{m}");
    assert!(
        m.contains("The metadata of the Delta table delta.`t` has been changed"),
        "{m}"
    );
    assert!(m.contains("Conflicting commit: {"), "{m}");
    assert!(m.contains("\"version\":2"), "{m}");

    // 2. ... except an identity high-water-mark update.
    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(
        2,
        &[
            json!({"commitInfo": {"operation": "WRITE", "tags": {"delta.identity.schemaUpdate": "true"}}}),
            metadata,
        ],
    );
    let ok = env
        .commit(
            &s,
            staged("b", vec![add("b.parquet", Some("x"))], ReadSet::default()),
        )
        .unwrap();
    assert_eq!(ok.version, 3);

    // 1. Protocol: a winner that drops to writer 1 changed the protocol.
    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(
        2,
        &[
            commit_info.clone(),
            json!({"protocol": {"minReaderVersion": 1, "minWriterVersion": 1}}),
        ],
    );
    let (c, _) = class(env.commit(
        &s,
        staged("b", vec![add("b.parquet", Some("x"))], ReadSet::default()),
    ));
    assert_eq!(c, ErrorClass::DeltaProtocolChanged);

    // 1. Protocol: a winner adding a feature Krishiv does not write.
    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(
        2,
        &[
            commit_info.clone(),
            json!({"protocol": {"minReaderVersion": 1, "minWriterVersion": 7,
                "writerFeatures": ["appendOnly", "invariants", "rowTracking", "domainMetadata"]}}),
        ],
    );
    let (c, m) = class(env.commit(
        &s,
        staged("b", vec![add("b.parquet", Some("x"))], ReadSet::default()),
    ));
    assert_eq!(c, ErrorClass::DeltaUnsupportedFeaturesForWrite, "{m}");
    assert!(m.contains("rowTracking"), "{m}");

    // 4. App ids: a winner txn for an app id the transaction read.
    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(
        2,
        &[
            commit_info,
            json!({"txn": {"appId": "stream-1", "version": 3}}),
        ],
    );
    let reader = ReadSet {
        app_ids: vec!["stream-1".to_owned()],
        ..ReadSet::default()
    };
    let (c, _) = class(env.commit(&s, staged("b", vec![add("b.parquet", Some("x"))], reader)));
    assert_eq!(c, ErrorClass::DeltaConcurrentTransaction);
}

/// The answer to a commit that landed is lost, and a second writer commits before the retry:
/// the retry finds its own write id and does not append twice.
#[test]
fn delta_commit_response_lost() {
    let env = Env::new();
    let s = env.snapshot();
    let first = staged("w", vec![add("w.parquet", Some("x"))], ReadSet::default());
    env.commit(&s, first.clone()).unwrap();
    env.commit(
        &s,
        staged(
            "other",
            vec![add("o.parquet", Some("x"))],
            ReadSet::default(),
        ),
    )
    .unwrap();
    let retried = env.commit(&s, first).unwrap();
    assert_eq!((retried.version, retried.found_by_landed_check), (2, true));
    assert_eq!(env.latest(), 3);
}

/// A loser checks every winner up to the last commit file, not only the first: here the first
/// winner does not conflict and the second does.
#[test]
fn delta_rebuild_at_checked_version() {
    let env = Env::new();
    let s = env.snapshot();
    env.commit(
        &s,
        staged("a", vec![add("a.parquet", Some("z"))], ReadSet::default()),
    )
    .unwrap();
    env.commit(
        &s,
        staged("b", vec![add("b.parquet", Some("x"))], ReadSet::default()),
    )
    .unwrap();
    let (c, m) = class(env.commit(
        &s,
        staged("c", vec![add("c.parquet", Some("q"))], reads_partition("x")),
    ));
    assert_eq!(c, ErrorClass::DeltaConcurrentAppendWithPartitionHint, "{m}");
    assert!(m.contains("committed at version 3"), "{m}");
    // A loser that does not conflict commits after the last checked version.
    let ok = env
        .commit(
            &s,
            staged("d", vec![add("d.parquet", Some("q"))], reads_partition("q")),
        )
        .unwrap();
    assert_eq!(ok.version, 4);
}

#[test]
fn the_attempt_bound_is_deltas_max_commit_attempts() {
    let env = Env::new();
    let s = env.snapshot();
    env.commit(
        &s,
        staged("a", vec![add("a.parquet", Some("z"))], ReadSet::default()),
    )
    .unwrap();
    let limits = RetryLimits {
        max_attempts: 0,
        ..fast()
    };
    let (c, m) = class(env.commit_with(
        &s,
        staged("b", vec![add("b.parquet", Some("z"))], ReadSet::default()),
        limits,
    ));
    assert_eq!(c, ErrorClass::DeltaMaxCommitRetriesExceeded, "{m}");
    assert!(m.contains("it has been tried 1 times"), "{m}");
    assert!(m.contains("Commit started at version: 2"), "{m}");
    assert!(m.contains("Commit failed at version: 3"), "{m}");
    assert!(
        m.contains("Number of actions attempted to commit: 2"),
        "{m}"
    );
}

#[test]
fn backoff_doubles_up_to_its_bound_with_jitter() {
    let limits = RetryLimits::default();
    let ms = |n| delay(&limits, "w", n).as_millis();
    assert!((50..=100).contains(&ms(1)), "{}", ms(1));
    assert!((100..=200).contains(&ms(2)), "{}", ms(2));
    assert!((30_000..=60_000).contains(&ms(40)), "{}", ms(40));
    assert_eq!(delay(&limits, "w", 3), delay(&limits, "w", 3));
}

#[test]
fn a_remove_carries_its_path_and_partition_values() {
    let env = Env::new();
    let s = env.snapshot();
    let r = env.removes(&s, "p=a%253Db/");
    let actions = remove_actions(&r).unwrap();
    assert_eq!(actions.len(), 1);
    let FileAction {
        path,
        partition_values,
    } = &actions[0];
    assert!(path.starts_with("p=a%253Db/"), "{path}");
    assert_eq!(partition_values["p"].as_deref(), Some("a=b"));
    assert_eq!(partition_values["n"].as_deref(), Some("-1"));
}

#[test]
fn an_oracle_tables_writer_protocol_is_read_from_its_snapshot() {
    use crate::features::{WriteProtocol, check_write};
    let env = Env::new();
    let p = WriteProtocol::of(&env.snapshot());
    assert_eq!((p.min_writer_version, p.invariants_in_use), (2, false));
    assert!(check_write("t", &p).is_ok());
}
