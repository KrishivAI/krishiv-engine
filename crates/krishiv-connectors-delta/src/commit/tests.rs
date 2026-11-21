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

use super::hook_store::HookStore;
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
    hooks: Arc<HookStore>,
    token: Arc<CancellationToken>,
}

impl Env {
    fn new() -> Self {
        Self::with_limits(DeltaLimits::default())
    }

    fn with_limits(limits: DeltaLimits) -> Self {
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
        let hooks = HookStore::new(Arc::new(krishiv_io::SettledLocalFileSystem::new()));
        sched.register_whole_listing_store(id.clone(), Arc::clone(&hooks) as _);
        let token = Arc::new(CancellationToken::new());
        let process: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
        sched.attach_memory(&process);
        let pool: &'static LogPool = Box::leak(Box::new(LogPool::new(2)));
        let engine = DeltaEngine::new(
            sched.context().with_memory(Arc::clone(&process)),
            id,
            Url::from_directory_path(&dir).unwrap(),
            rt.handle().clone(),
            limits,
            Arc::clone(&token),
            pool,
        )
        .unwrap();
        Self {
            _tmp: tmp,
            dir,
            rt,
            engine,
            hooks,
            token,
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
    add_values(path, &[("p", p), ("n", None), ("d", None), ("ts", None)])
}

/// One added file with partition values `values`.
fn add_values(path: &str, values: &[(&str, Option<&str>)]) -> RecordBatch {
    let schema = add_schema();
    let values = MapData::try_new(
        MapType::new(DataType::STRING, DataType::STRING, true),
        values.iter().map(|(k, v)| {
            (
                Scalar::String((*k).to_owned()),
                v.map_or(Scalar::Null(DataType::STRING), |v| {
                    Scalar::String(v.to_owned())
                }),
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

/// Another writer's blind append in partition `p`, as a commit file's actions.
fn other_append(path: &str, p: &str) -> Vec<Value> {
    vec![
        json!({"commitInfo": {"operation": "WRITE", "isBlindAppend": true}}),
        json!({"add": {"path": path, "partitionValues": {"p": p, "n": null, "d": null, "ts": null},
            "size": 1, "modificationTime": 1, "dataChange": true}}),
    ]
}

/// The table's own metaData action with `partitionColumns` replaced.
fn metadata_partitioned_by(columns: &[&str]) -> Value {
    let mut m = std::fs::read_to_string(fixture().join("_delta_log/00000000000000000000.json"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|a| a.get("metaData").is_some())
        .unwrap();
    m["metaData"]["partitionColumns"] = json!(columns);
    m
}

fn staged(write_id: &str, adds: Vec<RecordBatch>, read_set: ReadSet) -> StagedCommit {
    StagedCommit {
        table: "delta.`t`".to_owned(),
        write_id: write_id.to_owned(),
        idempotency_key: None,
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

/// A statement retried from its old snapshot after its commit landed finds its own write id
/// among the winners and does not append twice.
#[test]
fn a_retried_commit_finds_its_own_landing() {
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

fn keyed(write_id: &str, key: &str, file: &str) -> StagedCommit {
    let mut c = staged(write_id, vec![add(file, Some("x"))], ReadSet::default());
    c.idempotency_key = Some(key.to_owned());
    c
}

/// E34: a statement retried with a new write id but the same key appends once, from a fresh
/// snapshot or from the stale one; the key is a `txn` action; another key commits.
#[test]
fn a_retried_statement_with_the_same_idempotency_key_commits_once() {
    let env = Env::new();
    let s = env.snapshot();
    let first = env.commit(&s, keyed("w1", "job-7", "a.parquet")).unwrap();
    assert!(!first.deduplicated);
    let log = env.log(first.version);
    assert!(
        log.iter()
            .any(|a| a["txn"]["appId"] == "krishiv.idempotency:job-7" && a["txn"]["version"] == 1)
    );
    let latest = env.latest();

    let stale = env.commit(&s, keyed("w2", "job-7", "b.parquet")).unwrap();
    assert!(stale.deduplicated && stale.attempts == 0);
    let fresh = env
        .commit(&env.snapshot(), keyed("w3", "job-7", "c.parquet"))
        .unwrap();
    assert!(fresh.deduplicated);
    assert_eq!(fresh.version, latest);
    assert_eq!(env.latest(), latest, "no commit was made");

    let other = env
        .commit(&env.snapshot(), keyed("w4", "job-8", "d.parquet"))
        .unwrap();
    assert!(!other.deduplicated);
    assert_eq!(env.latest(), latest + 1);
}

/// E34: a keyed write that reads the table and loses its race to a retry of the same statement
/// (same key, another write id) is deduplicated before the winner is checked for conflicts: the
/// winner's append would otherwise be a concurrent-append conflict.
#[test]
fn a_same_key_winner_is_deduplicated_before_conflict_checks() {
    let env = Env::new();
    let s = env.snapshot();
    let reads = |write_id: &str, file: &str| {
        let mut c = staged(write_id, vec![add(file, Some("x"))], whole_table());
        c.idempotency_key = Some("job-9".to_owned());
        c
    };
    let first = env.commit(&s, reads("w1", "a.parquet")).unwrap();
    let latest = env.latest();
    let retry = env.commit(&s, reads("w2", "b.parquet")).unwrap();
    assert!(retry.deduplicated, "{retry:?}");
    assert_eq!(retry.version, first.version);
    assert_eq!(env.latest(), latest, "no commit was made");
    // An unrelated winner still conflicts.
    let (c, _) = class(env.commit(
        &s,
        staged("w3", vec![add("c.parquet", Some("x"))], whole_table()),
    ));
    assert!(
        format!("{c:?}").starts_with("DeltaConcurrentAppend"),
        "{c:?}"
    );
}

/// E34: with no attempt left, a winner carrying this write's key is still a deduplicated write,
/// not `DELTA_MAX_COMMIT_RETRIES_EXCEEDED`.
#[test]
fn a_same_key_winner_is_deduplicated_when_out_of_attempts() {
    let env = Env::new();
    let s = env.snapshot();
    let first = env.commit(&s, keyed("w1", "job-10", "a.parquet")).unwrap();
    let limits = RetryLimits {
        max_attempts: 0,
        ..fast()
    };
    let retry = env
        .commit_with(&s, keyed("w2", "job-10", "b.parquet"), limits)
        .unwrap();
    assert!(retry.deduplicated, "{retry:?}");
    assert_eq!(retry.version, first.version);
    assert_eq!(env.latest(), first.version);
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
    assert!(m.contains("Commit failed at version: 2"), "{m}");
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

/// The answer to a put that landed is lost, and a second writer commits before the landed
/// check: the check finds this write's commit and nothing is appended twice
/// (`delta_commit_response_lost`).
#[test]
fn delta_commit_response_lost() {
    let env = Env::new();
    let s = env.snapshot();
    let dir = env.dir.clone();
    env.hooks
        .on_put("_delta_log/00000000000000000002.json", true, move || {
            let text: String = other_append("o.parquet", "y")
                .iter()
                .map(|a| format!("{a}\n"))
                .collect();
            std::fs::write(dir.join("_delta_log/00000000000000000003.json"), text).unwrap();
        });
    let c = env
        .commit(
            &s,
            staged("w", vec![add("w.parquet", Some("x"))], ReadSet::default()),
        )
        .unwrap();
    assert_eq!(
        (c.version, c.attempts, c.found_by_landed_check),
        (2, 1, true)
    );
    assert_eq!(env.latest(), 3);
    let ours = env.log(2);
    assert!(ours.iter().any(|a| a["add"]["path"] == "w.parquet"));
}

/// A put that fails before writing: the landed check finds nothing and the failure is raised.
#[test]
fn a_failed_put_that_did_not_land_is_raised() {
    let env = Env::new();
    let s = env.snapshot();
    env.hooks
        .on_put("_delta_log/00000000000000000002.json", false, || {});
    let e = env
        .commit(
            &s,
            staged("w", vec![add("w.parquet", Some("x"))], ReadSet::default()),
        )
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TaskWriteFailed, "{e}");
    assert_eq!(env.latest(), 1);
}

/// The landed check reads more than one chunk of commits (four under test).
#[test]
fn the_landed_check_reads_every_chunk() {
    let env = Env::new();
    let s = env.snapshot();
    for v in 2..=7 {
        env.write_commit(v, &other_append(&format!("o{v}.parquet"), "y"));
    }
    env.hooks
        .on_put("_delta_log/00000000000000000008.json", true, || {});
    let c = env
        .commit(
            &s,
            staged("w", vec![add("w.parquet", Some("x"))], ReadSet::default()),
        )
        .unwrap();
    assert_eq!((c.version, c.found_by_landed_check), (8, true));
    assert_eq!(env.latest(), 8);
}

/// A writer that commits after the loser read the winners, before its rebuild: the rebuild is
/// at the last checked version, so that commit is checked on the next round
/// (`delta_rebuild_at_checked_version`).
#[test]
fn delta_rebuild_at_checked_version() {
    let env = Env::new();
    let s = env.snapshot();
    env.commit(
        &s,
        staged("a", vec![add("a.parquet", Some("z"))], ReadSet::default()),
    )
    .unwrap();
    let dir = env.dir.clone();
    env.hooks
        .on_missing_head("_delta_log/00000000000000000003.json", move || {
            let text: String = other_append("late.parquet", "x")
                .iter()
                .map(|a| format!("{a}\n"))
                .collect();
            std::fs::write(dir.join("_delta_log/00000000000000000003.json"), text).unwrap();
        });
    let (c, m) = class(env.commit(
        &s,
        staged("c", vec![add("c.parquet", Some("q"))], reads_partition("x")),
    ));
    assert_eq!(c, ErrorClass::DeltaConcurrentAppendWithPartitionHint, "{m}");
    assert!(m.contains("committed at version 3"), "{m}");
    assert_eq!(env.latest(), 3);
}

/// Every winner up to the last commit file is checked, not only the first.
#[test]
fn every_winner_is_checked() {
    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(2, &other_append("a.parquet", "z"));
    env.write_commit(3, &other_append("b.parquet", "x"));
    let (c, m) = class(env.commit(
        &s,
        staged("c", vec![add("c.parquet", Some("q"))], reads_partition("x")),
    ));
    assert_eq!(c, ErrorClass::DeltaConcurrentAppendWithPartitionHint, "{m}");
    assert!(m.contains("committed at version 3"), "{m}");
    let ok = env
        .commit(
            &s,
            staged("d", vec![add("d.parquet", Some("q"))], reads_partition("q")),
        )
        .unwrap();
    assert_eq!(ok.version, 4);
}

#[test]
fn the_time_bound_is_checked_after_a_lost_race() {
    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(2, &other_append("a.parquet", "z"));
    let limits = RetryLimits {
        timeout: std::time::Duration::ZERO,
        ..fast()
    };
    let (c, m) = class(env.commit_with(
        &s,
        staged("b", vec![add("b.parquet", Some("z"))], ReadSet::default()),
        limits,
    ));
    assert_eq!(c, ErrorClass::DeltaMaxCommitRetriesExceeded, "{m}");
    assert!(m.contains("tried 1 times"), "{m}");
}

#[test]
fn a_cancelled_query_stops_waiting_between_attempts() {
    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(2, &other_append("a.parquet", "z"));
    let token = Arc::clone(&env.token);
    let cancel = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(200));
        token.cancel();
    });
    let limits = RetryLimits {
        backoff_min: std::time::Duration::from_secs(60),
        backoff_max: std::time::Duration::from_secs(60),
        ..fast()
    };
    let started = std::time::Instant::now();
    let (c, _) = class(env.commit_with(
        &s,
        staged("b", vec![add("b.parquet", Some("z"))], ReadSet::default()),
        limits,
    ));
    cancel.join().unwrap();
    assert_eq!(c, ErrorClass::OperationCanceled);
    assert!(started.elapsed() < std::time::Duration::from_secs(20));
    assert_eq!(env.latest(), 2);
}

/// Hints list the partition columns in the table's `partitionColumns` order, not schema order.
#[test]
fn hints_follow_the_partition_column_order() {
    let env = Env::new();
    env.write_commit(
        2,
        &[
            json!({"commitInfo": {"operation": "REPLACE"}}),
            metadata_partitioned_by(&["n", "p", "d", "ts"]),
        ],
    );
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
    assert!(m.contains("[n=null, p=x, d=null, ts=null]"), "{m}");
}

/// On an unpartitioned table any add conflicts with a reader, without a hint.
#[test]
fn an_unpartitioned_table_conflicts_without_hints() {
    let env = Env::new();
    env.write_commit(
        2,
        &[
            json!({"commitInfo": {"operation": "REPLACE"}}),
            metadata_partitioned_by(&[]),
        ],
    );
    let s = env.snapshot();
    env.commit(
        &s,
        staged("a", vec![add_values("a.parquet", &[])], ReadSet::default()),
    )
    .unwrap();
    let reader = ReadSet {
        predicates: reads_partition("x").predicates,
        ..ReadSet::default()
    };
    let (c, m) = class(env.commit(&s, staged("b", vec![add_values("b.parquet", &[])], reader)));
    assert_eq!(c, ErrorClass::DeltaConcurrentAppendWithoutHint, "{m}");
    assert!(
        m.contains("a concurrent WRITE added data to table delta.`t` committed at version 3"),
        "{m}"
    );
}

/// A NULL partition value matches `p IS NULL` and not `p = 'x'` (NULL is not TRUE).
#[test]
fn null_partition_values_match_only_true_predicates() {
    let is_null = ReadSet {
        predicates: vec![Arc::new(Predicate::is_null(Expression::column(["p"])))],
        ..ReadSet::default()
    };
    let env = Env::new();
    let s = env.snapshot();
    env.commit(
        &s,
        staged("a", vec![add("a.parquet", None)], ReadSet::default()),
    )
    .unwrap();
    let (c, m) = class(env.commit(&s, staged("b", vec![add("b.parquet", None)], is_null)));
    assert_eq!(c, ErrorClass::DeltaConcurrentAppendWithPartitionHint, "{m}");
    assert!(m.contains("[p=null, n=null, d=null, ts=null]"), "{m}");
    let ok = env
        .commit(
            &s,
            staged("c", vec![add("c.parquet", Some("x"))], reads_partition("x")),
        )
        .unwrap();
    assert_eq!(ok.version, 3);
}

/// A legacy (1, 2) table implies `appendOnly` and `invariants`: a winner protocol without
/// `invariants` drops it, one listing both and `domainMetadata` upgrades the table and the
/// loser commits on it.
#[test]
fn legacy_protocols_compare_with_their_implied_features() {
    let info = json!({"commitInfo": {"operation": "UPGRADE PROTOCOL"}});
    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(
        2,
        &[
            info.clone(),
            json!({"protocol": {"minReaderVersion": 1, "minWriterVersion": 7,
                "writerFeatures": ["appendOnly", "domainMetadata"]}}),
        ],
    );
    let (c, _) = class(env.commit(
        &s,
        staged("b", vec![add("b.parquet", Some("x"))], ReadSet::default()),
    ));
    assert_eq!(c, ErrorClass::DeltaProtocolChanged);

    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(
        2,
        &[
            info,
            json!({"protocol": {"minReaderVersion": 1, "minWriterVersion": 7,
                "writerFeatures": ["appendOnly", "invariants", "domainMetadata"]}}),
        ],
    );
    let ok = env
        .commit(
            &s,
            staged("b", vec![add("b.parquet", Some("x"))], ReadSet::default()),
        )
        .unwrap();
    assert_eq!(ok.version, 3);
}

#[test]
fn implied_features_of_each_legacy_version() {
    use super::conflict::supported_features;
    let names = |r, w, wf: Option<&[&str]>| {
        let wf: Option<Vec<String>> = wf.map(|f| f.iter().map(|s| (*s).to_owned()).collect());
        supported_features(r, w, None, wf.as_deref())
            .into_iter()
            .collect::<Vec<_>>()
    };
    assert_eq!(names(1, 2, None), ["appendOnly", "invariants"]);
    // columnMapping only from reader 2 with writer 5 or 6; writer 7 implies nothing.
    assert!(!names(1, 5, None).contains(&"columnMapping".to_owned()));
    assert_eq!(names(2, 2, None), ["appendOnly", "invariants"]);
    assert!(names(2, 6, None).contains(&"columnMapping".to_owned()));
    assert_eq!(names(2, 7, Some(&["appendOnly"])), ["appendOnly"]);
    assert_eq!(
        names(1, 2, None),
        names(1, 7, Some(&["appendOnly", "invariants"]))
    );
    assert_eq!(
        names(2, 5, None),
        [
            "appendOnly",
            "changeDataFeed",
            "checkConstraints",
            "columnMapping",
            "generatedColumns",
            "invariants"
        ]
    );
    assert!(names(1, 6, None).contains(&"identityColumns".to_owned()));
}

/// Winner parsing: the first commitInfo and protocol win, unknown actions are skipped, and a line
/// that is not JSON, a file action without a path or a protocol without versions fail.
#[test]
fn winner_parsing() {
    use super::winner::parse;
    let env = Env::new();
    let text = "{\"commitInfo\":{\"operation\":\"A\"}}\n{\"commitInfo\":{\"operation\":\"B\"}}\n\
                {\"protocol\":{\"minReaderVersion\":1,\"minWriterVersion\":2}}\n\
                {\"protocol\":{\"minReaderVersion\":3,\"minWriterVersion\":7}}\n\
                {\"cdc\":{\"path\":\"c\"}}\n{\"unknownAction\":{}}\n\n\
                {\"remove\":{\"path\":\"r\",\"partitionValues\":{\"p\":null}}}\n";
    let w = parse(&env.engine, 5, text.as_bytes(), "t").unwrap();
    assert_eq!(w.operation(), "A");
    assert_eq!(w.protocol.as_ref().unwrap().min_writer_version, 2);
    assert_eq!(w.removes[0].path, "r");
    assert_eq!(w.removes[0].partition_values["p"], None);
    assert!(w.adds.is_empty());
    assert!(
        w.conflicting_commit()
            .starts_with("\nConflicting commit: {")
    );
    assert!(w.conflicting_commit().contains("\"version\":5"));
    let none = parse(&env.engine, 5, b"{\"add\":{\"path\":\"a\"}}\n", "t").unwrap();
    assert_eq!(
        (none.operation(), none.conflicting_commit()),
        ("TRANSACTION".to_owned(), String::new())
    );
    for bad in [
        "not json\n",
        "[1]\n",
        "{\"add\":{\"size\":1}}\n",
        "{\"protocol\":{\"minReaderVersion\":1}}\n",
    ] {
        let e = parse(&env.engine, 5, bad.as_bytes(), "t").unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{bad}");
    }
}

/// A winner over `krishiv.catalog.maxMetadataBytes` fails the commit with the limit's class.
#[test]
fn a_winner_over_the_metadata_limit_fails_the_commit() {
    let env = Env::with_limits(DeltaLimits {
        max_metadata_bytes: 4096,
        ..DeltaLimits::default()
    });
    let s = env.snapshot();
    let big = json!({"commitInfo": {"operation": "WRITE", "pad": "x".repeat(5000)}});
    env.write_commit(2, &[big]);
    let (c, _) = class(env.commit(
        &s,
        staged("b", vec![add("b.parquet", Some("x"))], ReadSet::default()),
    ));
    assert_eq!(c, ErrorClass::KrishivCatalogLimitExceeded);
}

/// A winner add whose partition value does not parse counts as a match.
#[test]
fn an_unparseable_partition_value_counts_as_a_match() {
    let env = Env::new();
    let s = env.snapshot();
    env.write_commit(
        2,
        &[
            json!({"commitInfo": {"operation": "WRITE"}}),
            json!({"add": {"path": "bad.parquet",
                "partitionValues": {"p": "z", "n": "not-a-number", "d": null, "ts": null},
                "size": 1, "modificationTime": 1, "dataChange": true}}),
        ],
    );
    let (c, m) = class(env.commit(
        &s,
        staged("b", vec![add("b.parquet", Some("q"))], reads_partition("x")),
    ));
    assert_eq!(c, ErrorClass::DeltaConcurrentAppendWithPartitionHint, "{m}");
    assert!(m.contains("n=not-a-number"), "{m}");
}

/// A winner nested past the depth limit fails the commit with the limit's class.
#[test]
fn a_winner_past_the_depth_limit_fails_the_commit() {
    let env = Env::with_limits(DeltaLimits {
        max_depth: 4,
        ..DeltaLimits::default()
    });
    let s = env.snapshot();
    env.write_commit(
        2,
        &[json!({"commitInfo": {"operation": "WRITE", "x": {"a": {"b": {"c": {"d": 1}}}}}})],
    );
    let (c, _) = class(env.commit(
        &s,
        staged("b", vec![add("b.parquet", Some("x"))], ReadSet::default()),
    ));
    assert_eq!(c, ErrorClass::KrishivCatalogLimitExceeded);
}

#[test]
fn retry_limits_come_from_the_config() {
    let mut c = krishiv_common::config::Config::new();
    assert_eq!(RetryLimits::from_config(&c), RetryLimits::default());
    c.set("spark.databricks.delta.maxCommitAttempts", "7")
        .unwrap();
    c.set("krishiv.write.commitRetryTimeout", "5s").unwrap();
    let l = RetryLimits::from_config(&c);
    assert_eq!(l.max_attempts, 7);
    assert_eq!(l.timeout, std::time::Duration::from_secs(5));
    assert_eq!(l.backoff_min, RetryLimits::default().backoff_min);
}
