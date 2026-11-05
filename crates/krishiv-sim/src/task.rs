// SPDX-License-Identifier: Apache-2.0
//! The task descriptor a coordinator ships with a fragment's bytes
//! (design P2-T176 §2, Q2): a [`TaskSpec`] as an unstable protobuf message,
//! `krishiv.task.v1alpha`. No compatibility promise: P5-T004 freezes the
//! task envelope. It carries the submitted query's execution settings
//! ([`ExecConfig`]), so a remote task runs as a local one would, and the
//! file splits the coordinator dealt to each scan (P2-T178): the files with
//! their pinned versions, so a worker reads exactly what was listed.
//! Relation-batch splits are refused: a relation lives in one process.

use std::collections::BTreeMap;
use std::time::Duration;

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::ExecConfig;
use krishiv_exec::physical::{
    ExchangeId, ExchangeInput, ScanId, ScanSplits, SplitFileRef, TaskSpec,
};
use krishiv_io::split::{FilePartition, FileSplit};
use krishiv_io::{ByteRange, StoreId, VersionToken};
use krishiv_io::{DisabledReason, FilterId, FilterSnapshot, KeyFilter};
use krishiv_scheduler::Partials;
use object_store::path::Path;
use prost::Message as _;
use std::sync::Arc;

/// `krishiv.task.v1alpha.Task`.
#[derive(Clone, PartialEq, prost::Message)]
struct Task {
    #[prost(uint32, tag = "1")]
    stage: u32,
    #[prost(uint32, tag = "2")]
    stage_attempt: u32,
    #[prost(uint32, tag = "3")]
    partition: u32,
    #[prost(uint32, tag = "4")]
    partitions: u32,
    #[prost(uint32, tag = "5")]
    attempt: u32,
    #[prost(message, repeated, tag = "6")]
    exchanges: Vec<Input>,
    #[prost(uint32, optional, tag = "7")]
    output: Option<u32>,
    #[prost(message, optional, tag = "8")]
    settings: Option<Settings>,
    #[prost(message, repeated, tag = "9")]
    scans: Vec<Scan>,
    #[prost(message, repeated, tag = "10")]
    filters: Vec<FilterValue>,
}

/// `krishiv.task.v1alpha.FilterValue`: a runtime filter's value (an entry
/// of [`TaskSpec::filters`]) or one task's partial of it; no `keys` is a
/// lost partial.
#[derive(Clone, PartialEq, prost::Message)]
struct FilterValue {
    #[prost(uint64, tag = "1")]
    id: u64,
    /// `KeyFilter` v1 bytes.
    #[prost(bytes = "vec", optional, tag = "2")]
    keys: Option<Vec<u8>>,
}

/// `krishiv.task.v1alpha.Partials`: what a map task's reply carries.
#[derive(Clone, PartialEq, prost::Message)]
struct PartialList {
    #[prost(message, repeated, tag = "1")]
    partials: Vec<FilterValue>,
}

/// One map task's filter partials as the bytes its reply carries.
#[must_use]
pub fn encode_partials(partials: &Partials) -> Vec<u8> {
    PartialList {
        partials: partials
            .iter()
            .map(|(id, s)| FilterValue {
                id: id.0,
                keys: match s {
                    FilterSnapshot::Final(k) => Some(k.encode()),
                    _ => None,
                },
            })
            .collect(),
    }
    .encode_to_vec()
}

/// The partials of `bytes` ([`encode_partials`]).
///
/// # Errors
/// `INTERNAL_ERROR` for bytes that are not partials, or a filter's bytes
/// that are not a `KeyFilter`.
pub fn decode_partials(bytes: &[u8]) -> Result<Partials, KrishivError> {
    let list = PartialList::decode(bytes).map_err(|e| internal(format!("partials: {e}")))?;
    list.partials
        .into_iter()
        .map(|p| {
            let snapshot = match p.keys {
                Some(k) => FilterSnapshot::Final(Arc::new(
                    KeyFilter::decode(&k).map_err(|e| internal(format!("filter {}: {e}", p.id)))?,
                )),
                None => FilterSnapshot::Disabled(DisabledReason::ProducerFailed),
            };
            Ok((FilterId(p.id), snapshot))
        })
        .collect()
}

/// `krishiv.task.v1alpha.Scan`: one entry of [`TaskSpec::scans`], file
/// splits only.
#[derive(Clone, PartialEq, prost::Message)]
struct Scan {
    #[prost(uint64, tag = "1")]
    scan: u64,
    #[prost(message, repeated, tag = "2")]
    files: Vec<File>,
    #[prost(message, repeated, tag = "3")]
    partitions: Vec<Partition>,
}

/// `krishiv.task.v1alpha.File`: a [`SplitFileRef`].
#[derive(Clone, PartialEq, prost::Message)]
struct File {
    #[prost(string, tag = "1")]
    store: String,
    #[prost(string, tag = "2")]
    path: String,
    #[prost(uint64, tag = "3")]
    size: u64,
    #[prost(oneof = "Version", tags = "4, 5, 6")]
    version: Option<Version>,
    #[prost(int64, optional, tag = "7")]
    modified_micros: Option<i64>,
}

/// `krishiv.task.v1alpha.File.version`: a [`VersionToken`].
#[derive(Clone, PartialEq, prost::Oneof)]
enum Version {
    #[prost(string, tag = "4")]
    ETag(String),
    #[prost(int64, tag = "5")]
    Modified(i64),
    #[prost(bool, tag = "6")]
    Immutable(bool),
}

/// `krishiv.task.v1alpha.Partition`: a [`FilePartition`].
#[derive(Clone, PartialEq, prost::Message)]
struct Partition {
    #[prost(uint64, tag = "1")]
    index: u64,
    #[prost(message, repeated, tag = "2")]
    splits: Vec<Split>,
}

/// `krishiv.task.v1alpha.Split`: a [`FileSplit`].
#[derive(Clone, PartialEq, prost::Message)]
struct Split {
    #[prost(uint64, tag = "1")]
    file: u64,
    #[prost(uint64, tag = "2")]
    start: u64,
    #[prost(uint64, tag = "3")]
    end: u64,
}

fn scan_of(id: ScanId, splits: &ScanSplits) -> Result<Scan, KrishivError> {
    let ScanSplits::Files { files, partitions } = splits else {
        return Err(internal(format!(
            "scan {}: relation batches live in one process and are not shipped",
            id.0
        )));
    };
    Ok(Scan {
        scan: id.0,
        files: files
            .iter()
            .map(|f| File {
                store: f.store.key().to_owned(),
                path: f.path.to_string(),
                size: f.size,
                version: Some(match &f.version {
                    VersionToken::ETag(e) => Version::ETag(e.to_string()),
                    VersionToken::Modified(m) => Version::Modified(*m),
                    VersionToken::Immutable => Version::Immutable(true),
                }),
                modified_micros: f.modified_micros,
            })
            .collect(),
        partitions: partitions
            .iter()
            .map(|p| Partition {
                index: wide(p.index),
                splits: p
                    .splits
                    .iter()
                    .map(|s| Split {
                        file: wide(s.file),
                        start: s.range.start,
                        end: s.range.end,
                    })
                    .collect(),
            })
            .collect(),
    })
}

fn splits_of(scan: Scan) -> Result<(ScanId, ScanSplits), KrishivError> {
    let files = scan
        .files
        .into_iter()
        .map(|f| {
            Ok(SplitFileRef {
                store: StoreId::new(&f.store),
                path: Path::from(f.path),
                size: f.size,
                version: match f.version {
                    Some(Version::ETag(e)) => VersionToken::ETag(e.into()),
                    Some(Version::Modified(m)) => VersionToken::Modified(m),
                    Some(Version::Immutable(_)) => VersionToken::Immutable,
                    None => return Err(internal("a split file without a version".to_owned())),
                },
                modified_micros: f.modified_micros,
            })
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    let count = files.len();
    let partitions = scan
        .partitions
        .into_iter()
        .map(|p| {
            Ok(FilePartition {
                index: narrow(p.index, "partition index")?,
                splits: p
                    .splits
                    .into_iter()
                    .map(|s| {
                        let file = narrow(s.file, "split file")?;
                        if file >= count || s.start > s.end {
                            return Err(internal(format!(
                                "a split of file {file} of {count}, bytes {}..{}",
                                s.start, s.end
                            )));
                        }
                        Ok(FileSplit {
                            file,
                            range: ByteRange {
                                start: s.start,
                                end: s.end,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, KrishivError>>()?,
            })
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    Ok((
        ScanId(scan.scan),
        ScanSplits::Files {
            files: files.into(),
            partitions,
        },
    ))
}

/// `krishiv.task.v1alpha.Settings`: the query's [`ExecConfig`].
#[derive(Clone, PartialEq, prost::Message)]
struct Settings {
    #[prost(uint64, tag = "1")]
    batch_size: u64,
    #[prost(uint64, tag = "2")]
    morsel_rows: u64,
    #[prost(uint64, tag = "3")]
    result_buffer_per_slot: u64,
    #[prost(uint64, tag = "4")]
    stall_threshold_nanos: u64,
    #[prost(bool, tag = "5")]
    fail_on_stall: bool,
    #[prost(uint64, optional, tag = "6")]
    seed: Option<u64>,
    #[prost(bool, tag = "7")]
    agg_row_layout: bool,
    #[prost(bool, tag = "8")]
    detailed_metrics: bool,
    #[prost(bool, tag = "9")]
    join_row_layout: bool,
    #[prost(uint32, tag = "10")]
    join_radix_bits: u32,
    #[prost(bool, tag = "11")]
    sort_hash_buckets: bool,
    #[prost(uint64, tag = "12")]
    join_merge_buffer_bytes: u64,
    #[prost(bool, tag = "13")]
    deterministic_order: bool,
}

fn wide(v: usize) -> u64 {
    u64::try_from(v).unwrap_or(u64::MAX)
}

fn narrow(v: u64, what: &str) -> Result<usize, KrishivError> {
    usize::try_from(v).map_err(|_| internal(format!("a task's {what} of {v}")))
}

impl Settings {
    fn of(c: &ExecConfig) -> Self {
        // Exhaustive: a new field fails to compile here until it travels.
        let ExecConfig {
            batch_size,
            morsel_rows,
            result_buffer_per_slot,
            stall_threshold,
            fail_on_stall,
            seed,
            agg_row_layout,
            detailed_metrics,
            join_row_layout,
            join_radix_bits,
            sort_hash_buckets,
            join_merge_buffer_bytes,
            deterministic_order,
        } = c.clone();
        Self {
            batch_size: wide(batch_size),
            morsel_rows: wide(morsel_rows),
            result_buffer_per_slot: wide(result_buffer_per_slot),
            stall_threshold_nanos: u64::try_from(stall_threshold.as_nanos()).unwrap_or(u64::MAX),
            fail_on_stall,
            seed,
            agg_row_layout,
            detailed_metrics,
            join_row_layout,
            join_radix_bits,
            sort_hash_buckets,
            join_merge_buffer_bytes: wide(join_merge_buffer_bytes),
            deterministic_order,
        }
    }

    fn config(&self) -> Result<ExecConfig, KrishivError> {
        Ok(ExecConfig {
            batch_size: narrow(self.batch_size, "batch size")?,
            morsel_rows: narrow(self.morsel_rows, "morsel rows")?,
            result_buffer_per_slot: narrow(self.result_buffer_per_slot, "result buffer")?,
            stall_threshold: Duration::from_nanos(self.stall_threshold_nanos),
            fail_on_stall: self.fail_on_stall,
            seed: self.seed,
            agg_row_layout: self.agg_row_layout,
            detailed_metrics: self.detailed_metrics,
            join_row_layout: self.join_row_layout,
            join_radix_bits: self.join_radix_bits,
            sort_hash_buckets: self.sort_hash_buckets,
            join_merge_buffer_bytes: narrow(self.join_merge_buffer_bytes, "merge buffer")?,
            deterministic_order: self.deterministic_order,
        })
    }
}

/// `krishiv.task.v1alpha.Input`: one entry of [`TaskSpec::exchanges`].
#[derive(Clone, PartialEq, prost::Message)]
struct Input {
    #[prost(uint64, tag = "1")]
    exchange: u64,
    #[prost(uint32, tag = "2")]
    stage_attempt: u32,
    #[prost(uint32, tag = "3")]
    maps: u32,
    #[prost(uint32, repeated, tag = "4")]
    reduce: Vec<u32>,
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// `task` of a query run with `settings`, as bytes.
///
/// # Errors
/// `INTERNAL_ERROR` for a task reading relation batches (they live in one
/// process).
pub fn encode_task(task: &TaskSpec, settings: &ExecConfig) -> Result<Vec<u8>, KrishivError> {
    let scans = task
        .scans
        .iter()
        .map(|(id, splits)| scan_of(*id, splits))
        .collect::<Result<Vec<_>, KrishivError>>()?;
    let message = Task {
        stage: task.stage,
        stage_attempt: task.stage_attempt,
        partition: task.partition,
        partitions: task.partitions,
        attempt: task.attempt,
        exchanges: task
            .exchanges
            .iter()
            .map(|(id, input)| Input {
                exchange: id.0,
                stage_attempt: input.stage_attempt,
                maps: input.maps,
                reduce: input.reduce.clone(),
            })
            .collect(),
        output: task.output,
        settings: Some(Settings::of(settings)),
        scans,
        filters: task
            .filters
            .iter()
            .map(|(id, keys)| FilterValue {
                id: id.0,
                keys: Some(keys.clone()),
            })
            .collect(),
    };
    Ok(message.encode_to_vec())
}

/// The task of `bytes` and its query's execution settings.
///
/// # Errors
/// `INTERNAL_ERROR` for bytes that are not a task, lack settings, or name
/// an exchange or a scan twice, or a split outside its scan's files.
pub fn decode_task(bytes: &[u8]) -> Result<(TaskSpec, ExecConfig), KrishivError> {
    let message = Task::decode(bytes).map_err(|e| internal(format!("a task descriptor: {e}")))?;
    let settings = message
        .settings
        .as_ref()
        .ok_or_else(|| internal("a task descriptor without settings".to_owned()))?
        .config()?;
    let mut exchanges = BTreeMap::new();
    for input in message.exchanges {
        let entry = ExchangeInput {
            stage_attempt: input.stage_attempt,
            maps: input.maps,
            reduce: input.reduce,
        };
        if exchanges
            .insert(ExchangeId(input.exchange), entry)
            .is_some()
        {
            return Err(internal(format!(
                "a task descriptor reads exchange {} twice",
                input.exchange
            )));
        }
    }
    let mut scans = BTreeMap::new();
    for scan in message.scans {
        let (id, splits) = splits_of(scan)?;
        if scans.insert(id, splits).is_some() {
            return Err(internal(format!(
                "a task descriptor reads scan {} twice",
                id.0
            )));
        }
    }
    let mut filters = BTreeMap::new();
    for f in message.filters {
        let keys = f
            .keys
            .ok_or_else(|| internal(format!("filter {} of a task without a value", f.id)))?;
        if filters.insert(FilterId(f.id), keys).is_some() {
            return Err(internal(format!(
                "a task descriptor carries filter {} twice",
                f.id
            )));
        }
    }
    let task = TaskSpec {
        stage: message.stage,
        stage_attempt: message.stage_attempt,
        partition: message.partition,
        partitions: message.partitions,
        attempt: message.attempt,
        scans,
        exchanges,
        output: message.output,
        filters,
    };
    Ok((task, settings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_task_round_trips() {
        let task = TaskSpec {
            stage: 3,
            stage_attempt: 1,
            partition: 2,
            partitions: 4,
            attempt: 1,
            scans: BTreeMap::new(),
            exchanges: BTreeMap::from([(
                ExchangeId(7),
                ExchangeInput {
                    stage_attempt: 0,
                    maps: 5,
                    reduce: vec![2],
                },
            )]),
            output: Some(4),
            filters: BTreeMap::from([(FilterId(5), vec![1, 2, 3])]),
        };
        let settings = ExecConfig {
            batch_size: 77,
            seed: Some(9),
            detailed_metrics: true,
            join_radix_bits: 3,
            stall_threshold: Duration::from_millis(1_234),
            ..ExecConfig::default()
        };
        let (back, got) = decode_task(&encode_task(&task, &settings).unwrap()).unwrap();
        assert_eq!(format!("{back:?}"), format!("{task:?}"));
        assert_eq!(format!("{got:?}"), format!("{settings:?}"));
        let root = TaskSpec {
            output: None,
            ..TaskSpec::whole()
        };
        let defaults = ExecConfig::default();
        let (root_back, got) = decode_task(&encode_task(&root, &defaults).unwrap()).unwrap();
        assert_eq!(root_back.output, None);
        assert_eq!(format!("{got:?}"), format!("{defaults:?}"));
    }

    #[test]
    fn file_splits_round_trip_and_relation_batches_are_refused() {
        let files: Arc<[SplitFileRef]> = Arc::from(vec![
            SplitFileRef {
                store: StoreId::new("memory://"),
                path: Path::from("t/a.parquet"),
                size: 300,
                version: VersionToken::ETag("\"e1\"".into()),
                modified_micros: Some(17),
            },
            SplitFileRef {
                store: StoreId::new("file://"),
                path: Path::from("t/b.parquet"),
                size: 10,
                version: VersionToken::Modified(-4),
                modified_micros: None,
            },
            SplitFileRef {
                store: StoreId::new("s3://bucket"),
                path: Path::from("t/c.parquet"),
                size: 0,
                version: VersionToken::Immutable,
                modified_micros: None,
            },
        ]);
        let split = |file, start, end| FileSplit {
            file,
            range: ByteRange { start, end },
        };
        let task = TaskSpec {
            scans: BTreeMap::from([
                (
                    ScanId(4),
                    ScanSplits::Files {
                        files: Arc::clone(&files),
                        partitions: vec![
                            FilePartition {
                                index: 2,
                                splits: vec![split(0, 100, 300), split(1, 0, 10)],
                            },
                            FilePartition {
                                index: 3,
                                splits: vec![split(2, 0, 0)],
                            },
                        ],
                    },
                ),
                (
                    ScanId(9),
                    ScanSplits::Files {
                        files,
                        partitions: Vec::new(),
                    },
                ),
            ]),
            ..TaskSpec::whole()
        };
        let config = ExecConfig::default();
        let (back, _) = decode_task(&encode_task(&task, &config).unwrap()).unwrap();
        assert_eq!(back, task);

        let relation = TaskSpec {
            scans: BTreeMap::from([(
                ScanId(1),
                ScanSplits::Batches {
                    relation: krishiv_memory::RelationRef {
                        hash: krishiv_memory::RelationHash::from_bytes([7; 32]),
                    },
                    range: 0..2,
                },
            )]),
            ..TaskSpec::whole()
        };
        assert!(encode_task(&relation, &config).is_err());

        // A split past its scan's files is refused, not read.
        let mut message = Task::decode(encode_task(&task, &config).unwrap().as_slice()).unwrap();
        message.scans[0].partitions[0].splits[0].file = 3;
        assert!(decode_task(&message.encode_to_vec()).is_err());
    }

    #[test]
    fn garbage_is_refused() {
        assert!(decode_task(&[0xff, 0xff, 0xff]).is_err());
    }
}
