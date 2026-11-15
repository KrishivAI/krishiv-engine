// SPDX-License-Identifier: Apache-2.0
//! The commits that won a race: each one's actions, as the conflict checker reads them (port of
//! delta-spark's `WinningCommitSummary`, `ConflictChecker.scala:152-196`).
//!
//! A commit file is read on the log pool through the storage handler: its size is checked
//! against `krishiv.catalog.maxMetadataBytes` before it is read, and each line's nesting against
//! the depth limit before it is parsed. A line that is not an action Krishiv reads (`cdc`,
//! `checkpointMetadata`, an unknown action) is skipped, as Delta skips actions its checker does
//! not look at.

use std::collections::HashMap;

use delta_kernel::{FileSlice, KernelError};
use krishiv_common::error::{InternalError, KrishivError};
use serde_json::{Map, Value};
use url::Url;

use crate::engine::DeltaEngine;
use crate::engine::json::{check_depth, check_size};
use crate::error::from_kernel;

/// The commitInfo tag that marks an identity high-water-mark update
/// (`DeltaSourceUtils.IDENTITY_COMMITINFO_TAG`).
pub const IDENTITY_TAG: &str = "delta.identity.schemaUpdate";

/// An add or remove action: its path as the log writes it, and its partition values by
/// physical name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileAction {
    /// The path, as the log writes it.
    pub path: String,
    /// The partition values by physical name (`None`: a NULL value).
    pub partition_values: HashMap<String, Option<String>>,
}

/// A protocol action.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProtocolAction {
    /// `minReaderVersion`.
    pub min_reader_version: i64,
    /// `minWriterVersion`.
    pub min_writer_version: i64,
    /// `readerFeatures`.
    pub reader_features: Option<Vec<String>>,
    /// `writerFeatures`.
    pub writer_features: Option<Vec<String>>,
}

impl ProtocolAction {
    /// Every feature it names, reader and writer.
    pub fn features(&self) -> impl Iterator<Item = &str> {
        self.reader_features
            .iter()
            .chain(&self.writer_features)
            .flatten()
            .map(String::as_str)
    }
}

/// What one winning commit did.
#[derive(Debug, Clone, Default)]
pub struct WinningCommit {
    /// Its version.
    pub version: u64,
    /// Its `commitInfo` as written, if it has one.
    pub commit_info: Option<Map<String, Value>>,
    /// Its add actions.
    pub adds: Vec<FileAction>,
    /// Its remove actions.
    pub removes: Vec<FileAction>,
    /// Its protocol action.
    pub protocol: Option<ProtocolAction>,
    /// Whether it has a `metaData` action.
    pub metadata_changed: bool,
    /// The domains of its `domainMetadata` actions.
    pub domains: Vec<String>,
    /// The app ids of its `txn` actions.
    pub app_ids: Vec<String>,
}

impl WinningCommit {
    /// `commitInfo.operation`, or `TRANSACTION` without one (`DeltaErrors.concurrentAppendException`).
    #[must_use]
    pub fn operation(&self) -> String {
        self.commit_info
            .as_ref()
            .and_then(|c| c.get("operation"))
            .and_then(Value::as_str)
            .unwrap_or("TRANSACTION")
            .to_owned()
    }

    /// `commitInfo.isBlindAppend`; absent means not blind.
    #[must_use]
    pub fn is_blind_append(&self) -> bool {
        self.commit_info
            .as_ref()
            .and_then(|c| c.get("isBlindAppend"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// The value of the commitInfo tag `key`.
    #[must_use]
    pub fn tag(&self, key: &str) -> Option<&str> {
        self.commit_info
            .as_ref()
            .and_then(|c| c.get("tags"))
            .and_then(|t| t.get(key))
            .and_then(Value::as_str)
    }

    /// Whether its metadata change only moves identity high-water marks (the commitInfo tag
    /// [`IDENTITY_TAG`] is `true`).
    #[must_use]
    pub fn identity_only_metadata_update(&self) -> bool {
        self.tag(IDENTITY_TAG)
            .is_some_and(|v| v.eq_ignore_ascii_case("true"))
    }

    /// `<conflictingCommit>` of Delta's classes: `"\nConflicting commit: "` and the commitInfo
    /// with its `version` (keys sorted), or empty without one.
    #[must_use]
    pub fn conflicting_commit(&self) -> String {
        let Some(info) = &self.commit_info else {
            return String::new();
        };
        let mut with_version = Map::new();
        with_version.insert("version".to_owned(), Value::from(self.version));
        for (k, v) in info {
            if k != "version" {
                with_version.insert(k.clone(), v.clone());
            }
        }
        format!("\nConflicting commit: {}", Value::Object(with_version))
    }
}

/// The URL of the commit file of `version` under `root`.
///
/// # Errors
/// `INTERNAL_ERROR` if the URL cannot be built.
pub fn commit_url(root: &Url, version: u64) -> Result<Url, KrishivError> {
    root.join(&format!("_delta_log/{version:020}.json"))
        .map_err(|e| internal(&format!("commit file URL of version {version}: {e}")))
}

/// The commits from `from` on, at most `max` of them, stopping at the first version with no
/// commit file. Runs on the log pool.
///
/// # Errors
/// A store failure, a commit over the size or depth limit, or a line that is not JSON.
pub fn read_from(
    engine: &DeltaEngine,
    from: u64,
    max: usize,
    table: &str,
) -> Result<Vec<WinningCommit>, KrishivError> {
    use delta_kernel::StorageHandler as _;
    let storage = engine.storage();
    let mut out = Vec::new();
    let mut version = from;
    while out.len() < max {
        let url = commit_url(engine.table_root(), version)?;
        let meta = match storage.head(&url) {
            Ok(meta) => meta,
            Err(e) if matches!(e.without_backtrace(), KernelError::FileNotFound(_)) => break,
            Err(e) => return Err(from_kernel(&e, table)),
        };
        check_size(engine.limits(), url.as_str(), meta.size)?;
        let slice: FileSlice = (url.clone(), None);
        let bytes = storage
            .read_files(vec![slice])
            .map_err(|e| from_kernel(&e, table))?
            .next()
            .ok_or_else(|| internal("a commit file read returned nothing"))?
            .map_err(|e| from_kernel(&e, table))?;
        check_size(engine.limits(), url.as_str(), bytes.len() as u64)?;
        out.push(parse(engine, version, &bytes, url.as_str())?);
        version += 1;
    }
    Ok(out)
}

/// The actions of the commit file `bytes` of `version`.
///
/// # Errors
/// A line over the depth limit, or one that is not a JSON object.
pub fn parse(
    engine: &DeltaEngine,
    version: u64,
    bytes: &[u8],
    source: &str,
) -> Result<WinningCommit, KrishivError> {
    let mut commit = WinningCommit {
        version,
        ..WinningCommit::default()
    };
    for line in bytes.split(|b| *b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        check_depth(engine.limits(), source, line)?;
        let action: Map<String, Value> = serde_json::from_slice(line)
            .map_err(|e| internal(&format!("commit {source} has a line that is not JSON: {e}")))?;
        for (kind, body) in action {
            match kind.as_str() {
                "commitInfo" => {
                    if let Value::Object(info) = body {
                        commit.commit_info.get_or_insert(info);
                    }
                }
                "add" => commit.adds.push(file_action(&body)),
                "remove" => commit.removes.push(file_action(&body)),
                "protocol" => commit.protocol = Some(protocol(&body)),
                "metaData" => commit.metadata_changed = true,
                "domainMetadata" => {
                    if let Some(d) = body.get("domain").and_then(Value::as_str) {
                        commit.domains.push(d.to_owned());
                    }
                }
                "txn" => {
                    if let Some(id) = body.get("appId").and_then(Value::as_str) {
                        commit.app_ids.push(id.to_owned());
                    }
                }
                _ => {}
            }
        }
    }
    Ok(commit)
}

fn file_action(body: &Value) -> FileAction {
    let partition_values = body
        .get("partitionValues")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.as_str().map(str::to_owned)))
                .collect()
        })
        .unwrap_or_default();
    FileAction {
        path: body
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        partition_values,
    }
}

fn protocol(body: &Value) -> ProtocolAction {
    let names = |key: &str| {
        body.get(key).and_then(Value::as_array).map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
    };
    ProtocolAction {
        min_reader_version: body
            .get("minReaderVersion")
            .and_then(Value::as_i64)
            .unwrap_or(1),
        min_writer_version: body
            .get("minWriterVersion")
            .and_then(Value::as_i64)
            .unwrap_or(1),
        reader_features: names("readerFeatures"),
        writer_features: names("writerFeatures"),
    }
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}
