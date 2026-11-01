// SPDX-License-Identifier: Apache-2.0
//! The commit log of a plain directory (P2-T082b unit 3, design §2.3/§2.4).
//!
//! `<root>/_krishiv_log/<version:020>.json` holds one commit per object,
//! created with put-if-absent, so two writers taking one version cannot both
//! win. A checkpoint `<version:020>.checkpoint.json` (the files visible after
//! that version) bounds replay. The `_` prefix makes Spark's listings skip the
//! directory. A reader replays the log instead of listing the data files, so
//! it sees only committed files.

use std::collections::BTreeMap;

use bytes::Bytes;
use krishiv_common::error::KrishivError;
use object_store::path::Path;
use serde_json::{Value, json};

use crate::error::{Result, internal};
use crate::io::{IoContext, PutIfAbsent, StoreId};

/// The log's directory name under the data root.
pub const LOG_DIR: &str = "_krishiv_log";
/// A checkpoint is written every this many versions.
pub const CHECKPOINT_EVERY: u64 = 64;
/// The version of the entry format.
const FORMAT: u64 = 1;
/// Commits re-tried after losing a race to an append.
const MAX_RACES: u32 = 32;

/// A data file a commit adds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddedFile {
    /// Path relative to the data root.
    pub path: String,
    /// Size in bytes.
    pub size: u64,
    /// Rows in the file.
    pub rows: u64,
    /// The `name=value` partition values, `None` for NULL.
    pub partition: Vec<(String, Option<String>)>,
}

/// How a commit changes the directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitMode {
    /// Adds files; never conflicts.
    Append,
    /// Replaces files; conflicts with any commit after its base version.
    Overwrite,
}

/// One commit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogEntry {
    /// Identifies the writer's attempt, so a commit whose response was lost
    /// is recognised on read-back.
    pub query_id: String,
    /// See `query_id`.
    pub write_id: String,
    /// The write's job uuid.
    pub job_uuid: String,
    /// Append or overwrite.
    pub mode: CommitMode,
    /// Files made visible.
    pub add: Vec<AddedFile>,
    /// Paths (relative to the root) no longer visible.
    pub remove: Vec<String>,
}

/// The files visible at a log version.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// The newest version replayed; `None` for an empty log.
    pub version: Option<u64>,
    /// Visible files by relative path.
    pub files: BTreeMap<String, AddedFile>,
}

/// The log of the data root `root` in `store`.
#[derive(Debug)]
pub struct CommitLog<'a> {
    io: &'a IoContext,
    store: StoreId,
    root: Path,
    display: String,
}

impl<'a> CommitLog<'a> {
    /// The log under `root`; `display` names the root in errors.
    #[must_use]
    pub fn new(io: &'a IoContext, store: StoreId, root: Path, display: String) -> Self {
        Self {
            io,
            store,
            root,
            display,
        }
    }

    fn dir(&self) -> Path {
        self.root.clone().join(LOG_DIR)
    }

    fn entry_path(&self, version: u64) -> Path {
        self.dir().join(format!("{version:020}.json"))
    }

    fn checkpoint_path(&self, version: u64) -> Path {
        self.dir().join(format!("{version:020}.checkpoint.json"))
    }

    fn corrupt(&self, reason: &str) -> KrishivError {
        krishiv_common::error::KrishivCommitLogCorrupt {
            root: self.display.clone(),
            reason: reason.to_owned(),
        }
        .into()
    }

    /// Whether the root has a log (any object under `_krishiv_log/`).
    ///
    /// # Errors
    /// Storage failures.
    pub async fn exists(&self) -> Result<bool> {
        Ok(!self.io.list(&self.store, &self.dir()).await?.is_empty())
    }

    /// The files visible at the newest version.
    ///
    /// # Errors
    /// Storage failures; `KRISHIV_COMMIT_LOG_CORRUPT` for a gap or an
    /// unreadable entry.
    pub async fn snapshot(&self) -> Result<Snapshot> {
        self.snapshot_to(u64::MAX).await
    }

    /// Entry versions (ascending) and the newest checkpoint.
    async fn versions(&self) -> Result<(Vec<u64>, Option<u64>)> {
        let mut entries = Vec::new();
        let mut checkpoint: Option<u64> = None;
        for meta in self.io.list(&self.store, &self.dir()).await? {
            let Some(name) = meta.location.filename() else {
                continue;
            };
            if let Some(v) = name.strip_suffix(".checkpoint.json") {
                if let Ok(v) = v.parse::<u64>() {
                    checkpoint = checkpoint.max(Some(v));
                }
            } else if let Some(v) = name
                .strip_suffix(".json")
                .and_then(|v| v.parse::<u64>().ok())
            {
                entries.push(v);
            }
        }
        entries.sort_unstable();
        Ok((entries, checkpoint))
    }

    async fn read_entry(&self, version: u64) -> Result<(LogEntry, Bytes)> {
        let bytes = self
            .io
            .get_object(&self.store, &self.entry_path(version))
            .await?
            .ok_or_else(|| self.corrupt(&format!("version {version} vanished")))?;
        let entry =
            decode_entry(&bytes).map_err(|e| self.corrupt(&format!("version {version}: {e}")))?;
        Ok((entry, bytes))
    }

    fn parse_checkpoint(&self, bytes: &[u8], version: u64) -> Result<Snapshot> {
        let v: Value = serde_json::from_slice(bytes)
            .map_err(|e| self.corrupt(&format!("checkpoint {version}: {e}")))?;
        let files = v["files"]
            .as_array()
            .ok_or_else(|| self.corrupt("a checkpoint without files"))?;
        let mut snap = Snapshot {
            version: Some(version),
            files: BTreeMap::new(),
        };
        for f in files {
            let f = decode_file(f)
                .ok_or_else(|| self.corrupt(&format!("checkpoint {version}: bad file entry")))?;
            snap.files.insert(f.path.clone(), f);
        }
        Ok(snap)
    }

    /// Commits `entry` on top of the log version `base` (the version the
    /// writer read before it wrote; `None` for an empty log). Returns the
    /// version taken.
    ///
    /// An append that loses a race retries on the next version. An overwrite
    /// conflicts with any commit after `base`. A commit whose response was
    /// lost is recognised by its bytes (see `IoContext::put_if_absent`).
    ///
    /// # Errors
    /// `KRISHIV_CONCURRENT_COMMIT` for an overwrite that lost a race, or an
    /// append that lost [`MAX_RACES`] times; storage failures.
    pub async fn commit(&self, base: Option<u64>, entry: &LogEntry) -> Result<u64> {
        let bytes = Bytes::from(encode_entry(entry));
        let mut version = base.map_or(0, |b| b + 1);
        for _ in 0..=MAX_RACES {
            match self
                .io
                .put_if_absent(&self.store, &self.entry_path(version), bytes.clone())
                .await?
            {
                PutIfAbsent::Created => {
                    if (version + 1).is_multiple_of(CHECKPOINT_EVERY) {
                        // A failed checkpoint only costs a longer replay.
                        if let Err(e) = self.write_checkpoint(version).await {
                            tracing::warn!(%e, version, "commit log checkpoint failed");
                        }
                    }
                    return Ok(version);
                }
                PutIfAbsent::Exists => {
                    if entry.mode == CommitMode::Overwrite {
                        return Err(self.conflict(version));
                    }
                    version += 1;
                }
            }
        }
        Err(self.conflict(version))
    }

    fn conflict(&self, version: u64) -> KrishivError {
        krishiv_common::error::KrishivConcurrentCommit {
            root: self.display.clone(),
            version: version.to_string(),
        }
        .into()
    }

    /// Writes the checkpoint of `version`: a snapshot replayed to there.
    async fn write_checkpoint(&self, version: u64) -> Result<()> {
        let snap = self.snapshot_to(version).await?;
        let files: Vec<Value> = snap.files.values().map(encode_file).collect();
        let doc = json!({ "format": FORMAT, "version": version, "files": files });
        let bytes = serde_json::to_vec(&doc).map_err(|e| internal(e.to_string()))?;
        // Exists: another writer made the same checkpoint.
        self.io
            .put_if_absent(
                &self.store,
                &self.checkpoint_path(version),
                Bytes::from(bytes),
            )
            .await?;
        Ok(())
    }

    /// The snapshot as of `version` (not later).
    async fn snapshot_to(&self, version: u64) -> Result<Snapshot> {
        let (entries, checkpoint) = self.versions().await?;
        let mut snap = Snapshot::default();
        let mut next = 0;
        if let Some(c) = checkpoint.filter(|c| *c <= version) {
            let bytes = self
                .io
                .get_object(&self.store, &self.checkpoint_path(c))
                .await?
                .ok_or_else(|| self.corrupt("a checkpoint vanished"))?;
            snap = self.parse_checkpoint(&bytes, c)?;
            next = c + 1;
        }
        let start = next;
        for v in entries.into_iter().filter(|v| *v >= start && *v <= version) {
            if v != next {
                return Err(self.corrupt(&format!("version {next} is missing")));
            }
            let (entry, _) = self.read_entry(v).await?;
            apply(&mut snap, &entry);
            snap.version = Some(v);
            next = v + 1;
        }
        Ok(snap)
    }
}

fn apply(snap: &mut Snapshot, entry: &LogEntry) {
    for r in &entry.remove {
        snap.files.remove(r);
    }
    for f in &entry.add {
        snap.files.insert(f.path.clone(), f.clone());
    }
}

fn encode_file(f: &AddedFile) -> Value {
    let partition: Vec<Value> = f.partition.iter().map(|(k, v)| json!([k, v])).collect();
    json!({ "path": f.path, "size": f.size, "rows": f.rows, "partition": partition })
}

fn decode_file(v: &Value) -> Option<AddedFile> {
    let partition = v["partition"]
        .as_array()?
        .iter()
        .map(|p| {
            let p = p.as_array()?;
            Some((
                p.first()?.as_str()?.to_owned(),
                p.get(1)?.as_str().map(str::to_owned),
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(AddedFile {
        path: v["path"].as_str()?.to_owned(),
        size: v["size"].as_u64()?,
        rows: v["rows"].as_u64()?,
        partition,
    })
}

fn encode_entry(e: &LogEntry) -> Vec<u8> {
    let mode = match e.mode {
        CommitMode::Append => "append",
        CommitMode::Overwrite => "overwrite",
    };
    let doc = json!({
        "format": FORMAT,
        "commit_key": { "query_id": e.query_id, "write_id": e.write_id },
        "job_uuid": e.job_uuid,
        "mode": mode,
        "add": e.add.iter().map(encode_file).collect::<Vec<_>>(),
        "remove": e.remove,
    });
    // A `Value` of strings and numbers always serializes.
    serde_json::to_vec(&doc).unwrap_or_default()
}

fn decode_entry(bytes: &[u8]) -> std::result::Result<LogEntry, String> {
    let v: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let format = v["format"].as_u64().ok_or("no format")?;
    // N reads N-1 (ADR 0026): an entry from a newer writer is refused.
    if format > FORMAT {
        return Err(format!(
            "entry format {format} is newer than this reader's {FORMAT}"
        ));
    }
    let key = &v["commit_key"];
    let mode = match v["mode"].as_str() {
        Some("append") => CommitMode::Append,
        Some("overwrite") => CommitMode::Overwrite,
        _ => return Err("bad mode".to_owned()),
    };
    let add = v["add"]
        .as_array()
        .ok_or("no add")?
        .iter()
        .map(|f| decode_file(f).ok_or_else(|| "bad file entry".to_owned()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let remove = v["remove"]
        .as_array()
        .ok_or("no remove")?
        .iter()
        .map(|r| {
            r.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "bad remove".to_owned())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(LogEntry {
        query_id: key["query_id"].as_str().ok_or("no query_id")?.to_owned(),
        write_id: key["write_id"].as_str().ok_or("no write_id")?.to_owned(),
        job_uuid: v["job_uuid"].as_str().ok_or("no job_uuid")?.to_owned(),
        mode,
        add,
        remove,
    })
}

#[cfg(test)]
mod tests;
