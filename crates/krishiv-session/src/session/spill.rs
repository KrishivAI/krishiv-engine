// SPDX-License-Identifier: Apache-2.0
//! Per-query spill scopes, the column check of a write's schema, the target of a write and its job id
//! (the worker context a query's plans run with is `session/env.rs`).

use super::*;

/// The spill scope of one query: its own directory under the runtime's
/// spill directory (opened from the runtime's configuration by the first
/// query that spills; `krishiv.spill.*` are runtime keys, so every session
/// shares one directory and one `krishiv.spill.maxBytes`), removed when the
/// query's operators and streams are dropped (cancellation included).
pub(super) struct QuerySpill {
    pub(super) state: Arc<SessionState>,
    /// Where this query's spill set-up failure is kept for its memory error (PY-078).
    pub(super) failure: Arc<Mutex<Option<KrishivError>>>,
}

impl fmt::Debug for QuerySpill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuerySpill").finish_non_exhaustive()
    }
}

impl QuerySpill {
    /// Keeps the I/O failure of the spill directory, named for the user, as this query's recorded
    /// error (the memory error that follows is replaced by it).
    fn record(&self, e: &krishiv_memory::SpillError) {
        let failure: KrishivError = match e {
            krishiv_memory::SpillError::Io {
                operation,
                path,
                source,
            } => krishiv_common::error::KrishivSpillIoError {
                operation: (*operation).to_owned(),
                path: path.display().to_string(),
                reason: if source.kind() == std::io::ErrorKind::StorageFull {
                    format!(
                        "the spill disk is full ({source}); set `krishiv.spill.dir` to a directory with free space"
                    )
                } else {
                    format!(
                        "{source}; set `krishiv.spill.dir` to a writable local directory"
                    )
                },
            }
            .into(),
            other => KrishivError::from(other),
        };
        *self.failure.lock().unwrap_or_else(PoisonError::into_inner) = Some(failure);
    }
}

impl SpillScopes for QuerySpill {
    fn scope(&self) -> krishiv_exec::Result<SpillScope> {
        let mut slot = self
            .state
            .runtime
            .spill
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            let config = SpillConfig::from_config(&self.state.runtime.config);
            let manager = SpillManager::open(&config).map_err(|e| {
                self.record(&e);
                internal(format!(
                    "cannot set up the spill directory (set `krishiv.spill.dir` to a writable local directory): {e}"
                ))
            })?;
            *slot = Some(manager);
        }
        slot.as_ref().map_or_else(
            || Err(internal("the spill directory is not open".to_owned())),
            |m| match m.scope() {
                Ok(scope) => {
                    // The directory works: a failure an earlier run of this statement recorded is stale.
                    *self.failure.lock().unwrap_or_else(PoisonError::into_inner) = None;
                    Ok(scope)
                }
                Err(e) => {
                    self.record(&e);
                    Err(internal(format!(
                        "cannot create the query's spill directory: {e}"
                    )))
                }
            },
        )
    }
}

/// Spark's `COLUMN_ALREADY_EXISTS` for a written schema with a repeated
/// name (`check_column_name_duplication`).
pub(super) fn unique_columns(schema: &SchemaRef, case_sensitive: bool) -> Result<(), KrishivError> {
    check_column_name_duplication(
        schema.fields().iter().map(|f| f.name().as_str()),
        case_sensitive,
    )
}

/// The name of the column `bucketBy` appends to the rows written: each row's bucket id.
pub(crate) const BUCKET_ID_COLUMN: &str = "__krishiv_bucket_id";

/// A bucketed write (Spark's `BucketSpec`): the rows go to `buckets` buckets by `pmod(hash(columns), buckets)`
/// (Murmur3, seed 42), and each bucket is sorted by `sort_by`.
#[derive(Debug, Clone)]
pub(crate) struct BucketTarget {
    pub(crate) buckets: u32,
    pub(crate) columns: Vec<String>,
    pub(crate) sort_by: Vec<String>,
}

impl BucketTarget {
    /// `df` with each row's bucket id appended as its last column, named so that no column of `df`
    /// (in any letter case) has the same name: [`BUCKET_ID_COLUMN`], with underscores added until free.
    pub(crate) async fn with_bucket_id(
        &self,
        df: &crate::DataFrame,
    ) -> Result<crate::DataFrame, KrishivError> {
        let schema = df.schema().await?;
        let mut name = BUCKET_ID_COLUMN.to_owned();
        while schema
            .fields()
            .iter()
            .any(|f| f.name().eq_ignore_ascii_case(&name))
        {
            name.push('_');
        }
        let quoted: Vec<String> = self
            .columns
            .iter()
            .map(|c| format!("`{}`", c.replace('`', "``")))
            .collect();
        df.select_expr([
            "*".to_owned(),
            format!(
                "pmod(hash({}), {}) AS `{name}`",
                quoted.join(", "),
                self.buckets
            ),
        ])
    }
}

/// Where and how `df.write` writes.
pub(crate) struct WriteTarget {
    /// The bucketing of a write into a bucketed table.
    pub(crate) bucket: Option<BucketTarget>,
    pub(crate) format: String,
    pub(crate) mode: String,
    pub(crate) path: OptionValue,
    pub(crate) options: BTreeMap<String, OptionValue>,
    /// The stored options of the table written (`INSERT`), already authorized when they were
    /// stored: an option equal to one of them is not asked about again (P3-T067b).
    pub(crate) table_options: BTreeMap<String, OptionValue>,
    pub(crate) partition_by: Vec<String>,
    /// Duplicate column names are not refused (csv of `INSERT OVERWRITE DIRECTORY`).
    pub(crate) skip_duplicate_check: bool,
}

/// A fresh job id naming a write's files (`part-00000-<id>-c000...`), in
/// UUID form. Its bits come from the standard library's per-process
/// random hash keys, a clock reading and a counter, so two writes never
/// share one, within or across processes.
pub(crate) fn job_uuid() -> String {
    use std::hash::{BuildHasher, Hasher};
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let n = WRITES.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let half = |salt: u64| {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(salt);
        h.write_u64(n);
        h.write_u128(nanos);
        h.write_u32(std::process::id());
        h.finish()
    };
    let (hi, lo) = (half(1), half(2));
    // Version 4, RFC 4122 variant.
    let hi = (hi & !0xF000) | 0x4000;
    let lo = (lo & !(0xC << 60)) | (0x8 << 60);
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        hi >> 32,
        (hi >> 16) & 0xFFFF,
        hi & 0xFFFF,
        lo >> 48,
        lo & 0xFFFF_FFFF_FFFF
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failure an earlier run of the statement recorded is cleared once the directory opens
    /// (PY-078): a later memory error of the statement must not name a directory that works.
    #[test]
    fn a_usable_spill_directory_clears_a_recorded_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let session = Session::builder()
            .config(
                "krishiv.spill.dir",
                tmp.path().join("spill").to_str().unwrap(),
            )
            .build()
            .unwrap();
        let stale: KrishivError = InternalError {
            message: "stale".to_owned(),
        }
        .into();
        let failure = Arc::new(Mutex::new(Some(stale)));
        let spill = QuerySpill {
            state: Arc::clone(&session.state),
            failure: Arc::clone(&failure),
        };
        spill.scope().unwrap();
        assert!(failure.lock().unwrap().is_none());
        // and again, with the directory already open
        *failure.lock().unwrap() = Some(
            InternalError {
                message: "stale".to_owned(),
            }
            .into(),
        );
        spill.scope().unwrap();
        assert!(failure.lock().unwrap().is_none());
    }
}
