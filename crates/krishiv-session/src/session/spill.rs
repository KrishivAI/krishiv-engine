// SPDX-License-Identifier: Apache-2.0
//! Per-query spill scopes and the target of a write.

use super::*;

/// The spill scope of one query: its own directory under the runtime's
/// spill directory (opened from the runtime's configuration by the first
/// query that spills; `krishiv.spill.*` are runtime keys, so every session
/// shares one directory and one `krishiv.spill.maxBytes`), removed when the
/// query's operators and streams are dropped (cancellation included).
pub(super) struct QuerySpill {
    pub(super) state: Arc<SessionState>,
}

impl fmt::Debug for QuerySpill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuerySpill").finish_non_exhaustive()
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
            let manager = SpillManager::open(&SpillConfig::from_config(&self.state.runtime.config))
                .map_err(|e| internal(format!("cannot set up the spill directory: {e}")))?;
            *slot = Some(manager);
        }
        slot.as_ref().map_or_else(
            || Err(internal("the spill directory is not open".to_owned())),
            |m| {
                m.scope().map_err(|e| {
                    internal(format!("cannot create the query's spill directory: {e}"))
                })
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

/// Where and how `df.write` writes.
pub(crate) struct WriteTarget {
    pub(crate) format: String,
    pub(crate) mode: String,
    pub(crate) path: OptionValue,
    pub(crate) options: BTreeMap<String, OptionValue>,
    /// The stored options of the table written (`INSERT`), already authorized when they were
    /// stored: an option equal to one of them is not asked about again (P3-T067b).
    pub(crate) table_options: BTreeMap<String, OptionValue>,
    pub(crate) partition_by: Vec<String>,
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
