// SPDX-License-Identifier: Apache-2.0
//! Which snapshot a read sees, and the schema it reads with (design note §2.1,
//! FC S6, as Iceberg 1.12's `SparkCatalog`/`SnapshotUtil` on Spark 4.1):
//! - the current snapshot, with the current schema;
//! - `VERSION AS OF <id>`: that snapshot (`Cannot find snapshot with ID <id>`),
//!   with its own schema;
//! - `VERSION AS OF '<name>'`: a branch or tag (`Cannot find matching snapshot
//!   ID or reference name for version <name>`); a tag reads the tagged
//!   snapshot's schema, a branch the table's current schema;
//! - `TIMESTAMP AS OF`: the snapshot-log entry with the largest time at or before
//!   the timestamp, Spark's microseconds truncated to milliseconds (`Cannot find
//!   a snapshot older than <ts>`), with that snapshot's schema.
//!
//! A ref is checked against the snapshot list before anything else uses it:
//! iceberg-rust's `snapshot_for_ref` panics on a dangling ref (FC S6), and it is
//! never called.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_io::ColumnPath;
use krishiv_types::StructType;

use super::types::{PartitionSpecs, SchemaField};

/// The version a read asks for (connector-local until the shared
/// `TableVersionSpec` of PR-07 is wired).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VersionSpec {
    /// The table's current snapshot.
    Current,
    /// `VERSION AS OF <number>`: a snapshot id.
    Id(i64),
    /// `VERSION AS OF '<name>'`: a branch or tag.
    Ref(String),
    /// `TIMESTAMP AS OF`: microseconds since the epoch.
    TimestampMicros(i64),
}

/// A branch or a tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefKind {
    /// A branch (reads the current schema).
    Branch,
    /// A tag (reads the tagged snapshot's schema).
    Tag,
}

/// A named ref of the table metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefInfo {
    /// The snapshot it points at.
    pub snapshot_id: i64,
    /// Branch or tag.
    pub kind: RefKind,
}

/// What resolution needs of one snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SnapshotInfo {
    pub(crate) snapshot_id: i64,
    pub(crate) sequence_number: i64,
    pub(crate) timestamp_ms: i64,
    pub(crate) schema_id: Option<i32>,
    pub(crate) manifest_list: String,
}

/// What resolution needs of the table metadata.
#[derive(Debug, Clone, Default)]
pub(crate) struct MetadataView {
    pub(crate) snapshots: BTreeMap<i64, SnapshotInfo>,
    pub(crate) current_snapshot_id: Option<i64>,
    pub(crate) current_schema_id: i32,
    pub(crate) refs: BTreeMap<String, RefInfo>,
    /// The snapshot log: (timestamp in ms, snapshot id), in log order.
    pub(crate) log: Vec<(i64, i64)>,
}

/// A resolved version: the snapshot (none for a table without one) and the
/// schema id to read with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Resolved {
    pub(crate) snapshot: Option<SnapshotInfo>,
    pub(crate) schema_id: i32,
}

/// The snapshot and schema `version` reads.
///
/// # Errors
/// Iceberg's messages for an unknown id, ref or a timestamp before the first
/// snapshot (class: see [`time_travel_error`]).
pub(crate) fn resolve(
    view: &MetadataView,
    version: &VersionSpec,
) -> Result<Resolved, KrishivError> {
    let by_id = |id: i64| {
        view.snapshots
            .get(&id)
            .cloned()
            .ok_or_else(|| time_travel_error(format!("Cannot find snapshot with ID {id}")))
    };
    let own_schema = |s: &SnapshotInfo| s.schema_id.unwrap_or(view.current_schema_id);
    match version {
        VersionSpec::Current => Ok(Resolved {
            snapshot: view.current_snapshot_id.map(by_id).transpose()?,
            schema_id: view.current_schema_id,
        }),
        VersionSpec::Id(id) => {
            let s = by_id(*id)?;
            Ok(Resolved {
                schema_id: own_schema(&s),
                snapshot: Some(s),
            })
        }
        VersionSpec::Ref(name) => {
            let Some(r) = view.refs.get(name) else {
                return Err(time_travel_error(format!(
                    "Cannot find matching snapshot ID or reference name for version {name}"
                )));
            };
            let s = by_id(r.snapshot_id)?;
            let schema_id = match r.kind {
                RefKind::Branch => view.current_schema_id,
                RefKind::Tag => own_schema(&s),
            };
            Ok(Resolved {
                snapshot: Some(s),
                schema_id,
            })
        }
        VersionSpec::TimestampMicros(us) => {
            // Java's `TimeUnit.MICROSECONDS.toMillis` truncates toward zero.
            let ms = us / 1000;
            let mut best: Option<(i64, i64)> = None;
            for &(ts, id) in &view.log {
                if ts <= ms && best.is_none_or(|(b, _)| ts >= b) {
                    best = Some((ts, id));
                }
            }
            let Some((_, id)) = best else {
                return Err(time_travel_error(format!(
                    "Cannot find a snapshot older than {}",
                    format_timestamp_millis(ms)
                )));
            };
            let s = by_id(id)?;
            Ok(Resolved {
                schema_id: own_schema(&s),
                snapshot: Some(s),
            })
        }
    }
}

/// The error of a time-travel lookup Iceberg refuses with an
/// `IllegalArgumentException`, which carries no Spark error class. Until a
/// Krishiv class exists for it, it is `INTERNAL_ERROR` with Iceberg's exact
/// message (see the PR-25a report's open question).
#[must_use]
pub(crate) fn time_travel_error(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// Iceberg's `DateTimeUtil.formatTimestampMillis`: `Instant.toString()` with `Z`
/// written as `+00:00` (seconds always shown, milliseconds when not zero).
#[must_use]
pub(crate) fn format_timestamp_millis(ms: i64) -> String {
    let days = ms.div_euclid(86_400_000);
    let rem = ms.rem_euclid(86_400_000);
    let (y, m, d) = civil(days);
    let (h, mi, s, milli) = (
        rem / 3_600_000,
        rem / 60_000 % 60,
        rem / 1000 % 60,
        rem % 1000,
    );
    let mut out = String::new();
    if y > 9999 {
        let _ = write!(out, "+{y}");
    } else if y < 0 {
        let _ = write!(out, "-{:04}", -y);
    } else {
        let _ = write!(out, "{y:04}");
    }
    let _ = write!(out, "-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}");
    if milli != 0 {
        let _ = write!(out, ".{milli:03}");
    }
    out.push_str("+00:00");
    out
}

fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// The version of a table a scan reads, pinned for the query (ADR 0037).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TableSnapshot {
    /// The table uuid.
    pub table_uuid: String,
    /// The table format version (1, 2 or 3).
    pub format_version: u8,
    /// The snapshot (`None`: the table has none, so no rows).
    pub snapshot_id: Option<i64>,
    /// The snapshot's sequence number.
    pub sequence_number: Option<i64>,
    /// The snapshot's commit time (ms).
    pub timestamp_ms: Option<i64>,
    /// The snapshot's manifest list.
    pub manifest_list: Option<String>,
    /// The schema read with (see the module docs).
    pub schema_id: i32,
    /// That schema as Spark shows it.
    pub schema: StructType,
    /// Its primitive columns reachable through structs (what pruning can see).
    pub fields: Vec<SchemaField>,
    /// Every partition spec of the table, by id.
    pub partition_specs: PartitionSpecs,
}

impl TableSnapshot {
    /// The primitive column at `path`.
    #[must_use]
    pub fn field(&self, path: &ColumnPath) -> Option<&SchemaField> {
        self.fields.iter().find(|f| &f.path == path)
    }
}
