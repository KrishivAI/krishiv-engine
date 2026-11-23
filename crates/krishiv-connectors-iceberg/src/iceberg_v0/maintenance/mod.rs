// SPDX-License-Identifier: Apache-2.0
//! The Iceberg maintenance procedures behind `CALL cat.system.<name>(..)` (design P3-T027
//! PR-36e): [`procedure`] gives the four `system` procedures of the design, with Iceberg's
//! parameter lists in Java's order; `expire_snapshots` runs (`expire.rs`), the others are refused
//! by name until their units land.

use std::fmt;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use krishiv_common::error::{
    KrishivError, KrishivIcebergProcedureArgument, KrishivUnsupportedFeature,
};
use krishiv_plan::analyzer::BoxFuture;
use krishiv_plan::procedure::{
    Procedure, ProcedureContext, ProcedureParameter, ProcedureType, ProcedureValue,
};

use super::sink::io_clock;
use super::table::IcebergTable;

pub(crate) mod expire;

/// Where a procedure finds its table: the catalog that serves `CALL cat.system.p(table => ..)`.
pub trait ProcedureTables: fmt::Debug + Send + Sync {
    /// The table `name` (`ns.t`, as the call wrote it) of the catalog, loaded to commit through
    /// the catalog's commit endpoint.
    ///
    /// # Errors
    /// `KRISHIV_ICEBERG_PROCEDURE_TABLE_NOT_FOUND`, or the catalog's failure.
    fn open<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Result<IcebergTable, KrishivError>>;
}

const fn req(name: &'static str, data_type: ProcedureType) -> ProcedureParameter {
    ProcedureParameter {
        name,
        data_type,
        required: true,
    }
}

const fn opt(name: &'static str, data_type: ProcedureType) -> ProcedureParameter {
    ProcedureParameter {
        name,
        data_type,
        required: false,
    }
}

use ProcedureType as T;

const EXPIRE_SNAPSHOTS: &[ProcedureParameter] = &[
    req("table", T::String),
    opt("older_than", T::Timestamp),
    opt("retain_last", T::Int),
    opt("max_concurrent_deletes", T::Int),
    opt("stream_results", T::Boolean),
    opt("snapshot_ids", T::LongArray),
    opt("clean_expired_metadata", T::Boolean),
];
const REWRITE_MANIFESTS: &[ProcedureParameter] = &[
    req("table", T::String),
    opt("use_caching", T::Boolean),
    opt("spec_id", T::Int),
    opt("sort_by", T::StringArray),
];
const REWRITE_DATA_FILES: &[ProcedureParameter] = &[
    req("table", T::String),
    opt("strategy", T::String),
    opt("sort_order", T::String),
    opt("options", T::StringMap),
    opt("where", T::String),
    opt("branch", T::String),
];
const REMOVE_ORPHAN_FILES: &[ProcedureParameter] = &[
    req("table", T::String),
    opt("older_than", T::Timestamp),
    opt("location", T::String),
    opt("dry_run", T::Boolean),
    opt("max_concurrent_deletes", T::Int),
    opt("file_list_view", T::String),
    opt("equal_schemes", T::StringMap),
    opt("equal_authorities", T::StringMap),
    opt("prefix_mismatch_mode", T::String),
    opt("prefix_listing", T::Boolean),
    opt("stream_results", T::Boolean),
];

/// Spark's candidate order of each procedure's parameter names (`Seq(..).toSet.toSeq` in Scala
/// 2.13, printed by a Spark 4.1.0 + Iceberg 1.12.0 run, 2026-11-23, and checked through the
/// proposals of an equidistant unknown name). `rewrite_manifests` has four names: its signature
/// order.
const EXPIRE_SNAPSHOTS_ORDER: &[&str] = &[
    "stream_results",
    "clean_expired_metadata",
    "older_than",
    "max_concurrent_deletes",
    "table",
    "retain_last",
    "snapshot_ids",
];
const REWRITE_DATA_FILES_ORDER: &[&str] = &[
    "branch",
    "options",
    "strategy",
    "where",
    "sort_order",
    "table",
];
const REMOVE_ORPHAN_FILES_ORDER: &[&str] = &[
    "prefix_listing",
    "location",
    "stream_results",
    "equal_schemes",
    "file_list_view",
    "dry_run",
    "equal_authorities",
    "max_concurrent_deletes",
    "table",
    "prefix_mismatch_mode",
    "older_than",
];

/// The procedures of an Iceberg catalog's `system` namespace.
const NAMES: [&str; 4] = [
    "expire_snapshots",
    "rewrite_manifests",
    "rewrite_data_files",
    "remove_orphan_files",
];

/// The procedure `name` (lower-case) over `tables`; `None` for a name Iceberg's `system`
/// namespace does not have in this build (Spark's other procedures are not offered).
#[must_use]
pub fn procedure(name: &str, tables: Arc<dyn ProcedureTables>) -> Option<Arc<dyn Procedure>> {
    let name = NAMES.iter().copied().find(|n| *n == name)?;
    Some(Arc::new(Maintenance { name, tables }))
}

struct Maintenance {
    name: &'static str,
    tables: Arc<dyn ProcedureTables>,
}

impl fmt::Debug for Maintenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Maintenance")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

fn unsupported(name: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!("the Iceberg procedure {name} (not in this build yet)"),
    }
    .into()
}

fn long(v: &ProcedureValue) -> Option<i64> {
    match v {
        ProcedureValue::Long(n) => Some(*n),
        _ => None,
    }
}

/// `max_concurrent_deletes`, checked as Iceberg's `Preconditions` do.
fn check_concurrency(v: &ProcedureValue) -> Result<(), KrishivError> {
    if let Some(n) = long(v)
        && n <= 0
    {
        return Err(KrishivIcebergProcedureArgument {
            message: format!("max_concurrent_deletes should have value > 0, value: {n}"),
        }
        .into());
    }
    Ok(())
}

impl Procedure for Maintenance {
    fn name(&self) -> &str {
        self.name
    }

    fn parameters(&self) -> &[ProcedureParameter] {
        match self.name {
            "expire_snapshots" => EXPIRE_SNAPSHOTS,
            "rewrite_manifests" => REWRITE_MANIFESTS,
            "rewrite_data_files" => REWRITE_DATA_FILES,
            _ => REMOVE_ORPHAN_FILES,
        }
    }

    fn name_order(&self) -> Vec<&str> {
        match self.name {
            "expire_snapshots" => EXPIRE_SNAPSHOTS_ORDER.to_vec(),
            "rewrite_data_files" => REWRITE_DATA_FILES_ORDER.to_vec(),
            "remove_orphan_files" => REMOVE_ORPHAN_FILES_ORDER.to_vec(),
            _ => self.parameters().iter().map(|p| p.name).collect(),
        }
    }

    fn modifies_table(&self) -> bool {
        self.name != "remove_orphan_files"
    }

    fn table(&self, args: &[ProcedureValue]) -> Option<String> {
        match args.first() {
            Some(ProcedureValue::String(t)) => Some(t.clone()),
            _ => None,
        }
    }

    fn call<'a>(
        &'a self,
        _ctx: &'a ProcedureContext,
        args: Vec<ProcedureValue>,
    ) -> BoxFuture<'a, Result<RecordBatch, KrishivError>> {
        Box::pin(async move {
            let Some(ProcedureValue::String(table)) = args.first() else {
                return Err(KrishivIcebergProcedureArgument {
                    message: "Cannot handle an empty identifier for parameter 'table'".to_owned(),
                }
                .into());
            };
            if table.is_empty() {
                return Err(KrishivIcebergProcedureArgument {
                    message: "Cannot handle an empty identifier for parameter 'table'".to_owned(),
                }
                .into());
            }
            match self.name {
                "expire_snapshots" => self.expire(table, &args).await,
                other => Err(unsupported(other)),
            }
        })
    }
}

impl Maintenance {
    async fn expire(
        &self,
        table: &str,
        args: &[ProcedureValue],
    ) -> Result<RecordBatch, KrishivError> {
        // The parameters, in the order of `EXPIRE_SNAPSHOTS`.
        let arg = |i: usize| args.get(i).unwrap_or(&ProcedureValue::Null);
        check_concurrency(arg(3))?;
        let opts = expire::ExpireArgs {
            // Timestamps arrive as microseconds and are truncated to milliseconds.
            older_than_ms: match arg(1) {
                ProcedureValue::TimestampMicros(us) => Some(us.div_euclid(1000)),
                _ => None,
            },
            retain_last: long(arg(2)),
            snapshot_ids: match arg(5) {
                ProcedureValue::LongArray(ids) => ids.clone(),
                _ => Vec::new(),
            },
            clean_expired_metadata: matches!(arg(6), ProcedureValue::Boolean(true)),
            max_concurrent_deletes: long(arg(3)).and_then(|n| usize::try_from(n).ok()),
        };
        let t = self.tables.open(table).await?;
        let (committer, io, clock) = t.maintenance_parts(io_clock)?;
        let done = expire::expire(
            committer.as_ref(),
            &io,
            clock.as_ref(),
            t.max_metadata_bytes(),
            &opts,
        )
        .await?;
        let d = done.deleted;
        let names = [
            "deleted_data_files_count",
            "deleted_position_delete_files_count",
            "deleted_equality_delete_files_count",
            "deleted_manifest_files_count",
            "deleted_manifest_lists_count",
            "deleted_statistics_files_count",
        ];
        let values = [
            d.data_files,
            d.position_delete_files,
            d.equality_delete_files,
            d.manifests,
            d.manifest_lists,
            d.statistics_files,
        ];
        let schema = Arc::new(Schema::new(
            names
                .iter()
                .map(|n| Field::new(*n, DataType::Int64, true))
                .collect::<Vec<_>>(),
        ));
        let columns: Vec<ArrayRef> = values
            .iter()
            .map(|v| Arc::new(Int64Array::from(vec![*v])) as ArrayRef)
            .collect();
        RecordBatch::try_new(schema, columns).map_err(|e| {
            krishiv_common::error::InternalError {
                message: format!("the expire_snapshots result: {e}"),
            }
            .into()
        })
    }
}
