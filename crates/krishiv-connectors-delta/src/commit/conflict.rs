// SPDX-License-Identifier: Apache-2.0
//! The conflict checker (design note P3-T021 section 5.3): a port of delta-spark 4.4.1's
//! `ConflictChecker.checkConflicts` (`ConflictChecker.scala:280-319`) at Serializable, run
//! against each winning commit in version order.
//!
//! 1. **Protocol.** A winner protocol is checked against Krishiv's reader and writer
//!    allow-lists; a current transaction that also changes the protocol, or a winner that drops
//!    features, is `DELTA_PROTOCOL_CHANGED`.
//! 2. **Metadata.** Any winner `metaData` is `DELTA_METADATA_CHANGED`, except an identity
//!    high-water-mark update when the current transaction changes no metadata (the strict
//!    legacy rule `checkNoMetadataUpdates`, without its row-tracking exception).
//! 3. **Domain metadata.** An overlapping domain is `DELTA_CONCURRENT_TRANSACTION`.
//! 4. **App ids.** A winner `txn` for an app id the transaction read is
//!    `DELTA_CONCURRENT_TRANSACTION`.
//! 5. Row tracking is refused for writes in P3 (the writer allow-list), so it has no step here.
//! 6. **Append.** Unless the current transaction is a blind append, every winner add is checked:
//!    on a whole-table read or an unpartitioned table any add conflicts; otherwise an add whose
//!    partition values satisfy a read predicate does → `DELTA_CONCURRENT_APPEND`.
//! 7. **Delete-read.** A winner remove of a file the transaction read, or any winner remove after
//!    a whole-table read → `DELTA_CONCURRENT_DELETE_READ`.
//! 8. **Delete-delete.** A path both removed → `DELTA_CONCURRENT_DELETE_DELETE`.
//!
//! The hint subclasses follow Delta's `getPrettyPartitionMessage`: `[col=value, …]` over every
//! partition column in schema order (`null` for a missing or NULL value), from the winner's add
//! for an append, the transaction's read file for a delete-read and its own remove for a
//! delete-delete; no hint for an unpartitioned table, an empty value map or a whole-table
//! delete-read.
//!
//! # Known differences
//! - `<conflictingCommit>` prints the winner's `commitInfo` as the log holds it plus its
//!   `version`, keys in sorted order; delta-spark prints its `CommitInfo` case class, in that
//!   class's field order.
//! - A domain conflict uses the class's template; delta-spark raises a raw
//!   `ConcurrentTransactionException` message naming the domain.
//! - A read predicate that cannot be evaluated on a file's partition values (a value that does
//!   not parse) counts as a match: wider is safe.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::Array as _;
use delta_kernel::EvaluationHandler;
use delta_kernel::expressions::{PredicateRef, Scalar};
use delta_kernel::schema::{DataType, StructField, StructType};
use krishiv_common::error::{
    DeltaConcurrentAppendWithPartitionHint, DeltaConcurrentAppendWithoutHint,
    DeltaConcurrentDeleteDeleteWithPartitionHint, DeltaConcurrentDeleteDeleteWithoutHint,
    DeltaConcurrentDeleteReadWithPartitionHint, DeltaConcurrentDeleteReadWithoutHint,
    DeltaConcurrentTransaction, DeltaMetadataChanged, DeltaProtocolChanged, KrishivError,
};

use super::ReadSet;
use super::winner::{FileAction, ProtocolAction, WinningCommit};
use crate::engine::KrishivEngineData;
use crate::engine::eval::KrishivEvaluationHandler;
use crate::features::{ReadProtocol, WriteProtocol, check_read, check_write};
use crate::table::partition_scalar;

/// `<docLink>` of the conflict classes (`DeltaErrors.generateDocsLink` with
/// `/concurrency-control.html`).
pub const DOC_LINK: &str = "https://docs.delta.io/latest/concurrency-control.html";

/// `<additionalInfo>` of `DELTA_PROTOCOL_CHANGED` when the winner is version 0.
pub const EMPTY_DIRECTORY_HINT: &str = "This happens when multiple writers are writing to an \
                                        empty directory. Creating the table ahead of time will \
                                        avoid this conflict. ";

/// A partition column: its name, the key its values are stored under and its type.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionColumn {
    /// The column's name (what read predicates name).
    pub name: String,
    /// The physical name (the key of `partitionValues`).
    pub physical: String,
    /// Its type.
    pub data_type: DataType,
}

/// The transaction being committed, as the checker sees it.
#[derive(Debug)]
pub struct CurrentTxn<'a> {
    /// The table's name or path, for messages.
    pub table: &'a str,
    /// The protocol and configuration of the snapshot it read.
    pub read_protocol: &'a WriteProtocol,
    /// The table's partition columns, in schema order.
    pub partition_columns: &'a [PartitionColumn],
    /// What it read.
    pub read_set: &'a ReadSet,
    /// The files it removes.
    pub removes: &'a [FileAction],
    /// Whether it writes a `metaData` action.
    pub changes_metadata: bool,
    /// Whether it writes a `protocol` action.
    pub changes_protocol: bool,
    /// The domains it writes.
    pub domains: &'a [String],
    /// Whether it is a blind append (only adds, no read of the table).
    pub blind_append: bool,
}

/// Checks the transaction `txn` against the winning commit `winner` (see the module docs).
///
/// # Errors
/// The conflict's class, or the allow-lists' refusal of a winner protocol.
pub fn check(txn: &CurrentTxn<'_>, winner: &WinningCommit) -> Result<(), KrishivError> {
    check_protocol(txn, winner)?;
    if winner.metadata_changed && !(winner.identity_only_metadata_update() && !txn.changes_metadata)
    {
        return Err(DeltaMetadataChanged {
            table_name: txn.table.to_owned(),
            conflicting_commit: winner.conflicting_commit(),
            doc_link: DOC_LINK.to_owned(),
        }
        .into());
    }
    let domains_supported = txn
        .read_protocol
        .read
        .writer_features
        .iter()
        .flatten()
        .any(|f| f == "domainMetadata");
    if domains_supported && txn.domains.iter().any(|d| winner.domains.contains(d)) {
        return Err(concurrent_transaction(winner));
    }
    if winner
        .app_ids
        .iter()
        .any(|id| txn.read_set.app_ids.contains(id))
    {
        return Err(concurrent_transaction(winner));
    }
    check_appends(txn, winner)?;
    check_deletes(txn, winner)
}

fn concurrent_transaction(winner: &WinningCommit) -> KrishivError {
    DeltaConcurrentTransaction {
        conflicting_commit: winner.conflicting_commit(),
        doc_link: DOC_LINK.to_owned(),
    }
    .into()
}

fn check_protocol(txn: &CurrentTxn<'_>, winner: &WinningCommit) -> Result<(), KrishivError> {
    let Some(p) = &winner.protocol else {
        return Ok(());
    };
    let read = &txn.read_protocol.read;
    let winner_read = ReadProtocol {
        min_reader_version: i32::try_from(p.min_reader_version).unwrap_or(i32::MAX),
        reader_features: p.reader_features.clone(),
        writer_features: p.writer_features.clone(),
        configuration: read.configuration.clone(),
    };
    check_read(txn.table, &winner_read)?;
    check_write(
        txn.table,
        &WriteProtocol {
            read: winner_read,
            min_writer_version: i32::try_from(p.min_writer_version).unwrap_or(i32::MAX),
            invariants_in_use: txn.read_protocol.invariants_in_use,
        },
    )?;
    if txn.changes_protocol || drops_features(txn.read_protocol, p) {
        let additional_info = if winner.version == 0 && winner.commit_info.is_some() {
            EMPTY_DIRECTORY_HINT.to_owned()
        } else {
            String::new()
        };
        return Err(DeltaProtocolChanged {
            additional_info,
            conflicting_commit: winner.conflicting_commit(),
            doc_link: DOC_LINK.to_owned(),
        }
        .into());
    }
    Ok(())
}

/// Whether `winner` removes a feature or lowers a version of the protocol `read`
/// (`TableFeature.isProtocolRemovingFeatures`).
fn drops_features(read: &WriteProtocol, winner: &ProtocolAction) -> bool {
    if winner.min_reader_version < i64::from(read.read.min_reader_version)
        || winner.min_writer_version < i64::from(read.min_writer_version)
    {
        return true;
    }
    let kept: Vec<&str> = winner.features().collect();
    read.read
        .reader_features
        .iter()
        .chain(&read.read.writer_features)
        .flatten()
        .any(|f| !kept.contains(&f.as_str()))
}

fn check_appends(txn: &CurrentTxn<'_>, winner: &WinningCommit) -> Result<(), KrishivError> {
    if txn.blind_append || winner.adds.is_empty() {
        return Ok(());
    }
    let conflicting = if txn.read_set.whole_table || txn.partition_columns.is_empty() {
        winner.adds.first()
    } else {
        winner
            .adds
            .iter()
            .find(|add| matches_read_predicates(txn, &add.partition_values))
    };
    let Some(add) = conflicting else {
        return Ok(());
    };
    let (operation, table_name, version, doc_link) = parameters(txn, winner);
    Err(match pretty_partition(txn, &add.partition_values) {
        Some(partition_values) => DeltaConcurrentAppendWithPartitionHint {
            operation,
            table_name,
            version,
            partition_values,
            doc_link,
        }
        .into(),
        None => DeltaConcurrentAppendWithoutHint {
            operation,
            table_name,
            version,
            doc_link,
        }
        .into(),
    })
}

fn check_deletes(txn: &CurrentTxn<'_>, winner: &WinningCommit) -> Result<(), KrishivError> {
    let read: HashMap<&str, &HashMap<String, Option<String>>> = txn
        .read_set
        .files
        .iter()
        .map(|f| (f.path.as_str(), &f.partition_values))
        .collect();
    let overlap = winner
        .removes
        .iter()
        .find_map(|r| read.get(r.path.as_str()).copied());
    let whole_table_hit = !winner.removes.is_empty() && txn.read_set.whole_table;
    if overlap.is_some() || whole_table_hit {
        let (operation, table_name, version, doc_link) = parameters(txn, winner);
        let hint = overlap.and_then(|pv| pretty_partition(txn, pv));
        return Err(match hint {
            Some(partition_values) => DeltaConcurrentDeleteReadWithPartitionHint {
                operation,
                table_name,
                version,
                partition_values,
                doc_link,
            }
            .into(),
            None => DeltaConcurrentDeleteReadWithoutHint {
                operation,
                table_name,
                version,
                doc_link,
            }
            .into(),
        });
    }
    let removed: HashMap<&str, &HashMap<String, Option<String>>> = txn
        .removes
        .iter()
        .map(|f| (f.path.as_str(), &f.partition_values))
        .collect();
    if let Some(pv) = winner
        .removes
        .iter()
        .find_map(|r| removed.get(r.path.as_str()).copied())
    {
        let (operation, table_name, version, doc_link) = parameters(txn, winner);
        return Err(match pretty_partition(txn, pv) {
            Some(partition_values) => DeltaConcurrentDeleteDeleteWithPartitionHint {
                operation,
                table_name,
                version,
                partition_values,
                doc_link,
            }
            .into(),
            None => DeltaConcurrentDeleteDeleteWithoutHint {
                operation,
                table_name,
                version,
                doc_link,
            }
            .into(),
        });
    }
    Ok(())
}

/// `<operation>`, `<tableName>`, `<version>` and `<docLink>` of the file-level classes.
fn parameters(txn: &CurrentTxn<'_>, winner: &WinningCommit) -> (String, String, String, String) {
    (
        winner.operation(),
        txn.table.to_owned(),
        winner.version.to_string(),
        DOC_LINK.to_owned(),
    )
}

/// Delta's `getPrettyPartitionMessage`.
fn pretty_partition(
    txn: &CurrentTxn<'_>,
    values: &HashMap<String, Option<String>>,
) -> Option<String> {
    if txn.partition_columns.is_empty() || values.is_empty() {
        return None;
    }
    let parts: Vec<String> = txn
        .partition_columns
        .iter()
        .map(|c| {
            let v = values
                .get(&c.physical)
                .and_then(Option::as_deref)
                .unwrap_or("null");
            format!("{}={v}", c.name)
        })
        .collect();
    Some(format!("[{}]", parts.join(", ")))
}

/// Whether some read predicate holds on the partition values `values` (TRUE only; a NULL
/// result does not match, as Delta's `filterFileList` keeps only TRUE). A value that does not
/// parse or a predicate that fails to evaluate counts as a match.
fn matches_read_predicates(txn: &CurrentTxn<'_>, values: &HashMap<String, Option<String>>) -> bool {
    let predicates = &txn.read_set.predicates;
    if predicates.is_empty() {
        return false;
    }
    let schema =
        Arc::new(StructType::new_unchecked(txn.partition_columns.iter().map(
            |c| StructField::nullable(c.name.clone(), c.data_type.clone()),
        )));
    let row: Result<Vec<Scalar>, _> = txn
        .partition_columns
        .iter()
        .map(|c| {
            partition_scalar(
                values.get(&c.physical).and_then(Option::as_ref),
                &c.data_type,
            )
        })
        .collect();
    let Ok(row) = row else {
        return true;
    };
    let handler = KrishivEvaluationHandler::new(crate::engine::DeltaLimits::default());
    let Ok(batch) = handler.create_many(Arc::clone(&schema), vec![row]) else {
        return true;
    };
    predicates
        .iter()
        .any(|p| holds(&handler, &schema, p, batch.as_ref()).unwrap_or(true))
}

fn holds(
    handler: &KrishivEvaluationHandler,
    schema: &Arc<StructType>,
    predicate: &PredicateRef,
    batch: &dyn delta_kernel::EngineData,
) -> Option<bool> {
    let evaluator = handler
        .new_predicate_evaluator(Arc::clone(schema), Arc::clone(predicate))
        .ok()?;
    let out = evaluator.evaluate(batch).ok()?;
    let out = KrishivEngineData::batch_of(out.as_ref()).ok()?;
    let column = out
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::BooleanArray>()?;
    Some(column.len() == 1 && column.is_valid(0) && column.value(0))
}
