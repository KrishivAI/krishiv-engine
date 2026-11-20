// SPDX-License-Identifier: Apache-2.0
//! The conflict checker (design note P3-T021 section 5.3): a port of delta-spark 4.4.1's
//! `ConflictChecker.checkConflicts` (`ConflictChecker.scala:280-319`) at Serializable, run
//! against each winning commit in version order.
//!
//! 1. **Protocol.** A winner protocol is checked against Krishiv's reader and writer
//!    allow-lists; a current transaction that also changes the protocol, or a winner that drops
//!    a feature, is `DELTA_PROTOCOL_CHANGED`. Features are compared as Delta compares them
//!    (`TableFeature.isProtocolRemovingFeatures`): a legacy protocol's implied features count.
//! 2. **Metadata.** Any winner `metaData` is `DELTA_METADATA_CHANGED`, except an identity
//!    high-water-mark update when the current transaction changes no metadata (the strict
//!    legacy rule `checkNoMetadataUpdates`, without its row-tracking exception).
//! 3. **Domain metadata.** An overlapping domain is `DELTA_CONCURRENT_TRANSACTION`.
//! 4. **App ids.** A winner `txn` for an app id the transaction read is
//!    `DELTA_CONCURRENT_TRANSACTION`.
//! 5. Row tracking is refused for writes in P3 (the writer allow-list), so it has no step here.
//! 6. **Append.** Unless the current transaction is a blind append, every winner add is checked
//!    (at Serializable a blind-append winner's adds too): on a whole-table read or an
//!    unpartitioned table any add conflicts; otherwise an add whose partition values satisfy a
//!    read predicate does → `DELTA_CONCURRENT_APPEND`.
//! 7. **Delete-read.** A winner remove of a file the transaction read (with the read file's
//!    partition as the hint), or else any winner remove after a whole-table read (no hint) →
//!    `DELTA_CONCURRENT_DELETE_READ`.
//! 8. **Delete-delete.** A path both removed → `DELTA_CONCURRENT_DELETE_DELETE`.
//!
//! The hint subclasses follow Delta's `getPrettyPartitionMessage`: `[col=value, …]` over every
//! partition column in the table's `partitionColumns` order (`null` for a missing or NULL
//! value), from the winner's add for an append, the transaction's read file for a delete-read
//! and its own remove for a delete-delete; no hint for an unpartitioned table or an empty value
//! map.
//!
//! # Known differences
//! - `<conflictingCommit>` prints the winner's `commitInfo` as the log holds it plus its
//!   `version`, keys in sorted order; delta-spark prints its `CommitInfo` case class, in that
//!   class's field order.
//! - A domain conflict uses the class's template; delta-spark raises a raw
//!   `ConcurrentTransactionException` message naming the domain.
//! - A read predicate that cannot be evaluated on a file's partition values (a value that does
//!   not parse) counts as a match: wider is safe.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::{Array as _, BooleanArray, RecordBatch};
use delta_kernel::EvaluationHandler;
use delta_kernel::expressions::{PredicateRef, Scalar};
use delta_kernel::schema::{DataType, StructField, StructType};
use krishiv_common::error::{
    DeltaConcurrentAppendWithPartitionHint, DeltaConcurrentAppendWithoutHint,
    DeltaConcurrentDeleteDeleteWithPartitionHint, DeltaConcurrentDeleteDeleteWithoutHint,
    DeltaConcurrentDeleteReadWithPartitionHint, DeltaConcurrentDeleteReadWithoutHint,
    DeltaConcurrentTransaction, DeltaMetadataChanged, DeltaProtocolChanged, KrishivError,
};

use super::winner::{FileAction, ProtocolAction, WinningCommit};
use super::{PartitionValues, ReadSet};
use crate::engine::eval::KrishivEvaluationHandler;
use crate::engine::{DeltaLimits, KrishivEngineData};
use crate::features::{ReadProtocol, WriteProtocol, check_read, check_write};
use crate::table::partition_scalar;

/// `<docLink>` of the conflict classes (`DeltaErrors.generateDocsLink` with
/// `/concurrency-control.html`).
pub const DOC_LINK: &str = "https://docs.delta.io/latest/concurrency-control.html";

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
#[derive(Debug, Clone)]
pub struct CurrentTxn {
    /// The table's name or path, for messages.
    pub table: String,
    /// The protocol and configuration of the snapshot it read.
    pub read_protocol: WriteProtocol,
    /// The table's partition columns, in its `partitionColumns` order.
    pub partition_columns: Vec<PartitionColumn>,
    /// What it read.
    pub read_set: ReadSet,
    /// The files it removes.
    pub removes: Vec<FileAction>,
    /// Whether it writes a `metaData` action.
    pub changes_metadata: bool,
    /// Whether it writes a `protocol` action.
    pub changes_protocol: bool,
    /// The domains it writes.
    pub domains: Vec<String>,
    /// Whether it is a blind append (only adds, no read of the table).
    pub blind_append: bool,
    /// The parse limits of the evaluator.
    pub limits: DeltaLimits,
}

/// A transaction's checker: what every winner is checked against, indexed once.
#[derive(Debug)]
pub struct Checker {
    txn: CurrentTxn,
    read_files: HashMap<String, PartitionValues>,
    removed: HashMap<String, PartitionValues>,
    read_features: BTreeSet<String>,
    partition_schema: Arc<StructType>,
    handler: KrishivEvaluationHandler,
}

impl Checker {
    /// The checker of `txn`.
    #[must_use]
    pub fn new(txn: CurrentTxn) -> Self {
        let index = |files: &[FileAction]| {
            files
                .iter()
                .map(|f| (f.path.clone(), f.partition_values.clone()))
                .collect()
        };
        let read = &txn.read_protocol;
        let read_features = supported_features(
            i64::from(read.read.min_reader_version),
            i64::from(read.min_writer_version),
            read.read.reader_features.as_deref(),
            read.read.writer_features.as_deref(),
        );
        let partition_schema =
            Arc::new(StructType::new_unchecked(txn.partition_columns.iter().map(
                |c| StructField::nullable(c.name.clone(), c.data_type.clone()),
            )));
        Self {
            read_files: index(&txn.read_set.files),
            removed: index(&txn.removes),
            read_features,
            partition_schema,
            handler: KrishivEvaluationHandler::new(txn.limits),
            txn,
        }
    }

    /// The transaction.
    #[must_use]
    pub fn txn(&self) -> &CurrentTxn {
        &self.txn
    }

    /// Checks the transaction against the winning commit `winner` (see the module docs).
    ///
    /// # Errors
    /// The conflict's class, or the allow-lists' refusal of a winner protocol.
    pub fn check(&self, winner: &WinningCommit) -> Result<(), KrishivError> {
        let txn = &self.txn;
        self.check_protocol(winner)?;
        if winner.metadata_changed
            && !(winner.identity_only_metadata_update() && !txn.changes_metadata)
        {
            return Err(DeltaMetadataChanged {
                table_name: txn.table.clone(),
                conflicting_commit: winner.conflicting_commit(),
                doc_link: DOC_LINK.to_owned(),
            }
            .into());
        }
        if self.read_features.contains("domainMetadata")
            && txn.domains.iter().any(|d| winner.domains.contains(d))
        {
            return Err(concurrent_transaction(winner));
        }
        if winner
            .app_ids
            .iter()
            .any(|id| txn.read_set.app_ids.contains(id))
        {
            return Err(concurrent_transaction(winner));
        }
        self.check_appends(winner)?;
        self.check_deletes(winner)
    }

    fn check_protocol(&self, winner: &WinningCommit) -> Result<(), KrishivError> {
        let Some(p) = &winner.protocol else {
            return Ok(());
        };
        let txn = &self.txn;
        let read = &txn.read_protocol.read;
        let winner_read = ReadProtocol {
            min_reader_version: i32::try_from(p.min_reader_version).unwrap_or(i32::MAX),
            reader_features: p.reader_features.clone(),
            writer_features: p.writer_features.clone(),
            configuration: read.configuration.clone(),
        };
        check_read(&txn.table, &winner_read)?;
        check_write(
            &txn.table,
            &WriteProtocol {
                read: winner_read,
                min_writer_version: i32::try_from(p.min_writer_version).unwrap_or(i32::MAX),
                invariants_in_use: txn.read_protocol.invariants_in_use,
            },
        )?;
        if txn.changes_protocol || self.drops_features(p) {
            return Err(DeltaProtocolChanged {
                additional_info: String::new(),
                conflicting_commit: winner.conflicting_commit(),
                doc_link: DOC_LINK.to_owned(),
            }
            .into());
        }
        Ok(())
    }

    /// Whether `winner` lacks a feature the read protocol supports, implied ones included.
    fn drops_features(&self, winner: &ProtocolAction) -> bool {
        let kept = supported_features(
            winner.min_reader_version,
            winner.min_writer_version,
            winner.reader_features.as_deref(),
            winner.writer_features.as_deref(),
        );
        !self.read_features.is_subset(&kept)
    }

    fn check_appends(&self, winner: &WinningCommit) -> Result<(), KrishivError> {
        let txn = &self.txn;
        if txn.blind_append || winner.adds.is_empty() {
            return Ok(());
        }
        let conflicting = if txn.read_set.whole_table || txn.partition_columns.is_empty() {
            winner.adds.first()
        } else {
            self.first_matching(&winner.adds)
        };
        let Some(add) = conflicting else {
            return Ok(());
        };
        let (operation, table_name, version, doc_link) = self.parameters(winner);
        Err(match self.pretty_partition(&add.partition_values) {
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

    fn check_deletes(&self, winner: &WinningCommit) -> Result<(), KrishivError> {
        let overlap = winner
            .removes
            .iter()
            .find_map(|r| self.read_files.get(&r.path));
        let whole_table_hit = !winner.removes.is_empty() && self.txn.read_set.whole_table;
        if overlap.is_some() || whole_table_hit {
            let (operation, table_name, version, doc_link) = self.parameters(winner);
            return Err(match overlap.and_then(|pv| self.pretty_partition(pv)) {
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
        if let Some(pv) = winner
            .removes
            .iter()
            .find_map(|r| self.removed.get(&r.path))
        {
            let (operation, table_name, version, doc_link) = self.parameters(winner);
            return Err(match self.pretty_partition(pv) {
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
    fn parameters(&self, winner: &WinningCommit) -> (String, String, String, String) {
        (
            winner.operation(),
            self.txn.table.clone(),
            winner.version.to_string(),
            DOC_LINK.to_owned(),
        )
    }

    /// Delta's `getPrettyPartitionMessage`.
    fn pretty_partition(&self, values: &PartitionValues) -> Option<String> {
        let columns = &self.txn.partition_columns;
        if columns.is_empty() || values.is_empty() {
            return None;
        }
        let parts: Vec<String> = columns
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

    /// The first add whose partition values satisfy a read predicate (TRUE only: a NULL result
    /// does not match, as Delta's `filterFileList` keeps only TRUE). The adds' values are one
    /// batch; an add whose values do not parse, or a predicate that fails to evaluate, counts
    /// as a match.
    fn first_matching<'w>(&self, adds: &'w [FileAction]) -> Option<&'w FileAction> {
        let predicates = &self.txn.read_set.predicates;
        if predicates.is_empty() {
            return None;
        }
        let mut rows = Vec::with_capacity(adds.len());
        let mut parsed = Vec::with_capacity(adds.len());
        for add in adds {
            let row: Result<Vec<Scalar>, _> = self
                .txn
                .partition_columns
                .iter()
                .map(|c| {
                    partition_scalar(
                        add.partition_values
                            .get(&c.physical)
                            .and_then(Option::as_ref),
                        &c.data_type,
                    )
                })
                .collect();
            match row {
                Ok(row) => {
                    parsed.push(true);
                    rows.push(row);
                }
                // An unparseable value matches: no later add can come first.
                Err(_) => {
                    parsed.push(false);
                    break;
                }
            }
        }
        let matched = self.evaluate(predicates, rows);
        // Every add before an unparseable one parsed, so add `i` is row `i`.
        for (row, (add, ok)) in adds.iter().zip(parsed).enumerate() {
            if !ok || matched.as_ref().is_none_or(|m| m[row]) {
                return Some(add);
            }
        }
        None
    }

    /// Per row, whether some predicate is TRUE; `None` when evaluation fails.
    fn evaluate(&self, predicates: &[PredicateRef], rows: Vec<Vec<Scalar>>) -> Option<Vec<bool>> {
        if rows.is_empty() {
            return Some(Vec::new());
        }
        let n = rows.len();
        let batch = self
            .handler
            .create_many(Arc::clone(&self.partition_schema), rows)
            .ok()?;
        let mut any = vec![false; n];
        for p in predicates {
            let column = self.holds(p, batch.as_ref())?;
            for (i, hit) in any.iter_mut().enumerate() {
                *hit |= column.is_valid(i) && column.value(i);
            }
        }
        Some(any)
    }

    fn holds(
        &self,
        predicate: &PredicateRef,
        batch: &dyn delta_kernel::EngineData,
    ) -> Option<BooleanArray> {
        let evaluator = self
            .handler
            .new_predicate_evaluator(Arc::clone(&self.partition_schema), Arc::clone(predicate))
            .ok()?;
        let out = evaluator.evaluate(batch).ok()?;
        let out: &RecordBatch = KrishivEngineData::batch_of(out.as_ref()).ok()?;
        out.column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .cloned()
    }
}

fn concurrent_transaction(winner: &WinningCommit) -> KrishivError {
    DeltaConcurrentTransaction {
        conflicting_commit: winner.conflicting_commit(),
        doc_link: DOC_LINK.to_owned(),
    }
    .into()
}

/// Every feature a protocol supports, the ones its legacy versions imply included
/// (delta-spark's `implicitlyAndExplicitlySupportedFeatures`, `TableFeature.scala:470-497`): a
/// table-features protocol (writer 7) implies none and lists its writer features (and, at
/// reader 3, its reader features); a legacy one implies the features of its writer version, and
/// `columnMapping` only at reader 2 or above with writer 5 or 6. Reader 3 with a legacy writer
/// (below 7) is not a valid protocol; for it this function returns the legacy writer's features
/// where delta-spark implies none, which never matters because no table can be read at it.
#[must_use]
pub fn supported_features(
    min_reader: i64,
    min_writer: i64,
    reader_features: Option<&[String]>,
    writer_features: Option<&[String]>,
) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    if min_writer >= 7 {
        out.extend(writer_features.into_iter().flatten().cloned());
        if min_reader >= 3 {
            out.extend(reader_features.into_iter().flatten().cloned());
        }
        return out;
    }
    let implied: &[(i64, &str)] = &[
        (2, "appendOnly"),
        (2, "invariants"),
        (3, "checkConstraints"),
        (4, "changeDataFeed"),
        (4, "generatedColumns"),
        (6, "identityColumns"),
    ];
    out.extend(
        implied
            .iter()
            .filter(|(v, _)| min_writer >= *v)
            .map(|(_, f)| (*f).to_owned()),
    );
    if min_reader >= 2 && min_writer >= 5 {
        out.insert("columnMapping".to_owned());
    }
    out
}
