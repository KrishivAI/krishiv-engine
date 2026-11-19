// SPDX-License-Identifier: Apache-2.0
//! Krishiv's reader allow-list (design note P3-T021 sections 3.5 to 3.7), checked on a table's
//! protocol and configuration before any scan.
//!
//! The kernel refuses reader features it does not know, but accepts some whose data-side
//! requirements fall on the connector's scan; this list says which of those Krishiv's scan
//! honours. Writer-only features never block a read, with one exception: coordinated commits,
//! whose unbackfilled commits a reader of the log alone would silently miss.
//!
//! | Reader feature | P3 read |
//! |---|---|
//! | `columnMapping`, `deletionVectors`, `timestampNtz`, `typeWidening`, `typeWidening-preview`, `v2Checkpoint`, `vacuumProtocolCheck`, `variantType`, `variantType-preview` | yes |
//! | `catalogManaged`, `catalogOwned-preview` | refused: `UNSUPPORTED_FEATURE.TABLE_OPERATION` |
//! | `variantShredding`, `variantShredding-preview`, `geospatial`, anything else | refused: `DELTA_UNSUPPORTED_FEATURES_FOR_READ` |
//!
//! A `coordinatedCommits-preview` writer feature or any `delta.coordinatedCommits.*`
//! configuration, and requested row-tracking columns, are refused as
//! `UNSUPPORTED_FEATURE.TABLE_OPERATION`. A reader version above 3 is
//! `DELTA_INVALID_PROTOCOL_VERSION`.
//!
//! # The writer allow-list (section 4.2)
//! [`check_write`] runs before any file of a write is written, then the kernel's own check runs
//! when the transaction is built. The kernel marks features write-supported on the premise that
//! the connector meets their data requirements, and Krishiv writes the data, so its list is
//! narrower:
//!
//! | Writer feature | P3 write |
//! |---|---|
//! | `appendOnly`, `changeDataFeed`, `columnMapping`, `deletionVectors`, `domainMetadata`, `inCommitTimestamp`, `timestampNtz`, `typeWidening`, `typeWidening-preview`, `v2Checkpoint`, `vacuumProtocolCheck` | yes (`columnMapping`: the writer writes physical names and Parquet field ids, [`crate::write`]) |
//! | `invariants` | yes while no column carries `delta.invariants` |
//! | `variantType`, `variantType-preview` | refused until the writer passes the Variant round trip |
//! | anything else | refused |
//!
//! A refusal is `DELTA_UNSUPPORTED_FEATURES_FOR_WRITE` (56038), naming every refused feature in
//! the protocol's order. Writer versions 1 and 2 are writable; 3 to 6 are
//! `DELTA_INVALID_PROTOCOL_VERSION`, because the kernel infers `checkConstraints` and the other
//! legacy features there whether or not the table uses them (finding 1); above 7 too.
//!
//! # Known differences
//! - When the kernel refuses an unsupported reader feature itself (at snapshot load, before this
//!   list runs) it stops at the first one, so only that one is named; delta-spark names all.

use std::collections::{BTreeMap, HashMap};

use delta_kernel::schema::{DataType, StructType};
use delta_kernel::{KernelError, Snapshot};
use krishiv_common::error::{
    DeltaInvalidProtocolVersion, DeltaUnsupportedFeaturesForRead, DeltaUnsupportedFeaturesForWrite,
    KrishivError, UnsupportedFeatureTableOperation,
};

use crate::error::{SUPPORTED_READERS, SUPPORTED_WRITERS, delta_version, from_kernel};

/// Reader features Krishiv's scan reads (section 3.5's "yes" rows).
pub const READABLE: &[&str] = &[
    "columnMapping",
    "deletionVectors",
    "timestampNtz",
    "typeWidening",
    "typeWidening-preview",
    "v2Checkpoint",
    "vacuumProtocolCheck",
    "variantType",
    "variantType-preview",
];

/// The features that make a table catalog-managed.
pub const CATALOG_MANAGED: &[&str] = &["catalogManaged", "catalogOwned-preview"];

/// The writer feature of coordinated commits.
pub const COORDINATED_COMMITS: &str = "coordinatedCommits-preview";

/// The configuration prefix of coordinated commits.
pub const COORDINATED_COMMITS_PREFIX: &str = "delta.coordinatedCommits.";

/// The operation named when a catalog-managed table is refused.
pub const CATALOG_MANAGED_OPERATION: &str =
    "reading a catalog-managed Delta table (its catalog's ratified commits are not read)";

/// The operation named when a table with coordinated commits is refused.
pub const COORDINATED_COMMITS_OPERATION: &str = "reading a Delta table with coordinated commits \
                                                 (its unbackfilled commits are not read)";

/// The operation named when row-tracking columns are requested.
pub const ROW_TRACKING_OPERATION: &str = "reading Delta row-tracking columns";

/// What the allow-list reads of a table: its protocol and configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReadProtocol {
    /// `minReaderVersion`.
    pub min_reader_version: i32,
    /// `readerFeatures` (reader version 3), in the protocol's order.
    pub reader_features: Option<Vec<String>>,
    /// `writerFeatures` (writer version 7), in the protocol's order.
    pub writer_features: Option<Vec<String>>,
    /// The table's `metaData.configuration`.
    pub configuration: HashMap<String, String>,
}

impl ReadProtocol {
    /// The protocol and configuration of `snapshot`.
    #[must_use]
    pub fn of(snapshot: &Snapshot) -> Self {
        let tc = snapshot.table_configuration();
        let protocol = tc.protocol();
        let names = |features: Option<&[delta_kernel::table_features::TableFeature]>| {
            features.map(|f| f.iter().map(ToString::to_string).collect())
        };
        Self {
            min_reader_version: protocol.min_reader_version(),
            reader_features: names(protocol.reader_features()),
            writer_features: names(protocol.writer_features()),
            configuration: tc.metadata().configuration().clone(),
        }
    }

    fn has(&self, feature: &str) -> bool {
        let any =
            |f: &Option<Vec<String>>| f.as_ref().is_some_and(|f| f.iter().any(|n| n == feature));
        any(&self.reader_features) || any(&self.writer_features)
    }
}

/// Checks that Krishiv reads the table `table` (a name or path, for messages) with protocol
/// `protocol`.
///
/// # Errors
/// `UNSUPPORTED_FEATURE.TABLE_OPERATION` for a catalog-managed table or coordinated commits;
/// `DELTA_UNSUPPORTED_FEATURES_FOR_READ` (56038) naming every reader feature outside
/// [`READABLE`], in the protocol's order.
pub fn check_read(table: &str, protocol: &ReadProtocol) -> Result<(), KrishivError> {
    if CATALOG_MANAGED.iter().any(|f| protocol.has(f)) {
        return Err(table_operation(table, CATALOG_MANAGED_OPERATION));
    }
    if protocol.has(COORDINATED_COMMITS)
        || protocol
            .configuration
            .keys()
            .any(|k| k.starts_with(COORDINATED_COMMITS_PREFIX))
    {
        return Err(table_operation(table, COORDINATED_COMMITS_OPERATION));
    }
    let unsupported: Vec<&str> = protocol
        .reader_features
        .iter()
        .flatten()
        .map(String::as_str)
        .filter(|f| !READABLE.contains(f))
        .collect();
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(unsupported_for_read(table, &unsupported))
    }
}

/// The refusal of requested row-tracking columns (`_metadata.row_id`,
/// `_metadata.row_commit_version`) on the table `table`.
#[must_use]
pub fn row_tracking_refused(table: &str) -> KrishivError {
    table_operation(table, ROW_TRACKING_OPERATION)
}

/// The Krishiv error for `error`, raised while opening a snapshot of `table`: the kernel's
/// refusals of a catalog-managed table (no catalog version given) and of a reader feature it
/// does not support become this module's classes; anything else maps as
/// [`from_kernel`].
#[must_use]
pub fn snapshot_error(error: &KernelError, table: &str) -> KrishivError {
    match error.without_backtrace() {
        KernelError::MaxCatalogVersion(_) => table_operation(table, CATALOG_MANAGED_OPERATION),
        KernelError::Unsupported(what) => {
            if let Some(feature) = kernel_unsupported_feature(what) {
                unsupported_for_read(table, &[feature])
            } else if let Some(version) = kernel_unsupported_reader_version(what) {
                invalid_reader_version(table, version)
            } else {
                from_kernel(error, table)
            }
        }
        _ => from_kernel(error, table),
    }
}

/// The feature of the kernel's "Feature '<name>' is not supported" refusal
/// (`delta_kernel` 0.30.1 `TableConfiguration::check_feature_support`).
fn kernel_unsupported_feature(message: &str) -> Option<&str> {
    message
        .strip_prefix("Feature '")?
        .strip_suffix("' is not supported")
}

/// The version of the kernel's "Unsupported minimum reader version <n>" refusal
/// (`delta_kernel` 0.30.1 `check_reader_version_range`).
fn kernel_unsupported_reader_version(message: &str) -> Option<&str> {
    message
        .strip_prefix("Unsupported minimum reader version ")
        .filter(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit() || b == b'-'))
}

fn invalid_reader_version(table: &str, version: &str) -> KrishivError {
    DeltaInvalidProtocolVersion {
        table_name_or_path: table.to_owned(),
        reader_required: version.to_owned(),
        writer_required: "unknown".to_owned(),
        delta_version: delta_version(),
        supported_readers: SUPPORTED_READERS.to_owned(),
        supported_writers: SUPPORTED_WRITERS.to_owned(),
    }
    .into()
}

fn table_operation(table: &str, operation: &str) -> KrishivError {
    UnsupportedFeatureTableOperation {
        table_name: table.to_owned(),
        operation: operation.to_owned(),
    }
    .into()
}

fn unsupported_for_read(table: &str, features: &[&str]) -> KrishivError {
    DeltaUnsupportedFeaturesForRead {
        table_name_or_path: table.to_owned(),
        delta_version: delta_version(),
        unsupported: features.join(", "),
    }
    .into()
}

/// Writer features Krishiv's writes honour (section 4.2's "yes" rows; `invariants` only while
/// none is in use).
pub const WRITABLE: &[&str] = &[
    "appendOnly",
    "changeDataFeed",
    "columnMapping",
    "deletionVectors",
    "domainMetadata",
    "inCommitTimestamp",
    "invariants",
    "timestampNtz",
    "typeWidening",
    "typeWidening-preview",
    "v2Checkpoint",
    "vacuumProtocolCheck",
];

/// The column metadata key of a Delta invariant.
pub const INVARIANTS_KEY: &str = "delta.invariants";

/// What the writer allow-list reads of a table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WriteProtocol {
    /// The protocol's features and the table's configuration.
    pub read: ReadProtocol,
    /// `minWriterVersion`.
    pub min_writer_version: i32,
    /// Whether some column, nested ones included, carries `delta.invariants`.
    pub invariants_in_use: bool,
}

impl WriteProtocol {
    /// The protocol, configuration and schema of `snapshot`.
    #[must_use]
    pub fn of(snapshot: &Snapshot) -> Self {
        Self {
            read: ReadProtocol::of(snapshot),
            min_writer_version: snapshot
                .table_configuration()
                .protocol()
                .min_writer_version(),
            invariants_in_use: has_invariants(&snapshot.schema()),
        }
    }
}

fn has_invariants(schema: &StructType) -> bool {
    fn in_type(t: &DataType) -> bool {
        match t {
            DataType::Struct(s) => has_invariants(s),
            DataType::Array(a) => in_type(a.element_type()),
            DataType::Map(m) => in_type(m.key_type()) || in_type(m.value_type()),
            _ => false,
        }
    }
    schema
        .fields()
        .any(|f| f.metadata().contains_key(INVARIANTS_KEY) || in_type(f.data_type()))
}

/// Checks that Krishiv writes the table `table` (a name or path, for messages) with protocol
/// `protocol`. Runs before any file is written.
///
/// # Errors
/// `DELTA_INVALID_PROTOCOL_VERSION` for writer versions 3 to 6 and above 7;
/// `DELTA_UNSUPPORTED_FEATURES_FOR_WRITE` (56038) naming every writer feature outside
/// [`WRITABLE`] in the protocol's order, and `invariants` while one is in use (on a version 2
/// table, where the feature is implied, too).
pub fn check_write(table: &str, protocol: &WriteProtocol) -> Result<(), KrishivError> {
    let version = protocol.min_writer_version;
    match version {
        1 | 2 => {
            if version == 2 && protocol.invariants_in_use {
                return Err(unsupported_for_write(table, &["invariants"]));
            }
            Ok(())
        }
        7 => {
            let unsupported: Vec<&str> = protocol
                .read
                .writer_features
                .iter()
                .flatten()
                .map(String::as_str)
                .filter(|f| {
                    !WRITABLE.contains(f) || (*f == "invariants" && protocol.invariants_in_use)
                })
                .collect();
            if unsupported.is_empty() {
                Ok(())
            } else {
                Err(unsupported_for_write(table, &unsupported))
            }
        }
        _ => Err(DeltaInvalidProtocolVersion {
            table_name_or_path: table.to_owned(),
            reader_required: protocol.read.min_reader_version.to_string(),
            writer_required: version.to_string(),
            delta_version: delta_version(),
            supported_readers: SUPPORTED_READERS.to_owned(),
            supported_writers: WRITABLE_WRITERS.to_owned(),
        }
        .into()),
    }
}

/// Checks the protocol a create will write against the writer allow-list, before the create
/// transaction is built and before any file is written. The kernel's create admits features
/// Krishiv's writer does not honour (`rowTracking`, `variantType`, `icebergCompatV2`, ...), so
/// without this check Krishiv would create a table it then refuses to write. The features are
/// the ones the kernel's create (delta_kernel 0.30) enables from `properties` and `schema`: each
/// `delta.feature.<name>` signal, the enablement properties `delta.enableRowTracking`,
/// `delta.enableIcebergCompatV2` and `delta.enableIcebergCompatV3` (with the row tracking
/// that V3 implies), and `variantType` for a Variant column at any depth.
///
/// # Errors
/// `DELTA_UNSUPPORTED_FEATURES_FOR_WRITE` (56038) naming each such feature outside
/// [`WRITABLE`].
pub fn check_create(
    table: &str,
    properties: &BTreeMap<String, String>,
    schema: &StructType,
) -> Result<(), KrishivError> {
    let mut features: Vec<&str> = Vec::new();
    let mut add = |f: &'static str| {
        if !features.contains(&f) {
            features.push(f);
        }
    };
    let enabled = |key: &str| properties.get(key).is_some_and(|v| v == "true");
    for key in properties.keys() {
        if let Some(name) = key.strip_prefix("delta.feature.")
            && let Some(known) = CREATE_FEATURES.iter().find(|f| **f == name)
        {
            add(known);
            // The kernel's create adds Variant support beside shredding.
            if name == "variantShredding" {
                add("variantType");
            }
        }
    }
    if enabled("delta.enableRowTracking") {
        add("rowTracking");
    }
    if enabled("delta.enableIcebergCompatV2") {
        add("icebergCompatV2");
    }
    if enabled("delta.enableIcebergCompatV3") {
        add("icebergCompatV3");
        add("rowTracking");
    }
    if has_variant(schema) {
        add("variantType");
    }
    let unsupported: Vec<&str> = features
        .into_iter()
        .filter(|f| !WRITABLE.contains(f))
        .collect();
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(unsupported_for_write(table, &unsupported))
    }
}

/// The features delta_kernel 0.30's create accepts as a `delta.feature.<name>` signal
/// (`ALLOWED_DELTA_FEATURES`); the kernel refuses any other name itself.
const CREATE_FEATURES: &[&str] = &[
    "domainMetadata",
    "columnMapping",
    "inCommitTimestamp",
    "vacuumProtocolCheck",
    "catalogManaged",
    "deletionVectors",
    "v2Checkpoint",
    "appendOnly",
    "changeDataFeed",
    "typeWidening",
    "rowTracking",
    "variantType",
    "variantShredding",
    "invariants",
    "materializePartitionColumns",
    "icebergCompatV2",
    "icebergCompatV3",
];

fn has_variant(schema: &StructType) -> bool {
    fn in_type(t: &DataType) -> bool {
        match t {
            DataType::Variant(_) => true,
            DataType::Struct(s) => has_variant(s),
            DataType::Array(a) => in_type(a.element_type()),
            DataType::Map(m) => in_type(m.key_type()) || in_type(m.value_type()),
            _ => false,
        }
    }
    schema.fields().any(|f| in_type(f.data_type()))
}

/// The writer protocol versions Krishiv writes (`<supportedWriters>` of a write refusal).
pub const WRITABLE_WRITERS: &str = "[1, 2, 7]";

fn unsupported_for_write(table: &str, features: &[&str]) -> KrishivError {
    DeltaUnsupportedFeaturesForWrite {
        table_name_or_path: table.to_owned(),
        delta_version: delta_version(),
        unsupported: features.join(", "),
    }
    .into()
}

#[cfg(test)]
mod tests;
