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
//! `UNSUPPORTED_FEATURE.TABLE_OPERATION`.

use std::collections::HashMap;

use delta_kernel::{KernelError, Snapshot};
use krishiv_common::error::{
    DeltaUnsupportedFeaturesForRead, KrishivError, UnsupportedFeatureTableOperation,
};

use crate::error::{delta_version, from_kernel};

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
    "reading a catalog-managed Delta table (its catalog's ratified commits are not read in P3)";

/// The operation named when a table with coordinated commits is refused.
pub const COORDINATED_COMMITS_OPERATION: &str = "reading a Delta table with coordinated commits \
                                                 (its unbackfilled commits are not read in P3)";

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
/// [`from_kernel`](crate::error::from_kernel).
#[must_use]
pub fn snapshot_error(error: &KernelError, table: &str) -> KrishivError {
    match error.without_backtrace() {
        KernelError::MaxCatalogVersion(_) => table_operation(table, CATALOG_MANAGED_OPERATION),
        KernelError::Unsupported(what) => match kernel_unsupported_feature(what) {
            Some(feature) => unsupported_for_read(table, &[feature]),
            None => from_kernel(error, table),
        },
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

#[cfg(test)]
mod tests;
