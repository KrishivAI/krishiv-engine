// SPDX-License-Identifier: Apache-2.0
//! Puffin statistics files, read side (P3-T024, PR-54a): the NDV of a column from the table
//! metadata's `statistics` entry for **the snapshot read** (the `ndv` property of its
//! `apache-datasketches-theta-v1` blob; no Puffin I/O), and the blob itself as a compact Theta sketch
//! for the unions of a later write (PR-54b).
//!
//! As Java's `SparkScan` does, only the statistics file of the exact snapshot counts: another
//! snapshot's file is ignored, and so is a blob of another type, of several fields or without a
//! parsable `ndv`.

use std::collections::HashMap;

use datasketches::theta::CompactThetaSketch;
use iceberg::io::FileIO;
use iceberg::puffin::{APACHE_DATASKETCHES_THETA_V1, PuffinReader};
use iceberg::spec::{StatisticsFile, TableMetadata};
use krishiv_common::error::{InternalError, KrishivError};

use super::storage::from_iceberg;

/// The blob property that carries the sketch's estimate (Java's `NDVSketchUtil`: the estimate cast to
/// `long`, as text).
const NDV_PROPERTY: &str = "ndv";

/// The distinct count by field id of every single-field Theta blob of the statistics file that
/// `metadata` records for `snapshot_id`; empty when it has none.
#[must_use]
pub(crate) fn ndv_by_field(metadata: &TableMetadata, snapshot_id: i64) -> HashMap<i32, u64> {
    metadata
        .statistics_for_snapshot(snapshot_id)
        .map(|f| file_ndv(f, snapshot_id))
        .unwrap_or_default()
}

/// [`ndv_by_field`] of one statistics file entry.
fn file_ndv(file: &StatisticsFile, snapshot_id: i64) -> HashMap<i32, u64> {
    if file.snapshot_id != snapshot_id {
        return HashMap::new();
    }
    file.blob_metadata
        .iter()
        .filter(|b| b.r#type == APACHE_DATASKETCHES_THETA_V1 && b.snapshot_id == snapshot_id)
        .filter_map(|b| {
            let [field] = b.fields[..] else { return None };
            let ndv = b.properties.get(NDV_PROPERTY)?.parse::<u64>().ok()?;
            Some((field, ndv))
        })
        .collect()
}

/// The Theta sketch of `field` in `file`, read from the Puffin file (`None` when the file has no
/// single-field Theta blob of that field for its snapshot).
///
/// # Errors
/// The storage's error reading the file, or `INTERNAL_ERROR` for a blob that is not a Theta sketch
/// of the default seed.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "read by PR-54b's unions with the parent's sketches"
    )
)]
pub(crate) async fn read_sketch(
    io: &FileIO,
    file: &StatisticsFile,
    field: i32,
) -> Result<Option<CompactThetaSketch>, KrishivError> {
    let reader = PuffinReader::new(io.new_input(&file.statistics_path).map_err(from_iceberg)?)
        .await
        .map_err(from_iceberg)?;
    let footer = reader.file_metadata().await.map_err(from_iceberg)?;
    let Some(blob) = footer.blobs().iter().find(|b| {
        b.blob_type() == APACHE_DATASKETCHES_THETA_V1
            && b.fields() == [field]
            && b.snapshot_id() == file.snapshot_id
    }) else {
        return Ok(None);
    };
    let blob = reader.blob(blob).await.map_err(from_iceberg)?;
    CompactThetaSketch::deserialize(blob.data())
        .map(Some)
        .map_err(|e| {
            InternalError {
                message: format!(
                    "the Theta sketch of field {field} in {}: {e}",
                    file.statistics_path
                ),
            }
            .into()
        })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use iceberg::spec::{BlobMetadata, StatisticsFile};

    use super::{APACHE_DATASKETCHES_THETA_V1, file_ndv};

    fn blob(ty: &str, snapshot: i64, fields: &[i32], ndv: Option<&str>) -> BlobMetadata {
        BlobMetadata {
            r#type: ty.to_owned(),
            snapshot_id: snapshot,
            sequence_number: 1,
            fields: fields.to_vec(),
            properties: ndv
                .map(|n| HashMap::from([("ndv".to_owned(), n.to_owned())]))
                .unwrap_or_default(),
        }
    }

    fn file(snapshot: i64, blobs: Vec<BlobMetadata>) -> StatisticsFile {
        StatisticsFile {
            snapshot_id: snapshot,
            statistics_path: "file:/stats/s.stats".to_owned(),
            file_size_in_bytes: 100,
            file_footer_size_in_bytes: 10,
            key_metadata: None,
            blob_metadata: blobs,
        }
    }

    /// Only single-field Theta blobs of the snapshot with a numeric `ndv` count.
    #[test]
    fn only_single_field_theta_blobs_of_the_snapshot_with_an_ndv() {
        let t = APACHE_DATASKETCHES_THETA_V1;
        let f = file(
            5,
            vec![
                blob(t, 5, &[1], Some("42")),
                blob(t, 5, &[2], Some("0")),
                blob(t, 5, &[3, 4], Some("9")),
                blob(t, 5, &[5], None),
                blob(t, 5, &[6], Some("many")),
                blob(t, 4, &[7], Some("8")),
                blob("deletion-vector-v1", 5, &[8], Some("8")),
            ],
        );
        assert_eq!(file_ndv(&f, 5), HashMap::from([(1, 42), (2, 0)]));
        // Another snapshot's file is ignored whole.
        assert!(file_ndv(&f, 6).is_empty());
    }
}
