// SPDX-License-Identifier: Apache-2.0
//! Reading rows by position (design P2-T182 §2.1, P2-T063 §3): [`take`] for
//! the rows of one file, [`RowFile`] for the file half of a row id.
//!
//! A take is an ordinary scan with a selection and the row index, so a
//! format implements nothing new beyond [`Capabilities::selection`],
//! [`Capabilities::row_index`] and [`Capabilities::random_access`]. Every
//! read checks the file's version as any scan does, so a file changed since
//! the row ids were made fails with `KRISHIV_FILE_CHANGED_DURING_READ`.
//!
//! [`Capabilities::selection`]: crate::Capabilities::selection
//! [`Capabilities::row_index`]: crate::Capabilities::row_index
//! [`Capabilities::random_access`]: crate::Capabilities::random_access

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_schema::SchemaRef;
use krishiv_memory::Reservation;
use object_store::path::Path;

use crate::error::{Result, internal};
use crate::format::FileFormat;
use crate::io::{ByteRange, FileRef, IoContext, StoreId, VersionToken};
use crate::options::FormatOptions;
use crate::request::{
    ColumnPath, FieldMapping, ROW_INDEX_COLUMN, RowFilterPolicy, RowSelection, ScanRequest,
};
use crate::scan::{ScanOptions, ScanStream};
use crate::scan_gate::ScanGate;

/// The rows of one file to read.
#[derive(Debug, Clone)]
pub struct TakeRequest {
    /// The file, with the version the row ids were made from.
    pub file: FileRef,
    /// The schema `projection` refers to (the table's).
    pub table_schema: SchemaRef,
    /// Columns to read.
    pub projection: Vec<ColumnPath>,
    /// Column resolution between table and file.
    pub mapping: FieldMapping,
    /// File-relative positions, sorted and without duplicates
    /// ([`Self::new`]).
    rows: Vec<u64>,
}

impl TakeRequest {
    /// Reads `rows` (in any order, with repeats) of `file`.
    #[must_use]
    pub fn new(
        file: FileRef,
        table_schema: SchemaRef,
        projection: Vec<ColumnPath>,
        mapping: FieldMapping,
        mut rows: Vec<u64>,
    ) -> Self {
        rows.sort_unstable();
        rows.dedup();
        Self {
            file,
            table_schema,
            projection,
            mapping,
            rows,
        }
    }
}

/// The batches of a [`take`], in position order, each ending in
/// [`ROW_INDEX_COLUMN`].
#[derive(Debug)]
pub struct TakeStream {
    /// `None`: nothing to read.
    stream: Option<ScanStream>,
    rows: Vec<u64>,
    next: usize,
}

impl TakeStream {
    /// The next batch.
    ///
    /// # Errors
    /// Any scan error; `INTERNAL_ERROR` if a row is missing or not asked
    /// for, never a short result.
    pub async fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(stream) = self.stream.as_mut() else {
            return Ok(None);
        };
        let Some(batch) = stream.next_batch().await? else {
            return match self.rows.get(self.next) {
                Some(row) => Err(internal(format!(
                    "take: row {row} of {} is missing",
                    stream.file().display_path()
                ))),
                None => Ok(None),
            };
        };
        let index = batch
            .column_by_name(ROW_INDEX_COLUMN)
            .ok_or_else(|| internal("take: the scan returned no row index"))?
            .as_primitive::<Int64Type>();
        for &position in index.values() {
            if self.rows.get(self.next).map(|&r| i64::try_from(r)) != Some(Ok(position)) {
                return Err(internal(format!(
                    "take: row {position} of {} was not the next one asked for",
                    stream.file().display_path()
                )));
            }
            self.next += 1;
        }
        Ok(Some(batch))
    }

    /// The same read, sharing the query's decode budget with its scans.
    #[must_use]
    pub fn with_gate(mut self, gate: Arc<ScanGate>) -> Self {
        self.stream = self.stream.map(|s| s.with_gate(gate));
        self
    }

    /// The scan's reader, once something was to be read.
    #[must_use]
    pub fn stream(&self) -> Option<&ScanStream> {
        self.stream.as_ref()
    }
}

/// Opens a read of `req`'s rows. An empty request reads nothing, not even
/// the footer. A skipped file would change which rows come back, so the
/// `ignore*Files` settings of `options` don't apply.
///
/// # Errors
/// `INTERNAL_ERROR` for a format that can't read by position, or any error
/// opening the scan.
pub async fn take(
    format: &dyn FileFormat,
    io: &IoContext,
    req: TakeRequest,
    format_options: &FormatOptions,
    mem: Reservation,
    options: ScanOptions,
) -> Result<TakeStream> {
    let TakeRequest {
        file,
        table_schema,
        projection,
        mapping,
        rows,
    } = req;
    if rows.is_empty() {
        return Ok(TakeStream {
            stream: None,
            rows,
            next: 0,
        });
    }
    let caps = format.capabilities(format_options);
    if !(caps.selection && caps.row_index && caps.random_access.is_some()) {
        return Err(internal(format!(
            "take: format {} can't read a row by position",
            format.name()
        )));
    }
    let mut scan = ScanRequest::new(table_schema, projection);
    scan.mapping = mapping;
    scan.row_filter = RowFilterPolicy::Off;
    scan.metadata_cols.row_index = true;
    scan.selection = Some(RowSelection::from_ranges(rows.iter().map(|&r| r..r + 1)));
    let options = ScanOptions {
        ignore_corrupt_files: false,
        ignore_missing_files: false,
        ..options
    };
    let split = ByteRange::whole(file.size);
    let stream =
        ScanStream::open(io, format, file, split, &scan, format_options, mem, options).await?;
    Ok(TakeStream {
        stream: Some(stream),
        rows,
        next: 0,
    })
}

/// The encoded [`FileRef`] a row id carries: bytes, so it crosses any
/// exchange, and it names a store rather than holding one, so each worker
/// resolves it through its own registry (design P2-T063 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowFile;

impl RowFile {
    /// The layout written by [`Self::encode`]; a reader refuses others.
    const VERSION: u8 = 1;

    /// `file` as bytes.
    #[must_use]
    pub fn encode(file: &FileRef) -> Vec<u8> {
        let mut out = vec![Self::VERSION];
        put_str(&mut out, file.store.key());
        put_str(&mut out, file.path.as_ref());
        out.extend_from_slice(&file.size.to_le_bytes());
        match &file.version {
            VersionToken::ETag(e) => {
                out.push(0);
                put_str(&mut out, e);
            }
            VersionToken::Modified(us) => {
                out.push(1);
                out.extend_from_slice(&us.to_le_bytes());
            }
            VersionToken::Immutable => out.push(2),
        }
        match file.modified_micros {
            Some(us) => {
                out.push(1);
                out.extend_from_slice(&us.to_le_bytes());
            }
            None => out.push(0),
        }
        out
    }

    /// The file in `bytes`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for bytes [`Self::encode`] did not write.
    pub fn decode(bytes: &[u8]) -> Result<FileRef> {
        let mut r = Reader(bytes);
        let version = r.u8()?;
        if version != Self::VERSION {
            return Err(internal(format!("row file: unknown version {version}")));
        }
        let store = StoreId::new(r.str()?);
        let path = Path::parse(r.str()?).map_err(|e| internal(format!("row file: {e}")))?;
        let size = r.u64()?;
        let version = match r.u8()? {
            0 => VersionToken::ETag(Arc::from(r.str()?)),
            1 => VersionToken::Modified(r.i64()?),
            2 => VersionToken::Immutable,
            t => return Err(internal(format!("row file: unknown version kind {t}"))),
        };
        let modified_micros = match r.u8()? {
            0 => None,
            _ => Some(r.i64()?),
        };
        if !r.0.is_empty() {
            return Err(internal("row file: trailing bytes"));
        }
        Ok(FileRef {
            store,
            path,
            size,
            version,
            modified_micros,
        })
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let Some((head, rest)) = self.0.split_at_checked(n) else {
            return Err(internal("row file: truncated"));
        };
        self.0 = rest;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.bytes(8)?;
        Ok(b.iter().rev().fold(0, |v, &x| (v << 8) | u64::from(x)))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.u64()?.to_le_bytes()))
    }

    fn str(&mut self) -> Result<&'a str> {
        let n = usize::try_from(self.u64()?).map_err(|_| internal("row file: truncated"))?;
        std::str::from_utf8(self.bytes(n)?).map_err(|e| internal(format!("row file: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(version: VersionToken, modified_micros: Option<i64>) -> FileRef {
        FileRef {
            store: StoreId::new("s3://bucket"),
            path: Path::from("dir/a b.parquet"),
            size: 1 << 40,
            version,
            modified_micros,
        }
    }

    #[test]
    fn row_files_round_trip_and_refuse_what_they_did_not_write() {
        for f in [
            file(VersionToken::ETag(Arc::from("\"abc\"")), Some(-5)),
            file(VersionToken::Modified(i64::MIN), None),
            file(VersionToken::Immutable, Some(i64::MAX)),
            // The registry key, not the text a message shows.
            FileRef {
                store: StoreId::new("s3://bucket").private_to("job"),
                ..file(VersionToken::Immutable, None)
            },
        ] {
            let bytes = RowFile::encode(&f);
            assert_eq!(RowFile::decode(&bytes).unwrap(), f);
            for cut in 0..bytes.len() {
                assert!(RowFile::decode(&bytes[..cut]).is_err(), "cut at {cut}");
            }
            let mut other = bytes.clone();
            other[0] = 2;
            assert!(RowFile::decode(&other).is_err());
            let mut long = bytes;
            long.push(0);
            assert!(RowFile::decode(&long).is_err());
        }
    }
}
