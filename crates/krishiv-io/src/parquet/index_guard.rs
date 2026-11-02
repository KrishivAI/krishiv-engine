// SPDX-License-Identifier: Apache-2.0
//! The guard of the page index and the stored Bloom filters (P2-T080b,
//! `design/P2-T080b-parquet-indexes.md` section 4). Nothing from a file's
//! index or Bloom filter reaches arrow-rs's decoders, or a pruning stage,
//! before it passed here.
//!
//! A `ColumnIndex`, an `OffsetIndex` and a `BloomFilterHeader` are Thrift
//! structs read from the file, so they get the footer's treatment
//! ([`thrift_guard::check`](super::thrift_guard::check)) on their own
//! slice. After decoding, the structure is checked against the column
//! chunk it describes. Any failure is a [`Rejected`]: the caller reads the
//! unit as if the file had no index (the design's question 2), because an
//! index can only prune and a bad one must cost speed, never results.

use std::ops::Range;

use parquet::basic::Type as PhysicalType;
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::page_index::index_reader::{decode_column_index, decode_offset_index};
use parquet::file::page_index::offset_index::{OffsetIndexMetaData, PageLocation};

use super::thrift_guard;
use crate::error::FormatError;

/// The largest Bloom bitset read from a file, as arrow-rs's
/// `BITSET_MAX_LENGTH`.
pub(super) const BLOOM_MAX_BYTES: u64 = 128 * 1024 * 1024;
/// The Bloom bitset is whole 32-byte blocks.
const BLOOM_BLOCK_BYTES: u64 = 32;

/// Why an index or Bloom filter is not used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Rejected(pub(super) String);

impl Rejected {
    fn new(why: impl Into<String>) -> Self {
        Self(why.into())
    }
}

impl From<&'static str> for Rejected {
    fn from(why: &'static str) -> Self {
        Self::new(why)
    }
}

type R<T> = Result<T, Rejected>;

/// The byte ranges a column chunk declares for its index and Bloom
/// filter, each checked against the file and the chunk's data.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Declared {
    /// The `ColumnIndex`, when the chunk declares one.
    pub(super) column_index: Option<Range<u64>>,
    /// The `OffsetIndex`, when the chunk declares one.
    pub(super) offset_index: Option<Range<u64>>,
    /// The Bloom filter (header and bitset), when the chunk declares one.
    pub(super) bloom: Option<Range<u64>>,
}

/// The ranges `col` declares, for a file of `file_size` bytes whose chunk
/// data (dictionary and data pages) occupies `chunk`.
///
/// A range must be positive, no longer than `i32::MAX`, inside the file and
/// disjoint from the chunk's data. Declaring only one of a column index's
/// or an offset index's offset and length is a rejection; declaring none is
/// not (the chunk simply has no index).
///
/// # Errors
/// [`Rejected`] naming the first range that fails.
pub(super) fn declared(
    col: &ColumnChunkMetaData,
    file_size: u64,
    chunk: &Range<u64>,
) -> R<Declared> {
    Ok(Declared {
        column_index: range(
            "column index",
            col.column_index_offset(),
            col.column_index_length(),
            file_size,
            chunk,
        )?,
        offset_index: range(
            "offset index",
            col.offset_index_offset(),
            col.offset_index_length(),
            file_size,
            chunk,
        )?,
        bloom: range(
            "Bloom filter",
            col.bloom_filter_offset(),
            col.bloom_filter_length(),
            file_size,
            chunk,
        )?,
    })
}

fn range(
    what: &str,
    offset: Option<i64>,
    length: Option<i32>,
    file_size: u64,
    chunk: &Range<u64>,
) -> R<Option<Range<u64>>> {
    let (offset, length) = match (offset, length) {
        (None, None) => return Ok(None),
        // A Bloom filter written before the length existed declares an
        // offset only: its length is unknown, so it is not used.
        (Some(_), None) if what == "Bloom filter" => return Ok(None),
        (Some(o), Some(l)) => (o, l),
        _ => return Err(Rejected::new(format!("the {what} declares half a range"))),
    };
    let start = u64::try_from(offset)
        .map_err(|_| Rejected::new(format!("the {what} offset is negative")))?;
    let len = u64::try_from(length)
        .ok()
        .filter(|l| *l > 0)
        .ok_or_else(|| Rejected::new(format!("the {what} length is not positive")))?;
    let end = start
        .checked_add(len)
        .filter(|e| *e <= file_size)
        .ok_or_else(|| Rejected::new(format!("the {what} runs past the end of the file")))?;
    if start < chunk.end && end > chunk.start {
        return Err(Rejected::new(format!(
            "the {what} overlaps the column chunk's data"
        )));
    }
    Ok(Some(start..end))
}

/// The page locations of an `OffsetIndex` read from `bytes`, for a chunk
/// whose data is `chunk` and which holds `rows` rows.
///
/// The bytes are walked, decoded and checked: locations strictly
/// increasing, inside the chunk, with positive sizes; `first_row_index`
/// starting at 0, strictly increasing and below `rows`.
///
/// # Errors
/// [`Rejected`] for the first violation.
pub(super) fn offset_index(bytes: &[u8], chunk: &Range<u64>, rows: u64) -> R<OffsetIndexMetaData> {
    thrift_guard::check(bytes)?;
    let index = super::contain("reading an offset index", FormatError::Corrupt, || {
        decode_offset_index(bytes).map_err(|e| FormatError::Corrupt(e.to_string()))
    })
    .map_err(|e| Rejected::new(e.to_string()))?;
    check_locations(index.page_locations(), chunk, rows)?;
    Ok(index)
}

fn check_locations(pages: &[PageLocation], chunk: &Range<u64>, rows: u64) -> R<()> {
    if pages.is_empty() {
        return Err(Rejected::new("an offset index without pages"));
    }
    let mut previous_end = chunk.start;
    let mut previous_row: Option<u64> = None;
    for p in pages {
        let start =
            u64::try_from(p.offset).map_err(|_| Rejected::new("a page offset is negative"))?;
        let size = u64::try_from(p.compressed_page_size)
            .ok()
            .filter(|s| *s > 0)
            .ok_or_else(|| Rejected::new("a page size is not positive"))?;
        let end = start
            .checked_add(size)
            .filter(|e| *e <= chunk.end)
            .ok_or_else(|| Rejected::new("a page runs past the end of its column chunk"))?;
        if start < previous_end {
            return Err(Rejected::new("page locations are not increasing"));
        }
        previous_end = end;
        let row = u64::try_from(p.first_row_index)
            .map_err(|_| Rejected::new("a page's first row is negative"))?;
        match previous_row {
            None if row != 0 => {
                return Err(Rejected::new("the first page does not start at row 0"));
            }
            Some(before) if row <= before => {
                return Err(Rejected::new("first row indexes are not increasing"));
            }
            _ => {}
        }
        if row >= rows {
            return Err(Rejected::new("a page starts past the last row"));
        }
        previous_row = Some(row);
    }
    Ok(())
}

/// The `ColumnIndex` read from `bytes` for a column of physical type
/// `ty`, which must describe `pages` pages (the offset index's count).
///
/// # Errors
/// [`Rejected`] for a walk, decode or page-count failure.
pub(super) fn column_index(bytes: &[u8], ty: PhysicalType, pages: usize) -> R<ColumnIndexMetaData> {
    thrift_guard::check(bytes)?;
    let index = super::contain("reading a column index", FormatError::Corrupt, || {
        decode_column_index(bytes, ty).map_err(|e| FormatError::Corrupt(e.to_string()))
    })
    .map_err(|e| Rejected::new(e.to_string()))?;
    if usize::try_from(index.num_pages()).ok() != Some(pages) {
        return Err(Rejected::new(
            "the column index and the offset index count different pages",
        ));
    }
    Ok(index)
}

/// The bitset size a Bloom filter header declares, and the header's own
/// length, from the start of the filter's `declared` range.
///
/// The header is walked on its own, then its `numBytes` (field 1, the
/// first field of the struct) is checked: whole 32-byte blocks, at most
/// [`BLOOM_MAX_BYTES`], and header plus bitset inside the declared range.
///
/// # Errors
/// [`Rejected`] for a malformed header or an impossible size.
pub(super) fn bloom_header(bytes: &[u8], declared: &Range<u64>) -> R<(u64, usize)> {
    let (num_bytes, header_len) = thrift_guard::bloom_header(bytes)?;
    let num_bytes = u64::from(num_bytes);
    if num_bytes == 0 || num_bytes % BLOOM_BLOCK_BYTES != 0 {
        return Err(Rejected::new(
            "the Bloom bitset is not whole 32-byte blocks",
        ));
    }
    if num_bytes > BLOOM_MAX_BYTES {
        return Err(Rejected::new("the Bloom bitset is larger than 128 MiB"));
    }
    let total = u64::try_from(header_len)
        .ok()
        .and_then(|h| h.checked_add(num_bytes))
        .ok_or_else(|| Rejected::new("the Bloom filter size overflows"))?;
    if total > declared.end - declared.start {
        return Err(Rejected::new(
            "the Bloom header and bitset do not fit the declared length",
        ));
    }
    Ok((num_bytes, header_len))
}

/// Checks the page headers of the pages in `pages` that lie wholly inside
/// the pushed bytes `bytes`, which start at file offset `start`: with an
/// offset index attached, a pushed range may hold some pages of a chunk
/// and not all (design section 4.4).
///
/// # Errors
/// [`Rejected`] for a header that fails the page-header walk.
pub(super) fn sparse_pages(bytes: &[u8], start: u64, pages: &[PageLocation]) -> R<()> {
    let end = start + bytes.len() as u64;
    for p in pages {
        let (Ok(offset), Ok(size)) = (
            u64::try_from(p.offset),
            u64::try_from(p.compressed_page_size),
        ) else {
            continue;
        };
        let Some(page_end) = offset.checked_add(size) else {
            continue;
        };
        if offset < start || page_end > end {
            continue;
        }
        let lo = usize::try_from(offset - start).map_err(|_| Rejected::new("offset overflow"))?;
        let hi = usize::try_from(page_end - start).map_err(|_| Rejected::new("offset overflow"))?;
        if let Some(slice) = bytes.get(lo..hi) {
            thrift_guard::check_pages(slice)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
