// SPDX-License-Identifier: Apache-2.0
//! Stage 2b of `design/P2-T080b-parquet-indexes.md`: pushed predicates
//! evaluated on the entries of a column chunk's dictionary.
//!
//! A chunk is eligible when every data page is dictionary encoded
//! ([`all_dictionary`]); then a conjunct on that column holds for a row
//! only if it holds for the row's dictionary entry, so it is evaluated once
//! per entry ([`passes`]). The result is a per-code pass mask. T080b ships
//! the eligibility test, the evaluator and the mask.
//!
//! T085 (P2b row 33) prunes with it: for an equality or `IN` conjunct that
//! statistics left undecided, the page stage fetches the dictionary page
//! ([`dictionary_range`]) ahead of the data pages, decodes it
//! ([`read_dictionary`]) and skips the row group when no entry is one of
//! the literals ([`any_entry_in`]). The literals are compared as the plain
//! bytes the file stores (`bloom_probe::plain_values`), so the test is
//! exact for the types the Bloom stage takes; no entry is converted to a
//! `ScalarValue`, which keeps a 300K-entry dictionary cheap (ClickBench
//! q20). The cap is on bytes ([`MAX_DICTIONARY_BYTES`]), not on entries:
//! the owner's 2026-10-24 rule is that q20's 309K-entry dictionaries must
//! still prune.
//!
//! The evaluator is the sound comparison the statistics stages use
//! ([`prune`]): an entry is a one-value "row group" whose minimum and
//! maximum are the entry itself. An entry the comparison cannot decide
//! (a NaN, a collation other than binary) passes, so the mask only ever
//! drops rows the predicate rejects.

use std::ops::Range;
use std::sync::Arc;

use bytes::{Buf, Bytes};
use parquet::basic::{Encoding, Type as PhysicalType};
use parquet::column::page::{Page, PageReader};
use parquet::file::metadata::ColumnChunkMetaData;
use parquet::file::page_index::offset_index::PageLocation;
use parquet::file::reader::{ChunkReader, Length};
use parquet::file::serialized_reader::SerializedPageReader;

use super::index_guard::Rejected;
use super::thrift_guard;

use crate::request::{ColumnPath, PushdownExpr, ScalarValue};
use crate::stats::{ColumnStatistics, FileStatistics, Precision, TriState, prune};

/// Whether every data page of the chunk is dictionary encoded: it declares
/// a dictionary page and its page encodings are only `PLAIN_DICTIONARY` or
/// only `RLE_DICTIONARY`. A chunk without encoding statistics is not
/// eligible (a writer that fell back to plain pages part way cannot be told
/// from one that did not).
pub(super) fn all_dictionary(col: &ColumnChunkMetaData) -> bool {
    col.dictionary_page_offset().is_some()
        && col.page_encoding_stats_mask().is_some_and(|mask| {
            mask.is_only(Encoding::PLAIN_DICTIONARY) || mask.is_only(Encoding::RLE_DICTIONARY)
        })
}

/// The largest dictionary page (compressed, header included) the stage
/// fetches: the bytes are held until the stage decides, under the decoder's
/// reservation, so the cap bounds that memory per column chunk.
pub(super) const MAX_DICTIONARY_BYTES: u64 = 16 << 20;

/// The file range of the chunk's dictionary page, header included: from
/// the chunk's start to its first data page. `None` when the chunk does not
/// start with a dictionary page, the range leaves the chunk or the file, or
/// it is larger than [`MAX_DICTIONARY_BYTES`]. There is no cap relative to
/// the chunk: a high-cardinality column's dictionary is most of its chunk
/// (q20's `UserID`), and the gain is the decode of the rows, not bytes.
pub(super) fn dictionary_range(col: &ColumnChunkMetaData, file_size: u64) -> Option<Range<u64>> {
    let (start, len) = col.byte_range();
    let end = start.checked_add(len)?;
    let dict = u64::try_from(col.dictionary_page_offset()?).ok()?;
    let data = u64::try_from(col.data_page_offset()).ok()?;
    let bytes = data.checked_sub(dict)?;
    (dict == start && data <= end && end <= file_size && bytes > 0 && bytes <= MAX_DICTIONARY_BYTES)
        .then_some(dict..data)
}

/// The fetched bytes of one file range, served to arrow-rs's page reader at
/// their file offsets.
struct Window {
    base: u64,
    bytes: Bytes,
}

impl Length for Window {
    fn len(&self) -> u64 {
        self.base.saturating_add(self.bytes.len() as u64)
    }
}

impl ChunkReader for Window {
    type T = bytes::buf::Reader<Bytes>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        Ok(self.get_bytes(start, 0)?.reader())
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        let outside = || parquet::errors::ParquetError::EOF("outside the dictionary page".into());
        let from = start
            .checked_sub(self.base)
            .and_then(|o| usize::try_from(o).ok())
            .ok_or_else(outside)?;
        let to = if length == 0 {
            self.bytes.len()
        } else {
            from.checked_add(length).ok_or_else(outside)?
        };
        if from > to || to > self.bytes.len() {
            return Err(outside());
        }
        Ok(self.bytes.slice(from..to))
    }
}

/// The decoded dictionary page of `col` from the fetched `bytes` of
/// `range` ([`dictionary_range`]): its plain-encoded values and their
/// count. The page header is guarded before arrow-rs parses it, and the
/// page must be a `PLAIN`/`PLAIN_DICTIONARY` dictionary page.
pub(super) fn read_dictionary(
    col: &ColumnChunkMetaData,
    range: &Range<u64>,
    bytes: &[u8],
) -> Result<(Bytes, usize), Rejected> {
    thrift_guard::check_pages(bytes).map_err(Rejected::from)?;
    let window = Arc::new(Window {
        base: range.start,
        bytes: Bytes::copy_from_slice(bytes),
    });
    // The first data page's location makes arrow-rs read everything before
    // it as the dictionary page, and nothing past it.
    let first = PageLocation {
        offset: i64::try_from(range.end).map_err(|_| Rejected::from("a page offset"))?,
        compressed_page_size: 0,
        first_row_index: 0,
    };
    let rows = usize::try_from(col.num_values()).unwrap_or(0);
    let mut pages = SerializedPageReader::new(window, col, rows, Some(vec![first]))
        .map_err(|_| Rejected::from("the dictionary page reader"))?;
    match pages.get_next_page() {
        Ok(Some(Page::DictionaryPage {
            buf,
            num_values,
            encoding: Encoding::PLAIN | Encoding::PLAIN_DICTIONARY,
            ..
        })) => Ok((
            buf,
            usize::try_from(num_values).map_err(|_| Rejected::from("a count"))?,
        )),
        Ok(_) => Err(Rejected::from(
            "the chunk does not start with a plain dictionary page",
        )),
        Err(_) => Err(Rejected::from("the dictionary page does not decode")),
    }
}

/// Whether one of the `count` plain-encoded entries of `dictionary` equals
/// one of `literals` (plain-encoded the same way). The entries are walked
/// as the physical type lays them out; a page shorter than its count is
/// rejected, never read past.
pub(super) fn any_entry_in(
    dictionary: &[u8],
    count: usize,
    physical: PhysicalType,
    literals: &[Vec<u8>],
) -> Result<bool, Rejected> {
    let short = || Rejected::from("the dictionary page is short");
    match physical {
        PhysicalType::INT32 | PhysicalType::INT64 => {
            let width = if physical == PhysicalType::INT32 {
                4
            } else {
                8
            };
            let need = count.checked_mul(width).ok_or_else(short)?;
            let values = dictionary.get(..need).ok_or_else(short)?;
            let wanted: Vec<&[u8]> = literals
                .iter()
                .filter(|l| l.len() == width)
                .map(Vec::as_slice)
                .collect();
            Ok(values.chunks_exact(width).any(|v| wanted.contains(&v)))
        }
        PhysicalType::BYTE_ARRAY => {
            let mut rest = dictionary;
            for _ in 0..count {
                let (len, tail) = rest.split_first_chunk::<4>().ok_or_else(short)?;
                let len = usize::try_from(u32::from_le_bytes(*len)).map_err(|_| short())?;
                let (value, tail) = tail.split_at_checked(len).ok_or_else(short)?;
                if literals.iter().any(|l| l.as_slice() == value) {
                    return Ok(true);
                }
                rest = tail;
            }
            Ok(false)
        }
        _ => Err(Rejected::from(
            "a physical type without an exact plain encoding",
        )),
    }
}

/// For each dictionary entry, whether every conjunct in `conjuncts` (all on
/// `column`) may hold for it: `false` means no row with this entry can
/// match.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the general per-entry evaluator; T085 prunes on equality and IN only so far"
    )
)]
pub(super) fn passes(
    conjuncts: &[&PushdownExpr],
    column: &ColumnPath,
    entries: &[ScalarValue],
) -> Vec<bool> {
    entries
        .iter()
        .map(|entry| {
            let stats = entry_statistics(column, entry);
            conjuncts
                .iter()
                .all(|c| prune(c, &stats) != TriState::False)
        })
        .collect()
}

/// The statistics of a one-row unit holding `entry` in `column`.
#[cfg_attr(not(test), expect(dead_code, reason = "used by `passes`"))]
fn entry_statistics(column: &ColumnPath, entry: &ScalarValue) -> FileStatistics {
    let mut c = ColumnStatistics::new(column.clone());
    c.null_count = Precision::Exact(0);
    let nan = matches!(entry, ScalarValue::Float32(x) if x.is_nan())
        || matches!(entry, ScalarValue::Float64(x) if x.is_nan());
    if !nan {
        c.min = Precision::Exact(entry.clone());
        c.max = Precision::Exact(entry.clone());
        c.nan_count = Precision::Exact(0);
    }
    FileStatistics {
        rows: Precision::Exact(1),
        bytes: Precision::Absent,
        columns: vec![c],
    }
}

#[cfg(test)]
mod tests;
