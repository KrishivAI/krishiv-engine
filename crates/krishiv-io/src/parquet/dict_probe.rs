// SPDX-License-Identifier: Apache-2.0
//! Stage 2b of `design/P2-T080b-parquet-indexes.md`: pushed predicates
//! evaluated on the entries of a column chunk's dictionary.
//!
//! A chunk is eligible when every data page is dictionary encoded
//! ([`all_dictionary`]); then a conjunct on that column holds for a row
//! only if it holds for the row's dictionary entry, so it is evaluated once
//! per entry ([`passes`]). The result is a per-code pass mask. T080b ships
//! the eligibility test, the evaluator and the mask; whether to fetch the
//! dictionary page ahead of the data pages, the size cap and skipping a
//! unit whose mask is all-false belong to T085.
//!
//! The evaluator is the sound comparison the statistics stages use
//! ([`prune`]): an entry is a one-value "row group" whose minimum and
//! maximum are the entry itself. An entry the comparison cannot decide
//! (a NaN, a collation other than binary) passes, so the mask only ever
//! drops rows the predicate rejects.

use parquet::basic::Encoding;
use parquet::file::metadata::ColumnChunkMetaData;

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

/// For each dictionary entry, whether every conjunct in `conjuncts` (all on
/// `column`) may hold for it: `false` means no row with this entry can
/// match.
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
