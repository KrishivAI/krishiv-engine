// SPDX-License-Identifier: Apache-2.0
//! Stage 3 of `design/P2-T080b-parquet-indexes.md`: the page index as a
//! row selection.
//!
//! A scan whose pushed filter has single-column conjuncts, over a file
//! whose footer declares a column index and an offset index for those
//! columns, asks for the indexes of the row groups that statistics could
//! not decide, as one extra read of the decoder (`Step::NeedData`). Each
//! slice goes through the [`index_guard`](super::index_guard), is turned
//! into one `FileStatistics` per page, and the pushed conjuncts are
//! evaluated per page with the same sound [`prune`] the row-group stage
//! uses. Pages that cannot match become skipped row ranges, intersected
//! with the unit's selection. A unit whose index fails any check is read
//! as if the file had none.
//!
//! Only primitive, non-nested, fixed-width columns take part (integers,
//! booleans, floats, dates, timestamps, decimals stored as integers):
//! byte-array bounds in a column index are inexact (design question 3), so
//! `prune` would ignore them anyway.

use std::collections::BTreeMap;
use std::ops::Range;

use bytes::Bytes;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::basic::{ConvertedType, LogicalType, Type as PhysicalType};
use parquet::file::page_index::column_index::ColumnIndexMetaData;
use parquet::file::statistics::Statistics;
use parquet::schema::types::ColumnDescriptor;

use super::index_guard::{self, Rejected};
use super::meta::{ParquetMetadata, column_statistics, leaf_type, micros_per_unit};
use crate::error::FormatError;
use crate::request::{ColumnPath, PushdownExpr};
use crate::stats::{FileStatistics, Precision, TriState, prune};

/// A column of a unit whose indexes the stage reads.
#[derive(Debug, Clone)]
struct IndexedColumn {
    path: ColumnPath,
    leaf: usize,
    physical: PhysicalType,
    /// The column chunk's data (dictionary and data pages).
    chunk: Range<u64>,
    column_index: Range<u64>,
    offset_index: Range<u64>,
    /// The conjuncts (indexes into [`PageStage::conjuncts`]) on this column.
    conjuncts: Vec<usize>,
}

/// A row group the stage looks at.
#[derive(Debug, Clone)]
struct IndexedUnit {
    /// The row group's index in the file.
    index: usize,
    rows: u64,
    columns: Vec<IndexedColumn>,
}

/// The stage's state for one decoder.
#[derive(Debug)]
pub(super) struct PageStage {
    conjuncts: Vec<PushdownExpr>,
    units: Vec<IndexedUnit>,
    /// Every index range to read, in file order.
    ranges: Vec<Range<u64>>,
    held: BTreeMap<(u64, u64), Bytes>,
    requested: bool,
    done: bool,
    /// Units whose index failed a check.
    pub(super) rejected: u64,
    /// Pages the stage removed.
    pub(super) pages_pruned: u64,
}

/// What the stage decided for a unit: the rows to keep.
pub(super) struct Decision {
    /// The row group's index in the file.
    pub(super) index: usize,
    /// The rows that may match (all rows when nothing was pruned).
    pub(super) keep: RowSelection,
}

/// The top-level conjuncts of `predicate` (negation pushed to the
/// leaves).
fn conjuncts_of(predicate: &PushdownExpr) -> Vec<PushdownExpr> {
    match predicate.to_nnf() {
        PushdownExpr::And(xs) => xs,
        other => vec![other],
    }
}

/// The one top-level column `e` is about, when every leaf of it is a
/// comparison the statistics can answer on that column.
fn single_column(e: &PushdownExpr) -> Option<&ColumnPath> {
    match e {
        PushdownExpr::Cmp { column, .. }
        | PushdownExpr::In { column, .. }
        | PushdownExpr::IsNull(column)
        | PushdownExpr::IsNotNull(column)
        | PushdownExpr::StartsWith { column, .. } => (column.0.len() == 1).then_some(column),
        PushdownExpr::And(xs) | PushdownExpr::Or(xs) => {
            let mut found: Option<&ColumnPath> = None;
            for x in xs {
                let c = single_column(x)?;
                if found.is_some_and(|f| f != c) {
                    return None;
                }
                found = Some(c);
            }
            found
        }
        PushdownExpr::Not(x) => single_column(x),
        PushdownExpr::Contains { .. }
        | PushdownExpr::Constant(_)
        | PushdownExpr::Membership { .. } => None,
    }
}

/// Whether the column `d`, read as `dt`, takes part (see the module docs):
/// fixed-width values whose page bounds `column_statistics` keeps exact.
fn takes_part(d: &ColumnDescriptor, dt: &arrow_schema::DataType) -> bool {
    use arrow_schema::DataType as T;
    let unsigned = matches!(d.logical_type_ref(), Some(LogicalType::Integer(i)) if !i.is_signed)
        || matches!(
            d.converted_type(),
            ConvertedType::UINT_8
                | ConvertedType::UINT_16
                | ConvertedType::UINT_32
                | ConvertedType::UINT_64
        );
    let physical = matches!(
        d.physical_type(),
        PhysicalType::BOOLEAN
            | PhysicalType::INT32
            | PhysicalType::INT64
            | PhysicalType::FLOAT
            | PhysicalType::DOUBLE
    );
    let typed = match dt {
        T::Boolean
        | T::Int8
        | T::Int16
        | T::Int32
        | T::Int64
        | T::Float32
        | T::Float64
        | T::Date32
        | T::Decimal64(..)
        | T::Decimal128(..) => true,
        T::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => micros_per_unit(d).is_some(),
        _ => false,
    };
    physical && typed && !unsigned
}

impl PageStage {
    /// The stage for the row groups `units` of `meta` under `predicate`, or
    /// `None` when no unit takes part: no usable conjunct, no declared
    /// index, statistics already decided every unit, or the index would
    /// cost more than an eighth of what the unit's chunks hold.
    pub(super) fn plan(
        meta: &ParquetMetadata,
        units: &[usize],
        predicate: &PushdownExpr,
    ) -> Option<Self> {
        let conjuncts: Vec<PushdownExpr> = conjuncts_of(predicate)
            .into_iter()
            .filter(|c| single_column(c).is_some())
            .collect();
        if conjuncts.is_empty() {
            return None;
        }
        let descr = meta.reader.parquet_schema();
        let footer = meta.reader.metadata();
        let mut planned = Vec::new();
        let mut ranges = Vec::new();
        for &index in units {
            let rg = footer.row_group(index);
            let row_group = meta.row_groups.get(index)?;
            let mut columns: Vec<IndexedColumn> = Vec::new();
            for (n, c) in conjuncts.iter().enumerate() {
                // Statistics decided this conjunct for the unit already.
                if prune(c, &row_group.statistics) != TriState::Unknown {
                    continue;
                }
                let path = single_column(c)?;
                if let Some(existing) = columns.iter_mut().find(|x| &x.path == path) {
                    existing.conjuncts.push(n);
                    continue;
                }
                let Some(leaf) = (0..descr.num_columns()).find(|&l| {
                    let d = descr.column(l);
                    d.max_rep_level() == 0 && d.path().parts() == path.0.as_slice()
                }) else {
                    continue;
                };
                let Some(dt) = leaf_type(&meta.schema, &path.0) else {
                    continue;
                };
                if !takes_part(&descr.column(leaf), dt) {
                    continue;
                }
                let physical = descr.column(leaf).physical_type();
                let col = rg.column(leaf);
                let (start, len) = col.byte_range();
                let chunk = start..start.checked_add(len)?;
                let Ok(declared) = index_guard::declared(col, meta.size, &chunk) else {
                    continue;
                };
                let (Some(column_index), Some(offset_index)) =
                    (declared.column_index, declared.offset_index)
                else {
                    continue;
                };
                columns.push(IndexedColumn {
                    path: path.clone(),
                    leaf,
                    physical,
                    chunk,
                    column_index,
                    offset_index,
                    conjuncts: vec![n],
                });
            }
            if columns.is_empty() {
                continue;
            }
            // The read must be small next to what it can save: an eighth of
            // the filter columns' own chunks.
            let index_bytes: u64 = columns
                .iter()
                .map(|c| {
                    (c.column_index.end - c.column_index.start)
                        + (c.offset_index.end - c.offset_index.start)
                })
                .sum();
            let chunk_bytes: u64 = columns
                .iter()
                .map(|c| rg.column(c.leaf).byte_range().1)
                .sum();
            if index_bytes.saturating_mul(8) > chunk_bytes {
                continue;
            }
            for c in &columns {
                ranges.push(c.column_index.clone());
                ranges.push(c.offset_index.clone());
            }
            planned.push(IndexedUnit {
                index,
                rows: row_group.rows,
                columns,
            });
        }
        if planned.is_empty() {
            return None;
        }
        ranges.sort_by_key(|r| (r.start, r.end));
        ranges.dedup();
        Some(Self {
            conjuncts,
            units: planned,
            ranges,
            held: BTreeMap::new(),
            requested: false,
            done: false,
            rejected: 0,
            pages_pruned: 0,
        })
    }

    /// The ranges to read, once: the decoder's first step.
    pub(super) fn request(&mut self) -> Option<Vec<Range<u64>>> {
        if self.requested || self.done {
            return None;
        }
        self.requested = true;
        Some(self.ranges.clone())
    }

    /// Takes the bytes of an index range; `false` when `range` is not one
    /// the stage asked for (the caller hands it to the page decoder).
    pub(super) fn accept(&mut self, range: &Range<u64>, bytes: &Bytes) -> bool {
        if self.done
            || self
                .ranges
                .binary_search_by_key(&(range.start, range.end), |r| (r.start, r.end))
                .is_err()
        {
            return false;
        }
        self.held.insert((range.start, range.end), bytes.clone());
        true
    }

    /// Whether every asked range arrived.
    pub(super) fn complete(&self) -> bool {
        self.requested && !self.done && self.held.len() == self.ranges.len()
    }

    /// The bytes the stage holds, for the decoder's memory accounting.
    pub(super) fn held_bytes(&self) -> usize {
        self.held.values().map(Bytes::len).sum()
    }

    /// Decodes what arrived and decides each unit; releases the bytes. A
    /// unit whose index fails a check, or makes arrow-rs panic, is left as
    /// it was; one warning per file says so.
    pub(super) fn finish(&mut self, meta: &ParquetMetadata) -> Vec<Decision> {
        self.done = true;
        let held = std::mem::take(&mut self.held);
        let mut out = Vec::new();
        let (mut pruned, mut rejected) = (0, 0);
        let mut first_reason: Option<String> = None;
        for unit in &self.units {
            let decided = super::contain("applying a page index", FormatError::Corrupt, || {
                self.decide(unit, meta, &held)
                    .map_err(|why| FormatError::Corrupt(why.0))
            });
            match decided {
                Ok((decision, pages)) => {
                    pruned += pages;
                    out.extend(decision);
                }
                Err(why) => {
                    rejected += 1;
                    first_reason.get_or_insert_with(|| why.to_string());
                }
            }
        }
        if let Some(reason) = first_reason {
            tracing::warn!(units = rejected, %reason, "a Parquet page index is ignored");
        }
        self.pages_pruned += pruned;
        self.rejected += rejected;
        out
    }

    fn slice<'a>(
        held: &'a BTreeMap<(u64, u64), Bytes>,
        r: &Range<u64>,
    ) -> Result<&'a [u8], Rejected> {
        let bytes = held
            .get(&(r.start, r.end))
            .ok_or_else(|| Rejected::from("an index range did not arrive"))?;
        let len =
            usize::try_from(r.end - r.start).map_err(|_| Rejected::from("a range overflows"))?;
        bytes
            .get(..len)
            .ok_or_else(|| Rejected::from("an index range arrived short"))
    }

    fn decide(
        &self,
        unit: &IndexedUnit,
        meta: &ParquetMetadata,
        held: &BTreeMap<(u64, u64), Bytes>,
    ) -> Result<(Option<Decision>, u64), Rejected> {
        let rows = usize::try_from(unit.rows).map_err(|_| Rejected::from("rows"))?;
        // The row ranges to skip, from the pages alone: time and memory
        // follow the number of pages, never the row count a footer declares.
        let mut skipped: Vec<Range<usize>> = Vec::new();
        let mut pruned_pages = 0_u64;
        for col in &unit.columns {
            let offsets = index_guard::offset_index(
                Self::slice(held, &col.offset_index)?,
                &col.chunk,
                unit.rows,
            )?;
            let locations = offsets.page_locations();
            let columns = index_guard::column_index(
                Self::slice(held, &col.column_index)?,
                col.physical,
                locations.len(),
            )?;
            let descr = meta.reader.parquet_schema().column(col.leaf);
            let Some(dt) = leaf_type(&meta.schema, &col.path.0) else {
                continue;
            };
            for (page, location) in locations.iter().enumerate() {
                let first = usize::try_from(location.first_row_index)
                    .map_err(|_| Rejected::from("a first row"))?;
                let end = locations
                    .get(page + 1)
                    .map_or(Ok(rows), |next| usize::try_from(next.first_row_index))
                    .map_err(|_| Rejected::from("a first row"))?;
                if end <= first || end > rows {
                    return Err(Rejected::from("a page has no rows"));
                }
                let page_rows = (end - first) as u64;
                let stats = page_statistics(&columns, page, page_rows);
                let file = FileStatistics {
                    rows: Precision::Exact(page_rows),
                    bytes: Precision::Absent,
                    columns: vec![column_statistics(
                        col.path.clone(),
                        &descr,
                        dt,
                        stats.as_ref(),
                        meta.legacy_writer,
                    )],
                };
                if col
                    .conjuncts
                    .iter()
                    .any(|&n| prune(&self.conjuncts[n], &file) == TriState::False)
                {
                    pruned_pages += 1;
                    skipped.push(first..end);
                }
            }
        }
        if skipped.is_empty() {
            return Ok((None, 0));
        }
        skipped.sort_by_key(|r| (r.start, r.end));
        let mut selectors = Vec::new();
        let mut at = 0_usize;
        for r in skipped {
            let start = r.start.max(at);
            if r.end <= start {
                continue;
            }
            if start > at {
                selectors.push(RowSelector::select(start - at));
            }
            selectors.push(RowSelector::skip(r.end - start));
            at = r.end;
        }
        if at < rows {
            selectors.push(RowSelector::select(rows - at));
        }
        Ok((
            Some(Decision {
                index: unit.index,
                keep: selectors.into_iter().collect(),
            }),
            pruned_pages,
        ))
    }
}

/// The statistics of page `page` of a column index, as a footer would
/// carry them for a chunk of `rows` rows. `None` for a page whose bounds
/// cannot be read (the page then stays `Unknown`).
fn page_statistics(index: &ColumnIndexMetaData, page: usize, rows: u64) -> Option<Statistics> {
    let nulls = if index.is_null_page(page) {
        Some(rows)
    } else {
        index.null_count(page).and_then(|n| u64::try_from(n).ok())
    };
    macro_rules! typed {
        ($idx:expr, $make:ident) => {{
            if $idx.is_null_page(page) {
                Statistics::$make(None, None, None, nulls, false)
            } else {
                Statistics::$make(
                    $idx.min_value(page).copied(),
                    $idx.max_value(page).copied(),
                    None,
                    nulls,
                    false,
                )
            }
        }};
    }
    let nan = index.nan_count(page).and_then(|n| u64::try_from(n).ok());
    Some(match index {
        ColumnIndexMetaData::BOOLEAN(i) => typed!(i, boolean),
        ColumnIndexMetaData::INT32(i) => typed!(i, int32),
        ColumnIndexMetaData::INT64(i) => typed!(i, int64),
        ColumnIndexMetaData::FLOAT(i) => match typed!(i, float) {
            Statistics::Float(v) => Statistics::Float(v.with_nan_count(nan)),
            other => other,
        },
        ColumnIndexMetaData::DOUBLE(i) => match typed!(i, double) {
            Statistics::Double(v) => Statistics::Double(v.with_nan_count(nan)),
            other => other,
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
