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
use parquet::file::page_index::offset_index::OffsetIndexMetaData;
use parquet::file::statistics::Statistics;
use parquet::schema::types::ColumnDescriptor;

use super::bloom_probe;
use super::dict_probe;
use super::index_guard::{self, Rejected};
use super::meta::{ParquetMetadata, column_statistics, leaf_type, micros_per_unit};
use crate::error::FormatError;
use crate::request::{ColumnPath, PushdownExpr};
use crate::runtime_filter::RuntimeBloom;
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

/// A column the scan reads (and does not filter on) whose offset index
/// the stage fetches, so that arrow-rs can fetch only the selected pages.
#[derive(Debug, Clone)]
struct ReadColumn {
    leaf: usize,
    chunk: Range<u64>,
    offset_index: Range<u64>,
}

/// A stored Bloom filter to probe for a unit: the filter's declared range
/// and, per equality conjunct on the column, the hashes of its literals.
#[derive(Debug, Clone)]
struct BloomCheck {
    leaf: usize,
    range: Range<u64>,
    probes: Vec<Vec<u64>>,
}

/// A dictionary page to check for a unit (stage 2b, T085): the page's
/// range and, per equality conjunct on the column, its literals as the
/// file stores them.
#[derive(Debug, Clone)]
struct DictCheck {
    leaf: usize,
    range: Range<u64>,
    physical: PhysicalType,
    probes: Vec<Vec<Vec<u8>>>,
}

/// A row group the stage looks at.
#[derive(Debug, Clone)]
struct IndexedUnit {
    /// The row group's index in the file.
    index: usize,
    rows: u64,
    columns: Vec<IndexedColumn>,
    /// The other projected columns' offset indexes; `None` when some
    /// projected column declares none (nothing of this unit is attached).
    reads: Option<Vec<ReadColumn>>,
    /// The stored Bloom filters to probe (stage 2a).
    blooms: Vec<BloomCheck>,
    /// The dictionary pages to check (stage 2b).
    dicts: Vec<DictCheck>,
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
    /// Units a stored Bloom filter ruled out.
    pub(super) bloom_pruned: u64,
    /// Units whose Bloom filter failed a check and was not used.
    pub(super) bloom_rejected: u64,
    /// Units a dictionary page ruled out.
    pub(super) dict_pruned: u64,
    /// Units whose dictionary page failed a check and was not used.
    pub(super) dict_rejected: u64,
}

/// What the stage decided for a unit: the rows to keep.
pub(super) struct Decision {
    /// The row group's index in the file.
    pub(super) index: usize,
    /// The rows that may match (all rows when nothing was pruned).
    pub(super) keep: RowSelection,
}

/// The offset indexes of every projected leaf of a row group, guarded and
/// decoded: what lets arrow-rs fetch only the pages a selection needs.
pub(super) struct Offsets {
    /// The row group's index in the file.
    pub(super) index: usize,
    /// `(leaf, offset index)` for every projected leaf.
    pub(super) leaves: Vec<(usize, OffsetIndexMetaData)>,
}

/// What [`PageStage::finish`] found.
pub(super) struct Outcome {
    pub(super) decisions: Vec<Decision>,
    /// Row groups with complete offset indexes.
    pub(super) offsets: Vec<Offsets>,
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
        read_leaves: &[usize],
        page: bool,
        bloom: bool,
        dict: bool,
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
        // Dictionary bytes the stage fetches so far (stage 2b's budget).
        let mut dict_bytes = 0_u64;
        for &index in units {
            let rg = footer.row_group(index);
            let row_group = meta.row_groups.get(index)?;
            let mut columns: Vec<IndexedColumn> = Vec::new();
            for (n, c) in conjuncts.iter().enumerate().filter(|_| page) {
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
            // Stored Bloom filters (stage 2a): equality conjuncts that
            // statistics left undecided, on columns with an exact plain
            // encoding, whose filter is small next to the column chunk.
            let mut blooms: Vec<BloomCheck> = Vec::new();
            if bloom {
                for c in &conjuncts {
                    if prune(c, &row_group.statistics) != TriState::Unknown {
                        continue;
                    }
                    let Some((path, values)) = bloom_probe::probe_values(c) else {
                        continue;
                    };
                    let Some(leaf) = (0..descr.num_columns()).find(|&l| {
                        let d = descr.column(l);
                        d.max_rep_level() == 0 && d.path().parts() == path.0.as_slice()
                    }) else {
                        continue;
                    };
                    let d = descr.column(leaf);
                    let Some(dt) = leaf_type(&meta.schema, &path.0) else {
                        continue;
                    };
                    if !bloom_probe::eligible(&d) {
                        continue;
                    }
                    let Some(hashes) =
                        bloom_probe::hashes(&values, d.physical_type(), dt, micros_per_unit(&d))
                    else {
                        continue;
                    };
                    let col = rg.column(leaf);
                    let (start, len) = col.byte_range();
                    let Some(chunk_end) = start.checked_add(len) else {
                        continue;
                    };
                    let Some(range) = index_guard::declared(col, meta.size, &(start..chunk_end))
                        .ok()
                        .and_then(|d| d.bloom)
                    else {
                        continue;
                    };
                    if (range.end - range.start).saturating_mul(8) > len {
                        continue;
                    }
                    match blooms.iter_mut().find(|b| b.leaf == leaf) {
                        Some(b) => b.probes.push(hashes),
                        None => blooms.push(BloomCheck {
                            leaf,
                            range,
                            probes: vec![hashes],
                        }),
                    }
                }
            }
            // Dictionary pages (stage 2b, T085): the same equality
            // conjuncts, on chunks whose data pages are all dictionary
            // encoded, with a dictionary page under the byte cap.
            let mut dicts: Vec<DictCheck> = Vec::new();
            if dict && dict_bytes < dict_probe::MAX_STAGE_DICTIONARY_BYTES {
                for c in &conjuncts {
                    if prune(c, &row_group.statistics) != TriState::Unknown {
                        continue;
                    }
                    let Some((path, values)) = bloom_probe::probe_values(c) else {
                        continue;
                    };
                    let Some(leaf) = (0..descr.num_columns()).find(|&l| {
                        let d = descr.column(l);
                        d.max_rep_level() == 0 && d.path().parts() == path.0.as_slice()
                    }) else {
                        continue;
                    };
                    let d = descr.column(leaf);
                    let col = rg.column(leaf);
                    if !bloom_probe::eligible(&d) || !dict_probe::all_dictionary(col) {
                        continue;
                    }
                    let Some(dt) = leaf_type(&meta.schema, &path.0) else {
                        continue;
                    };
                    // Dates and timestamps are stored in the writer's calendar,
                    // which a Spark < 3.0 file may rebase (`calendar.rs`): their
                    // literals are not compared to stored bytes here.
                    if matches!(
                        dt,
                        arrow_schema::DataType::Date32 | arrow_schema::DataType::Timestamp(..)
                    ) {
                        continue;
                    }
                    let Some(literals) = bloom_probe::plain_values(
                        &values,
                        d.physical_type(),
                        dt,
                        micros_per_unit(&d),
                    ) else {
                        continue;
                    };
                    let Some(range) = dict_probe::dictionary_range(col, meta.size) else {
                        continue;
                    };
                    match dicts.iter_mut().find(|x| x.leaf == leaf) {
                        Some(x) => x.probes.push(literals),
                        None if dict_bytes.saturating_add(range.end - range.start)
                            > dict_probe::MAX_STAGE_DICTIONARY_BYTES => {}
                        None => {
                            dict_bytes += range.end - range.start;
                            dicts.push(DictCheck {
                                leaf,
                                range,
                                physical: d.physical_type(),
                                probes: vec![literals],
                            });
                        }
                    }
                }
            }
            if columns.is_empty() && blooms.is_empty() && dicts.is_empty() {
                continue;
            }
            // The other projected columns' offset indexes (all or none: a
            // row group with one missing is not attached, see decode.rs).
            let mut reads = (!columns.is_empty()).then(Vec::new);
            for &leaf in read_leaves {
                if columns.iter().any(|c| c.leaf == leaf) {
                    continue;
                }
                let col = rg.column(leaf);
                let (start, len) = col.byte_range();
                let found = start.checked_add(len).and_then(|end| {
                    index_guard::declared(col, meta.size, &(start..end))
                        .ok()
                        .and_then(|d| d.offset_index)
                        .map(|offset_index| ReadColumn {
                            leaf,
                            chunk: start..end,
                            offset_index,
                        })
                });
                match (found, reads.as_mut()) {
                    (Some(r), Some(list)) => list.push(r),
                    _ => reads = None,
                }
            }
            // The read must be small next to what it can save: an eighth of
            // the projected columns' chunks.
            let len = |r: &Range<u64>| r.end - r.start;
            let index_bytes: u64 = columns
                .iter()
                .map(|c| len(&c.column_index) + len(&c.offset_index))
                .chain(reads.iter().flatten().map(|r| len(&r.offset_index)))
                .sum();
            let chunk_bytes: u64 = columns
                .iter()
                .map(|c| len(&c.chunk))
                .chain(reads.iter().flatten().map(|r| len(&r.chunk)))
                .sum();
            if index_bytes.saturating_mul(8) > chunk_bytes {
                columns.clear();
                reads = None;
            }
            if columns.is_empty() && blooms.is_empty() && dicts.is_empty() {
                continue;
            }
            for c in &columns {
                ranges.push(c.column_index.clone());
                ranges.push(c.offset_index.clone());
            }
            for r in reads.iter().flatten() {
                ranges.push(r.offset_index.clone());
            }
            for b in &blooms {
                ranges.push(b.range.clone());
            }
            for d in &dicts {
                ranges.push(d.range.clone());
            }
            planned.push(IndexedUnit {
                index,
                rows: row_group.rows,
                columns,
                reads,
                blooms,
                dicts,
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
            bloom_pruned: 0,
            bloom_rejected: 0,
            dict_pruned: 0,
            dict_rejected: 0,
        })
    }

    /// The ranges to read, until they arrived: the decoder's first steps.
    pub(super) fn request(&mut self) -> Option<Vec<Range<u64>>> {
        // Until the bytes arrive: a step without a push returns the same
        // ranges (also after a cancelled fetch).
        if self.done {
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
    pub(super) fn finish(&mut self, meta: &ParquetMetadata) -> Outcome {
        self.done = true;
        let held = std::mem::take(&mut self.held);
        let mut out = Vec::new();
        let mut all_offsets = Vec::new();
        let (mut pruned, mut rejected) = (0, 0);
        let mut first_reason: Option<String> = None;
        let (mut bloom_pruned, mut bloom_rejected) = (0_u64, 0_u64);
        let mut first_bloom: Option<String> = None;
        let (mut dict_pruned, mut dict_rejected) = (0_u64, 0_u64);
        let mut first_dict: Option<String> = None;
        for unit in &self.units {
            // A stored Bloom filter that rules the row group out ends it.
            if !unit.blooms.is_empty() {
                match Self::may_match(&held, &unit.blooms) {
                    Ok(false) => {
                        if let Ok(rows) = usize::try_from(unit.rows) {
                            bloom_pruned += 1;
                            out.push(Decision {
                                index: unit.index,
                                keep: std::iter::once(RowSelector::skip(rows)).collect(),
                            });
                            continue;
                        }
                    }
                    Ok(true) => {}
                    Err(why) => {
                        bloom_rejected += 1;
                        first_bloom.get_or_insert(why.0);
                    }
                }
            }
            // A dictionary page with none of a conjunct's literals ends it.
            if !unit.dicts.is_empty() {
                let checked =
                    super::contain("checking a dictionary page", FormatError::Corrupt, || {
                        Self::dictionaries_match(&held, meta, unit)
                            .map_err(|why| FormatError::Corrupt(why.0))
                    });
                match checked {
                    Ok(false) => {
                        if let Ok(rows) = usize::try_from(unit.rows) {
                            dict_pruned += 1;
                            out.push(Decision {
                                index: unit.index,
                                keep: std::iter::once(RowSelector::skip(rows)).collect(),
                            });
                            continue;
                        }
                    }
                    Ok(true) => {}
                    Err(why) => {
                        dict_rejected += 1;
                        first_dict.get_or_insert_with(|| why.to_string());
                    }
                }
            }
            if unit.columns.is_empty() {
                continue;
            }
            let decided = super::contain("applying a page index", FormatError::Corrupt, || {
                self.decide(unit, meta, &held)
                    .map_err(|why| FormatError::Corrupt(why.0))
            });
            match decided {
                Ok((decision, pages, offsets)) => {
                    pruned += pages;
                    out.extend(decision);
                    all_offsets.extend(offsets);
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
        if let Some(reason) = first_bloom {
            tracing::warn!(units = bloom_rejected, %reason, "a Parquet Bloom filter is ignored");
        }
        if let Some(reason) = first_dict {
            tracing::warn!(units = dict_rejected, %reason, "a Parquet dictionary page is ignored");
        }
        self.bloom_pruned += bloom_pruned;
        self.bloom_rejected += bloom_rejected;
        self.dict_pruned += dict_pruned;
        self.dict_rejected += dict_rejected;
        self.pages_pruned += pruned;
        self.rejected += rejected;
        Outcome {
            decisions: out,
            offsets: all_offsets,
        }
    }

    /// Whether the unit may hold a row matching every Bloom-probed
    /// conjunct: `false` when some conjunct has no literal the filter can
    /// contain. The filter's header and bitset are guarded first.
    fn may_match(
        held: &BTreeMap<(u64, u64), Bytes>,
        blooms: &[BloomCheck],
    ) -> Result<bool, Rejected> {
        for b in blooms {
            let bytes = Self::slice(held, &b.range)?;
            let (num_bytes, header) = index_guard::bloom_header(bytes, &b.range)?;
            let bitset = usize::try_from(num_bytes)
                .ok()
                .and_then(|n| header.checked_add(n))
                .and_then(|end| bytes.get(header..end))
                .ok_or_else(|| Rejected::from("the Bloom bitset is short"))?;
            let bloom = RuntimeBloom::from_bytes(bitset)
                .ok_or_else(|| Rejected::from("the Bloom bitset is not a filter"))?;
            for probe in &b.probes {
                if !probe.iter().any(|h| bloom.check_hash(*h)) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Whether the unit may hold a row matching every dictionary-checked
    /// conjunct: `false` when some conjunct's literals are all absent from
    /// the column's dictionary. Every data page is dictionary encoded
    /// (`dict_probe::all_dictionary`), so a value not in the dictionary is
    /// in no row.
    fn dictionaries_match(
        held: &BTreeMap<(u64, u64), Bytes>,
        meta: &ParquetMetadata,
        unit: &IndexedUnit,
    ) -> Result<bool, Rejected> {
        let rg = meta.reader.metadata().row_group(unit.index);
        for d in &unit.dicts {
            let bytes = Self::slice_bytes(held, &d.range)?;
            let (values, count) = dict_probe::read_dictionary(rg.column(d.leaf), &d.range, bytes)?;
            for literals in &d.probes {
                if !dict_probe::any_entry_in(&values, count, d.physical, literals)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// [`Self::slice`] as shared bytes (no copy).
    fn slice_bytes(held: &BTreeMap<(u64, u64), Bytes>, r: &Range<u64>) -> Result<Bytes, Rejected> {
        let bytes = held
            .get(&(r.start, r.end))
            .ok_or_else(|| Rejected::from("an index range did not arrive"))?;
        let len =
            usize::try_from(r.end - r.start).map_err(|_| Rejected::from("a range overflows"))?;
        if bytes.len() < len {
            return Err(Rejected::from("an index range arrived short"));
        }
        Ok(bytes.slice(..len))
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
    ) -> Result<(Option<Decision>, u64, Option<Offsets>), Rejected> {
        let rows = usize::try_from(unit.rows).map_err(|_| Rejected::from("rows"))?;
        // The row ranges to skip, from the pages alone: time and memory
        // follow the number of pages, never the row count a footer declares.
        let mut skipped: Vec<Range<usize>> = Vec::new();
        let mut pruned_pages = 0_u64;
        let mut kept: Vec<(usize, OffsetIndexMetaData)> = Vec::new();
        for col in &unit.columns {
            let offsets = index_guard::offset_index(
                Self::slice(held, &col.offset_index)?,
                &col.chunk,
                unit.rows,
            )?;
            kept.push((col.leaf, offsets.clone()));
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
        // The other projected columns' offset indexes: a bad or missing one
        // only means this row group is not attached.
        let mut complete = unit.reads.is_some();
        for r in unit.reads.iter().flatten() {
            let one = Self::slice(held, &r.offset_index)
                .and_then(|bytes| index_guard::offset_index(bytes, &r.chunk, unit.rows));
            match one {
                Ok(o) => kept.push((r.leaf, o)),
                Err(_) => {
                    complete = false;
                    break;
                }
            }
        }
        let offsets = complete.then_some(Offsets {
            index: unit.index,
            leaves: kept,
        });
        if skipped.is_empty() {
            return Ok((None, 0, offsets));
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
            offsets,
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
mod dict_tests;
#[cfg(test)]
mod tests;
