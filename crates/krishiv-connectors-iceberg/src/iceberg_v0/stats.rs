// SPDX-License-Identifier: Apache-2.0
//! The optimizer's statistics of an Iceberg snapshot, from manifest metadata
//! only (PR-25d; design note §2.4). Estimates, never used for correctness:
//!
//! - **Rows**: the live data files' `record_count`s, less the `record_count`
//!   of each position delete file and DV that applies to one of them (each
//!   counted once; an upper bound on what they remove, which is fine for an
//!   estimate). Equality deletes are not subtracted: what they remove is
//!   unknown without reading the files.
//! - **Columns** (top-level primitives): null counts when every file has
//!   one; min/max from the files' bounds when every file with rows has both,
//!   under a metrics mode that keeps bounds. A float column is skipped unless
//!   every file says it has no NaN (bounds exclude NaN). A string column is
//!   skipped when a bound may be truncated (`truncate(n)` with a bound of `n`
//!   or more characters, or an unknown mode), and binary columns always.
//! - `TableProvider::statistics` takes no filter, so the sums are over every
//!   live data file of the snapshot (the manifests' own pruning summaries are
//!   not applied).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use krishiv_io::ScalarValue;
use krishiv_optimizer::{ColumnStats, StatSource, TableStats};
use krishiv_plan::Literal;
use krishiv_types::DecimalType;

use crate::plan::deletes::DeleteIndex;
use crate::plan::{DataFileEntry, IcebergType, MetricsMode, SchemaField, WriterMetrics, decode};

/// The statistics of `entries` (a snapshot's live data files) under
/// `index`'s deletes; `fields` are the snapshot's columns.
pub(super) fn table_stats(
    entries: &[std::sync::Arc<DataFileEntry>],
    index: &DeleteIndex,
    fields: &[SchemaField],
    metrics: &WriterMetrics,
) -> TableStats {
    let rows: u64 = entries
        .iter()
        .map(|e| e.file.record_count)
        .fold(0, u64::saturating_add);
    let mut counted: HashSet<&str> = HashSet::new();
    let mut deleted = 0_u64;
    let assigned: Vec<_> = entries
        .iter()
        .filter_map(|e| index.for_data_file(e).ok())
        .collect();
    for d in &assigned {
        for f in d.deletion_vector.iter().chain(&d.position_deletes) {
            if counted.insert(f.entry.file.path.as_str()) {
                deleted = deleted.saturating_add(f.entry.file.record_count);
            }
        }
    }
    let mut out = TableStats {
        rows: rows.saturating_sub(deleted),
        columns: HashMap::new(),
    };
    for f in fields.iter().filter(|f| f.path.0.len() == 1) {
        let c = column_stats(entries, f, metrics.mode(f.id));
        if c != ColumnStats::default() {
            out.columns.insert(f.path.root().to_owned(), c);
        }
    }
    out
}

fn column_stats(
    entries: &[std::sync::Arc<DataFileEntry>],
    f: &SchemaField,
    mode: MetricsMode,
) -> ColumnStats {
    let nulls = entries
        .iter()
        .map(|e| e.file.null_counts.get(&f.id).copied())
        .try_fold(0_u64, |acc, n| n.map(|n| acc.saturating_add(n)));
    let (min, max) = bounds(entries, f, mode).unzip();
    ColumnStats {
        source: StatSource::Provided,
        distinct: None,
        nulls,
        min,
        max,
    }
}

/// The smallest lower and largest upper bound of `f` over the files with
/// rows, when every one has both and they are usable (see the module docs).
fn bounds(
    entries: &[std::sync::Arc<DataFileEntry>],
    f: &SchemaField,
    mode: MetricsMode,
) -> Option<(Literal, Literal)> {
    if !mode.trusts_bounds() {
        return None;
    }
    let mut acc: Option<(Literal, Literal)> = None;
    for e in entries.iter().filter(|e| e.file.record_count > 0) {
        let file = &e.file;
        if matches!(f.ty, IcebergType::Float | IcebergType::Double)
            && file.nan_counts.get(&f.id) != Some(&0)
        {
            return None;
        }
        let lo = file.lower_bounds.get(&f.id)?;
        let hi = file.upper_bounds.get(&f.id)?;
        if f.ty == IcebergType::String {
            let chars = |b: &[u8]| std::str::from_utf8(b).map_or(usize::MAX, |s| s.chars().count());
            match mode {
                MetricsMode::Full => {}
                MetricsMode::Truncate(n) if chars(lo) < n && chars(hi) < n => {}
                _ => return None,
            }
        }
        let lo = literal(&decode(lo, &f.ty)?, &f.ty)?;
        let hi = literal(&decode(hi, &f.ty)?, &f.ty)?;
        acc = Some(match acc {
            None => (lo, hi),
            Some((a, b)) => (
                if compare(&lo, &a)? == Ordering::Less {
                    lo
                } else {
                    a
                },
                if compare(&hi, &b)? == Ordering::Greater {
                    hi
                } else {
                    b
                },
            ),
        });
    }
    acc
}

/// A decoded bound as a literal of the column's Spark type (the types the
/// optimizer's estimates read; others are left out).
fn literal(v: &ScalarValue, ty: &IcebergType) -> Option<Literal> {
    use IcebergType as T;
    use ScalarValue as S;
    Some(match (v, ty) {
        (S::Int32(x), T::Int) => Literal::Integer(*x),
        (S::Int64(x), T::Long) => Literal::Long(*x),
        (S::Float32(x), T::Float) => Literal::Float(*x),
        (S::Float64(x), T::Double) => Literal::Double(*x),
        (S::Date32(x), T::Date) => Literal::Date(*x),
        (S::TimestampMicros(x), T::Timestamptz) => Literal::Timestamp(*x),
        (S::TimestampMicros(x), T::Timestamp) => Literal::TimestampNtz(*x),
        (S::Utf8(x), T::String) => Literal::String(x.clone()),
        (S::Decimal128 { unscaled, .. }, T::Decimal { precision, scale }) => Literal::Decimal {
            unscaled: *unscaled,
            data_type: DecimalType::try_new(u32::from(*precision), u32::from(*scale)).ok()?,
        },
        _ => return None,
    })
}

/// The order of two literals of one column type.
fn compare(a: &Literal, b: &Literal) -> Option<Ordering> {
    use Literal as L;
    match (a, b) {
        (L::Integer(x), L::Integer(y)) | (L::Date(x), L::Date(y)) => Some(x.cmp(y)),
        (L::Long(x), L::Long(y))
        | (L::Timestamp(x), L::Timestamp(y))
        | (L::TimestampNtz(x), L::TimestampNtz(y)) => Some(x.cmp(y)),
        (L::Float(x), L::Float(y)) => x.partial_cmp(y),
        (L::Double(x), L::Double(y)) => x.partial_cmp(y),
        (L::String(x), L::String(y)) => Some(x.cmp(y)),
        (L::Decimal { unscaled: x, .. }, L::Decimal { unscaled: y, .. }) => Some(x.cmp(y)),
        _ => None,
    }
}

/// How many row groups a read of `entries` touches: the files' split
/// offsets (one per row group in Parquet), one for a file without them.
pub(super) fn scan_units(entries: &[std::sync::Arc<DataFileEntry>]) -> u64 {
    entries
        .iter()
        .map(|e| {
            e.file
                .split_offsets
                .as_ref()
                .map_or(1, |s| u64::try_from(s.len().max(1)).unwrap_or(u64::MAX))
        })
        .fold(0, u64::saturating_add)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv_io::ColumnPath;
    use krishiv_plan::Literal;

    use super::table_stats;
    use crate::plan::deletes::DeleteIndex;
    use crate::plan::tests::file;
    use crate::plan::{
        DataFileEntry, EntryStatus, IcebergType, SchemaField, WriterMetrics, encode,
    };

    fn entry(f: crate::plan::ContentFile) -> Arc<DataFileEntry> {
        Arc::new(DataFileEntry {
            file: f,
            status: EntryStatus::Added,
            snapshot_id: 1,
            data_sequence_number: Some(1),
            file_sequence_number: Some(1),
        })
    }

    fn field(id: i32, name: &str, ty: IcebergType) -> SchemaField {
        SchemaField {
            id,
            path: ColumnPath::new(name),
            ty,
        }
    }

    /// Rows sum the files; min/max are the widest bounds; nulls sum; a
    /// float column with NaN (or an unknown NaN count) and a string bound at
    /// the truncation length give no min/max.
    #[test]
    fn stats_come_from_the_manifest_entries() {
        let long = |v: i64| encode(&krishiv_io::ScalarValue::Int64(v), &IcebergType::Long).unwrap();
        let dbl =
            |v: f64| encode(&krishiv_io::ScalarValue::Float64(v), &IcebergType::Double).unwrap();
        let mut a = file(10);
        let mut b = file(5);
        for (f, lo, hi, nulls) in [(&mut a, 3, 9, 1), (&mut b, -2, 4, 2)] {
            f.lower_bounds.insert(1, long(lo));
            f.upper_bounds.insert(1, long(hi));
            f.null_counts.insert(1, nulls);
            f.lower_bounds.insert(2, dbl(0.5));
            f.upper_bounds.insert(2, dbl(1.5));
            f.lower_bounds.insert(3, b"abc".to_vec());
            f.upper_bounds.insert(3, b"abcdefghijklmnop".to_vec());
        }
        a.nan_counts.insert(2, 0);
        let fields = [
            field(1, "id", IcebergType::Long),
            field(2, "x", IcebergType::Double),
            field(3, "s", IcebergType::String),
        ];
        let entries = [entry(a.clone()), entry(b.clone())];
        let s = table_stats(
            &entries,
            &DeleteIndex::default(),
            &fields,
            &WriterMetrics::default(),
        );
        assert_eq!(s.rows, 15);
        let id = &s.columns["id"];
        assert_eq!(
            (id.min.clone(), id.max.clone(), id.nulls),
            (Some(Literal::Long(-2)), Some(Literal::Long(9)), Some(3))
        );
        // b's NaN count is unknown; the 16-character upper bound may be
        // truncated (default `truncate(16)`).
        assert!(!s.columns.contains_key("x"), "{:?}", s.columns.get("x"));
        assert!(!s.columns.contains_key("s"), "{:?}", s.columns.get("s"));
        b.nan_counts.insert(2, 0);
        let entries = [entry(a), entry(b)];
        let s = table_stats(
            &entries,
            &DeleteIndex::default(),
            &fields,
            &WriterMetrics::default(),
        );
        assert_eq!(s.columns["x"].min, Some(Literal::Double(0.5)));
    }
}
