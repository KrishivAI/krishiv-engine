// SPDX-License-Identifier: Apache-2.0
//! [`FileTable`]: a table of listed files of one format, with statistics
//! from their footers.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_common::config::Config;
use krishiv_io::split::SplitConfig;
use krishiv_io::{
    FileFormat, FileRef, FileStatistics, FormatOptions, IoContext, MetadataNeeds, ScalarValue,
    ScanOptions,
};
use krishiv_optimizer::{ColumnStats, TableStats};
use krishiv_plan::Literal;
use krishiv_types::DecimalType;

use super::table::TableProvider;
use crate::error::Result;
use crate::operators::{FileScan, ScanPushdown};

/// A table of `files` read by one [`FileFormat`] (spec 13 §7): the file
/// scan with Spark's split planning, and the row counts, null counts and
/// min/max values of the files' footers as the optimizer's statistics.
#[derive(Debug, Clone)]
pub struct FileTable {
    scan: FileScan,
    statistics: Option<TableStats>,
}

impl FileTable {
    /// Opens a table of `files` as `table` (the schema the analyzer
    /// resolved, in Arrow form; files map to it by column name), reading
    /// every file's metadata for the statistics. Splits follow `config`
    /// (`spark.sql.files.maxPartitionBytes`, ...) over `worker_slots`
    /// drivers.
    ///
    /// # Errors
    /// Errors reading a file's metadata.
    pub async fn open(
        format: Arc<dyn FileFormat>,
        io: IoContext,
        table: SchemaRef,
        files: Vec<FileRef>,
        options: FormatOptions,
        config: &Config,
        worker_slots: u64,
    ) -> Result<Self> {
        let mut stats: Option<FileStatistics> = None;
        let mut known = true;
        for f in &files {
            let meta = io
                .read_metadata(format.as_ref(), f, MetadataNeeds::default(), &options)
                .await?;
            let s = meta.statistics();
            known &= s.rows.exact().is_some();
            stats = Some(match stats {
                None => s,
                Some(acc) => merge(acc, &s),
            });
        }
        let statistics = known
            .then(|| stats.map(|s| table_stats(&s, &table)))
            .flatten();
        let scan = FileScan::new(
            format,
            io,
            table,
            files,
            options,
            ScanOptions::from_config(config),
            SplitConfig::from_config(config, worker_slots),
        );
        Ok(Self { scan, statistics })
    }
}

impl TableProvider for FileTable {
    fn scan(&self) -> Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::new(self.scan.clone()))
    }

    fn statistics(&self) -> Option<TableStats> {
        self.statistics.clone()
    }
}

/// The statistics of two files together.
fn merge(mut acc: FileStatistics, s: &FileStatistics) -> FileStatistics {
    use krishiv_io::Precision;
    let add = |a: &Precision<u64>, b: &Precision<u64>| match (a.exact(), b.exact()) {
        (Some(x), Some(y)) => Precision::Exact(x.saturating_add(*y)),
        _ => Precision::Absent,
    };
    acc.rows = add(&acc.rows, &s.rows);
    acc.bytes = add(&acc.bytes, &s.bytes);
    acc.columns.retain_mut(|c| {
        let Some(o) = s.columns.iter().find(|o| o.path == c.path) else {
            return false;
        };
        c.null_count = add(&c.null_count, &o.null_count);
        c.min = pick(&c.min, &o.min, Ordering::Less);
        c.max = pick(&c.max, &o.max, Ordering::Greater);
        true
    });
    acc
}

/// The smaller (`Less`) or larger of two exact bounds.
fn pick(
    a: &krishiv_io::Precision<ScalarValue>,
    b: &krishiv_io::Precision<ScalarValue>,
    want: Ordering,
) -> krishiv_io::Precision<ScalarValue> {
    use krishiv_io::Precision;
    match (a.exact(), b.exact()) {
        (Some(x), Some(y)) => match compare(x, y) {
            Some(o) if o == want || o == Ordering::Equal => Precision::Exact(x.clone()),
            Some(_) => Precision::Exact(y.clone()),
            None => Precision::Absent,
        },
        _ => Precision::Absent,
    }
}

fn compare(a: &ScalarValue, b: &ScalarValue) -> Option<Ordering> {
    use ScalarValue as S;
    Some(match (a, b) {
        (S::Int8(x), S::Int8(y)) => x.cmp(y),
        (S::Int16(x), S::Int16(y)) => x.cmp(y),
        (S::Int32(x), S::Int32(y)) | (S::Date32(x), S::Date32(y)) => x.cmp(y),
        (S::Int64(x), S::Int64(y)) | (S::TimestampMicros(x), S::TimestampMicros(y)) => x.cmp(y),
        (
            S::Decimal128 {
                unscaled: x,
                scale: sx,
                ..
            },
            S::Decimal128 {
                unscaled: y,
                scale: sy,
                ..
            },
        ) if sx == sy => x.cmp(y),
        (S::Utf8(x), S::Utf8(y)) => x.as_bytes().cmp(y.as_bytes()),
        _ => return None,
    })
}

/// The optimizer's statistics of the table columns.
fn table_stats(s: &FileStatistics, table: &SchemaRef) -> TableStats {
    let mut out = TableStats {
        rows: s.rows.exact().copied().unwrap_or_default(),
        ..TableStats::default()
    };
    for c in &s.columns {
        let [name] = c.path.0.as_slice() else {
            continue;
        };
        let Some(field) = table.fields().iter().find(|f| f.name() == name) else {
            continue;
        };
        let literal = |v: &krishiv_io::Precision<ScalarValue>| {
            v.exact().and_then(|v| literal(v, field.data_type()))
        };
        let nulls = c.null_count.exact().copied();
        // Footers carry no distinct counts; an integral range bounds them
        // (`c_nationkey` in 0..=24 has at most 25 values).
        let distinct = match (c.min.exact(), c.max.exact()) {
            (Some(lo), Some(hi)) => integral(lo).zip(integral(hi)).and_then(|(lo, hi)| {
                let span = u64::try_from(hi.checked_sub(lo)?).ok()?.checked_add(1)?;
                let values = out.rows.saturating_sub(nulls.unwrap_or(0));
                Some(span.min(values))
            }),
            _ => None,
        };
        out.columns.insert(
            name.clone(),
            ColumnStats {
                distinct,
                nulls,
                min: literal(&c.min),
                max: literal(&c.max),
            },
        );
    }
    out
}

/// An integral or date bound as an integer.
fn integral(v: &ScalarValue) -> Option<i128> {
    Some(match v {
        ScalarValue::Int8(x) => i128::from(*x),
        ScalarValue::Int16(x) => i128::from(*x),
        ScalarValue::Int32(x) | ScalarValue::Date32(x) => i128::from(*x),
        ScalarValue::Int64(x) => i128::from(*x),
        _ => return None,
    })
}

/// A bound as a literal of the table column's type (only where the file
/// value has that type: bounds of a mapped column are left out).
fn literal(v: &ScalarValue, t: &arrow_schema::DataType) -> Option<Literal> {
    use ScalarValue as S;
    use arrow_schema::DataType as A;
    Some(match (v, t) {
        (S::Int8(x), A::Int8) => Literal::Byte(*x),
        (S::Int16(x), A::Int16) => Literal::Short(*x),
        (S::Int32(x), A::Int32) => Literal::Integer(*x),
        (S::Int64(x), A::Int64) => Literal::Long(*x),
        (S::Float32(x), A::Float32) => Literal::Float(*x),
        (S::Float64(x), A::Float64) => Literal::Double(*x),
        (S::Date32(x), A::Date32) => Literal::Date(*x),
        (S::Utf8(x), A::Utf8View | A::Utf8) => Literal::String(x.clone()),
        (
            S::Decimal128 {
                unscaled,
                precision,
                scale,
            },
            A::Decimal128(p, s),
        ) if precision == p && scale == s => Literal::Decimal {
            unscaled: *unscaled,
            data_type: DecimalType::try_new(u32::from(*p), u32::try_from(*s).ok()?).ok()?,
        },
        _ => return None,
    })
}
