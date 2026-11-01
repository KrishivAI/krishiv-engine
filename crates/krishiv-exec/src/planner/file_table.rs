// SPDX-License-Identifier: Apache-2.0
//! [`FileTable`]: a table of listed files of one format, with statistics
//! from their footers.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use futures::{StreamExt as _, TryStreamExt as _, stream};
use krishiv_common::config::Config;
use krishiv_io::split::SplitConfig;
use krishiv_io::{
    FileFormat, FileRef, FileStatistics, FormatOptions, IoContext, MetadataNeeds, ScalarValue,
    ScanOptions,
};
use krishiv_optimizer::{ColumnStats, StatSource, TableStats};
use krishiv_plan::Literal;
use krishiv_types::DecimalType;

use super::table::TableProvider;
use crate::error::Result;
use crate::operators::{FileScan, ScanPushdown};

/// Pending footer reads of [`FileTable::open`], in multiples of the
/// stores' request limit: enough to keep that many requests in flight
/// across most of a chunk.
const FOOTER_CHUNK: usize = 4;

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
        // The footers are read concurrently, up to the stores' request
        // limit (the scheduler bounds requests per store too); `buffered`
        // keeps file order, so the merged statistics do not depend on
        // which read finishes first. Files go in chunks of
        // `FOOTER_CHUNK` times that limit, so a table of many files holds
        // a bounded number of pending reads, not one per file.
        // The read futures of a chunk are built before its stream: a
        // closure that borrows into `stream::iter` makes the future of
        // `open` lose `Send` (the borrow's lifetime is not provably
        // general enough), so it could not run on a multi-threaded
        // runtime's task.
        let width = io.config().max_concurrent_requests.max(1);
        let mut stats: Option<FileStatistics> = None;
        let mut known = true;
        let read = |f| io.read_metadata(format.as_ref(), f, MetadataNeeds::default(), &options);
        in_order(&files, width, read, |meta| {
            let s = meta.statistics();
            known &= s.rows.exact().is_some();
            stats = Some(match stats.take() {
                None => s,
                Some(acc) => merge(acc, &s),
            });
        })
        .await?;
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

impl FileTable {
    /// The same table with `columns` (the last columns of its schema) read
    /// as partition columns: their values come from the `name=value`
    /// directories above each file.
    #[must_use]
    pub fn with_partition_columns(mut self, columns: Vec<arrow_schema::FieldRef>) -> Self {
        self.scan = self.scan.with_partition_columns(columns);
        self
    }

    /// The listed files, in listing order (split file indexes refer to
    /// them).
    #[must_use]
    pub fn files(&self) -> &[FileRef] {
        self.scan.files()
    }

    /// The format the files are read with.
    #[must_use]
    pub fn format(&self) -> Arc<dyn FileFormat> {
        Arc::clone(self.scan.format())
    }

    /// The parsed read options.
    #[must_use]
    pub fn options(&self) -> FormatOptions {
        self.scan.format_options().clone()
    }
}

/// Runs `start(item)` for every item, at most `width` at a time, and
/// hands each result to `each` in item order. Items go in chunks of
/// `width * FOOTER_CHUNK`: no more futures than that exist at once. A
/// chunk's futures are built before its stream, which keeps the caller's
/// future `Send`.
async fn in_order<'a, T, O, E, F>(
    items: &'a [T],
    width: usize,
    start: impl Fn(&'a T) -> F,
    mut each: impl FnMut(O),
) -> std::result::Result<(), E>
where
    F: Future<Output = std::result::Result<O, E>>,
{
    for chunk in items.chunks(width.saturating_mul(FOOTER_CHUNK)) {
        let futures: Vec<F> = chunk.iter().map(&start).collect();
        let mut results = stream::iter(futures).buffered(width);
        while let Some(o) = results.try_next().await? {
            each(o);
        }
    }
    Ok(())
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
        let range = match (c.min.exact(), c.max.exact()) {
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
                source: StatSource::IntegralRange,
                distinct: range,
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

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing; never runs on a compute worker"
)]
mod tests {
    use super::*;

    /// `FileTable::open`'s future is `Send`, so a multi-threaded runtime
    /// can run it as a task: this function only compiles if it is.
    fn open_on_any_thread(
        format: Arc<dyn FileFormat>,
        io: IoContext,
        table: SchemaRef,
        files: Vec<FileRef>,
        options: FormatOptions,
        config: &Config,
    ) -> impl Future<Output = Result<FileTable>> + Send + '_ {
        FileTable::open(format, io, table, files, options, config, 1)
    }

    /// Counts live futures: one more while a [`Live`] exists.
    struct Live(Arc<std::sync::Mutex<(usize, usize)>>);

    impl Live {
        fn new(c: &Arc<std::sync::Mutex<(usize, usize)>>) -> Self {
            let mut g = c.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            g.0 += 1;
            g.1 = g.1.max(g.0);
            Self(Arc::clone(c))
        }
    }

    impl Drop for Live {
        fn drop(&mut self) {
            let mut g = self
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            g.0 -= 1;
        }
    }

    #[test]
    fn reads_hold_a_bounded_number_of_futures_in_order() {
        let counts = Arc::new(std::sync::Mutex::new((0, 0)));
        let items: Vec<usize> = (0..301).collect();
        let mut seen = Vec::new();
        let run = in_order(
            &items,
            3,
            |i| {
                // Counted from creation, not from the first poll.
                let live = Live::new(&counts);
                async move {
                    // Later items finish first within a window.
                    for _ in 0..(301 - *i) % 4 {
                        tokio::task::yield_now().await;
                    }
                    drop(live);
                    Ok::<_, ()>(*i)
                }
            },
            |i| seen.push(i),
        );
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap_or_else(|e| unreachable!("{e}"));
        rt.block_on(run)
            .unwrap_or_else(|()| unreachable!("no error"));
        assert_eq!(seen, items, "in item order");
        let (live, peak) = *counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(live, 0);
        assert_eq!(peak, 3 * FOOTER_CHUNK, "at most a chunk of futures");
        // An error stops the reads and comes back.
        let failing = in_order(
            &items,
            3,
            |i| async move { if *i == 7 { Err(*i) } else { Ok(*i) } },
            |_| {},
        );
        assert_eq!(rt.block_on(failing), Err(7));
    }

    #[test]
    fn open_future_is_send() {
        // A compile-time check: naming the function type-checks its
        // `Send` bound; nothing runs.
        let _ = open_on_any_thread;
    }
}
