// SPDX-License-Identifier: Apache-2.0
//! Runtime filters in scans (design P2-T163): the list a planner attaches
//! to a scan ([`RuntimeFilters`]) and the row stage every scan runs over
//! its table-level batches ([`RowStage`]), with the adaptive switch-off.
//!
//! - **Attached after the scan is built.** The planner builds a scan before
//!   the join above it decides to filter it, so a scan holds a shared list
//!   that the planner fills and the scan reads when it starts running
//!   (plans run once, after planning).
//! - **Row stage.** Each driver evaluates every filter whose column the
//!   batch holds and drops the rows no filter passes (a superset of the
//!   rows the producer keeps). A filter that is pending applies nothing; it
//!   is adopted at the next batch once published.
//! - **Adaptive switch-off.** A driver measures each filter's pass rate over
//!   windows of at least `min_rows` rows; above `max_pass_rate` it stops
//!   evaluating that filter on rows (statistics stages keep using it: they
//!   cost nothing) until the filter publishes a new generation.

use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::Schema;
use krishiv_io::{FilterCounts, RuntimeFilter, ScanFilter};

use crate::error::{Result, internal};

/// The row stage's settings (`krishiv.runtimeFilter.adaptive.*`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowStageSettings {
    /// Rows per measurement window.
    pub min_rows: u64,
    /// A filter passing more than this fraction of a window stops being
    /// evaluated on rows.
    pub max_pass_rate: f64,
}

impl Default for RowStageSettings {
    fn default() -> Self {
        Self {
            min_rows: 65_536,
            max_pass_rate: 0.9,
        }
    }
}

#[derive(Debug, Default)]
struct Inner {
    filters: Vec<Arc<RuntimeFilter>>,
    settings: RowStageSettings,
}

/// The runtime filters a planner pushes into one scan (see the module
/// docs). Cloning shares the list.
#[derive(Debug, Clone, Default)]
pub struct RuntimeFilters(Arc<Mutex<Inner>>);

impl RuntimeFilters {
    /// A list of `filters`, with default row-stage settings.
    #[must_use]
    pub fn new(filters: Vec<Arc<RuntimeFilter>>) -> Self {
        Self(Arc::new(Mutex::new(Inner {
            filters,
            settings: RowStageSettings::default(),
        })))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Adds `filter`.
    pub fn push(&self, filter: Arc<RuntimeFilter>) {
        self.lock().filters.push(filter);
    }

    /// Sets the row stage's settings.
    pub fn set_settings(&self, settings: RowStageSettings) {
        self.lock().settings = settings;
    }

    /// The filters attached so far.
    #[must_use]
    pub fn filters(&self) -> Vec<Arc<RuntimeFilter>> {
        self.lock().filters.clone()
    }

    /// The row stage's settings.
    #[must_use]
    pub fn settings(&self) -> RowStageSettings {
        self.lock().settings
    }

    /// The filters at the table level of `table` (filters whose target is
    /// not a column of it are dropped).
    #[must_use]
    pub fn scan_filters(&self, table: &Schema) -> Vec<ScanFilter> {
        self.filters()
            .into_iter()
            .filter_map(|f| ScanFilter::new(f, table))
            .collect()
    }

    /// The `EXPLAIN ANALYZE` counters, summed over the filters (nothing
    /// without filters).
    #[must_use]
    pub fn extra_metrics(&self) -> Vec<(&'static str, u64)> {
        let filters = self.filters();
        if filters.is_empty() {
            return Vec::new();
        }
        let mut t = FilterCounts::default();
        for f in &filters {
            let c = f.stats().counts();
            t.rows_tested += c.rows_tested;
            t.rows_dropped += c.rows_dropped;
            t.units_pruned += c.units_pruned;
            t.bytes_pruned += c.bytes_pruned;
            t.splits_skipped += c.splits_skipped;
        }
        vec![
            ("runtime_filter_rows_tested", t.rows_tested),
            ("runtime_filter_rows_dropped", t.rows_dropped),
            ("runtime_filter_units_pruned", t.units_pruned),
            ("runtime_filter_bytes_pruned", t.bytes_pruned),
            ("runtime_filter_splits_skipped", t.splits_skipped),
        ]
    }
}

/// One filter in one driver's row stage.
#[derive(Debug)]
struct RowFilter {
    filter: ScanFilter,
    /// The filtered column's index in the batches.
    column: usize,
    window_rows: u64,
    window_passed: u64,
    /// Off for rows since this generation (until the filter changes).
    off_at: Option<u64>,
}

/// A driver's row stage (see the module docs).
#[derive(Debug)]
pub(crate) struct RowStage {
    filters: Vec<RowFilter>,
    settings: RowStageSettings,
}

impl RowStage {
    /// The stage over batches of `schema` (filters on columns it does not
    /// hold at the top level do nothing at this stage).
    pub(crate) fn new(filters: &[ScanFilter], schema: &Schema, settings: RowStageSettings) -> Self {
        let filters = filters
            .iter()
            .filter_map(|f| {
                let (path, _) = f.column()?;
                let [name] = path.0.as_slice() else {
                    return None;
                };
                Some(RowFilter {
                    filter: f.clone(),
                    column: schema.index_of(name).ok()?,
                    window_rows: 0,
                    window_passed: 0,
                    off_at: None,
                })
            })
            .collect();
        Self { filters, settings }
    }

    /// The rows of `batch` some producer may keep, with the selection
    /// applied (`None`: every row passes, nothing was filtered).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the batch lacks a filtered column.
    pub(crate) fn mask(&mut self, batch: &RecordBatch) -> Result<Option<BooleanArray>> {
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(None);
        }
        let mut all: Option<BooleanArray> = None;
        for f in &mut self.filters {
            let generation = f.filter.generation();
            if f.off_at.is_some_and(|g| g == generation) {
                continue;
            }
            f.off_at = None;
            let column = batch
                .columns()
                .get(f.column)
                .ok_or_else(|| internal("a runtime filter's column is missing".to_owned()))?;
            let Some(mask) = f.filter.evaluate(column.as_ref()) else {
                continue;
            };
            let passed = mask.true_count() as u64;
            f.filter
                .filter()
                .stats()
                .add_rows(rows as u64, rows as u64 - passed);
            f.window_rows += rows as u64;
            f.window_passed += passed;
            if f.window_rows >= self.settings.min_rows {
                #[expect(clippy::cast_precision_loss, reason = "a rate")]
                let rate = f.window_passed as f64 / f.window_rows as f64;
                if rate > self.settings.max_pass_rate {
                    f.off_at = Some(generation);
                }
                f.window_rows = 0;
                f.window_passed = 0;
            }
            // Runtime filter masks have no nulls.
            all = Some(match all {
                None => mask,
                Some(prev) => BooleanArray::new(prev.values() & mask.values(), None),
            });
        }
        Ok(all.filter(|m| m.true_count() < rows))
    }

    /// `batch` without the rows no producer keeps.
    ///
    /// # Errors
    /// As [`Self::mask`].
    pub(crate) fn apply(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        match self.mask(&batch)? {
            None => Ok(batch),
            Some(mask) => arrow_select::filter::filter_record_batch(&batch, &mask)
                .map_err(|e| internal(e.to_string())),
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_io::{
        ColumnPath, FilterId, FilterSnapshot, KeyFilterBuilder, KeyFilterOptions, RuntimeFilter,
    };

    use super::{RowStage, RowStageSettings, RuntimeFilters};

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
        ]))
    }

    fn batch(a: &[i64]) -> RecordBatch {
        let s: Vec<String> = a.iter().map(|v| format!("s{v}")).collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(a.to_vec())),
                Arc::new(StringArray::from(s)),
            ],
        )
        .unwrap_or_else(|e| unreachable!("{e}"))
    }

    fn publish(f: &RuntimeFilter, keys: &[i64]) {
        let mut b = KeyFilterBuilder::new(DataType::Int64, KeyFilterOptions::default())
            .unwrap_or_else(|| unreachable!("int64"));
        b.add(&Int64Array::from(keys.to_vec()), None)
            .unwrap_or_else(|e| unreachable!("{e}"));
        f.publish(FilterSnapshot::Final(Arc::new(b.finish())));
    }

    fn rows(stage: &mut RowStage, a: &[i64]) -> usize {
        stage
            .apply(batch(a))
            .unwrap_or_else(|e| unreachable!("{e}"))
            .num_rows()
    }

    /// A pending filter applies nothing and is adopted at the next batch
    /// once published; a filter that passes most rows switches off for
    /// the driver; a selective one stays on.
    #[test]
    fn late_adoption_and_adaptive_switch_off() {
        let loose = Arc::new(RuntimeFilter::new(
            FilterId(1),
            ColumnPath::new("a"),
            DataType::Int64,
        ));
        let slot = RuntimeFilters::new(vec![Arc::clone(&loose)]);
        assert!(slot.extra_metrics().iter().all(|(_, v)| *v == 0));
        let settings = RowStageSettings {
            min_rows: 4,
            max_pass_rate: 0.5,
        };
        slot.set_settings(settings);
        assert_eq!(slot.settings(), settings);
        let mut stage = RowStage::new(&slot.scan_filters(&schema()), &schema(), settings);
        assert_eq!(rows(&mut stage, &[1, 2, 3, 4]), 4, "pending");
        publish(&loose, &[1, 2, 3]);
        assert_eq!(rows(&mut stage, &[1, 2, 3, 4]), 3, "adopted");
        // 3 of 4 passed: over the 50% cut-off, the row stage stops.
        assert_eq!(rows(&mut stage, &[4, 4, 4, 4]), 4, "switched off");
        let tight = Arc::new(RuntimeFilter::new(
            FilterId(2),
            ColumnPath::new("a"),
            DataType::Int64,
        ));
        publish(&tight, &[1]);
        slot.push(Arc::clone(&tight));
        let mut stage = RowStage::new(&slot.scan_filters(&schema()), &schema(), settings);
        assert_eq!(rows(&mut stage, &[1, 2, 3, 4]), 1);
        assert_eq!(rows(&mut stage, &[5, 6]), 0, "a selective filter stays on");
        assert_eq!(rows(&mut stage, &[]), 0);
        let m = slot.extra_metrics();
        let get = |n: &str| m.iter().find(|(k, _)| *k == n).map(|(_, v)| *v);
        assert_eq!(get("runtime_filter_rows_tested"), Some(4 + 4 + 2 + 4));
        assert_eq!(get("runtime_filter_rows_dropped"), Some(1 + 3 + 2 + 1));
        assert!(RuntimeFilters::default().extra_metrics().is_empty());
    }

    /// Filters on nested or absent columns have no row stage; a batch
    /// without the filtered column is an internal error.
    #[test]
    fn columns_the_stage_cannot_filter() {
        let nested = Arc::new(RuntimeFilter::new(
            FilterId(1),
            ColumnPath::nested(["a", "x"]),
            DataType::Int64,
        ));
        let slot = RuntimeFilters::new(vec![nested]);
        assert!(slot.scan_filters(&schema()).is_empty());
        let f = Arc::new(RuntimeFilter::new(
            FilterId(2),
            ColumnPath::new("s"),
            DataType::Utf8,
        ));
        let mut b = KeyFilterBuilder::new(DataType::Utf8, KeyFilterOptions::default())
            .unwrap_or_else(|| unreachable!("utf8"));
        b.add(&StringArray::from(vec!["s1"]), None)
            .unwrap_or_else(|e| unreachable!("{e}"));
        f.publish(FilterSnapshot::Final(Arc::new(b.finish())));
        let filters = RuntimeFilters::new(vec![f]).scan_filters(&schema());
        let mut stage = RowStage::new(&filters, &schema(), RowStageSettings::default());
        assert_eq!(rows(&mut stage, &[1, 2]), 1);
        let only_a = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        let narrow = RecordBatch::try_new(only_a, vec![Arc::new(Int64Array::from(vec![1]))])
            .unwrap_or_else(|e| unreachable!("{e}"));
        assert!(stage.apply(narrow).is_err());
    }
}
