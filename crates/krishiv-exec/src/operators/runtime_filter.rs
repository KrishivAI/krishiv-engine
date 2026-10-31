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
//!   batch holds and drops the rows some filter rejects (a superset of the
//!   rows the producers keep). Filters run cheapest form first (range,
//!   small set, hash set, Bloom; ties in attachment order), and once fewer
//!   than half the rows survive, a later filter sees only the survivors (as
//!   Velox orders and narrows conjuncts, research `papers.md` A5 claim 3,
//!   but by a static cost so the order never depends on timing). A filter
//!   that is pending applies nothing; it is adopted at the next batch once
//!   published.
//! - **Adaptive switch-off.** Per scan partition, a driver measures each
//!   filter's pass rate over windows of at least `min_rows` rows that reach
//!   it. A filter pays for itself when the rows it drops save more than it
//!   costs on every row it tests: `(1 - pass) * DROP_SAVES_NS > cost`, with
//!   the form's per-row cost from the `runtime_filter` microbenchmark
//!   ([`KeyFilter::row_cost_ns`]). Above `min(max_pass_rate, 1 - cost /
//!   DROP_SAVES_NS)` it stops evaluating that filter on rows until the
//!   measurement restarts (statistics stages keep using it: they cost
//!   nothing), or until the filter publishes a new generation.
//! - **Decisions change speed, never results (P2-T171 I10).** The measurement
//!   and the decision restart when a scan says so
//!   ([`RowStage::start_partition`]): a memory scan at every morsel, a file
//!   scan every few morsels a driver claims, about one planned split's
//!   rows. They depend on row counts, never on time, but in a file scan also
//!   on which morsels a driver won. A filter only drops rows its join would
//!   drop, so the query's output does not depend on them.

use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::{BooleanArray, RecordBatch};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use arrow_schema::Schema;
use krishiv_io::{FilterCounts, KeyFilter, RuntimeFilter, ScanFilter};

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
    pub fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
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
        let mut out = crate::metrics::counts([
            ("runtime_filter_rows_tested", t.rows_tested),
            ("runtime_filter_rows_dropped", t.rows_dropped),
            ("runtime_filter_units_pruned", t.units_pruned),
            ("runtime_filter_bytes_pruned", t.bytes_pruned),
            ("runtime_filter_splits_skipped", t.splits_skipped),
        ]);
        let forms: Vec<String> = filters.iter().map(|f| describe(f)).collect();
        out.push((
            std::borrow::Cow::Borrowed("runtime_filters"),
            crate::metrics::Metric::Text(forms.join(", ")),
        ));
        out
    }
}

/// One filter for `EXPLAIN ANALYZE`: its id, column, the form its
/// producer chose and its row counters, e.g.
/// `#3 [l_partkey]: bloom 4096 B, tested 100, dropped 90`.
fn describe(f: &RuntimeFilter) -> String {
    use krishiv_io::{FilterForm, FilterSnapshot};
    let form = match &f.published().snapshot {
        FilterSnapshot::Pending => "pending".to_owned(),
        FilterSnapshot::Disabled(_) => "disabled".to_owned(),
        FilterSnapshot::Final(k) => match k.form() {
            FilterForm::Range => "range".to_owned(),
            FilterForm::SmallSet(n) => format!("small set of {n}"),
            FilterForm::HashSet(n) => format!("hash set of {n}"),
            FilterForm::Bloom(b) => format!("bloom {b} B"),
            _ => "other".to_owned(),
        },
        _ => "other".to_owned(),
    };
    let c = f.stats().counts();
    format!(
        "{} {}: {form}, tested {}, dropped {}",
        f.id(),
        f.target(),
        c.rows_tested,
        c.rows_dropped
    )
}

/// What one row a runtime filter drops saves downstream, in nanoseconds:
/// at least the join's own probe of that row (key encoding, hash, a
/// probe that misses), from the `hash_join` microbenchmark's probe time
/// per row, rounded down.
pub(crate) const DROP_SAVES_NS: f64 = 20.0;

/// One filter in one driver's row stage.
#[derive(Debug)]
struct RowFilter {
    filter: ScanFilter,
    /// The filtered column's index in the batches.
    column: usize,
    window_rows: u64,
    window_passed: u64,
    /// Off for rows since this generation (until the filter changes or the
    /// next partition starts).
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

    /// A new stretch of the scan starts: every filter is measured afresh, so
    /// the switch-off decisions depend on that stretch's rows only.
    pub(crate) fn start_partition(&mut self) {
        for f in &mut self.filters {
            f.window_rows = 0;
            f.window_passed = 0;
            f.off_at = None;
        }
    }

    /// The pass rate above which `keys` stops paying for itself.
    fn cut_off(&self, keys: &KeyFilter) -> f64 {
        self.settings
            .max_pass_rate
            .min(1.0 - keys.row_cost_ns() / DROP_SAVES_NS)
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
        // The applicable filters, cheapest first (stable: ties keep their
        // attachment order).
        let mut active: Vec<(usize, Arc<KeyFilter>, u64)> = Vec::new();
        for (i, f) in self.filters.iter_mut().enumerate() {
            let generation = f.filter.generation();
            if f.off_at.is_some_and(|g| g == generation) {
                continue;
            }
            f.off_at = None;
            if let Some(keys) = f.filter.current() {
                active.push((i, keys, generation));
            }
        }
        active.sort_by(|a, b| a.1.row_cost_ns().total_cmp(&b.1.row_cost_ns()));
        let mut all: Option<BooleanBuffer> = None;
        for (i, keys, generation) in active {
            let cut_off = self.cut_off(&keys);
            let Some(f) = self.filters.get_mut(i) else {
                continue;
            };
            let column = batch
                .columns()
                .get(f.column)
                .ok_or_else(|| internal("a runtime filter's column is missing".to_owned()))?;
            // Once most rows are gone, only the survivors reach this filter.
            let narrowed = match &all {
                Some(m) if m.count_set_bits() * 2 < rows => Some(
                    arrow_select::filter::filter(
                        column.as_ref(),
                        &BooleanArray::new(m.clone(), None),
                    )
                    .map_err(|e| internal(e.to_string()))?,
                ),
                _ => None,
            };
            let input = narrowed.as_deref().unwrap_or(column.as_ref());
            let Some(mask) = keys.evaluate(input) else {
                continue;
            };
            let tested = input.len() as u64;
            let passed = mask.true_count() as u64;
            f.filter.filter().stats().add_rows(tested, tested - passed);
            f.window_rows += tested;
            f.window_passed += passed;
            if f.window_rows >= self.settings.min_rows {
                #[expect(clippy::cast_precision_loss, reason = "a rate")]
                let rate = f.window_passed as f64 / f.window_rows as f64;
                if rate > cut_off {
                    f.off_at = Some(generation);
                }
                f.window_rows = 0;
                f.window_passed = 0;
            }
            // Runtime filter masks have no nulls.
            all = Some(match (all, narrowed.is_some()) {
                (None, _) => mask.values().clone(),
                (Some(prev), false) => &prev & mask.values(),
                (Some(prev), true) => scatter(&prev, mask.values()),
            });
        }
        Ok(all
            .filter(|m| m.count_set_bits() < rows)
            .map(|m| BooleanArray::new(m, None)))
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

/// `prev` with its set rows replaced by `answers` (one per set row, in
/// order).
fn scatter(prev: &BooleanBuffer, answers: &BooleanBuffer) -> BooleanBuffer {
    let mut out = BooleanBufferBuilder::new(prev.len());
    out.append_n(prev.len(), false);
    for (k, i) in prev.set_indices().enumerate() {
        if k < answers.len() && answers.value(k) {
            out.set_bit(i, true);
        }
    }
    out.finish()
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
        assert!(slot.extra_metrics().iter().all(|(_, v)| matches!(
            v,
            crate::metrics::Metric::Count(0) | crate::metrics::Metric::Text(_)
        )));
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
        let get = |n: &str| m.iter().find(|(k, _)| *k == n).map(|(_, v)| v.clone());
        use crate::metrics::Metric;
        assert_eq!(
            get("runtime_filter_rows_tested"),
            Some(Metric::Count(4 + 4 + 2 + 4))
        );
        assert_eq!(
            get("runtime_filter_rows_dropped"),
            Some(Metric::Count(1 + 3 + 2 + 1))
        );
        assert_eq!(
            get("runtime_filters"),
            Some(Metric::Text(
                "#1 [a]: range, tested 8, dropped 2, #2 [a]: range, tested 6, dropped 5".to_owned()
            ))
        );
        assert!(RuntimeFilters::default().extra_metrics().is_empty());
    }

    fn publish_with(f: &RuntimeFilter, keys: &[i64], options: KeyFilterOptions) {
        let mut b = KeyFilterBuilder::new(DataType::Int64, options)
            .unwrap_or_else(|| unreachable!("int64"));
        b.add(&Int64Array::from(keys.to_vec()), None)
            .unwrap_or_else(|e| unreachable!("{e}"));
        f.publish(FilterSnapshot::Final(Arc::new(b.finish())));
    }

    fn tested(f: &RuntimeFilter) -> u64 {
        f.stats().counts().rows_tested
    }

    /// A filter that drops too few rows to pay for its per-row cost
    /// switches off even under a 100% pass-rate cap; the decision restarts
    /// with every partition and depends only on that partition's rows; a
    /// later (costlier) filter sees only the rows the cheaper ones kept.
    #[test]
    fn cost_based_per_partition_switch_off_and_narrowing() {
        let set = Arc::new(RuntimeFilter::new(
            FilterId(1),
            ColumnPath::new("a"),
            DataType::Int64,
        ));
        // Spread keys: an exact hash set (no range help on 0..=63).
        let keys: Vec<i64> = (0..64).filter(|k| k % 8 != 7).chain([1000]).collect();
        publish_with(&set, &keys, KeyFilterOptions::default());
        let form = set.published();
        let FilterSnapshot::Final(k) = &form.snapshot else {
            unreachable!("published")
        };
        assert!(matches!(k.form(), krishiv_io::FilterForm::HashSet(_)));
        let cut_off = 1.0 - k.row_cost_ns() / super::DROP_SAVES_NS;
        assert!(cut_off < 7.0 / 8.0, "{cut_off}");
        let settings = RowStageSettings {
            min_rows: 8,
            max_pass_rate: 1.0,
        };
        let slot = RuntimeFilters::new(vec![Arc::clone(&set)]);
        let mut stage = RowStage::new(&slot.scan_filters(&schema()), &schema(), settings);
        let loose: Vec<i64> = (0..8).collect();
        // 7 of 8 pass: above the cost cut-off, the filter stops.
        assert_eq!(rows(&mut stage, &loose), 7);
        assert_eq!(rows(&mut stage, &[7, 15, 23]), 3, "off for this partition");
        // A new partition measures afresh: the same rows, the same
        // decisions.
        stage.start_partition();
        assert_eq!(rows(&mut stage, &loose), 7);
        assert_eq!(rows(&mut stage, &[7, 15, 23]), 3);
        stage.start_partition();
        let tight = [7i64, 15, 23, 31, 39, 47, 55, 1];
        assert_eq!(rows(&mut stage, &tight), 1, "1 of 8 pass: stays on");
        assert_eq!(rows(&mut stage, &[7, 15]), 0);

        // Narrowing: a cheap range filter first, then the set on the
        // survivors only.
        let range = Arc::new(RuntimeFilter::new(
            FilterId(2),
            ColumnPath::new("a"),
            DataType::Int64,
        ));
        publish_with(&range, &[0, 1, 2], KeyFilterOptions::default());
        // Attached after the set: the cheaper range still runs first.
        let both = RuntimeFilters::new(vec![Arc::clone(&set), Arc::clone(&range)]);
        let mut stage = RowStage::new(&both.scan_filters(&schema()), &schema(), settings);
        let before = tested(&set);
        let batch: Vec<i64> = (0..100).collect();
        assert_eq!(rows(&mut stage, &batch), 3, "keys 0, 1 and 2");
        assert_eq!(tested(&range), 100);
        assert_eq!(tested(&set) - before, 3, "the set tests the survivors only");
        // A batch the range mostly keeps is tested by both in full.
        assert_eq!(rows(&mut stage, &[0, 1, 2, 7000]), 3);
        assert_eq!(tested(&set) - before, 3 + 4);
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
