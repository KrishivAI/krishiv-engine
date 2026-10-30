// SPDX-License-Identifier: Apache-2.0
//! Window functions (P2-T055, ADR 0011): the [`Window`] operator.
//!
//! # Plan shape
//! Sort-based, as in Leis et al. (VLDB 2015, `research/papers.md` A10,
//! §3.1): the planner sorts the input by the partitioning keys, then the
//! ordering keys ([`crate::operators::Sort`]: external merge sort, spills
//! under memory pressure), and puts a [`Window`] over the sorted stream.
//! The window keeps one partition at a time (plus the rows of the next one
//! seen so far), so its state is partition-local: an input distributed by
//! the partitioning keys (`Exchange(Hash(partition keys))` below the sort,
//! P2-T174) gives the same rows, partition by partition. A window without
//! partitioning keys needs all rows in one partition (one driver above the
//! sort, or a single-partition exchange).
//!
//! # Boundaries
//! Partition and peer-group boundaries come from comparing adjacent rows'
//! keys in the sort's normalized encoding, so they follow Spark's equality
//! of keys: `NULL` equals `NULL`, every NaN is equal and `-0.0 = 0.0`.
//!
//! # Evaluation
//! Complete partitions (all rows before the start of the last partition
//! seen; everything once the input ends) are evaluated together, function
//! by function, over the whole unit:
//! - `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist` and
//!   `ntile` from the boundaries (`ntile` as Spark's `NTile`: the first
//!   `n % k` buckets take one row more);
//! - an aggregate over `UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`: one
//!   fold per batch into one group per partition;
//! - an aggregate over `UNBOUNDED PRECEDING AND CURRENT ROW`: the cumulative
//!   algorithm (Leis et al. §4.3.2), one row (`ROWS`) or one peer group
//!   (`RANGE`) at a time, the state kept after each. Rows are folded one by
//!   one in partition order, as Spark's frames fold them, so float sums
//!   round and integer sums overflow exactly as Spark's do. Decimal `sum`
//!   and `avg` are kept exact (Spark keeps a window's decimal buffer as an
//!   unbounded `Decimal`) and checked when evaluated: past the result's
//!   precision the value is `NULL`, or with ANSI
//!   `NUMERIC_VALUE_OUT_OF_RANGE`.
//!
//! - any other frame: `ROWS` frames with offsets and frames to the
//!   partition's last row (`frames.rs`: segment trees, removable
//!   cumulative sums, or an in-order fold where the order of additions
//!   shows, as Spark's frames fold).
//!
//! `RANGE` frames with value bounds and the offset functions (`lag`,
//! `lead`, `nth_value`) are planned with a later part of P2-T055.
//!
//! # Memory
//! Before a batch is taken (`NeedMemory` otherwise) the operator reserves
//! what it holds for it: its columns, its boundary flags, and what
//! evaluating its rows holds (row maps, states, outputs; exact for its
//! rows, partitions and peer groups). String and binary results are views
//! of the input's buffers, never copies. A batch's reservation is released
//! when its last output slice is emitted.
//! With a spill scope, the buffered rows spill under memory pressure and a
//! partition larger than memory is evaluated from disk (`spill.rs`,
//! `stream.rs`); without one, a partition must fit in memory.
//!
//! # Weights
//! Window functions are defined for insert-only input: weights other than
//! 1 are `KRISHIV_UNSUPPORTED_FEATURE` (incremental windows are P4).

mod eval;
mod frames;
mod group_limit;
mod spill;
mod stream;
#[cfg(all(test, not(loom)))]
mod tests;

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{Array, RecordBatch, RecordBatchOptions};
use arrow_row::OwnedRow;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{ConsumerHandle, Reservation, SpillFile, SpillScope, SpillSignal};
use krishiv_types::batch::{Batch, Weights};

use crate::aggregate::{AggregateExpr, AggregateFunction};
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{FlushOutput, LocalState, Operator, OperatorOutput};
use crate::operators::{KeyEncoder, SortKey};

pub(crate) use group_limit::Ranker;
pub use group_limit::{GroupLimitPlan, RankLike, WindowGroupLimit};

/// A window function.
#[derive(Clone, Debug, PartialEq)]
pub enum WindowFunction {
    /// `row_number()`: `int`, 1 for a partition's first row.
    RowNumber,
    /// `rank()`: `int`, 1 + the rows before the row's peer group.
    Rank,
    /// `dense_rank()`: `int`, 1 + the peer groups before the row's.
    DenseRank,
    /// `percent_rank()`: `double`, `(rank - 1) / (rows - 1)`, 0 for one row.
    PercentRank,
    /// `cume_dist()`: `double`, the rows up to the end of the row's peer
    /// group over the partition's rows.
    CumeDist,
    /// `ntile(buckets)`: `int`, `buckets > 0`.
    Ntile(i32),
    /// An aggregate over the frame; its arguments are input columns.
    Aggregate(AggregateExpr),
    /// `lag`/`lead`: column `input` of the row `offset` rows after the
    /// current one in its partition (before it when negative); column
    /// `default` of the current row (NULL without one) when there is no
    /// such row. With `ignore_nulls`, rows whose input is NULL are not
    /// counted (an offset of 0 is the current row).
    Offset {
        /// The input column.
        input: usize,
        /// Rows after the current one (`lead(x, n)`: `n`; `lag(x, n)`:
        /// `-n`).
        offset: i64,
        /// The default column (cast to the input's type).
        default: Option<usize>,
        /// `IGNORE NULLS`.
        ignore_nulls: bool,
        /// The input's (and the result's) type.
        data_type: DataType,
    },
    /// `nth_value(input, n)`: column `input` of the frame's `n`-th row
    /// (counting only non-NULL inputs with `ignore_nulls`), NULL if the
    /// frame has fewer.
    NthValue {
        /// The input column.
        input: usize,
        /// `n`, above 0.
        n: i64,
        /// `IGNORE NULLS`.
        ignore_nulls: bool,
        /// The input's (and the result's) type.
        data_type: DataType,
    },
}

/// The units of a [`WindowFrame`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameUnits {
    /// `ROWS`.
    Rows,
    /// `RANGE`: `CURRENT ROW` includes the row's peers.
    Range,
}

/// A bound of a [`WindowFrame`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrameBound {
    /// The partition's first row.
    UnboundedPreceding,
    /// The current row (with its peers, in `RANGE` units).
    CurrentRow,
    /// The partition's last row.
    UnboundedFollowing,
    /// `ROWS` only: the row this many rows after the current one (before
    /// it when negative: `n PRECEDING` is `-n`), as Spark keeps a frame's
    /// value bound.
    Offset(i64),
}

/// A window frame (Spark's `SpecifiedWindowFrame`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowFrame {
    /// `ROWS` or `RANGE`.
    pub units: FrameUnits,
    /// The start.
    pub lower: FrameBound,
    /// The end.
    pub upper: FrameBound,
}

impl WindowFrame {
    /// `units BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`.
    #[must_use]
    pub const fn whole(units: FrameUnits) -> Self {
        Self {
            units,
            lower: FrameBound::UnboundedPreceding,
            upper: FrameBound::UnboundedFollowing,
        }
    }

    /// `units BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
    #[must_use]
    pub const fn running(units: FrameUnits) -> Self {
        Self {
            units,
            lower: FrameBound::UnboundedPreceding,
            upper: FrameBound::CurrentRow,
        }
    }
}

/// One window function of a [`WindowPlan`].
#[derive(Clone, Debug, PartialEq)]
pub struct WindowFunctionPlan {
    /// The function.
    pub function: WindowFunction,
    /// Its frame (ranking functions ignore it).
    pub frame: WindowFrame,
    /// A boolean input column: an aggregate reads only the rows where it is
    /// true (`FILTER (WHERE ...)`).
    pub filter: Option<usize>,
    /// The output column's name.
    pub name: String,
}

/// The window functions over one partitioning and ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowPlan {
    /// The partitioning columns.
    pub partition_by: Vec<usize>,
    /// The ordering within a partition (the input is sorted by the
    /// partitioning columns, then by these).
    pub order_by: Vec<SortKey>,
    /// The functions; each adds one column after the input's.
    pub functions: Vec<WindowFunctionPlan>,
}

crate::physical::data::plan_data!(
    enum WindowFunction {
        RowNumber,
        Rank,
        DenseRank,
        PercentRank,
        CumeDist,
        Ntile(buckets),
        Aggregate(aggregate),
        Offset { input, offset, default, ignore_nulls, data_type },
        NthValue { input, n, ignore_nulls, data_type },
    }
);
crate::physical::data::plan_data!(
    enum FrameUnits {
        Rows,
        Range,
    }
);
crate::physical::data::plan_data!(
    enum FrameBound {
        UnboundedPreceding,
        CurrentRow,
        UnboundedFollowing,
        Offset(offset),
    }
);
crate::physical::data::plan_data!(struct WindowFrame { units, lower, upper });
crate::physical::data::plan_data!(struct WindowFunctionPlan { function, frame, filter, name });
crate::physical::data::plan_data!(struct WindowPlan { partition_by, order_by, functions });

/// The distribution a window requires of its input: its state is one
/// partition at a time, so every row of a partition must reach one driver
/// (plan data, for the exchange pass of P2-T174).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequiredDistribution {
    /// No partitioning: all rows in one partition.
    Single,
    /// Rows with equal values of these input columns in one partition.
    Clustered(Vec<usize>),
}

impl WindowPlan {
    /// What the window requires of its input's distribution.
    #[must_use]
    pub fn required_distribution(&self) -> RequiredDistribution {
        if self.partition_by.is_empty() {
            RequiredDistribution::Single
        } else {
            RequiredDistribution::Clustered(self.partition_by.clone())
        }
    }
}

/// How an aggregate's frame is evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shape {
    /// The whole partition.
    Whole,
    /// From the partition's start to the current row.
    RunningRows,
    /// From the partition's start to the end of the current peer group.
    RunningPeers,
    /// Any other frame: each row's own range of rows (`frames.rs`).
    Frame,
}

impl WindowFunctionPlan {
    /// The output type and nullability.
    ///
    /// # Errors
    /// The aggregate's typing errors.
    pub fn output(&self) -> Result<(DataType, bool)> {
        Ok(match &self.function {
            WindowFunction::RowNumber
            | WindowFunction::Rank
            | WindowFunction::DenseRank
            | WindowFunction::Ntile(_) => (DataType::Int32, false),
            WindowFunction::PercentRank | WindowFunction::CumeDist => (DataType::Float64, false),
            WindowFunction::Aggregate(a) => {
                (a.result_type()?, a.function != AggregateFunction::Count)
            }
            WindowFunction::Offset { data_type, .. }
            | WindowFunction::NthValue { data_type, .. } => (data_type.clone(), true),
        })
    }

    /// How the frame is evaluated.
    pub(crate) fn shape(&self) -> Result<Shape> {
        use FrameBound as B;
        match (self.frame.units, self.frame.lower, self.frame.upper) {
            (_, B::UnboundedPreceding, B::UnboundedFollowing) => Ok(Shape::Whole),
            (FrameUnits::Rows, B::UnboundedPreceding, B::CurrentRow) => Ok(Shape::RunningRows),
            (FrameUnits::Range, B::UnboundedPreceding, B::CurrentRow) => Ok(Shape::RunningPeers),
            (FrameUnits::Rows, B::UnboundedPreceding | B::CurrentRow | B::Offset(_), upper)
                if !matches!(upper, B::UnboundedPreceding) =>
            {
                Ok(Shape::Frame)
            }
            (FrameUnits::Range, B::CurrentRow, B::UnboundedFollowing) => Ok(Shape::Frame),
            (units, lower, upper) => Err(crate::aggregate::unsupported(format!(
                "window frame {units:?} between {lower:?} and {upper:?}"
            ))),
        }
    }
}

/// Window functions over an input sorted by its partitioning, then its
/// ordering (see the module docs).
pub struct Window {
    env: Arc<Env>,
    /// Where the buffered rows spill to (none: they never spill).
    spill: Option<SpillScope>,
    partitions: AtomicU64,
    rows: AtomicU64,
    counters: Arc<spill::Counters>,
}

/// What evaluating a window takes, shared with the steps that run on the
/// blocking pool.
struct Env {
    input: SchemaRef,
    schema: SchemaRef,
    plan: WindowPlan,
    partition_keys: Option<Arc<KeyEncoder>>,
    order_keys: Option<Arc<KeyEncoder>>,
    /// `spark.sql.ansi.enabled`: overflow raises instead of wrapping or
    /// becoming NULL.
    ansi: bool,
}

impl std::fmt::Debug for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Window")
            .field("plan", &self.env.plan)
            .finish_non_exhaustive()
    }
}

/// One input batch with rows not emitted yet.
struct Pending {
    batch: RecordBatch,
    /// Rows already emitted.
    offset: usize,
    /// Per row: it starts a partition.
    part_start: Vec<bool>,
    /// Per row: it starts a peer group (or a partition).
    peer_start: Vec<bool>,
    /// The bytes reserved for it.
    reserved: usize,
}

struct Local {
    /// A row was taken (the next one is not the input's first).
    started: bool,
    pending: VecDeque<Pending>,
    /// Where the last partition seen starts: an index into `pending` and a
    /// row. Rows before it are complete.
    last_start: Option<(usize, usize)>,
    /// Rows of the partition open at the end of the last batch taken (a
    /// growing aggregate's snapshots are priced by position).
    open_len: usize,
    /// The keys of the last row taken, to compare the next batch's first
    /// row with.
    prev_partition: Option<OwnedRow>,
    prev_order: Option<OwnedRow>,
    /// Evaluated batches waiting to be emitted, each with the bytes its
    /// emission releases (an input batch's reservation goes with its last
    /// output slice).
    out: VecDeque<(Batch, usize)>,
    reservation: Reservation,
    /// The input ended.
    finished: bool,
    /// Complete rows below this many wait for more (the configured batch
    /// size).
    batch_rows: usize,
    /// Rows written to disk, in input order, while the input runs: from the
    /// first spill on, the rows wait for the flush phase (see
    /// [`spill`](self::spill)).
    spilled: Option<VecDeque<SpillFile>>,
    /// The flush phase: the spilled rows read back and evaluated.
    replay: Option<Box<spill::Replay>>,
    /// Rows were lost (a failed or cancelled spill or step): every call
    /// fails with this error, never with a short result.
    failed: Option<KrishivError>,
    signal: Arc<SpillSignal>,
    /// The operator has a spill scope.
    spillable: bool,
}

impl Local {
    /// Fails if rows were lost.
    fn check(&self) -> Result<()> {
        self.failed.clone().map_or(Ok(()), Err)
    }

    /// The bytes the buffered rows hold, which a spill frees.
    fn pending_bytes(&self) -> usize {
        self.pending.iter().map(|p| p.reserved).sum()
    }

    /// Publishes what a spill would free (nothing in the flush phase).
    fn publish(&self) {
        let bytes = if self.replay.is_some() || !self.spillable {
            0
        } else {
            self.pending_bytes()
        };
        self.signal.set_spillable_bytes(bytes);
    }
}

fn downcast(l: &mut LocalState) -> Result<&mut Local> {
    l.downcast_mut::<Local>()
        .ok_or_else(|| internal("wrong local state for Window".to_owned()))
}

impl Window {
    /// A window over batches of `input`, sorted by `plan`'s partitioning
    /// and ordering; with `spill`, its buffered rows spill there under
    /// memory pressure.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a column outside the input or a filter that is
    /// not boolean; `KRISHIV_UNSUPPORTED_FEATURE` for a key type the sort
    /// cannot order, a frame not evaluated yet, or an aggregate of a type
    /// it does not take.
    pub fn new(
        input: SchemaRef,
        plan: WindowPlan,
        ansi: bool,
        spill: Option<SpillScope>,
    ) -> Result<Self> {
        let width = input.fields().len();
        let column = |i: usize, what: &str| {
            if i < width {
                Ok(())
            } else {
                Err(internal(format!(
                    "window {what} column {i} is out of range"
                )))
            }
        };
        let partition: Vec<SortKey> = plan
            .partition_by
            .iter()
            .map(|&c| column(c, "partition").map(|()| SortKey::asc(c)))
            .collect::<Result<_>>()?;
        for k in &plan.order_by {
            column(k.column, "order")?;
        }
        let mut fields: Vec<Field> = input.fields().iter().map(|f| f.as_ref().clone()).collect();
        for f in &plan.functions {
            if let WindowFunction::Aggregate(a) = &f.function {
                for &arg in &a.args {
                    column(arg, "argument")?;
                }
                f.shape()?;
            }
            if let WindowFunction::Ntile(n) = f.function
                && n <= 0
            {
                return Err(internal(format!("ntile of {n} buckets")));
            }
            match &f.function {
                WindowFunction::Offset {
                    input: i, default, ..
                } => {
                    column(*i, "input")?;
                    if let Some(d) = default {
                        column(*d, "default")?;
                    }
                }
                WindowFunction::NthValue { input: i, n, .. } => {
                    column(*i, "input")?;
                    f.shape()?;
                    if *n <= 0 {
                        return Err(internal(format!("nth_value of row {n}")));
                    }
                }
                _ => {}
            }
            if let Some(c) = f.filter {
                column(c, "filter")?;
                if input.field(c).data_type() != &DataType::Boolean {
                    return Err(internal("a window filter that is not boolean".to_owned()));
                }
            }
            let (t, nullable) = f.output()?;
            fields.push(Field::new(&f.name, t, nullable));
        }
        let encoder = |keys: &[SortKey]| {
            (!keys.is_empty())
                .then(|| KeyEncoder::new(&input, keys))
                .transpose()
        };
        Ok(Self {
            env: Arc::new(Env {
                partition_keys: encoder(&partition)?,
                order_keys: encoder(&plan.order_by)?,
                schema: Arc::new(Schema::new(fields)),
                input,
                plan,
                ansi,
            }),
            spill,
            partitions: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            counters: Arc::default(),
        })
    }

    /// Takes `batch` if the bytes it holds while buffered can be reserved
    /// (`Err(bytes)` otherwise, nothing changed): its columns, its boundary
    /// flags against the rows before it, and what evaluating its rows
    /// holds ([`eval::cost`], exact for its rows, partitions and peer
    /// groups). `ingest`: the rows are the operator's input (counted in its
    /// metrics), not spilled rows read back in the flush phase.
    fn append(&self, l: &mut Local, batch: RecordBatch, ingest: bool) -> Result<Result<(), usize>> {
        let n = batch.num_rows();
        let flags = |enc: &Option<Arc<KeyEncoder>>,
                     prev: &Option<OwnedRow>|
         -> Result<(Vec<bool>, Option<OwnedRow>)> {
            let Some(enc) = enc else {
                return Ok((vec![false; n], None));
            };
            let rows = enc.encode(&batch)?;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(match i {
                    0 => prev.as_ref().is_none_or(|p| p.row() != rows.row(0)),
                    _ => rows.row(i) != rows.row(i - 1),
                });
            }
            let last = (n > 0).then(|| rows.row(n - 1).owned());
            Ok((out, last))
        };
        let env = &*self.env;
        let (mut part_start, prev_partition) = flags(&env.partition_keys, &l.prev_partition)?;
        let (order_start, prev_order) = flags(&env.order_keys, &l.prev_order)?;
        if !l.started
            && let Some(first) = part_start.first_mut()
        {
            // The first row of the input starts a partition.
            *first = true;
        }
        let peer_start: Vec<bool> = part_start
            .iter()
            .zip(&order_start)
            .map(|(p, o)| *p || *o)
            .collect();
        let parts = part_start.iter().filter(|&&s| s).count();
        let peers = peer_start.iter().filter(|&&s| s).count();
        let (held, open_len) = eval::snapshot_slots(&part_start, l.open_len);
        let cost = batch
            .get_array_memory_size()
            .saturating_add(2 * n)
            .saturating_add(eval::cost(&env.plan, n, parts, peers, held)?);
        if l.reservation.try_grow(cost).is_err() {
            return Ok(Err(cost));
        }
        l.open_len = open_len;
        l.started |= n > 0;
        if prev_partition.is_some() {
            l.prev_partition = prev_partition;
        }
        if prev_order.is_some() {
            l.prev_order = prev_order;
        }
        if let Some(i) = part_start.iter().rposition(|&s| s) {
            l.last_start = Some((l.pending.len(), i));
        }
        if ingest {
            self.partitions.fetch_add(parts as u64, Ordering::Relaxed);
            self.rows.fetch_add(n as u64, Ordering::Relaxed);
        }
        l.pending.push_back(Pending {
            batch,
            offset: 0,
            part_start,
            peer_start,
            reserved: cost,
        });
        Ok(Ok(()))
    }

    /// The pending rows that are complete partitions: everything when the
    /// input ended, else everything before the last partition's start.
    fn complete_rows(l: &Local) -> usize {
        let upto = if l.finished {
            None
        } else {
            match l.last_start {
                Some(at) => Some(at),
                None => return 0,
            }
        };
        let mut total = 0;
        for (i, p) in l.pending.iter().enumerate() {
            match upto {
                Some((bi, row)) if bi == i => {
                    total += row.saturating_sub(p.offset);
                    break;
                }
                _ => total += p.batch.num_rows() - p.offset,
            }
        }
        total
    }

    /// Evaluates the complete rows and queues their output batches.
    fn flush_complete(&self, l: &mut Local) -> Result<()> {
        let rows = Self::complete_rows(l);
        if rows == 0 {
            return Ok(());
        }
        // The slices of pending batches the unit covers.
        let mut slices = Vec::new();
        let mut left = rows;
        for (i, p) in l.pending.iter().enumerate() {
            if left == 0 {
                break;
            }
            let len = (p.batch.num_rows() - p.offset).min(left);
            slices.push((i, p.offset, len));
            left -= len;
        }
        let columns = {
            let unit: Vec<eval::Slice<'_>> = slices
                .iter()
                .map(|&(i, offset, len)| eval::Slice {
                    batch: &l.pending[i].batch,
                    offset,
                    len,
                    part_start: &l.pending[i].part_start[offset..offset + len],
                    peer_start: &l.pending[i].peer_start[offset..offset + len],
                })
                .collect();
            eval::evaluate(&self.env.plan, &unit, self.env.ansi)?
        };
        // One output batch per slice: its input columns, then the slice of
        // each window column.
        let options = |n: usize| RecordBatchOptions::new().with_row_count(Some(n));
        let mut at = 0;
        for &(i, offset, len) in &slices {
            let p = &l.pending[i];
            let mut cols: Vec<_> = p
                .batch
                .columns()
                .iter()
                .map(|c| c.slice(offset, len))
                .collect();
            cols.extend(columns.iter().map(|c| c.slice(at, len)));
            at += len;
            let batch = RecordBatch::try_new_with_options(
                Arc::clone(&self.env.schema),
                cols,
                &options(len),
            )
            .map_err(|e| internal(format!("window produced an invalid batch: {e}")))?;
            l.out.push_back((Batch::new(batch), 0));
        }
        // Advance past the evaluated rows; a fully evaluated input batch
        // leaves, its reservation released with its last output slice.
        let mut dropped = 0;
        let first_out = l.out.len() - slices.len();
        for (k, &(i, offset, len)) in slices.iter().enumerate() {
            let p = &mut l.pending[i - dropped];
            p.offset = offset + len;
            if p.offset == p.batch.num_rows() {
                let gone = l.pending.pop_front().map_or(0, |p| p.reserved);
                if let Some((_, release)) = l.out.get_mut(first_out + k) {
                    *release += gone;
                }
                dropped += 1;
            }
        }
        if let Some((bi, row)) = l.last_start {
            l.last_start = bi.checked_sub(dropped).map(|bi| (bi, row));
        }
        Ok(())
    }
}

impl Window {
    /// The next evaluated batch, releasing what its emission frees.
    fn emit(l: &mut Local) -> Option<Batch> {
        let (b, release) = l.out.pop_front()?;
        l.reservation.shrink(release);
        Some(b)
    }
}

impl Operator for Window {
    fn name(&self) -> &'static str {
        "Window"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.env.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        let consumer: Arc<ConsumerHandle> = Arc::clone(ctx.consumer().consumer());
        Ok(Box::new(Local {
            started: false,
            spilled: None,
            replay: None,
            failed: None,
            signal: Arc::clone(ctx.consumer().signal()),
            spillable: self.spill.is_some(),
            pending: VecDeque::new(),
            last_start: None,
            open_len: 0,
            prev_partition: None,
            prev_order: None,
            out: VecDeque::new(),
            reservation: consumer.reservation(),
            finished: false,
            batch_rows: ctx.query().config().batch_size.max(1),
        }))
    }

    fn execute(&self, input: &Batch, l: &mut LocalState) -> Result<OperatorOutput> {
        let l = downcast(l)?;
        l.check()?;
        if let Some(b) = Self::emit(l) {
            return Ok(OperatorOutput::HaveMoreOutput(b));
        }
        if input.num_rows() == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        if !matches!(input.weights(), Weights::Const(1)) {
            return Err(crate::aggregate::unsupported(
                "window functions over rows with weights other than 1 (needs incremental \
                 view maintenance)"
                    .to_owned(),
            ));
        }
        if input.data().schema().fields().len() != self.env.input.fields().len() {
            return Err(internal("window input of the wrong width".to_owned()));
        }
        if let Err(bytes) = self.append(l, input.data().clone(), true)? {
            return Ok(OperatorOutput::NeedMemory { bytes });
        }
        // After a spill, rows wait for the flush phase: the spilled ones
        // come first.
        if l.spilled.is_none() && Self::complete_rows(l) >= l.batch_rows {
            self.flush_complete(l)?;
        }
        l.publish();
        Ok(OperatorOutput::NeedMoreInput(Self::emit(l)))
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        let _ = target;
        Box::pin(async move { self.spill_pending(downcast(local)?, ctx, false).await })
    }

    fn finish(&self, l: &mut LocalState) -> Result<Option<Batch>> {
        let l = downcast(l)?;
        l.check()?;
        if let Some(b) = Self::emit(l) {
            return Ok(Some(b));
        }
        if l.spilled.is_some() {
            // The rest is evaluated in the flush phase.
            return Ok(None);
        }
        if !std::mem::replace(&mut l.finished, true) {
            self.flush_complete(l)?;
        }
        l.publish();
        Ok(Self::emit(l))
    }

    fn needs_flush(&self) -> bool {
        self.spill.is_some()
    }

    fn end_input<'a>(
        &'a self,
        local: &'a mut LocalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.start_replay(downcast(local)?, ctx).await })
    }

    fn flush<'a>(
        &'a self,
        local: &'a mut LocalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<FlushOutput>> {
        Box::pin(async move { self.replay(downcast(local)?, ctx).await })
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        crate::metrics::counts([
            ("partitions", self.partitions.load(Ordering::Relaxed)),
            ("rows", self.rows.load(Ordering::Relaxed)),
            ("spills", self.counters.spills.load(Ordering::Relaxed)),
            (
                "spill_file_bytes",
                self.counters.bytes.load(Ordering::Relaxed),
            ),
            (
                "external_partitions",
                self.counters.external.load(Ordering::Relaxed),
            ),
        ])
    }
}
