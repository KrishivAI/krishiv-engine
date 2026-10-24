// SPDX-License-Identifier: Apache-2.0
//! The window group limit (Spark's `WindowGroupLimit`): the rows of a
//! partition whose rank-like value (`row_number`, `rank`, `dense_rank`) is
//! at most `limit`, ahead of a window whose filter keeps only those.
//!
//! - [`WindowGroupLimit`] is the exact limit (Spark's `Final` mode): an
//!   operator over input sorted by the partitioning, then the ordering
//!   keys, as the window above it reads it.
//! - [`Ranker`] also prunes each sorted batch inside the sort below it
//!   (Spark's `Partial` mode, `operators::sort`). A row past the limit in
//!   a subset of its partition is past it in the whole partition: at least
//!   `limit` rows (`row_number`), rows ranked strictly before it (`rank`),
//!   or distinct ordering values before it (`dense_rank`) are in the subset
//!   already. For `row_number`, ties are ordered arbitrarily, as Spark
//!   orders them.
//!
//! Boundaries come from the sort's normalized key encoding, as the
//! window's do (`NULL = NULL`, every NaN equal, `-0.0 = 0.0`).
//!
//! **Weights:** defined for insert-only input; weighted input is
//! `KRISHIV_UNSUPPORTED_FEATURE`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::BooleanArray;
use arrow_row::{OwnedRow, Rows};
use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivUnsupportedFeature;
use krishiv_types::batch::Batch;

use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{LocalState, Operator, OperatorOutput};
use crate::operators::{KeyEncoder, SortKey};

/// A rank-like window function a group limit applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankLike {
    /// `row_number()`: one row per rank.
    RowNumber,
    /// `rank()`: peers share a rank, which then skips.
    Rank,
    /// `dense_rank()`: peers share a rank, without gaps.
    DenseRank,
}

impl RankLike {
    /// The function's name as Spark prints it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::RowNumber => "row_number",
            Self::Rank => "rank",
            Self::DenseRank => "dense_rank",
        }
    }
}

/// The rows of each partition ranked at most `limit` by `function` over
/// `order_by`.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupLimitPlan {
    /// The partitioning columns (none: one partition).
    pub partition_by: Vec<usize>,
    /// The ordering within a partition.
    pub order_by: Vec<SortKey>,
    /// The rank-like function.
    pub function: RankLike,
    /// The highest rank kept (at least 1).
    pub limit: u64,
}

crate::physical::data::plan_data!(
    enum RankLike {
        RowNumber,
        Rank,
        DenseRank,
    }
);
crate::physical::data::plan_data!(struct GroupLimitPlan { partition_by, order_by, function, limit });

impl GroupLimitPlan {
    /// The partitioning keys as the sort below orders them.
    pub(crate) fn partition_keys(&self) -> Vec<SortKey> {
        self.partition_by.iter().map(|&c| SortKey::asc(c)).collect()
    }

    /// Whether a sort by `keys` is the sort this limit reads: the
    /// partitioning keys ascending, then the ordering.
    pub(crate) fn sorted_by(&self, keys: &[SortKey]) -> bool {
        let mut want = self.partition_keys();
        want.extend(self.order_by.iter().copied());
        want == keys
    }
}

/// The rank of each row of one sorted stream, row by row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ranker {
    function: RankLike,
    limit: u64,
    /// Rows of the current partition so far.
    rows: u64,
    /// The current row's rank.
    rank: u64,
}

impl Ranker {
    pub(crate) const fn new(function: RankLike, limit: u64) -> Self {
        Self {
            function,
            limit,
            rows: 0,
            rank: 0,
        }
    }

    /// Ranks the next row, which starts a partition (`part_start`) or a
    /// peer group (`peer_start`, implied by `part_start`); returns whether
    /// it is within the limit.
    pub(crate) fn keep(&mut self, part_start: bool, peer_start: bool) -> bool {
        if part_start {
            self.rows = 0;
            self.rank = 0;
        }
        self.rows = self.rows.saturating_add(1);
        let new_peer = part_start || peer_start;
        match self.function {
            RankLike::RowNumber => self.rank = self.rows,
            RankLike::Rank if new_peer => self.rank = self.rows,
            RankLike::DenseRank if new_peer => self.rank = self.rank.saturating_add(1),
            RankLike::Rank | RankLike::DenseRank => {}
        }
        self.rank <= self.limit
    }

    /// Whether the function needs peer boundaries (`row_number` does not).
    pub(crate) fn needs_peers(&self) -> bool {
        self.function != RankLike::RowNumber
    }
}

/// The exact window group limit over input sorted by the plan's
/// partitioning, then its ordering (see the module docs).
#[derive(Debug)]
pub struct WindowGroupLimit {
    schema: SchemaRef,
    plan: GroupLimitPlan,
    partition_keys: Option<Arc<KeyEncoder>>,
    /// The ordering keys, for peer boundaries (not for `row_number`).
    order_keys: Option<Arc<KeyEncoder>>,
    rows_in: AtomicU64,
    rows_out: AtomicU64,
}

impl WindowGroupLimit {
    /// The limit `plan` over batches of `schema`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a column outside `schema` or a limit of 0;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a key type the sort cannot order.
    pub fn new(schema: SchemaRef, plan: GroupLimitPlan) -> Result<Self> {
        if plan.limit == 0 {
            return Err(internal("a window group limit of 0 rows".to_owned()));
        }
        let encoder = |keys: &[SortKey]| {
            (!keys.is_empty())
                .then(|| KeyEncoder::new(&schema, keys))
                .transpose()
        };
        let partition_keys = encoder(&plan.partition_keys())?;
        let order_keys = if Ranker::new(plan.function, plan.limit).needs_peers() {
            encoder(&plan.order_by)?
        } else {
            None
        };
        Ok(Self {
            schema,
            plan,
            partition_keys,
            order_keys,
            rows_in: AtomicU64::new(0),
            rows_out: AtomicU64::new(0),
        })
    }

    /// The plan.
    #[must_use]
    pub fn plan(&self) -> &GroupLimitPlan {
        &self.plan
    }
}

/// One driver's position in its sorted stream.
struct Local {
    ranker: Ranker,
    /// The keys of the last row seen, compared with the next batch's first.
    prev_partition: Option<OwnedRow>,
    prev_order: Option<OwnedRow>,
    /// A row was seen (the next one is not the stream's first).
    started: bool,
}

/// Per row of `rows`: it differs from the row before it (`prev` for the
/// first). Also returns the last row.
fn starts(rows: &Rows, prev: Option<&OwnedRow>) -> (Vec<bool>, Option<OwnedRow>) {
    let n = rows.num_rows();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match i {
            0 => prev.is_none_or(|p| p.row() != rows.row(0)),
            _ => rows.row(i) != rows.row(i - 1),
        });
    }
    let last = (n > 0).then(|| rows.row(n - 1).owned());
    (out, last)
}

impl Operator for WindowGroupLimit {
    fn name(&self) -> &'static str {
        "WindowGroupLimit"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, _ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(Local {
            ranker: Ranker::new(self.plan.function, self.plan.limit),
            prev_partition: None,
            prev_order: None,
            started: false,
        }))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        if !input.is_unweighted() {
            return Err(KrishivUnsupportedFeature {
                feature: "a window group limit over weighted (change stream) input".to_owned(),
            }
            .into());
        }
        let l = local
            .downcast_mut::<Local>()
            .ok_or_else(|| internal("wrong local state for WindowGroupLimit".to_owned()))?;
        let n = input.num_rows();
        if n == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let (part_start, last_partition) = match &self.partition_keys {
            Some(enc) => starts(&enc.encode(input.data())?, l.prev_partition.as_ref()),
            None => (vec![false; n], None),
        };
        let (peer_start, last_order) = match &self.order_keys {
            Some(enc) => starts(&enc.encode(input.data())?, l.prev_order.as_ref()),
            None => (vec![false; n], None),
        };
        let mut keep = Vec::with_capacity(n);
        for i in 0..n {
            // The stream's first row starts a partition.
            let part = part_start[i] || (i == 0 && !l.started);
            keep.push(l.ranker.keep(part, peer_start[i]));
        }
        l.started = true;
        if last_partition.is_some() {
            l.prev_partition = last_partition;
        }
        if last_order.is_some() {
            l.prev_order = last_order;
        }
        let kept = keep.iter().filter(|&&k| k).count();
        self.rows_in.fetch_add(n as u64, Ordering::Relaxed);
        self.rows_out.fetch_add(kept as u64, Ordering::Relaxed);
        let out = if kept == n {
            Some(input.clone())
        } else if kept == 0 {
            None
        } else {
            Some(input.filter(&BooleanArray::from(keep))?)
        };
        Ok(OperatorOutput::NeedMoreInput(out))
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        crate::metrics::counts([
            ("rows_in", self.rows_in.load(Ordering::Relaxed)),
            ("rows_kept", self.rows_out.load(Ordering::Relaxed)),
        ])
    }
}
