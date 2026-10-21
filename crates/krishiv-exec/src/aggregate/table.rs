// SPDX-License-Identifier: Apache-2.0
//! One aggregation hash table: distinct keys ([`KeyTable`]) and one
//! accumulator per aggregate, sized together and accounted as one.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_row::Rows;
use arrow_schema::SchemaRef;
use krishiv_types::batch::{Batch, Weights};

use super::accumulator::Accumulator;
use super::{AggregateOutput, AggregateSpec};
use crate::error::{Result, internal};
use crate::hash::{HashSeed, KeyEncoder, KeyTable, ProbeCount, ProbeTotals, spill_partition};

/// What every table of one aggregate shares.
#[derive(Debug)]
pub(crate) struct AggCore {
    pub(crate) spec: AggregateSpec,
    encoder: KeyEncoder,
    pub(crate) state_schema: SchemaRef,
    pub(crate) output_schema: SchemaRef,
    /// Each aggregate's columns in the state schema.
    ranges: Vec<Range<usize>>,
}

impl AggCore {
    pub(crate) fn new(spec: AggregateSpec) -> Result<Self> {
        let types: Vec<_> = spec
            .group_by
            .iter()
            .map(|&k| {
                let f = spec.input.fields().get(k);
                let f = f.ok_or_else(|| internal(format!("grouping column {k} out of range")))?;
                Ok(f.data_type().clone())
            })
            .collect::<Result<_>>()?;
        let encoder = KeyEncoder::new(&types)?;
        let state_schema = spec.state_schema()?;
        let output_schema = spec.output_schema()?;
        let mut ranges = Vec::with_capacity(spec.aggregates.len());
        let mut at = spec.group_by.len();
        for a in &spec.aggregates {
            let n = a.state_fields()?.len();
            ranges.push(at..at + n);
            at += n;
        }
        Ok(Self {
            spec,
            encoder,
            state_schema,
            output_schema,
            ranges,
        })
    }

    fn keys(&self) -> usize {
        self.spec.group_by.len()
    }

    /// Row indices of a state batch per spill partition at `level`.
    pub(crate) fn partitions_of(
        &self,
        batch: &RecordBatch,
        seed: &HashSeed,
        level: u32,
    ) -> Result<Vec<Vec<u32>>> {
        let mut parts = vec![Vec::new(); crate::hash::PARTITIONS];
        let rows = self.encode(&batch.columns()[..self.keys()], batch.num_rows())?;
        for (i, r) in rows.iter().enumerate() {
            let p = rows
                .as_ref()
                .map_or(0, |_| spill_partition(seed.hash(r), level));
            parts[p].push(u32::try_from(i).map_err(|_| internal("batch too large".to_owned()))?);
        }
        Ok(parts)
    }

    /// Encoded keys, or `None` for a global aggregate (one group).
    fn encode(&self, keys: &[ArrayRef], n: usize) -> Result<KeyRows> {
        if keys.is_empty() {
            return Ok(KeyRows(None, n));
        }
        Ok(KeyRows(Some(self.encoder.encode(keys)?), n))
    }
}

/// Encoded keys of a batch; `None` for a global aggregate.
pub(crate) struct KeyRows(Option<Rows>, usize);

impl KeyRows {
    fn bytes(&self) -> usize {
        self.0.as_ref().map_or(0, Rows::size)
    }

    fn iter(&self) -> impl Iterator<Item = &[u8]> {
        let n = if self.0.is_some() { 0 } else { self.1 };
        self.0
            .iter()
            .flat_map(|r| r.iter().map(|row| row.data()))
            .chain(std::iter::repeat_n(&[][..], n))
    }

    fn as_ref(&self) -> Option<&Rows> {
        self.0.as_ref()
    }
}

/// A batch prepared for [`AggTable::apply`]: its keys are encoded, so the
/// table can tell how much memory the batch may need first.
pub(crate) struct Prepared {
    keys: KeyRows,
    cols: Vec<ArrayRef>,
    weights: Weights,
    /// The batch is partial states in the state layout (a spill file or a
    /// re-partitioned run).
    states: bool,
}

/// Groups and their accumulators.
#[derive(Debug)]
pub(crate) struct AggTable {
    core: Arc<AggCore>,
    seed: HashSeed,
    keys: KeyTable,
    accs: Vec<Accumulator>,
    ids: Vec<u32>,
    /// Where probe lengths go, for a detailed profile only.
    probes: Option<Arc<ProbeTotals>>,
}

impl AggTable {
    pub(crate) fn new(core: &Arc<AggCore>, seed: &HashSeed) -> Result<Self> {
        let accs = core
            .spec
            .aggregates
            .iter()
            .map(|a| Accumulator::new(a, core.spec.ansi))
            .collect::<Result<Vec<_>>>()?;
        let mut t = Self {
            core: Arc::clone(core),
            seed: seed.clone(),
            keys: KeyTable::new(),
            accs,
            ids: Vec::new(),
            probes: None,
        };
        if core.keys() == 0 {
            // A global aggregate has its one group even with no input.
            t.accs.iter_mut().for_each(|a| a.resize(1));
        }
        Ok(t)
    }

    /// Counts probe lengths into `totals` (`ExecConfig::detailed_metrics`;
    /// `None`: not counted).
    pub(crate) fn with_probes(mut self, totals: Option<Arc<ProbeTotals>>) -> Self {
        self.probes = totals;
        self
    }

    /// Where this table counts probe lengths, for its successor.
    pub(crate) fn probes(&self) -> Option<Arc<ProbeTotals>> {
        self.probes.clone()
    }

    /// Buckets of the key index (its capacity).
    pub(crate) fn buckets(&self) -> usize {
        self.keys.buckets()
    }

    /// The key hash seed.
    pub(crate) fn seed(&self) -> &HashSeed {
        &self.seed
    }

    /// Whether spilling this table frees nothing worth a file: no groups,
    /// or a global aggregate's single group.
    pub(crate) fn not_worth_spilling(&self) -> bool {
        self.core.keys() == 0 || self.keys.is_empty()
    }

    /// Number of groups.
    pub(crate) fn len(&self) -> usize {
        if self.core.keys() == 0 {
            1
        } else {
            self.keys.len()
        }
    }

    /// Bytes held.
    pub(crate) fn memory_size(&self) -> usize {
        self.keys.memory_size()
            + self.ids.capacity() * size_of::<u32>()
            + self.accs.iter().map(Accumulator::size).sum::<usize>()
    }

    /// Encodes the keys of an input batch (`states`: a batch in the state
    /// layout, else the operator's input).
    pub(crate) fn prepare(&self, batch: &Batch, states: bool) -> Result<Prepared> {
        let core = &self.core;
        let cols = batch.data().columns();
        let keys: Vec<ArrayRef> = if states {
            cols[..core.keys().min(cols.len())].to_vec()
        } else {
            core.spec
                .group_by
                .iter()
                .map(|&k| {
                    cols.get(k)
                        .cloned()
                        .ok_or_else(|| internal(format!("grouping column {k} missing")))
                })
                .collect::<Result<_>>()?
        };
        Ok(Prepared {
            keys: core.encode(&keys, batch.num_rows())?,
            cols: cols.to_vec(),
            weights: batch.weights().clone(),
            states,
        })
    }

    /// [`memory_size`](Self::memory_size) after [`apply`](Self::apply)ing
    /// `p` (an upper bound).
    pub(crate) fn projected(&self, p: &Prepared) -> usize {
        let new = if self.core.keys() == 0 { 0 } else { p.keys.1 };
        self.keys.projected_size(new, p.keys.bytes())
            + crate::hash::grown(&self.ids, p.keys.1) * size_of::<u32>()
            + self
                .accs
                .iter()
                .zip(&self.core.ranges)
                .map(|(a, r)| {
                    let at = if p.states {
                        Some(r.start)
                    } else {
                        a.args().first().copied()
                    };
                    a.projected(new, at.and_then(|i| p.cols.get(i)))
                })
                .sum::<usize>()
    }

    /// Folds a prepared batch in. Call after reserving
    /// [`projected`](Self::projected) bytes.
    pub(crate) fn apply(&mut self, p: &Prepared) -> Result<()> {
        let n = p.keys.1;
        let mut ids = std::mem::take(&mut self.ids);
        if let Some(rows) = p.keys.as_ref() {
            self.keys.reserve(n, p.keys.bytes());
            self.accs.iter_mut().for_each(|a| a.reserve(n));
            match &self.probes {
                Some(totals) => {
                    let mut count = ProbeCount::default();
                    self.keys
                        .find_or_insert_counted(rows, &self.seed, &mut ids, &mut count)?;
                    totals.add(&count);
                }
                None => {
                    self.keys.find_or_insert(rows, &self.seed, &mut ids)?;
                }
            }
            let len = self.keys.len();
            self.accs.iter_mut().for_each(|a| a.resize(len));
        } else {
            ids.clear();
            ids.resize(n, 0);
        }
        let result = if p.states {
            let core = Arc::clone(&self.core);
            self.accs
                .iter_mut()
                .zip(&core.ranges)
                .try_for_each(|(a, r)| a.merge(&ids, &p.cols[r.clone()]))
        } else {
            self.accs
                .iter_mut()
                .try_for_each(|a| a.update(&ids, &p.cols, &p.weights))
        };
        self.ids = ids;
        result
    }

    /// The partial states of groups `ids`, in the state layout.
    pub(crate) fn state_batch(&self, ids: &[u32]) -> Result<RecordBatch> {
        let mut cols = self.key_columns(ids)?;
        for a in &self.accs {
            cols.extend(a.state(ids)?);
        }
        batch(&self.core.state_schema, cols, ids.len())
    }

    /// The node's output for groups `ids`: final values or partial states.
    pub(crate) fn output_batch(&self, ids: &[u32]) -> Result<RecordBatch> {
        if self.core.spec.output == AggregateOutput::Partial {
            return self.state_batch(ids);
        }
        let mut cols = self.key_columns(ids)?;
        for a in &self.accs {
            cols.push(a.evaluate(ids)?);
        }
        batch(&self.core.output_schema, cols, ids.len())
    }

    fn key_columns(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        if self.core.keys() == 0 {
            return Ok(Vec::new());
        }
        self.core
            .encoder
            .decode(&self.keys, ids.iter().map(|&i| i as usize))
    }

    /// The partial states of each row of `batch` as its own group: a
    /// partial aggregate's pass-through (bypass) output.
    pub(crate) fn states_of_rows(core: &AggCore, batch: &Batch) -> Result<RecordBatch> {
        let n = batch.num_rows();
        let cols = batch.data().columns();
        let mut out = Vec::new();
        for &k in &core.spec.group_by {
            out.push(
                cols.get(k)
                    .cloned()
                    .ok_or_else(|| internal(format!("grouping column {k} missing")))?,
            );
        }
        let ids: Vec<u32> =
            (0..u32::try_from(n).map_err(|_| internal("batch too large".to_owned()))?).collect();
        for a in &core.spec.aggregates {
            let mut acc = Accumulator::new(a, core.spec.ansi)?;
            acc.resize(n);
            acc.update(&ids, cols, batch.weights())?;
            out.extend(acc.state(&ids)?);
        }
        self::batch(&core.state_schema, out, n)
    }

    /// Group ids per spill partition at `level`.
    pub(crate) fn ids_by_partition(&self, level: u32) -> Vec<Vec<u32>> {
        let mut parts = vec![Vec::new(); crate::hash::PARTITIONS];
        if self.core.keys() == 0 {
            parts[0].push(0);
            return parts;
        }
        for (id, &h) in (0u32..).zip(self.keys.hashes()) {
            parts[spill_partition(h, level)].push(id);
        }
        parts
    }
}

fn batch(schema: &SchemaRef, cols: Vec<ArrayRef>, rows: usize) -> Result<RecordBatch> {
    RecordBatch::try_new_with_options(
        Arc::clone(schema),
        cols,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .map_err(|e| internal(e.to_string()))
}
