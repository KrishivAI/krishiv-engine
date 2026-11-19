// SPDX-License-Identifier: Apache-2.0
//! The rows of a Python grouped aggregate as its tasks yield them (design
//! P3-T085 §3.1, review Md2): each lane's task yields one value per group,
//! in group order, so the oldest groups whose keys are still pending are
//! joined with the values every lane has yielded so far and stored at once.
//! A driver keeps one key batch per input batch (the group-start rows taken
//! together), reserved until its last group is joined.
//!
//! The aligned kinds (a window aggregate, a scalar iterator) are the same
//! join with a row for a group: every input row is kept whole, and each
//! lane yields one value per row.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_schema::SchemaRef;
use krishiv_memory::Reservation;
use krishiv_types::batch::Batch;

use super::lane::{Lane, SharedMem};
use super::sink::{RelationLane, mismatch};
use crate::error::{Result, internal};
use crate::sync::lock;

/// The group keys of one input batch, not joined with their values yet.
pub(super) struct KeyPart {
    rows: RecordBatch,
    /// Rows already joined.
    offset: usize,
    _memory: KeyMem,
}

/// The memory of a [`KeyPart`]: its own, or the input batch's reservation
/// it shares with the frames cut from the same rows (the aligned kinds).
/// Held for its drop.
struct KeyMem {
    _own: Option<Reservation>,
    _shared: Option<SharedMem>,
}

impl KeyPart {
    fn left(&self) -> usize {
        self.rows.num_rows() - self.offset
    }
}

/// The group-start rows `starts` of `batch` projected to `keys`, as one
/// batch of `schema`; `None` without starts.
pub(super) fn key_rows(
    batch: &RecordBatch,
    keys: &[usize],
    schema: &SchemaRef,
    starts: &[u32],
) -> Result<Option<RecordBatch>> {
    if starts.is_empty() {
        return Ok(None);
    }
    let indices = UInt32Array::from(starts.to_vec());
    let mut columns = Vec::with_capacity(keys.len());
    for &k in keys {
        let column = batch
            .columns()
            .get(k)
            .ok_or_else(|| internal(format!("key column {k} out of range")))?;
        columns.push(
            arrow_select::take::take(column.as_ref(), &indices, None)
                .map_err(|e| internal(format!("aggregate keys: {e}")))?,
        );
    }
    let options = RecordBatchOptions::new().with_row_count(Some(starts.len()));
    RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)
        .map(Some)
        .map_err(|e| internal(format!("aggregate keys: {e}")))
}

/// A driver's pending groups.
#[derive(Default)]
pub(super) struct Pending {
    keys: VecDeque<KeyPart>,
    /// Groups started, and joined, so far.
    groups: usize,
    joined: usize,
}

impl Pending {
    /// Adds the keys of an input batch's new groups, reserved in `memory`.
    pub(super) fn push(&mut self, rows: RecordBatch, memory: Reservation) {
        self.groups += rows.num_rows();
        self.keys.push_back(KeyPart {
            rows,
            offset: 0,
            _memory: KeyMem {
                _own: Some(memory),
                _shared: None,
            },
        });
    }

    /// Adds an input batch's rows, each a group of one (the aligned kinds),
    /// sharing the batch's reservation `memory`.
    pub(super) fn push_shared(&mut self, rows: RecordBatch, memory: SharedMem) {
        self.groups += rows.num_rows();
        self.keys.push_back(KeyPart {
            rows,
            offset: 0,
            _memory: KeyMem {
                _own: None,
                _shared: Some(memory),
            },
        });
    }

    fn waiting(&self) -> usize {
        self.keys.iter().map(KeyPart::left).sum()
    }

    /// The first `n` pending keys, columns concatenated; parts used up are
    /// dropped (their memory with them).
    fn take_keys(&mut self, n: usize, schema: &SchemaRef) -> Result<Vec<ArrayRef>> {
        let mut parts: Vec<RecordBatch> = Vec::new();
        let mut want = n;
        while want > 0 {
            let Some(front) = self.keys.front_mut() else {
                return Err(internal("aggregate keys ran out".to_owned()));
            };
            let len = want.min(front.left());
            parts.push(front.rows.slice(front.offset, len));
            front.offset += len;
            want -= len;
            if front.left() == 0 {
                self.keys.pop_front();
            }
        }
        let keys = arrow_select::concat::concat_batches(schema, &parts)
            .map_err(|e| internal(format!("aggregate keys: {e}")))?;
        self.joined += n;
        Ok(keys.columns().to_vec())
    }
}

/// The rows a lane's results hold, each checked to be one column of the
/// lane's declared type.
fn yielded(lane: &RelationLane, state: &Lane) -> Result<usize> {
    let q = lock(&state.io);
    let mut rows = 0;
    for (b, _) in &q.outputs {
        check_value(lane, b)?;
        rows += b.num_rows();
    }
    Ok(rows)
}

fn check_value(lane: &RelationLane, b: &RecordBatch) -> Result<()> {
    if b.num_columns() != 1 {
        return Err(mismatch(
            lane,
            format!("{} columns where 1 was declared", b.num_columns()),
        ));
    }
    match lane.client.first() {
        Some(client) if b.column(0).data_type() == client => Ok(()),
        Some(client) => Err(mismatch(
            lane,
            format!("{} where {client} was declared", b.column(0).data_type()),
        )),
        None => Err(internal("an aggregate lane without a type".to_owned())),
    }
}

/// The first `n` values of a lane's results; batches used up are dropped
/// from the queue, their reservations into `memory`.
fn take_values(state: &Lane, n: usize, memory: &mut Vec<Reservation>) -> Result<ArrayRef> {
    let mut q = lock(&state.io);
    let mut parts: Vec<ArrayRef> = Vec::new();
    let mut want = n;
    while want > 0 {
        let Some((front, _)) = q.outputs.front_mut() else {
            return Err(internal("aggregate values ran out".to_owned()));
        };
        let len = want.min(front.num_rows());
        parts.push(Arc::clone(front.column(0)).slice(0, len));
        want -= len;
        if len == front.num_rows() {
            if let Some((_, r)) = q.pop_output() {
                memory.push(r);
            }
        } else {
            *front = front.slice(len, front.num_rows() - len);
            q.took(len);
        }
    }
    let refs: Vec<&dyn Array> = parts.iter().map(AsRef::as_ref).collect();
    arrow_select::concat::concat(&refs).map_err(|e| internal(format!("aggregate results: {e}")))
}

/// The output rows of every group whose values each lane has yielded: the
/// keys, then a value per lane (canonical carriers), with the memory of the
/// results they came from; `None` when no group is complete. `unit` is what
/// a group is, in errors ("group", or "row" for the aligned kinds).
pub(super) fn joined(
    pending: &mut Pending,
    key_schema: &SchemaRef,
    schema: &SchemaRef,
    lanes: &[RelationLane],
    states: &[Lane],
    unit: &str,
) -> Result<Option<(Batch, Vec<Reservation>)>> {
    let mut n = pending.waiting();
    for (lane, state) in lanes.iter().zip(states) {
        let rows = yielded(lane, state)?;
        if rows + pending.joined > pending.groups {
            return Err(too_many(lane, rows + pending.joined, pending.groups, unit));
        }
        n = n.min(rows);
    }
    if n == 0 {
        return Ok(None);
    }
    let mut columns = pending.take_keys(n, key_schema)?;
    let width = columns.len();
    let mut memory = Vec::new();
    for (i, (lane, state)) in lanes.iter().zip(states).enumerate() {
        let values = take_values(state, n, &mut memory)?;
        let (Some(client), Some(field)) = (lane.client.first(), schema.fields().get(width + i))
        else {
            return Err(internal("an aggregate result out of its schema".to_owned()));
        };
        columns.push(super::sink::recarried(
            lane,
            &values,
            client,
            field.data_type(),
        )?);
    }
    let options = RecordBatchOptions::new().with_row_count(Some(n));
    let out = RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)
        .map_err(|e| internal(format!("a Python aggregate result: {e}")))?;
    Ok(Some((Batch::new(out), memory)))
}

/// Refuses an aggregate driver whose tasks all ended with groups or values
/// left over: a task yielded another number of values than groups. The
/// error names the first lane whose total is not the number of groups (a
/// lane with the right count beside a short one is not the culprit).
pub(super) fn check_complete(
    pending: &Pending,
    lanes: &[RelationLane],
    states: &[Lane],
    unit: &str,
) -> Result<()> {
    for (lane, state) in lanes.iter().zip(states) {
        let total = yielded(lane, state)? + pending.joined;
        if total != pending.groups {
            return Err(too_many(lane, total, pending.groups, unit));
        }
    }
    // Every lane has a value for every group, so `joined` took them all:
    // groups left over here are the engine's error, not the UDF's.
    if pending.joined < pending.groups {
        return Err(internal(format!(
            "{} of {} Python aggregate groups left unjoined",
            pending.groups - pending.joined,
            pending.groups
        )));
    }
    Ok(())
}

fn too_many(
    lane: &RelationLane,
    rows: usize,
    groups: usize,
    unit: &str,
) -> krishiv_common::error::KrishivError {
    mismatch(
        lane,
        format!("{rows} values for {groups} {unit}s (one value per {unit})"),
    )
}
