// SPDX-License-Identifier: Apache-2.0
//! Values computed while a plan runs and read by other operators of it:
//! runtime filters, subquery results and broadcast builds (design P2-T171,
//! I3; P2-T177).
//!
//! A value is named by a plan-assigned [`ValueId`] and has a [`Version`]
//! (one for a batch query). Each producer partition publishes one
//! [`Partial`] of it; the value at a version exists once every producer
//! partition of that version published, and is then the merge of their
//! partials. Consumers [`subscribe`](ValueService::subscribe) and read the
//! newest complete version, never a partial on its own.
//!
//! # Contract
//! - A [`ValueService`] serves one instantiation of a plan (a task): the
//!   cells it creates are never shared with another instantiation, so a
//!   plan can be instantiated any number of times.
//! - The number of producer partitions of a value comes from the task,
//!   not the plan; it is 1 unless the task says otherwise.
//! - Partials are deduplicated by producer partition: the first partial a
//!   partition publishes is the one merged, and a later one (a retried
//!   attempt, or a producer's failure path after it published) is ignored.
//!   The attempt never enters the value.
//! - A runtime filter never makes its consumers wait: until the value is
//!   complete its cell stays [`FilterSnapshot::Pending`] and applies
//!   nothing.
//!
//! [`LocalValues`] is the implementation for operators that run in one
//! process; its cells are the delivery of values, never the contract.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use krishiv_io::{ColumnPath, DisabledReason, FilterId, FilterSnapshot, RuntimeFilter};

use crate::error::{Result, internal};
pub use crate::physical::{ValueId, Version};
use crate::pipeline::ExecNode;
use crate::planner::nodes::FanInNode;

/// One task's place among the producers of a value: its partition and
/// attempt. Only the partition enters a value (I10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Producer {
    /// The producer partition, below the value's producer count.
    pub partition: u32,
    /// The task attempt that published (diagnostics only).
    pub attempt: u32,
}

/// What a value holds.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Value {
    /// A runtime filter: [`FilterSnapshot::Final`] or
    /// [`FilterSnapshot::Disabled`].
    Filter(FilterSnapshot),
    /// A subquery's result: one row.
    Subquery(ArrayRef),
    /// A broadcast exchange's build input, as instantiated on this worker.
    Broadcast(BroadcastBuild),
}

impl Value {
    fn kind_of(&self, id: ValueId) -> bool {
        matches!(
            (self, id),
            (Self::Filter(_), ValueId::Filter(_))
                | (Self::Subquery(_), ValueId::Subquery(_))
                | (Self::Broadcast(_), ValueId::Broadcast(_))
        )
    }
}

/// The input of a broadcast exchange as an operator graph of this process:
/// the hash join over it builds its table once and shares it among its
/// drivers. It does not leave the process.
#[derive(Clone)]
pub struct BroadcastBuild(pub Arc<dyn ExecNode>);

impl fmt::Debug for BroadcastBuild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BroadcastBuild")
    }
}

/// One producer partition's contribution to a value at a version.
#[derive(Debug, Clone)]
pub struct Partial {
    /// The value.
    pub value: ValueId,
    /// The version it contributes to.
    pub version: Version,
    /// Who published it.
    pub producer: Producer,
    /// What it contributes.
    pub payload: Value,
}

/// What a [`ValueService::publish`] did with a partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Publication {
    /// Taken; the value waits for other producer partitions.
    Waiting,
    /// Taken; the value is complete.
    Complete,
    /// Its producer partition already published: ignored.
    Ignored,
}

/// A consumer's view of one value.
pub trait Subscription: Send + Sync + fmt::Debug {
    /// The newest complete version and its value; `None` while no version
    /// is complete. Never waits.
    fn current(&self) -> Option<(Version, Arc<Value>)>;
}

/// Where a plan's values are published and read (see the module docs).
pub trait ValueService: Send + Sync + fmt::Debug {
    /// Declares the runtime filter `id` on the column `target`, comparing
    /// in `key_type`: creates the cell its consumers read.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when `id` is already declared.
    fn declare_filter(&self, id: FilterId, target: ColumnPath, key_type: DataType) -> Result<()>;

    /// The cell of the runtime filter `id`: pending until the value is
    /// complete, then the merged filter.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when `id` is not declared.
    fn filter(&self, id: FilterId) -> Result<Arc<RuntimeFilter>>;

    /// Takes one producer partition's partial.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an undeclared filter, a payload of another kind
    /// than its value, a pending filter, a producer partition out of range
    /// or a second version of a value.
    fn publish(&self, partial: Partial) -> Result<Publication>;

    /// The view of `id` its consumers read.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an undeclared filter.
    fn subscribe(&self, id: ValueId) -> Result<Arc<dyn Subscription>>;
}

/// The values of one instantiation in this process (see the module docs).
#[derive(Debug, Default)]
pub struct LocalValues {
    /// Producer partitions per value; 1 for a value not listed.
    producers: BTreeMap<ValueId, u32>,
    entries: Mutex<HashMap<ValueId, Arc<Entry>>>,
}

#[derive(Debug)]
struct Entry {
    producers: u32,
    /// A runtime filter's cell.
    cell: Option<Arc<RuntimeFilter>>,
    arrived: Mutex<Arrived>,
    complete: OnceLock<(Version, Arc<Value>)>,
}

#[derive(Debug, Default)]
struct Arrived {
    version: Option<Version>,
    /// By producer partition: merged in partition order.
    parts: BTreeMap<u32, Value>,
}

impl Subscription for Entry {
    fn current(&self) -> Option<(Version, Arc<Value>)> {
        self.complete.get().cloned()
    }
}

impl LocalValues {
    /// Values of one producer partition each.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Values with the producer partitions `producers` lists (1 for a
    /// value not listed).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a value of no producer.
    pub fn with_producers(producers: BTreeMap<ValueId, u32>) -> Result<Self> {
        if let Some((id, _)) = producers.iter().find(|(_, n)| **n == 0) {
            return Err(internal(format!("{id} has no producer")));
        }
        Ok(Self {
            producers,
            entries: Mutex::default(),
        })
    }

    /// Declares the runtime filter of `cell` with that cell (a join built
    /// outside a plan publishes into the cells it was given).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when the filter is already declared.
    pub fn adopt_filter(&self, cell: Arc<RuntimeFilter>) -> Result<()> {
        let id = ValueId::Filter(cell.id());
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        if entries.contains_key(&id) {
            return Err(internal(format!("{id} declared twice")));
        }
        entries.insert(id, Arc::new(self.entry(id, Some(cell))));
        Ok(())
    }

    fn entry(&self, id: ValueId, cell: Option<Arc<RuntimeFilter>>) -> Entry {
        Entry {
            producers: self.producers.get(&id).copied().unwrap_or(1),
            cell,
            arrived: Mutex::default(),
            complete: OnceLock::new(),
        }
    }

    /// The entry of `id`: a filter's must be declared, other values' are
    /// created when first named.
    fn get(&self, id: ValueId) -> Result<Arc<Entry>> {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(e) = entries.get(&id) {
            return Ok(Arc::clone(e));
        }
        if let ValueId::Filter(f) = id {
            return Err(internal(format!("runtime filter {f} has no scan")));
        }
        let e = Arc::new(self.entry(id, None));
        entries.insert(id, Arc::clone(&e));
        Ok(e)
    }
}

impl ValueService for LocalValues {
    fn declare_filter(&self, id: FilterId, target: ColumnPath, key_type: DataType) -> Result<()> {
        self.adopt_filter(Arc::new(RuntimeFilter::new(id, target, key_type)))
            .map_err(|_| internal(format!("runtime filter {id} declared twice")))
    }

    fn filter(&self, id: FilterId) -> Result<Arc<RuntimeFilter>> {
        self.get(ValueId::Filter(id))?
            .cell
            .clone()
            .ok_or_else(|| internal(format!("runtime filter {id} without a cell")))
    }

    fn publish(&self, partial: Partial) -> Result<Publication> {
        let Partial {
            value: id,
            version,
            producer,
            payload,
        } = partial;
        if !payload.kind_of(id) {
            return Err(internal(format!("a partial of another kind for {id}")));
        }
        if matches!(payload, Value::Filter(FilterSnapshot::Pending)) {
            return Err(internal(format!("a pending partial for {id}")));
        }
        let entry = self.get(id)?;
        if producer.partition >= entry.producers {
            return Err(internal(format!(
                "producer partition {} of {id}, which has {}",
                producer.partition, entry.producers
            )));
        }
        let mut arrived = entry.arrived.lock().unwrap_or_else(PoisonError::into_inner);
        match arrived.version {
            Some(v) if v != version => {
                return Err(internal(format!(
                    "{id} published at epochs {} and {}",
                    v.epoch, version.epoch
                )));
            }
            _ => arrived.version = Some(version),
        }
        if entry.complete.get().is_some() || arrived.parts.contains_key(&producer.partition) {
            return Ok(Publication::Ignored);
        }
        arrived.parts.insert(producer.partition, payload);
        if u64::try_from(arrived.parts.len()).unwrap_or(u64::MAX) < u64::from(entry.producers) {
            return Ok(Publication::Waiting);
        }
        let parts = std::mem::take(&mut arrived.parts).into_values().collect();
        let value = merge(id, parts)?;
        if let (Some(cell), Value::Filter(snapshot)) = (&entry.cell, &value) {
            cell.publish(snapshot.clone());
        }
        // Set under `arrived`'s lock: one partial completes the value.
        let _ = entry.complete.set((version, Arc::new(value)));
        Ok(Publication::Complete)
    }

    fn subscribe(&self, id: ValueId) -> Result<Arc<dyn Subscription>> {
        Ok(self.get(id)? as Arc<dyn Subscription>)
    }
}

/// The value of `parts`, one per producer partition in partition order
/// (all of `id`'s kind).
fn merge(id: ValueId, mut parts: Vec<Value>) -> Result<Value> {
    if parts.len() == 1
        && let Some(only) = parts.pop()
    {
        return Ok(only);
    }
    match id {
        // A disabled partial disables the value. Filters of several
        // producer partitions do not merge yet: their value applies
        // nothing, which keeps every partial's keys.
        ValueId::Filter(_) => Ok(Value::Filter(FilterSnapshot::Disabled(
            parts
                .iter()
                .find_map(|p| match p {
                    Value::Filter(FilterSnapshot::Disabled(r)) => Some(*r),
                    _ => None,
                })
                .unwrap_or(DisabledReason::Off),
        ))),
        // The partitions' rows in partition order.
        ValueId::Subquery(_) => {
            let arrays: Vec<&dyn Array> = parts
                .iter()
                .filter_map(|p| match p {
                    Value::Subquery(a) => Some(a.as_ref()),
                    _ => None,
                })
                .collect();
            arrow_select::concat::concat(&arrays)
                .map(Value::Subquery)
                .map_err(|e| internal(format!("{id}: {e}")))
        }
        ValueId::Broadcast(_) => Ok(Value::Broadcast(BroadcastBuild(Arc::new(FanInNode {
            inputs: parts
                .into_iter()
                .filter_map(|p| match p {
                    Value::Broadcast(b) => Some(b.0),
                    _ => None,
                })
                .collect(),
        })))),
    }
}

#[cfg(test)]
mod tests;
