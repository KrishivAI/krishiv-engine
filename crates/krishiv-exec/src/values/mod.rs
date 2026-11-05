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
//!   nothing. A partial alone is never applied: one build partition's
//!   filter would drop the probe rows of the other partitions' keys.
//! - A runtime filter's value is the [`KeyFilter::merge`] of its partials,
//!   within the options of the join that publishes it
//!   ([`ValueService::filter_options`]); one disabled partial disables it.
//!
//! [`LocalValues`] is the implementation for operators that run in one
//! process; its cells are the delivery of values, never the contract.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use krishiv_io::{
    ColumnPath, DisabledReason, FilterId, FilterSnapshot, KeyFilter, KeyFilterOptions,
    RuntimeFilter,
};

use crate::error::{Result, internal};
pub use crate::physical::{ValueId, Version};
use crate::pipeline::ExecNode;

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

/// The input of a broadcast exchange as operator graphs of this process,
/// one per producer partition: the hash join over it builds its table once
/// and shares it among its drivers. It does not leave the process.
///
/// It names the inputs without owning them: an input can hold the service
/// it is published to (a join's filter publisher, a subquery's sink), and
/// owning it here would be a reference cycle that never frees the plan.
/// The instantiation that lowered an input owns it.
#[derive(Clone, Default)]
pub struct BroadcastBuild {
    inputs: Vec<Weak<dyn ExecNode>>,
}

impl BroadcastBuild {
    /// The build of one producer partition's `input`.
    #[must_use]
    pub fn new(input: &Arc<dyn ExecNode>) -> Self {
        Self {
            inputs: vec![Arc::downgrade(input)],
        }
    }

    /// The inputs in producer partition order; `None` once one was freed.
    #[must_use]
    pub fn inputs(&self) -> Option<Vec<Arc<dyn ExecNode>>> {
        self.inputs.iter().map(Weak::upgrade).collect()
    }
}

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

    /// The number of producer partitions of `id` this service waits for.
    fn producers(&self, id: ValueId) -> u32;

    /// Records the options of the join that publishes the runtime filter
    /// `id`: its partials merge within their caps and planned Bloom size.
    /// The first options recorded hold; until then the defaults do.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when `id` is not declared.
    fn filter_options(&self, id: FilterId, options: KeyFilterOptions) -> Result<()>;

    /// Takes one producer partition's partial.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an undeclared filter, a payload of another kind
    /// than its value, a pending filter, a producer partition out of range
    /// or a second version of a value.
    fn publish(&self, partial: Partial) -> Result<Publication>;

    /// Offers what a producer knows so far about the runtime filter `id` (a
    /// top-N's bound). The partials merge to the tightest, and any one alone
    /// is valid, so this neither waits for other producer partitions nor
    /// completes the value.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an undeclared filter.
    fn tighten(&self, id: FilterId, keys: Arc<KeyFilter>) -> Result<()>;

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
    /// What a runtime filter's partials merge within.
    options: OnceLock<KeyFilterOptions>,
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
    /// `INTERNAL_ERROR` for a value of no producer, or a subquery of more
    /// than one (a subquery's rows are gathered to one producer).
    pub fn with_producers(producers: BTreeMap<ValueId, u32>) -> Result<Self> {
        if let Some((id, _)) = producers.iter().find(|(_, n)| **n == 0) {
            return Err(internal(format!("{id} has no producer")));
        }
        if let Some((id, n)) = producers
            .iter()
            .find(|(id, n)| matches!(id, ValueId::Subquery(_)) && **n > 1)
        {
            return Err(internal(format!(
                "{id} has {n} producers; a subquery has one"
            )));
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
            options: OnceLock::new(),
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

    fn producers(&self, id: ValueId) -> u32 {
        self.producers.get(&id).copied().unwrap_or(1)
    }

    fn filter_options(&self, id: FilterId, options: KeyFilterOptions) -> Result<()> {
        let _ = self.get(ValueId::Filter(id))?.options.set(options);
        Ok(())
    }

    fn tighten(&self, id: FilterId, keys: Arc<KeyFilter>) -> Result<()> {
        self.filter(id)?.tighten(keys);
        Ok(())
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
        let options = entry.options.get().copied().unwrap_or_default();
        let value = merge(id, parts, &options)?;
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
/// (all of `id`'s kind); a runtime filter's merge within `options`.
fn merge(id: ValueId, mut parts: Vec<Value>, options: &KeyFilterOptions) -> Result<Value> {
    if parts.len() == 1
        && let Some(only) = parts.pop()
    {
        return Ok(only);
    }
    match id {
        ValueId::Filter(_) => merge_filters(parts, options).map(Value::Filter),
        ValueId::Subquery(_) => Err(internal(format!("{id}: a subquery has one producer"))),
        ValueId::Broadcast(_) => Ok(Value::Broadcast(BroadcastBuild {
            inputs: parts
                .into_iter()
                .flat_map(|p| match p {
                    Value::Broadcast(b) => b.inputs,
                    _ => Vec::new(),
                })
                .collect(),
        })),
    }
}

/// The filter of the union of the partials' keys: a disabled partial (a
/// failed producer) disables it, else the partials' filters merge.
fn merge_filters(parts: Vec<Value>, options: &KeyFilterOptions) -> Result<FilterSnapshot> {
    let mut merged: Option<Arc<KeyFilter>> = None;
    for p in parts {
        let keys = match p {
            Value::Filter(FilterSnapshot::Final(keys)) => keys,
            Value::Filter(FilterSnapshot::Disabled(r)) => return Ok(FilterSnapshot::Disabled(r)),
            // A top-N's bound is offered with `tighten`, never published.
            other => return Err(internal(format!("a runtime filter partial of {other:?}"))),
        };
        merged = Some(match merged {
            None => keys,
            // The accumulator is owned once the second partial arrives, so
            // the fold copies no Bloom filter.
            Some(m) => Arc::new(Arc::unwrap_or_clone(m).merge_into(&keys, options)?),
        });
    }
    Ok(merged.map_or(
        FilterSnapshot::Disabled(DisabledReason::Off),
        FilterSnapshot::Final,
    ))
}

#[cfg(test)]
mod tests;
