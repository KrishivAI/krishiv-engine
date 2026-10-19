// SPDX-License-Identifier: Apache-2.0
//! Hash join (P2-T052a; ADR 0011, E02): inner, left outer, left semi, left
//! anti and null-aware anti (`NOT IN`) equi-joins with an optional residual
//! filter and partitioned grace-hash spill.
//!
//! # Shape
//! The **build** side is the right input: a [`Sink`] ending a child
//! pipeline. The **probe** side is the left (preserved) input: an
//! [`Operator`] in the parent pipeline that reads the finalized build
//! through [`TaskContext::dependency`](crate::TaskContext::dependency).
//! Output columns are the probe columns, then (inner and left outer) every
//! build column; build columns are nullable in a left outer join.
//! [`HashJoinNode`] wires both.
//!
//! # Spark semantics
//! - Keys compared with `=` never match when NULL (the row is dropped by
//!   inner and semi joins, null-padded by left outer, kept by anti); keys
//!   marked null-safe (`<=>`) match NULL to NULL. Floats compare after
//!   normalizing `-0.0` and NaN, as Spark does for join keys; keys with
//!   floats inside nested types are rejected (`KRISHIV_UNSUPPORTED_FEATURE`)
//!   until they are normalized too.
//! - Semi and anti joins emit each probe row at most once, whatever the
//!   number of matching build rows; probe duplicates are kept.
//! - Null-aware anti (`a NOT IN (subquery)`, one key, no residual filter;
//!   Spark's `ExtractSingleColumnNullAwareAntiJoin`): an empty build side
//!   keeps every probe row (NULL keys too); a NULL key on the build side
//!   keeps none; otherwise a probe row is kept if its key is not NULL and
//!   matches no build key.
//! - The residual filter sees probe columns then build columns of each
//!   candidate pair; NULL is not a match.
//!
//! # Weights (Z-set multiplicities)
//! Probe weights pass through every join type (an output row derived from
//! one probe row carries its weight). Inner joins multiply probe and build
//! weights. A weighted **build** side is rejected for the other join types
//! (`KRISHIV_UNSUPPORTED_FEATURE`) until IVM (ADR 0017): whether a build key
//! "exists" depends on the net multiplicity of all its rows, which a hash
//! table of individual rows does not track.
//!
//! # Key formats (P2-T158)
//! By default keys are arrow-row bytes indexed by the shared `KeyTable`.
//! With `krishiv.exec.join.rowLayout` on (an internal setting, off by
//! default, read when the join is instantiated), a join whose keys are all
//! fixed-width (integers, dates, timestamps, decimals, floats, booleans)
//! stores them as fixed-width rows ([`RowLayout`](crate::rows::RowLayout):
//! Spark-normalized, NULL-safe; string keys keep arrow rows for now) with a
//! seedless stored hash `h`, and indexes each partition with a
//! [`JoinIndex`](crate::rows::JoinIndex): `hashbrown` while the index is
//! small, a salted linear-probing directory above that, both keyed by the
//! seeded `g = fin(h ^ seed)`. Results are the same in both formats.
//! Payload columns stay arrow-row bytes in both.
//!
//! # Memory and spill
//! Build rows are radix-partitioned (8 ways) as they arrive, into row-arena
//! chunks reserved in the driver's consumer. Arrow-row keys take the
//! partition from the shared `spill_partition` bits of their seeded hash;
//! row-layout keys take level 0 from the top bits of `h` and every deeper
//! grace level from bits of `g` (P2-T157 Appendix C). A spill
//! request writes whole partitions to spill files. If memory ran short at
//! all, `finalize` spills **every** partition, so the probe never pins tables
//! that fill the budget; otherwise it indexes them all. Probe rows of
//! spilled partitions are buffered (reserved, spillable) and written to
//! files. Once every probe driver's input ended, the drivers share out the
//! spilled partitions ([`Operator::flush`]): each loads a partition's build
//! rows and the headroom to read its probe rows back. Once one does not
//! fit, that driver becomes the only one that claims partitions: it waits
//! until no other holds one, then repartitions build and probe rows 8 ways
//! on the next hash bits, at most six levels deep, then fails with
//! `KRISHIV_JOIN_PARTITION_TOO_LARGE` (a partition that does not split is
//! skewed on very few keys). Built tables are not spilled while probing.
//!
//! Spill I/O needs scratch: a spill writer's 64 KiB buffer plus twice the
//! slice it encodes (writes are sliced to 64 KiB). A spill the arbitrator
//! requests funds it from the bytes being spilled, kept as grant credit
//! across writes (`Reservation::releasing_to_credit`), and writes at least
//! 256 KiB when it can. A consumer holding less than the scratch takes the
//! rest from the pool's free memory, or declines the request (frees
//! nothing, so the arbitrator looks elsewhere) instead of failing
//! mid-write; `end_input` likewise keeps rows it cannot fund in memory for
//! the grace phase. Only the build's `finalize` must write (once memory
//! ran short every partition goes to disk) and takes missing scratch from
//! the pool per write. Repartitioning reserves
//! its scratch and read-back headroom through arbitration first, and the
//! grace phase reserves the exact read-back headroom of a partition's probe
//! files when it loads the partition. Writers' scratch may also use the
//! pool's spill-scratch floor (`krishiv.memory.spillScratchReserve`).
//!
//! In the grace phase, drivers claim spilled partitions in parallel and
//! load them with free memory. A partition that does not fit next to the
//! others is retried by its driver alone (the others finish the partitions
//! they hold and wait, woken by an event, never polling), this time asking
//! the arbitrator for the memory (`FlushOutput::NeedMemory`: spillable
//! consumers such as an aggregate downstream spill, no query is aborted).
//! Only a partition that does not fit even then is repartitioned; after
//! that every driver claims in parallel again.
//!
//! # Runtime filters (design P2-T163)
//! An inner or left semi join drops every probe row whose key matches no
//! build key, so its build can publish what it learned about its keys to
//! the scans feeding the probe side ([`HashJoin::with_runtime_filters`]).
//! Each build driver keeps a partial per filter while it sinks rows (the
//! rows it stores: a key compared with `=` that is NULL is not stored):
//! min/max, a value set that stops past
//! `krishiv.runtimeFilter.inList.maxValues`, and the XXH64 hashes of the
//! normalized keys when a Bloom filter is planned, reserved in the driver's
//! consumer (a failed reservation drops the Bloom, never the query).
//! `combine` merges the partials, `finalize` builds each filter (a Bloom
//! sized exactly for the hashes collected), reserves it with the build and
//! publishes it before indexing the table. A build that fails or is
//! cancelled publishes `Disabled`, so no consumer waits for it.
//!
//! # Metrics (`EXPLAIN ANALYZE`, [`OperatorEntry::extra`](crate::OperatorEntry::extra))
//! Build sink: `build_rows`, `spilled_partitions`, `spill_file_bytes`. Probe:
//! `probe_rows`, `probe_spilled_rows`, `spill_file_bytes`,
//! `grace_partitions` (spilled partitions joined), `repartitions` and
//! `recursion_depth` (deepest level joined).

mod build;
mod grace;
mod keys;
mod nested;
mod probe;
mod table;

use std::fmt;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_row::{RowConverter, RowParser, SortField};
use arrow_schema::{Field, Schema, SchemaRef};
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_memory::SpillScope;
use krishiv_types::batch::{Batch, Weights};

pub use nested::{NestedLoopJoin, NestedLoopJoinNode, NestedLoopSpec};

use crate::error::{Result, internal};
use crate::operator::{Operator, Sink};
use crate::operators::Predicate;
use crate::pipeline::{ExecNode, PipelineBuilder};

/// The join type. The probe (left) side is preserved by the outer and anti
/// joins; the build (right) side never is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinType {
    /// Matching pairs.
    Inner,
    /// Matching pairs, and each unmatched probe row with NULL build columns.
    LeftOuter,
    /// Probe rows with at least one match.
    LeftSemi,
    /// Probe rows with no match (`NOT EXISTS`).
    LeftAnti,
    /// `NOT IN`: see the module docs.
    NullAwareAnti,
}

impl JoinType {
    fn emits_build(self) -> bool {
        matches!(self, Self::Inner | Self::LeftOuter)
    }
}

/// What a hash join joins.
#[derive(Debug)]
pub struct JoinSpec {
    /// The join type.
    pub join_type: JoinType,
    /// The probe (left) input's schema.
    pub probe_schema: SchemaRef,
    /// The build (right) input's schema.
    pub build_schema: SchemaRef,
    /// Probe key columns.
    pub probe_keys: Vec<usize>,
    /// Build key columns, pairwise of the same type as the probe keys.
    pub build_keys: Vec<usize>,
    /// Per key: compared with `<=>` (NULL matches NULL) instead of `=`.
    pub null_safe: Vec<bool>,
    /// The residual condition, over the probe columns then the build
    /// columns of each candidate pair; NULL does not match.
    pub filter: Option<Arc<dyn Predicate>>,
    /// Where to spill; `None` never spills (memory pressure then fails the
    /// query through the arbitrator).
    pub spill: Option<SpillScope>,
}

/// A join's `EXPLAIN ANALYZE` counters.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Counter {
    BuildRows,
    SpilledPartitions,
    BuildSpillBytes,
    ProbeRows,
    ProbeSpilledRows,
    ProbeSpillBytes,
    GracePartitions,
    Repartitions,
    RecursionDepth,
}

/// The build sink's counters, by name.
const BUILD_COUNTERS: [(Counter, &str); 3] = [
    (Counter::BuildRows, "build_rows"),
    (Counter::SpilledPartitions, "spilled_partitions"),
    (Counter::BuildSpillBytes, "spill_file_bytes"),
];
/// The probe operator's counters, by name.
const PROBE_COUNTERS: [(Counter, &str); 6] = [
    (Counter::ProbeRows, "probe_rows"),
    (Counter::ProbeSpilledRows, "probe_spilled_rows"),
    (Counter::ProbeSpillBytes, "spill_file_bytes"),
    (Counter::GracePartitions, "grace_partitions"),
    (Counter::Repartitions, "repartitions"),
    (Counter::RecursionDepth, "recursion_depth"),
];

/// Counters updated by every driver (relaxed: read once the query ended).
#[derive(Debug, Default)]
pub(crate) struct Counters([std::sync::atomic::AtomicU64; 9]);

impl Counters {
    pub(crate) fn add(&self, c: Counter, v: u64) {
        self.0[c as usize].fetch_add(v, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn max(&self, c: Counter, v: u64) {
        self.0[c as usize].fetch_max(v, std::sync::atomic::Ordering::Relaxed);
    }

    fn report(&self, which: &[(Counter, &'static str)]) -> Vec<(&'static str, u64)> {
        let get = |c: Counter| self.0[c as usize].load(std::sync::atomic::Ordering::Relaxed);
        which.iter().map(|&(c, name)| (name, get(c))).collect()
    }

    pub(crate) fn build(&self) -> Vec<(&'static str, u64)> {
        self.report(&BUILD_COUNTERS)
    }

    pub(crate) fn probe(&self) -> Vec<(&'static str, u64)> {
        self.report(&PROBE_COUNTERS)
    }
}

/// State shared by a join's build sink and probe operator.
pub(crate) struct Shared {
    join_type: JoinType,
    probe_keys: Vec<usize>,
    build_keys: Vec<usize>,
    keys: keys::KeyCodec,
    /// Encodes whole build rows when the join needs build columns.
    payload: Option<(RowConverter, RowParser)>,
    /// Build column types after decoding (dictionaries decode to values).
    build_fields: Vec<Field>,
    filter: Option<Arc<dyn Predicate>>,
    filter_schema: SchemaRef,
    schema: SchemaRef,
    probe_schema: SchemaRef,
    /// The row arena's schema with arrow-row keys.
    arrow_arena: SchemaRef,
    /// The row arena's schema with row-layout keys, if they have a layout.
    row_arena: Option<SchemaRef>,
    spill: Option<SpillScope>,
    counters: Counters,
    runtime: Vec<JoinRuntimeFilter>,
}

/// A runtime filter a hash join's build publishes (see the module docs).
#[derive(Debug, Clone)]
pub struct JoinRuntimeFilter {
    /// The join key it describes (an index into the key lists).
    pub key: usize,
    /// The filter, shared with the scans it targets.
    pub filter: Arc<krishiv_io::RuntimeFilter>,
    /// What to collect (the cost model's decisions).
    pub options: krishiv_io::KeyFilterOptions,
}

impl fmt::Debug for Shared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashJoin")
            .field("join_type", &self.join_type)
            .field("probe_keys", &self.probe_keys)
            .field("build_keys", &self.build_keys)
            .finish_non_exhaustive()
    }
}

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

impl Shared {
    fn new(spec: JoinSpec) -> Result<Self> {
        let JoinSpec {
            join_type,
            probe_schema,
            build_schema,
            probe_keys,
            build_keys,
            null_safe,
            filter,
            spill,
        } = spec;
        let n = probe_keys.len();
        if n == 0 || build_keys.len() != n || null_safe.len() != n {
            return Err(internal(
                "a hash join needs matching, non-empty key lists".to_owned(),
            ));
        }
        if join_type == JoinType::NullAwareAnti
            && (n != 1 || filter.is_some() || null_safe.contains(&true))
        {
            return Err(unsupported(
                "null-aware anti join over several keys or with a residual condition",
            ));
        }
        let mut types = Vec::with_capacity(n);
        for (&p, &b) in probe_keys.iter().zip(&build_keys) {
            let field = |s: &Schema, i: usize| {
                s.fields()
                    .get(i)
                    .cloned()
                    .ok_or_else(|| internal(format!("join key {i} out of range")))
            };
            let (pf, bf) = (field(&probe_schema, p)?, field(&build_schema, b)?);
            if pf.data_type() != bf.data_type() {
                return Err(internal(format!(
                    "join keys differ in type: {} vs {}",
                    pf.data_type(),
                    bf.data_type()
                )));
            }
            types.push(pf.data_type().clone());
        }
        let keys = keys::KeyCodec::new(&types, null_safe)?;
        let needs_payload = join_type.emits_build() || filter.is_some();
        let (payload, build_fields) = if needs_payload {
            let fields = build_schema
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()));
            let conv = RowConverter::new(fields.collect()).map_err(|e| internal(e.to_string()))?;
            let empty = conv
                .convert_rows(std::iter::empty())
                .map_err(|e| internal(e.to_string()))?;
            let fields = build_schema
                .fields()
                .iter()
                .zip(&empty)
                .map(|(f, a)| Field::new(f.name(), a.data_type().clone(), f.is_nullable()))
                .collect();
            let parser = conv.parser();
            (Some((conv, parser)), fields)
        } else {
            (None, Vec::new())
        };
        let joined = |nullable: bool| {
            let build = build_fields
                .iter()
                .map(|f| f.clone().with_nullable(f.is_nullable() || nullable));
            let fields: Vec<Field> = probe_schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .chain(build)
                .collect();
            Arc::new(Schema::new(fields))
        };
        let filter_schema = joined(false);
        let schema = match join_type {
            JoinType::Inner => joined(false),
            JoinType::LeftOuter => joined(true),
            _ => Arc::clone(&probe_schema),
        };
        let row_arena = match keys.layout() {
            Some(l) => Some(table::arena_schema(payload.is_some(), Some(l))?),
            None => None,
        };
        Ok(Self {
            join_type,
            probe_keys,
            build_keys,
            arrow_arena: table::arena_schema(payload.is_some(), None)?,
            row_arena,
            keys,
            payload,
            build_fields,
            filter,
            filter_schema,
            schema,
            probe_schema,
            spill,
            counters: Counters::default(),
            runtime: Vec::new(),
        })
    }

    /// The row arena's schema in the key format `hashing` chose.
    fn arena_schema(&self, hashing: &keys::Hashing) -> &SchemaRef {
        match (&self.row_arena, hashing.rows()) {
            (Some(rows), true) => rows,
            _ => &self.arrow_arena,
        }
    }

    /// The key format of a join instance under `config`: the row layout
    /// when `krishiv.exec.join.rowLayout` is on and the keys have one.
    fn hashing(&self, config: &crate::ExecConfig) -> keys::Hashing {
        let rows = config.join_row_layout && self.keys.layout().is_some();
        keys::Hashing::new(config.seed, rows)
    }

    /// Decodes build rows (payload bytes) into build columns.
    fn decode<'a>(&self, rows: impl Iterator<Item = &'a [u8]>) -> Result<Vec<ArrayRef>> {
        let Some((conv, parser)) = &self.payload else {
            return Ok(Vec::new());
        };
        conv.convert_rows(rows.map(|b| parser.parse(b)))
            .map_err(|e| internal(e.to_string()))
    }
}

/// A record batch of `schema` with `rows` rows (also for no columns).
fn record(schema: &SchemaRef, columns: Vec<ArrayRef>, rows: usize) -> Result<RecordBatch> {
    let options = RecordBatchOptions::new().with_row_count(Some(rows));
    RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)
        .map_err(|e| internal(e.to_string()))
}

/// The rows `idx` of `batch`, with their weights.
fn take_batch(batch: &Batch, idx: &UInt32Array) -> Result<Batch> {
    let data = arrow_select::take::take_record_batch(batch.data(), idx)
        .map_err(|e| internal(e.to_string()))?;
    let weights = match batch.weights() {
        Weights::Const(w) => Weights::Const(*w),
        Weights::Column(c) => {
            let taken =
                arrow_select::take::take(c, idx, None).map_err(|e| internal(e.to_string()))?;
            let taken =
                arrow_array::cast::AsArray::as_primitive_opt::<arrow_array::types::Int64Type>(
                    &taken,
                )
                .cloned()
                .ok_or_else(|| internal("weights are not Int64".to_owned()))?;
            Weights::Column(taken)
        }
    };
    Batch::with_weights(data, weights).map_err(|e| internal(e.to_string()))
}

/// A hash join: its build [`Sink`] and probe [`Operator`] share one
/// definition. See the module docs.
#[derive(Debug, Clone)]
pub struct HashJoin(Arc<Shared>);

impl HashJoin {
    /// Validates `spec`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for keys that are missing, out of range or of
    /// different types on the two sides; `KRISHIV_UNSUPPORTED_FEATURE` for a
    /// null-aware anti join over several keys, null-safe keys or with a
    /// residual filter (Spark plans those as a nested-loop join).
    pub fn try_new(spec: JoinSpec) -> Result<Self> {
        Ok(Self(Arc::new(Shared::new(spec)?)))
    }

    /// The same join publishing `filters` from its build (see the module
    /// docs). Call before the join is shared.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a join whose build cannot filter its probe
    /// side (only inner and left semi joins drop unmatched probe rows), a
    /// key out of range or a filter of another key type, or a join already
    /// shared.
    pub fn with_runtime_filters(mut self, filters: Vec<JoinRuntimeFilter>) -> Result<Self> {
        let shared = Arc::get_mut(&mut self.0)
            .ok_or_else(|| internal("runtime filters added to a shared join".to_owned()))?;
        if !filters.is_empty() && !matches!(shared.join_type, JoinType::Inner | JoinType::LeftSemi)
        {
            return Err(internal(format!(
                "a {:?} join cannot filter its probe side",
                shared.join_type
            )));
        }
        for f in &filters {
            // Key types are the same on both sides.
            let key = shared
                .probe_keys
                .get(f.key)
                .and_then(|&k| shared.probe_schema.fields().get(k).cloned())
                .ok_or_else(|| internal(format!("runtime filter key {} out of range", f.key)))?;
            if key.data_type() != f.filter.key_type() {
                return Err(internal(format!(
                    "runtime filter of type {} on a {} key",
                    f.filter.key_type(),
                    key.data_type()
                )));
            }
        }
        shared.runtime = filters;
        Ok(self)
    }

    /// The runtime filters the build publishes.
    pub fn runtime_filters(&self) -> &[JoinRuntimeFilter] {
        &self.0.runtime
    }

    /// The output schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.0.schema)
    }

    /// The build side's sink.
    pub fn build_sink(&self) -> Arc<dyn Sink> {
        Arc::new(build::BuildSink(Arc::clone(&self.0)))
    }

    /// The probe operator.
    pub fn probe(&self) -> Arc<dyn Operator> {
        Arc::new(probe::ProbeOp(Arc::clone(&self.0)))
    }
}

/// A plan node joining `probe` (left) with `build` (right).
pub struct HashJoinNode {
    /// The join.
    pub join: HashJoin,
    /// The build (right) input.
    pub build: Arc<dyn ExecNode>,
    /// The probe (left) input.
    pub probe: Arc<dyn ExecNode>,
}

impl fmt::Debug for HashJoinNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashJoinNode")
            .field("join", &self.join)
            .finish_non_exhaustive()
    }
}

impl ExecNode for HashJoinNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.child_pipeline(self.join.build_sink(), Arc::clone(&self.build))?;
        Arc::clone(&self.probe).build_pipelines(b)?;
        b.push_operator_reading(self.join.probe(), sink).map(drop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;

    fn spec(join_type: JoinType, keys: usize, filter: bool) -> JoinSpec {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        let all = |b: &RecordBatch| Ok(arrow_array::BooleanArray::from(vec![true; b.num_rows()]));
        JoinSpec {
            join_type,
            probe_schema: Arc::clone(&schema),
            build_schema: schema,
            probe_keys: (0..keys).collect(),
            build_keys: (0..keys).collect(),
            null_safe: vec![false; keys],
            filter: filter.then(|| Arc::new(crate::FnPredicate(all)) as Arc<dyn Predicate>),
            spill: None,
        }
    }

    #[test]
    fn specs_are_validated() {
        assert!(HashJoin::try_new(spec(JoinType::Inner, 0, false)).is_err());
        assert!(HashJoin::try_new(spec(JoinType::NullAwareAnti, 2, false)).is_err());
        assert!(HashJoin::try_new(spec(JoinType::NullAwareAnti, 1, true)).is_err());
        let mut s = spec(JoinType::Inner, 1, false);
        s.build_keys = vec![1];
        assert!(HashJoin::try_new(s).is_err(), "Int64 vs Utf8");
        let mut s = spec(JoinType::Inner, 1, false);
        s.build_keys = vec![5];
        assert!(HashJoin::try_new(s).is_err());
        let nested = DataType::List(Arc::new(Field::new("x", DataType::Float64, true)));
        let schema = Arc::new(Schema::new(vec![Field::new("a", nested, true)]));
        let mut s = spec(JoinType::Inner, 1, false);
        (s.probe_schema, s.build_schema) = (Arc::clone(&schema), schema);
        let e = HashJoin::try_new(s).err().map(|e| e.error_class());
        assert_eq!(
            e,
            Some(krishiv_common::error::ErrorClass::KrishivUnsupportedFeature)
        );
        let j =
            HashJoin::try_new(spec(JoinType::LeftOuter, 2, true)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(j.schema().fields().len(), 4);
        assert!(j.schema().field(2).is_nullable());
        let j =
            HashJoin::try_new(spec(JoinType::LeftSemi, 1, false)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(j.schema().fields().len(), 2);
        assert!(format!("{j:?}").contains("LeftSemi"));
    }

    /// The row layout is used only when asked for and when every key type
    /// has a row encoding; other joins keep arrow rows.
    #[test]
    fn the_row_layout_is_chosen_per_instance_and_key_types() {
        let on = crate::ExecConfig {
            join_row_layout: true,
            ..crate::ExecConfig::default()
        };
        let j =
            HashJoin::try_new(spec(JoinType::Inner, 1, false)).unwrap_or_else(|e| panic!("{e}"));
        assert!(j.0.hashing(&on).rows());
        assert!(!j.0.hashing(&crate::ExecConfig::default()).rows());
        // hash, key rows, payload.
        assert_eq!(j.0.arena_schema(&j.0.hashing(&on)).fields().len(), 3);
        // A string key keeps arrow rows.
        let j =
            HashJoin::try_new(spec(JoinType::Inner, 2, false)).unwrap_or_else(|e| panic!("{e}"));
        assert!(!j.0.hashing(&on).rows());
        let list = DataType::List(Arc::new(Field::new("x", DataType::Int64, true)));
        let schema = Arc::new(Schema::new(vec![Field::new("a", list, true)]));
        let mut s = spec(JoinType::Inner, 1, false);
        (s.probe_schema, s.build_schema) = (Arc::clone(&schema), schema);
        let j = HashJoin::try_new(s).unwrap_or_else(|e| panic!("{e}"));
        let h = j.0.hashing(&on);
        assert!(!h.rows());
        assert_eq!(
            j.0.arena_schema(&h).field(1).data_type(),
            &DataType::LargeBinary
        );
    }
}
