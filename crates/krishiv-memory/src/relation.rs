// SPDX-License-Identifier: Apache-2.0
//! Content-addressed batches shared between a session's tables and its
//! workers.
//!
//! A [`RelationStore`] holds the batches of the session's in-memory tables
//! decoded, shared ([`Arc<[Batch]>`]): registering the same batches twice
//! shares one entry, and different batches hash differently. Workers resolve
//! a [`RelationRef`] through the store and read a batch range of it, so task
//! splits name batches without ever inlining rows into plan bytes.
//!
//! The batches are memory-accounted: each entry holds a [`Reservation`]
//! grown by the batches' array memory plus their weights, from a consumer
//! the store registers on its own pool. The store's pool is unbounded: table
//! bytes were never charged against the query pool's budget (tight-limit
//! spill tests register large tables), so holding them must not consume it
//! either. Entries are reference-counted ([`RelationLease`]): an entry lives
//! while a table (or a snapshot pin of one) holds its lease, and its bytes
//! are released with the last lease.
//!
//! Arrow IPC is the wire form: [`RelationStore::encode_ipc`] encodes an
//! entry's batches on demand, and [`RelationStore::insert_ipc`] decodes wire
//! bytes (unweighted memory tables) back into the store, re-hashing them
//! through [`RelationStore::insert`]. Scans with no task-split entry read
//! the shared batches directly through [`RelationStore::resolve`]: cloning
//! a [`Batch`] only bumps refcounts, so the in-process path pays no copy.

use std::collections::HashMap;
use std::fmt;
use std::io::Cursor;
use std::ops::Range;
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::SchemaRef;
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_types::batch::{Batch, Weights};
use sha2::{Digest, Sha256};

use crate::{ConsumerHandle, MemoryConfig, MemoryPool, Reservation, TreePool};

/// The content hash of a relation's batches: SHA-256 over the relation's
/// canonical bytes (its Arrow IPC bytes, which carry the schema and the
/// values, plus its weights).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RelationHash([u8; 32]);

impl RelationHash {
    /// A hash of these bytes (tests name relations the store never held).
    #[must_use]
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The hash as 64 hexadecimal characters (as Spark Connect addresses its
    /// cached local relations).
    #[must_use]
    pub fn hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for b in self.0 {
            out.push(HEX[usize::from(b >> 4)] as char);
            out.push(HEX[usize::from(b & 0xF)] as char);
        }
        out
    }
}

impl fmt::Debug for RelationHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RelationHash({})", self.hex())
    }
}

impl fmt::Display for RelationHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

/// A reference to a relation in a [`RelationStore`]: its content hash.
/// Task splits carry this, never rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelationRef {
    /// The content hash of the relation's batches.
    pub hash: RelationHash,
}

/// The weights of one stored batch.
#[derive(Debug, Clone, PartialEq)]
enum StoredWeights {
    /// Every row has this weight.
    Const(i64),
    /// One weight per row.
    Column(Vec<i64>),
}

impl StoredWeights {
    /// The weights of `batch`.
    fn of(batch: &Batch) -> Self {
        match batch.weights() {
            Weights::Const(c) => Self::Const(*c),
            Weights::Column(column) => Self::Column(column.values().to_vec()),
        }
    }

    /// The bytes hashed with the data: the constant, or the values.
    fn hash_bytes(&self, out: &mut Vec<u8>) {
        match self {
            Self::Const(c) => {
                out.push(0);
                out.extend_from_slice(&c.to_le_bytes());
            }
            Self::Column(values) => {
                out.push(1);
                out.extend_from_slice(&values.len().to_le_bytes());
                for v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
}

/// One stored relation: its schema and its batches, shared.
struct Entry {
    schema: SchemaRef,
    batches: Arc<[Batch]>,
    /// The bytes, held against the pool while the entry lives.
    _reservation: Reservation,
}

/// The live entries of a [`RelationStore`], by hash.
struct Entries {
    /// Entries with their owner count (leases held).
    map: HashMap<RelationHash, (Entry, usize)>,
}

/// A content-addressed store of relations' batches (see the module docs).
pub struct RelationStore {
    entries: Mutex<Entries>,
    consumer: Arc<ConsumerHandle>,
}

impl fmt::Debug for RelationStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self
            .entries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .map
            .len();
        f.debug_struct("RelationStore")
            .field("relations", &len)
            .finish_non_exhaustive()
    }
}

impl RelationStore {
    /// A store with its own unbounded pool: the bytes are tracked and
    /// released through reservations, but holding tables never consumes
    /// the query pool's budget.
    #[must_use]
    pub fn new() -> Arc<Self> {
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
            limit: None,
            ..MemoryConfig::default()
        });
        Arc::new(Self {
            entries: Mutex::new(Entries {
                map: HashMap::new(),
            }),
            consumer: ConsumerHandle::register(pool, "relations", None),
        })
    }

    /// The number of relations stored (tests and metrics).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .map
            .len()
    }

    /// Whether the store holds no relation.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bytes the store's entries hold now (tests and metrics).
    #[must_use]
    pub fn reserved(&self) -> usize {
        self.consumer.held()
    }

    /// Stores `batches` (of `schema`) and returns its lease: the content
    /// hash, shared with an identical registration. The batches are held
    /// decoded and shared, memory-accounted by their array memory plus
    /// their weights; the lease releases the entry with the last lease.
    ///
    /// # Errors
    /// `UNABLE_TO_ACQUIRE_MEMORY` if the pool cannot grant the bytes (the
    /// store's pool is unbounded, so only a shut-down pool refuses);
    /// `INTERNAL_ERROR` for an Arrow encoding failure.
    pub fn insert(
        self: &Arc<Self>,
        schema: &SchemaRef,
        batches: &[Batch],
    ) -> Result<RelationLease, KrishivError> {
        let ipc = encode(schema, batches)?;
        let stored: Vec<StoredWeights> = batches.iter().map(StoredWeights::of).collect();
        let hash = hash_relation(&ipc, &stored);
        let mut reservation = self.consumer.reservation();
        reservation.try_grow(decoded_bytes(batches))?;
        let entry = Entry {
            schema: Arc::clone(schema),
            batches: Arc::from(batches.to_vec()),
            _reservation: reservation,
        };
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        match entries.map.get_mut(&hash) {
            Some((_, owners)) => {
                *owners += 1;
            }
            None => {
                entries.map.insert(hash, (entry, 1));
            }
        }
        Ok(RelationLease {
            store: Arc::clone(self),
            hash,
            batches: batches.len(),
        })
    }

    /// The requested batch range of `hash`, sharing the stored batches:
    /// cloning a [`Batch`] only bumps refcounts, so scans pay no copy and
    /// no decode. A scan with no task-split entry resolves its whole range
    /// this way.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an unknown hash (a coordinator bug: every hash
    /// in a task was registered by the session) or a range past the
    /// relation's batches.
    pub fn resolve(
        &self,
        hash: &RelationHash,
        range: Range<usize>,
    ) -> Result<Vec<Batch>, KrishivError> {
        // Clone the shared batches under the lock; cloning out of the range
        // afterwards only bumps refcounts.
        let batches = {
            let entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
            let Some((entry, _)) = entries.map.get(hash) else {
                return Err(internal(format!(
                    "a relation {hash} the store does not hold"
                )));
            };
            Arc::clone(&entry.batches)
        };
        if range.start > range.end || range.end > batches.len() {
            return Err(internal(format!(
                "a batch range {}..{} over {} batches",
                range.start,
                range.end,
                batches.len()
            )));
        }
        Ok(batches[range].to_vec())
    }

    /// The batches of `hash` as Arrow IPC bytes (the wire form): a remote
    /// worker decodes them with [`Self::insert_ipc`].
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an unknown hash or an Arrow encoding failure.
    pub fn encode_ipc(&self, hash: &RelationHash) -> Result<Vec<u8>, KrishivError> {
        let (schema, batches) = {
            let entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
            let Some((entry, _)) = entries.map.get(hash) else {
                return Err(internal(format!(
                    "a relation {hash} the store does not hold"
                )));
            };
            (Arc::clone(&entry.schema), Arc::clone(&entry.batches))
        };
        encode(&schema, &batches)
    }

    /// Stores wire bytes from [`Self::encode_ipc`] and returns its lease.
    /// The bytes are decoded (memory tables are unweighted, so every batch
    /// gets weight 1) and re-hashed through [`Self::insert`], which shares
    /// an identical registration; the returned lease's hash is the content
    /// hash of the decoded batches.
    ///
    /// # Errors
    /// `UNABLE_TO_ACQUIRE_MEMORY` if the pool cannot grant the bytes;
    /// `INTERNAL_ERROR` for undecodable bytes.
    pub fn insert_ipc(self: &Arc<Self>, ipc: &[u8]) -> Result<RelationLease, KrishivError> {
        let (schema, records) = decode_records(ipc)?;
        let batches: Vec<Batch> = records.into_iter().map(Batch::new).collect();
        self.insert(&schema, &batches)
    }

    /// Releases one owner of `hash`, dropping its entry (and its bytes)
    /// with the last one.
    fn release(&self, hash: &RelationHash) {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        let remove = match entries.map.get_mut(hash) {
            Some((_, owners)) => {
                *owners = owners.saturating_sub(1);
                *owners == 0
            }
            None => false,
        };
        if remove {
            entries.map.remove(hash);
        }
    }

    /// Adds an owner of `hash` (a cloned lease). Unknown hashes stay
    /// unknown: only [`Self::insert`] creates entries.
    fn retain(&self, hash: &RelationHash) {
        let mut entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, owners)) = entries.map.get_mut(hash) {
            *owners += 1;
        }
    }
}

/// A registration of batches in a [`RelationStore`]: cloning shares the
/// entry, and dropping the last lease releases its bytes. A table holds
/// one, so the relation lives exactly while the table does.
pub struct RelationLease {
    store: Arc<RelationStore>,
    hash: RelationHash,
    batches: usize,
}

impl RelationLease {
    /// What task splits carry.
    #[must_use]
    pub fn reference(&self) -> RelationRef {
        RelationRef { hash: self.hash }
    }

    /// The store holding the relation.
    #[must_use]
    pub fn store(&self) -> &Arc<RelationStore> {
        &self.store
    }

    /// The content hash.
    #[must_use]
    pub fn hash(&self) -> RelationHash {
        self.hash
    }

    /// The number of stored batches.
    #[must_use]
    pub fn batches(&self) -> usize {
        self.batches
    }
}

impl Clone for RelationLease {
    fn clone(&self) -> Self {
        self.store.retain(&self.hash);
        Self {
            store: Arc::clone(&self.store),
            hash: self.hash,
            batches: self.batches,
        }
    }
}

impl Drop for RelationLease {
    fn drop(&mut self) {
        self.store.release(&self.hash);
    }
}

impl fmt::Debug for RelationLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelationLease")
            .field("hash", &self.hash)
            .field("batches", &self.batches)
            .finish_non_exhaustive()
    }
}

/// The accounted bytes of stored `batches`: their array memory plus their
/// weights.
fn decoded_bytes(batches: &[Batch]) -> usize {
    batches
        .iter()
        .map(|b| {
            b.data()
                .get_array_memory_size()
                .saturating_add(match b.weights() {
                    Weights::Const(_) => 8,
                    Weights::Column(column) => column.len().saturating_mul(8),
                })
        })
        .sum()
}

/// `batches` (of `schema`) as one Arrow IPC stream.
fn encode(schema: &SchemaRef, batches: &[Batch]) -> Result<Vec<u8>, KrishivError> {
    let mut ipc = Vec::new();
    let mut writer = StreamWriter::try_new(&mut ipc, schema.as_ref())
        .map_err(|e| internal(format!("cannot encode a relation's schema: {e}")))?;
    for b in batches {
        writer
            .write(b.data())
            .map_err(|e| internal(format!("cannot encode a relation's batches: {e}")))?;
    }
    writer
        .finish()
        .map_err(|e| internal(format!("cannot finish a relation's encoding: {e}")))?;
    Ok(ipc)
}

/// The schema and record batches of one IPC stream (unweighted: every batch
/// gets weight 1 at the caller's [`Batch::new`]).
fn decode_records(ipc: &[u8]) -> Result<(SchemaRef, Vec<RecordBatch>), KrishivError> {
    let mut reader = StreamReader::try_new(Cursor::new(ipc), None)
        .map_err(|e| internal(format!("cannot read a stored relation: {e}")))?;
    let schema = reader.schema();
    let mut out = Vec::new();
    for batch in &mut reader {
        let data = batch.map_err(|e| internal(format!("cannot read a stored relation: {e}")))?;
        out.push(data);
    }
    Ok((schema, out))
}

/// The content hash: SHA-256 over the IPC bytes (which carry the schema and
/// the values) plus the weights.
fn hash_relation(ipc: &[u8], weights: &[StoredWeights]) -> RelationHash {
    let mut hasher = Sha256::new();
    hasher.update(ipc);
    let mut w = Vec::new();
    for weight in weights {
        weight.hash_bytes(&mut w);
    }
    hasher.update(&w);
    let digest = hasher.finalize();
    let mut hash = [0_u8; 32];
    hash.copy_from_slice(&digest);
    RelationHash(hash)
}

/// `INTERNAL_ERROR_MEMORY`, for a broken invariant (not a user error).
fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, Int64Array, RecordBatch, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("s", DataType::Utf8View, false),
        ]))
    }

    fn batch(schema: &SchemaRef, ids: &[i32]) -> Batch {
        let data = RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringViewArray::from_iter_values(
                    ids.iter().map(|i| format!("s{i}")),
                )),
            ],
        )
        .expect("test batch builds");
        Batch::new(data)
    }

    fn weighted(schema: &SchemaRef, ids: &[i32], weights: &[i64]) -> Batch {
        let plain = batch(schema, ids);
        let (data, _) = plain.into_parts();
        Batch::with_weights(data, Weights::Column(Int64Array::from(weights.to_vec())))
            .expect("test weights fit")
    }

    fn store() -> Arc<RelationStore> {
        RelationStore::new()
    }

    #[test]
    fn batches_round_trip_with_schema_rows_and_weights() {
        let store = store();
        let schema = schema();
        let batches = vec![
            batch(&schema, &[1, 2, 3]),
            weighted(&schema, &[4, 5], &[2, -1]),
        ];
        let lease = store.insert(&schema, &batches).expect("insert stores");
        assert_eq!(lease.batches(), 2);
        let got = store
            .resolve(&lease.hash(), 0..lease.batches())
            .expect("resolve reads");
        assert_eq!(got, batches);
        // A range reads only its batches, sharing the stored arrays.
        let one = store.resolve(&lease.hash(), 1..2).expect("range reads");
        assert_eq!(one, batches[1..]);
        // No batches at all still round-trips its schema.
        let empty = store.insert(&schema, &[]).expect("empty insert stores");
        assert_eq!(empty.batches(), 0);
        assert!(
            store
                .resolve(&empty.hash(), 0..0)
                .expect("empty resolves")
                .is_empty()
        );
    }

    #[test]
    fn the_hash_is_content_addressed() {
        let store = store();
        let schema = schema();
        let one = store
            .insert(&schema, &[batch(&schema, &[1, 2])])
            .expect("insert stores");
        let same = store
            .insert(&schema, &[batch(&schema, &[1, 2])])
            .expect("same insert stores");
        assert_eq!(one.hash(), same.hash());
        assert_eq!(one.reference(), same.reference());
        // One entry: the same bytes are shared, not stored twice.
        assert_eq!(store.len(), 1);
        let other_rows = store
            .insert(&schema, &[batch(&schema, &[1, 3])])
            .expect("other rows store");
        assert_ne!(one.hash(), other_rows.hash());
        let other_weights = store
            .insert(&schema, &[weighted(&schema, &[1, 2], &[1, 1])])
            .expect("other weights store");
        // Same rows with column weights differ from the unweighted batch.
        assert_ne!(one.hash(), other_weights.hash());
        let other_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let narrow = RecordBatch::try_new(
            Arc::clone(&other_schema),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )
        .expect("narrow batch builds");
        let other_schema = store
            .insert(&other_schema, &[Batch::new(narrow)])
            .expect("other schema stores");
        assert_ne!(one.hash(), other_schema.hash());
    }

    #[test]
    fn the_hash_is_sha256_hex() {
        let store = store();
        let schema = schema();
        let lease = store
            .insert(&schema, &[batch(&schema, &[1, 2])])
            .expect("insert stores");
        // 32 bytes as 64 hex characters, as Spark Connect addresses its
        // cached local relations.
        assert_eq!(lease.hash().hex().len(), 64);
        assert!(
            lease.hash().hex().chars().all(|c| c.is_ascii_hexdigit()),
            "{}",
            lease.hash().hex()
        );
        assert_eq!(format!("{hash}", hash = lease.hash()), lease.hash().hex());
        assert_eq!(
            format!("{hash:?}", hash = lease.hash()),
            format!("RelationHash({})", lease.hash().hex())
        );
    }

    #[test]
    fn ipc_wire_bytes_round_trip_between_stores() {
        let first = store();
        let schema = schema();
        let batches = vec![batch(&schema, &[1, 2, 3]), batch(&schema, &[4])];
        let lease = first.insert(&schema, &batches).expect("insert stores");
        let wire = first.encode_ipc(&lease.hash()).expect("encode reads");
        assert!(!wire.is_empty(), "the wire carries the batches");
        // A remote worker inserts the wire bytes: the same content hash,
        // the same rows.
        let second = store();
        let remote = second.insert_ipc(&wire).expect("wire inserts");
        assert_eq!(remote.hash(), lease.hash());
        assert_eq!(
            second
                .resolve(&remote.hash(), 0..remote.batches())
                .expect("remote resolves"),
            batches
        );
        // Undecodable bytes are an internal error, not a panic.
        assert!(second.insert_ipc(b"not ipc").is_err());
    }

    #[test]
    fn resolves_share_the_underlying_buffers() {
        let store = store();
        let schema = schema();
        let batches = vec![batch(&schema, &[1, 2, 3])];
        let lease = store.insert(&schema, &batches).expect("insert stores");
        let first = store
            .resolve(&lease.hash(), 0..1)
            .expect("first scan reads");
        let second = store
            .resolve(&lease.hash(), 0..1)
            .expect("second scan reads");
        let first_ptr = first[0].data().column(0).to_data().buffers()[0]
            .as_slice()
            .as_ptr();
        let second_ptr = second[0].data().column(0).to_data().buffers()[0]
            .as_slice()
            .as_ptr();
        assert_eq!(
            first_ptr, second_ptr,
            "two scans share the stored arrays instead of decoding per scan"
        );
    }

    #[test]
    fn a_range_past_the_batches_is_an_internal_error() {
        let store = store();
        let schema = schema();
        let batches = vec![batch(&schema, &[1])];
        let lease = store.insert(&schema, &batches).expect("insert stores");
        for range in [0..2_usize, 1..2_usize, 2..3_usize] {
            let e = store
                .resolve(&lease.hash(), range.clone())
                .expect_err("past the batches");
            assert_eq!(e.error_class().name(), "INTERNAL_ERROR", "{range:?}: {e}");
        }
        // An empty range at the end reads nothing.
        assert!(
            store
                .resolve(&lease.hash(), 1..1)
                .expect("empty reads")
                .is_empty()
        );
    }

    #[test]
    fn the_last_lease_releases_the_entry() {
        let store = store();
        let schema = schema();
        let batches = vec![batch(&schema, &[7])];
        let first = store.insert(&schema, &batches).expect("insert stores");
        let hash = first.hash();
        assert_eq!(store.len(), 1);
        let second = first.clone();
        assert_eq!(store.len(), 1);
        drop(first);
        // One lease left: still resolvable.
        assert_eq!(
            store.resolve(&hash, 0..1).expect("shared entry lives"),
            batches
        );
        assert_eq!(store.len(), 1);
        drop(second);
        assert!(store.is_empty());
        let e = store
            .resolve(&hash, 0..1)
            .expect_err("released entry is gone");
        assert_eq!(
            e.error_class().name(),
            "INTERNAL_ERROR",
            "an unknown hash is a coordinator bug: {e}"
        );
    }

    #[test]
    fn an_unknown_hash_is_an_internal_error() {
        let store = store();
        let e = store
            .resolve(&RelationHash::from_bytes([9; 32]), 0..0)
            .expect_err("nothing stored");
        assert_eq!(e.error_class().name(), "INTERNAL_ERROR", "{e}");
        assert!(
            e.to_string()
                .contains(&RelationHash::from_bytes([9; 32]).hex())
        );
        assert!(
            store
                .encode_ipc(&RelationHash::from_bytes([9; 32]))
                .is_err()
        );
    }

    #[test]
    fn stored_bytes_are_accounted_and_released() {
        let store = store();
        assert_eq!(store.reserved(), 0);
        let schema = schema();
        let batches = vec![batch(&schema, &[1, 2, 3]), batch(&schema, &[4])];
        let lease = store.insert(&schema, &batches).expect("insert stores");
        assert!(
            store.reserved() > 0,
            "the entry holds pool bytes while leased"
        );
        drop(lease);
        assert_eq!(store.reserved(), 0, "the last lease returns the bytes");
        assert!(store.is_empty());
    }
}
