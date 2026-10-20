// SPDX-License-Identifier: Apache-2.0
//! P2-T159 Step 0: which hash-table design the grouped aggregate should use.
//!
//! ```text
//! cargo run --profile bench -p krishiv-exec --features row-table-prototype \
//!     --example agg_table_matrix
//! AGG_GROUPS=1,64,10000 AGG_KEYS=i64,i64x2,dec,str AGG_DISTS=uniform,zipf,runs \
//!     AGG_THREADS=1,12 AGG_CANDS=enc,a,ap,b,c,c128k,d,e,op AGG_REPS=6 \
//!     AGG_RBITS=5 cargo run ...
//! ```
//!
//! Inputs have `max(2^22, 2 × groups)` rows (`4 × groups` for `runs`):
//! `uniform` and `zipf` (Zipf(1)) draw key ranks at random; `runs` repeats
//! each rank 1..=7 times in order, the shape of `lineitem` grouped by
//! `l_orderkey`. `AGG_RBITS` sets the radix partitions (2^bits, 1..=8).
//!
//! Query shape: `select keys, sum(v), count(*), avg(d) group by keys`, with
//! `v` an `Int64` and `d` a `decimal(15,2)`. Each candidate runs a
//! thread-local phase (threads take 4,096-row batches from a shared
//! cursor) and a final phase, timed apart:
//!
//! - `a`: today's design. Arrow-row keys in a `KeyTable` per thread, states
//!   in a side array; the final phase merges every table into the largest
//!   on one thread (the shape of `HashAggregate`'s in-memory merge).
//! - `ap`: `a` radix-partitioned, with a partition-parallel final phase.
//! - `b`: row-layout keys with inline states in radix-partitioned pages,
//!   one `hashbrown::HashTable<u32>` index per thread that grows.
//! - `c`: Polars' hot/cold scheme over the same pages: a fixed two-choice
//!   hot table (4,096 slots; `c128k` = 2^17) with second-chance admission;
//!   a miss that is not admitted appends a one-row partial to its
//!   partition (a "cold" row). Evicted groups stay in their pages.
//! - `d`: `b` with the size-adaptive index of P2-T158 (flat up to 8 MiB of
//!   index, salted directory at load 1/2 above, touch pass from 16 MiB).
//! - `e`: DuckDB-style abandon: a fixed-capacity index that is cleared when
//!   full (rows stay in their pages), adapted every 2^20 rows from an HLL
//!   of the input hashes (grow when deduplication could double; skip
//!   lookups when over 95% of keys are unique).
//! - `enc`: the floor of the row candidates: key encoding and hashing only.
//! - `op`: the real `HashAggregate` operator end to end through the query
//!   runner (includes Arrow output and the scan; a reference point only).
//!
//! The final phase of `b`..`e` builds, for each radix partition, one index
//! over the partition's pages from every thread (rows are merged in place;
//! no copy), presized from merged per-partition HLL sketches of `h`.
//! Partitions are claimed from an atomic counter by all threads.
//!
//! Hashes follow P2-T157 Appendix C: the stored row hash `h` is seedless,
//! radix partitions are its top bits, and index bits come from
//! `g = fin(h ^ seed)`. (`h` here is the rows module's fold hash with seed
//! 0, not XXH64; P2-T175 measures that cost.)
//!
//! Every candidate's result (group count and state sums) must agree with the
//! others and with the generated data; a mismatch aborts the run.
//! Repetitions run the candidates in ABBA order (an odd `AGG_REPS` is
//! rounded up to even).

#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "a measurement tool: a broken fixture must stop it loudly"
)]

use std::hint::black_box;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::Instant;

use arrow_array::{Array, ArrayRef, Decimal128Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use hashbrown::HashTable;
use hashbrown::hash_table::Entry;
use krishiv_exec::aggregate::{
    AggregateExpr, AggregateFunction, AggregateNode, AggregateOutput, AggregateSpec, HashAggregate,
};
use krishiv_exec::hash::{HashSeed, KeyEncoder, KeyTable};
use krishiv_exec::rows::{EncodedKeys, RowLayout};
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{
    ExecConfig, ExecutorConfig, QueryContext, QueryId, QueryRunner, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_types::batch::Batch;

const BATCH: usize = 4096;
/// Radix bits of `h` for the in-memory partitions (`AGG_RBITS`, 1..=8).
static R_BITS: AtomicU32 = AtomicU32::new(5);
/// The sketch reads the bits below the most radix bits allowed.
const MAX_R_BITS: u32 = 8;

fn parts() -> usize {
    1 << R_BITS.load(Ordering::Relaxed)
}
/// Row id = `store << LOCAL_BITS | row in store`.
const LOCAL_BITS: u32 = 32 - MAX_R_BITS;
const LOCAL_MASK: u32 = (1 << LOCAL_BITS) - 1;
/// The per-operator seed of `g`.
const SEED: u64 = 0x5EED_1234_ABCD_0042;
const DEC: DataType = DataType::Decimal128(15, 2);

// ---------------------------------------------------------------- states

/// Inline state: `sum(v)` i64, `count(*)` i64, `avg(d)` as i128 sum and
/// i64 count, overflow flags.
const STATE: usize = 48;

#[inline]
fn rd64(s: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(s[at..at + 8].try_into().unwrap())
}
#[inline]
fn wr64(s: &mut [u8], at: usize, v: u64) {
    s[at..at + 8].copy_from_slice(&v.to_le_bytes());
}
#[inline]
fn rd128(s: &[u8], at: usize) -> i128 {
    i128::from_le_bytes(s[at..at + 16].try_into().unwrap())
}
#[inline]
fn wr128(s: &mut [u8], at: usize, v: i128) {
    s[at..at + 16].copy_from_slice(&v.to_le_bytes());
}

#[inline]
fn update(s: &mut [u8], v: i64, d: i128) {
    let (sum, o1) = (rd64(s, 0) as i64).overflowing_add(v);
    wr64(s, 0, sum as u64);
    wr64(s, 8, rd64(s, 8) + 1);
    let (asum, o2) = rd128(s, 16).overflowing_add(d);
    wr128(s, 16, asum);
    wr64(s, 32, rd64(s, 32) + 1);
    wr64(s, 40, rd64(s, 40) | u64::from(o1) | (u64::from(o2) << 1));
}

#[inline]
fn merge(s: &mut [u8], o: &[u8]) {
    let (sum, o1) = (rd64(s, 0) as i64).overflowing_add(rd64(o, 0) as i64);
    wr64(s, 0, sum as u64);
    wr64(s, 8, rd64(s, 8) + rd64(o, 8));
    let (asum, o2) = rd128(s, 16).overflowing_add(rd128(o, 16));
    wr128(s, 16, asum);
    wr64(s, 32, rd64(s, 32) + rd64(o, 32));
    wr64(
        s,
        40,
        rd64(s, 40) | rd64(o, 40) | u64::from(o1) | (u64::from(o2) << 1),
    );
}

/// Result fingerprint: groups, then wrapping sums of every state field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Check {
    groups: u64,
    sum: u64,
    cnt: u64,
    asum: i128,
    acnt: u64,
}

impl Check {
    fn add(&mut self, s: &[u8]) {
        self.groups += 1;
        self.sum = self.sum.wrapping_add(rd64(s, 0));
        self.cnt += rd64(s, 8);
        self.asum = self.asum.wrapping_add(rd128(s, 16));
        self.acnt += rd64(s, 32);
    }
    fn plus(mut self, o: Self) -> Self {
        self.groups += o.groups;
        self.sum = self.sum.wrapping_add(o.sum);
        self.cnt += o.cnt;
        self.asum = self.asum.wrapping_add(o.asum);
        self.acnt += o.acnt;
        self
    }
}

// ---------------------------------------------------------------- hashing

#[inline]
fn fold(a: u64, b: u64) -> u64 {
    let p = u128::from(a) * u128::from(b);
    (p as u64) ^ ((p >> 64) as u64)
}

/// `g = fin(h ^ seed)` (Appendix C).
#[inline]
fn fin(h: u64) -> u64 {
    fold(h ^ SEED, 0x9E37_79B9_7F4A_7C15)
}

#[inline]
fn part_of(h: u64) -> usize {
    (h >> (64 - R_BITS.load(Ordering::Relaxed))) as usize
}

/// HyperLogLog with 2^12 one-byte registers over the bits of `h` below the
/// radix bits (seedless, so sketches from any producer merge).
#[derive(Clone)]
struct Hll([u8; 4096]);

impl Hll {
    fn new() -> Self {
        Self([0; 4096])
    }
    #[inline]
    fn add(&mut self, h: u64) {
        let rest = h << MAX_R_BITS;
        let i = (rest >> 52) as usize;
        let w = rest << 12;
        let rank = (w.leading_zeros() + 1).min(64 - 12 - MAX_R_BITS + 1) as u8;
        let r = &mut self.0[i];
        if rank > *r {
            *r = rank;
        }
    }
    fn merge(&mut self, o: &Self) {
        for (a, b) in self.0.iter_mut().zip(o.0.iter()) {
            *a = (*a).max(*b);
        }
    }
    fn estimate(&self) -> f64 {
        let m = 4096.0;
        let mut sum = 0.0;
        let mut zeros = 0;
        for &r in &self.0 {
            sum += 2f64.powi(-i32::from(r));
            zeros += usize::from(r == 0);
        }
        let e = 0.7213 / (1.0 + 1.079 / m) * m * m / sum;
        if e <= 2.5 * m && zeros > 0 {
            m * (m / zeros as f64).ln()
        } else {
            e
        }
    }
}

// ---------------------------------------------------------------- pages

/// Rows of one (thread, partition): key region of the layout, then the
/// inline state. Pages of about 256 KiB; long strings in one heap.
struct Store {
    stride: usize,
    key_stride: usize,
    shift: u32,
    views: Arc<[usize]>,
    pages: Vec<Vec<u8>>,
    heap: Vec<u8>,
    len: usize,
}

impl Store {
    fn new(layout: &RowLayout) -> Self {
        let key_stride = layout.stride();
        let stride = key_stride + STATE;
        let per_page = ((256 << 10) / stride).max(64);
        let shift = usize::BITS - 1 - per_page.leading_zeros();
        Self {
            stride,
            key_stride,
            shift,
            views: layout.views().into(),
            pages: Vec::new(),
            heap: Vec::new(),
            len: 0,
        }
    }

    /// Appends key `k` of `keys` with a zero state; returns its local id.
    #[inline]
    fn push(&mut self, keys: &EncodedKeys, k: usize) -> u32 {
        let id = self.len;
        assert!(id <= LOCAL_MASK as usize, "store full");
        let per_page = 1usize << self.shift;
        if id & (per_page - 1) == 0 {
            // Geometric first page, then full pages.
            let cap = if self.pages.is_empty() { 64 } else { per_page };
            self.pages.push(Vec::with_capacity(cap * self.stride));
        }
        let page = self.pages.last_mut().unwrap();
        let start = page.len();
        page.extend_from_slice(keys.row(k));
        page.resize(start + self.stride, 0);
        for &v in self.views.iter() {
            let row = &page[start..];
            let len = rd64(row, v) as u32 as usize;
            if len > 12 {
                let from = rd64(row, v + 8) as usize;
                let at = self.heap.len() as u64;
                self.heap.extend_from_slice(&keys.heap()[from..from + len]);
                wr64(&mut page[start..], v + 8, at);
            }
        }
        self.len += 1;
        id as u32
    }

    #[inline]
    fn row(&self, id: u32) -> &[u8] {
        let id = id as usize;
        let at = (id & ((1 << self.shift) - 1)) * self.stride;
        &self.pages[id >> self.shift][at..at + self.stride]
    }

    #[inline]
    fn row_mut(&mut self, id: u32) -> &mut [u8] {
        let id = id as usize;
        let at = (id & ((1 << self.shift) - 1)) * self.stride;
        &mut self.pages[id >> self.shift][at..at + self.stride]
    }

    #[inline]
    fn state_mut(&mut self, id: u32) -> &mut [u8] {
        let k = self.key_stride;
        &mut self.row_mut(id)[k..]
    }
}

/// Stores addressed by `store << LOCAL_BITS | local`.
#[inline]
fn row_of(stores: &[Store], id: u32) -> (&[u8], &[u8]) {
    let s = &stores[(id >> LOCAL_BITS) as usize];
    (s.row(id & LOCAL_MASK), &s.heap)
}

#[inline]
fn hash_of(stores: &[Store], id: u32) -> u64 {
    rd64(row_of(stores, id).0, 0)
}

// ---------------------------------------------------------------- indexes

/// Largest flat index before the adaptive index goes salted (P2-T158).
const FLAT_MAX_BYTES: usize = 8 << 20;
/// Smallest salted directory that touches a chunk before resolving it.
const TOUCH_MIN_BYTES: usize = 16 << 20;
const SALT: u64 = 0xFFFF << 48;
const CHUNK: usize = 256;

fn flat_bytes(n: usize) -> usize {
    let b = if n < 8 {
        8
    } else {
        (n * 8 / 7).next_power_of_two()
    };
    b * 5 + 16
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Policy {
    Flat,
    Adaptive,
}

enum Index {
    Flat(HashTable<u32>),
    Salted { slots: Vec<u64>, len: usize },
}

impl Index {
    fn new() -> Self {
        Self::Flat(HashTable::new())
    }

    fn len(&self) -> usize {
        match self {
            Self::Flat(t) => t.len(),
            Self::Salted { len, .. } => *len,
        }
    }

    /// Makes room for `extra` more keys, switching to the salted directory
    /// when the flat index would pass [`FLAT_MAX_BYTES`] (adaptive only).
    fn reserve(&mut self, extra: usize, policy: Policy, stores: &[Store]) {
        self.reserve_with(extra, policy, stores, false);
    }

    /// [`reserve`](Self::reserve); with `dense` (every row of `stores` is
    /// indexed) a grown index is rebuilt by a sequential scan of the pages,
    /// as DuckDB's resize does, instead of reading each row's stored hash
    /// in directory order (a cache miss per key once the rows outgrow the
    /// caches).
    fn reserve_with(&mut self, extra: usize, policy: Policy, stores: &[Store], dense: bool) {
        let need = self.len() + extra;
        let salted = match self {
            Self::Flat(t) => {
                let flat = policy == Policy::Flat || flat_bytes(need) <= FLAT_MAX_BYTES;
                if flat && t.capacity() - t.len() >= extra {
                    return;
                }
                if flat && !dense {
                    t.reserve(extra, |&id| fin(hash_of(stores, id)));
                    return;
                }
                !flat
            }
            Self::Salted { slots, .. } => {
                if need * 2 <= slots.len() {
                    return;
                }
                true
            }
        };
        // The ids to re-insert, in page order when dense.
        let ids: Vec<u32> = if dense {
            Vec::new()
        } else {
            match self {
                Self::Flat(t) => t.iter().copied().collect(),
                Self::Salted { slots, .. } => slots
                    .iter()
                    .filter(|&&s| s != 0)
                    .map(|&s| ((s & 0xFFFF_FFFF) - 1) as u32)
                    .collect(),
            }
        };
        let each = |f: &mut dyn FnMut(u32, u64)| {
            if dense {
                for (si, s) in stores.iter().enumerate() {
                    let base = (si as u32) << LOCAL_BITS;
                    let mut local = 0u32;
                    for page in &s.pages {
                        for row in page.chunks_exact(s.stride) {
                            f(base | local, rd64(row, 0));
                            local += 1;
                        }
                    }
                }
            } else {
                for &id in &ids {
                    f(id, hash_of(stores, id));
                }
            }
        };
        if salted {
            let old = match self {
                Self::Salted { slots, .. } => slots.len(),
                Self::Flat(_) => 1024,
            };
            let mut n = old.max(1024);
            while need * 2 > n {
                n *= 2;
            }
            let mut slots = vec![0u64; n];
            let mask = n - 1;
            let mut len = 0;
            each(&mut |id, h| {
                let g = fin(h);
                let mut i = g as usize & mask;
                while slots[i] != 0 {
                    i = (i + 1) & mask;
                }
                slots[i] = (g & SALT) | (u64::from(id) + 1);
                len += 1;
            });
            *self = Self::Salted { slots, len };
        } else {
            let cap = match self {
                Self::Flat(t) => t.capacity(),
                Self::Salted { .. } => 0,
            };
            let mut t = HashTable::with_capacity(need.max(cap * 2));
            each(&mut |id, h| {
                t.insert_unique(fin(h), id, |&o| fin(hash_of(stores, o)));
            });
            *self = Self::Flat(t);
        }
    }

    /// Touches the directory slots and first salt-matching rows of a chunk
    /// (large salted directories only).
    #[inline]
    fn touch(&self, gs: &[u64], stores: &[Store]) {
        if let Self::Salted { slots, .. } = self {
            if slots.len() * 8 < TOUCH_MIN_BYTES {
                return;
            }
            let mask = slots.len() - 1;
            for &g in gs {
                black_box(slots[g as usize & mask]);
            }
            for &g in gs {
                let mut i = g as usize & mask;
                loop {
                    let s = slots[i];
                    if s == 0 {
                        break;
                    }
                    if s & SALT == g & SALT {
                        black_box(row_of(stores, ((s & 0xFFFF_FFFF) - 1) as u32).0.first());
                        break;
                    }
                    i = (i + 1) & mask;
                }
            }
        }
    }

    /// Finds `key`; on a miss, `make` appends its row and the index takes
    /// the id. Returns `(id, new)`. Room must have been reserved.
    #[inline]
    fn find_or_insert(
        &mut self,
        g: u64,
        layout: &RowLayout,
        key: &[u8],
        kheap: &[u8],
        stores: &mut [Store],
        make: impl FnOnce(&mut [Store]) -> u32,
    ) -> (u32, bool) {
        match self {
            Self::Flat(t) => {
                let entry = {
                    let st = &*stores;
                    t.entry(
                        g,
                        |&id| {
                            let (r, h) = row_of(st, id);
                            layout.eq(key, kheap, r, h)
                        },
                        |&id| fin(hash_of(st, id)),
                    )
                };
                match entry {
                    Entry::Occupied(o) => (*o.get(), false),
                    Entry::Vacant(v) => {
                        let id = make(stores);
                        v.insert(id);
                        (id, true)
                    }
                }
            }
            Self::Salted { slots, len } => {
                let mask = slots.len() - 1;
                let mut i = g as usize & mask;
                loop {
                    let s = slots[i];
                    if s == 0 {
                        let id = make(stores);
                        slots[i] = (g & SALT) | (u64::from(id) + 1);
                        *len += 1;
                        return (id, true);
                    }
                    if s & SALT == g & SALT {
                        let id = ((s & 0xFFFF_FFFF) - 1) as u32;
                        let (r, h) = row_of(stores, id);
                        if layout.eq(key, kheap, r, h) {
                            return (id, false);
                        }
                    }
                    i = (i + 1) & mask;
                }
            }
        }
    }

    /// [`find_or_insert`](Self::find_or_insert) for a row already in
    /// `stores` (the final phase): a miss indexes `id`.
    #[inline]
    fn find_or_insert_ro(
        &mut self,
        g: u64,
        layout: &RowLayout,
        key: &[u8],
        kheap: &[u8],
        stores: &[Store],
        id: u32,
    ) -> (u32, bool) {
        match self {
            Self::Flat(t) => match t.entry(
                g,
                |&o| {
                    let (r, h) = row_of(stores, o);
                    layout.eq(key, kheap, r, h)
                },
                |&o| fin(hash_of(stores, o)),
            ) {
                Entry::Occupied(o) => (*o.get(), false),
                Entry::Vacant(v) => {
                    v.insert(id);
                    (id, true)
                }
            },
            Self::Salted { slots, len } => {
                let mask = slots.len() - 1;
                let mut i = g as usize & mask;
                loop {
                    let s = slots[i];
                    if s == 0 {
                        slots[i] = (g & SALT) | (u64::from(id) + 1);
                        *len += 1;
                        return (id, true);
                    }
                    if s & SALT == g & SALT {
                        let o = ((s & 0xFFFF_FFFF) - 1) as u32;
                        let (r, h) = row_of(stores, o);
                        if layout.eq(key, kheap, r, h) {
                            return (o, false);
                        }
                    }
                    i = (i + 1) & mask;
                }
            }
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Flat(t) => t.clear(),
            Self::Salted { slots, len } => {
                slots.fill(0);
                *len = 0;
            }
        }
    }
}

/// Polars' fixed two-choice hot table (`polars-expr` `hot_groups/
/// fixed_index_table.rs`, py-2.0.0): one branch for hit/miss over both
/// candidate slots, fill empty slots, else pick one candidate at random
/// and evict it only if the same key missed it last time (second chance),
/// or the next key is equal.
#[derive(Clone, Copy)]
struct HotSlot {
    hash: u64,
    id: u32,
    tag: u32,
}

struct Hot {
    slots: Vec<HotSlot>,
    shift: u32,
    filled: usize,
    prng: u64,
}

enum HotHit {
    Hit(u32),
    Put(usize),
    Cold,
}

impl Hot {
    fn new(size: usize) -> Self {
        Self {
            slots: vec![
                HotSlot {
                    hash: 0,
                    id: u32::MAX,
                    tag: 0
                };
                size
            ],
            shift: 64 - size.trailing_zeros(),
            filled: 0,
            prng: 0,
        }
    }

    #[inline]
    fn lookup(&mut self, g: u64, next_g: u64, eq: impl Fn(u32) -> bool) -> HotHit {
        let tag = g as u32;
        let h1 = (g >> self.shift) as usize;
        let h2 = (g.wrapping_mul(0xA2AA_033B_E8F6_33C9) >> self.shift) as usize;
        let d1 = self.slots[h1].hash ^ g;
        let d2 = self.slots[h2].hash ^ g;
        if d1 & d2 == 0 {
            let ha = if d1 == 0 { h1 } else { h2 };
            let id = self.slots[ha].id;
            if id != u32::MAX && eq(id) {
                self.slots[ha].tag = tag;
                return HotHit::Hit(id);
            }
            if d1 == d2 {
                let hb = h1 ^ h2 ^ ha;
                let id = self.slots[hb].id;
                if id != u32::MAX && eq(id) {
                    self.slots[hb].tag = tag;
                    return HotHit::Hit(id);
                }
            }
        }
        if self.filled < self.slots.len() {
            for h in [h1, h2] {
                if self.slots[h].id == u32::MAX {
                    self.filled += 1;
                    return HotHit::Put(h);
                }
            }
        }
        let hr = if self.prng >> 63 == 0 { h1 } else { h2 };
        self.prng = self.prng.wrapping_add(g);
        if self.slots[hr].tag == tag || g == next_g {
            HotHit::Put(hr)
        } else {
            self.slots[hr].tag = tag;
            HotHit::Cold
        }
    }

    #[inline]
    fn put(&mut self, at: usize, g: u64, id: u32) {
        self.slots[at] = HotSlot {
            hash: g,
            id,
            tag: g as u32,
        };
    }
}

// ---------------------------------------------------------------- data

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyType {
    I64,
    I64x2,
    Dec,
    Str,
}

impl KeyType {
    fn parse(s: &str) -> Self {
        match s {
            "i64" => Self::I64,
            "i64x2" => Self::I64x2,
            "dec" => Self::Dec,
            "str" => Self::Str,
            o => panic!("unknown key type {o}"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::I64 => "i64",
            Self::I64x2 => "i64x2",
            Self::Dec => "dec",
            Self::Str => "str",
        }
    }
    fn types(self) -> Vec<DataType> {
        match self {
            Self::I64 => vec![DataType::Int64],
            Self::I64x2 => vec![DataType::Int64, DataType::Int64],
            Self::Dec => vec![DEC],
            Self::Str => vec![DataType::Utf8View],
        }
    }
}

fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Data {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    rows: usize,
    distinct: usize,
}

/// Rows over `groups` key ranks: `uniform`, `zipf` (Zipf(1)), or `runs`
/// (ranks in order, each repeated 1..=7 times, wrapping around: the shape of
/// `lineitem` grouped by `l_orderkey`).
fn data(kt: KeyType, groups: usize, dist: &str) -> Data {
    let zipf = dist == "zipf";
    let runs = dist == "runs";
    assert!(
        zipf || runs || dist == "uniform",
        "unknown distribution {dist}"
    );
    let rows = (1usize << 22).max(if runs { 4 } else { 2 } * groups);
    let (mut run_rank, mut run_left) = (0usize, 0u64);
    let cdf: Vec<f64> = if zipf {
        let mut acc = 0.0;
        let mut v: Vec<f64> = (1..=groups)
            .map(|r| {
                acc += 1.0 / r as f64;
                acc
            })
            .collect();
        let total = acc;
        v.iter_mut().for_each(|x| *x /= total);
        v
    } else {
        Vec::new()
    };
    let mut seen = vec![false; groups];
    let mut state = 0x1234_5678_u64;
    let ranks: Vec<u64> = (0..rows)
        .map(|_| {
            state = mix(state);
            let r = if runs {
                if run_left == 0 {
                    run_rank = (run_rank + 1) % groups;
                    run_left = 1 + state % 7;
                }
                run_left -= 1;
                run_rank
            } else if zipf {
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                cdf.partition_point(|&c| c < u).min(groups - 1)
            } else {
                (state % groups as u64) as usize
            };
            seen[r] = true;
            r as u64
        })
        .collect();
    let distinct = seen.iter().filter(|&&s| s).count();
    let mut fields: Vec<Field> = kt
        .types()
        .into_iter()
        .enumerate()
        .map(|(i, t)| Field::new(format!("k{i}"), t, false))
        .collect();
    let nk = fields.len();
    fields.push(Field::new("v", DataType::Int64, false));
    fields.push(Field::new("d", DEC, false));
    let schema = Arc::new(Schema::new(fields));
    let batches = ranks
        .chunks(BATCH)
        .enumerate()
        .map(|(b, rs)| {
            let mut cols: Vec<ArrayRef> = match kt {
                KeyType::I64 => vec![Arc::new(Int64Array::from_iter_values(
                    rs.iter().map(|&r| mix(r) as i64),
                ))],
                KeyType::I64x2 => vec![
                    Arc::new(Int64Array::from_iter_values(
                        rs.iter().map(|&r| (r % 1009) as i64),
                    )),
                    Arc::new(Int64Array::from_iter_values(
                        rs.iter().map(|&r| mix(r) as i64),
                    )),
                ],
                KeyType::Dec => vec![Arc::new(
                    Decimal128Array::from_iter_values(rs.iter().map(|&r| i128::from(r) * 101 + 7))
                        .with_data_type(DEC),
                )],
                KeyType::Str => vec![Arc::new(StringViewArray::from_iter_values(rs.iter().map(
                    |&r| {
                        let pad = (mix(r ^ 0xF00D) % 24) as usize;
                        format!("k{:016x}{}", mix(r), "-".repeat(pad))
                    },
                )))],
            };
            assert_eq!(cols.len(), nk);
            let base = (b * BATCH) as u64;
            cols.push(Arc::new(Int64Array::from_iter_values(
                (0..rs.len() as u64).map(|i| (mix(base + i) % 1000) as i64),
            )));
            cols.push(Arc::new(
                Decimal128Array::from_iter_values(
                    (0..rs.len() as u64)
                        .map(|i| 90_000 + i128::from(mix(!(base + i)) % 10_410_000)),
                )
                .with_data_type(DEC),
            ));
            RecordBatch::try_new(Arc::clone(&schema), cols).expect("batch")
        })
        .collect();
    Data {
        schema,
        batches,
        rows,
        distinct,
    }
}

fn vals(b: &RecordBatch, nk: usize) -> (&[i64], &[i128]) {
    let v = b.column(nk).as_any().downcast_ref::<Int64Array>().unwrap();
    let d = b
        .column(nk + 1)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    (v.values(), d.values())
}

// ---------------------------------------------------------------- runs

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cand {
    A,
    Ap,
    B,
    C(usize),
    D,
    E,
    Op,
    Enc,
}

impl Cand {
    fn parse(s: &str) -> Self {
        match s {
            "a" => Self::A,
            "ap" => Self::Ap,
            "b" => Self::B,
            "c" => Self::C(4096),
            "c128k" => Self::C(1 << 17),
            "d" => Self::D,
            "e" => Self::E,
            "op" => Self::Op,
            "enc" => Self::Enc,
            o => panic!("unknown candidate {o}"),
        }
    }
    fn name(self) -> String {
        match self {
            Self::A => "a".into(),
            Self::Ap => "ap".into(),
            Self::B => "b".into(),
            Self::C(4096) => "c".into(),
            Self::C(n) => format!("c{}k", n >> 10),
            Self::D => "d".into(),
            Self::E => "e".into(),
            Self::Op => "op".into(),
            Self::Enc => "enc".into(),
        }
    }
}

struct Ctx<'a> {
    data: &'a Data,
    kt: KeyType,
    threads: usize,
}

/// Runs `f(thread)` on `threads` scoped threads.
fn par<T: Send>(threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn({
                    let f = &f;
                    move || f(t)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("thread")).collect()
    })
}

/// Thread-local output of a row candidate: pages and sketches per partition.
struct LocalRows {
    stores: Vec<Store>,
    hll: Vec<Hll>,
}

/// A final-phase task, taken once by whichever thread claims it.
type Claim<T> = Mutex<Option<T>>;
/// A `KeyTable` and its groups' states (`STATE` bytes each).
type KeyedStates = (KeyTable, Vec<u8>);

/// One result: thread-local and final phase seconds, and the fingerprint.
struct Timing {
    local: f64,
    fin: f64,
    check: Check,
}

fn run(c: Cand, x: &Ctx<'_>) -> Timing {
    match c {
        Cand::A | Cand::Ap => run_keytable(c == Cand::Ap, x),
        Cand::Op => run_op(x),
        Cand::Enc => run_enc(x),
        _ => run_rows(c, x),
    }
}

fn key_cols(b: &RecordBatch, nk: usize) -> Vec<ArrayRef> {
    b.columns()[..nk].to_vec()
}

/// `a` and `ap`: arrow-row keys in `KeyTable`s, states in a side array.
fn run_keytable(partitioned: bool, x: &Ctx<'_>) -> Timing {
    let nk = x.kt.types().len();
    let encoder = KeyEncoder::new(&x.kt.types()).expect("encoder");
    let seed = HashSeed::new(Some(0));
    let nparts = if partitioned { parts() } else { 1 };
    let cursor = AtomicUsize::new(0);
    let t0 = Instant::now();
    let locals: Vec<Vec<(KeyTable, Vec<u8>)>> = par(x.threads, |_| {
        let mut tables: Vec<(KeyTable, Vec<u8>)> =
            (0..nparts).map(|_| (KeyTable::new(), Vec::new())).collect();
        let mut ids = Vec::new();
        loop {
            let i = cursor.fetch_add(1, Ordering::Relaxed);
            let Some(b) = x.data.batches.get(i) else {
                break;
            };
            let rows = encoder.encode(&key_cols(b, nk)).expect("encode");
            let (v, d) = vals(b, nk);
            if partitioned {
                for (k, r) in rows.iter().enumerate() {
                    let bytes = r.data();
                    let h = seed.hash(bytes);
                    let (t, st) = &mut tables[part_of(h)];
                    let (id, new) = t.insert_hashed(bytes, h).expect("insert");
                    if new {
                        st.resize(st.len() + STATE, 0);
                    }
                    let at = id as usize * STATE;
                    update(&mut st[at..at + STATE], v[k], d[k]);
                }
            } else {
                let (t, st) = &mut tables[0];
                t.reserve(rows.num_rows(), rows.size());
                t.find_or_insert(&rows, &seed, &mut ids).expect("insert");
                st.resize(t.len() * STATE, 0);
                for (k, &id) in ids.iter().enumerate() {
                    let at = id as usize * STATE;
                    update(&mut st[at..at + STATE], v[k], d[k]);
                }
            }
        }
        tables
    });
    let local = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    // Per partition: every thread's table, merged into the largest.
    let mut by_part: Vec<Vec<(KeyTable, Vec<u8>)>> = (0..nparts).map(|_| Vec::new()).collect();
    for tables in locals {
        for (p, t) in tables.into_iter().enumerate() {
            by_part[p].push(t);
        }
    }
    let merge_part = |mut ts: Vec<(KeyTable, Vec<u8>)>| -> Check {
        let big = (0..ts.len()).max_by_key(|&i| ts[i].0.len()).unwrap_or(0);
        let (mut dst, mut dst_st) = ts.swap_remove(big);
        for (src, src_st) in ts {
            for id in 0..src.len() {
                let (gid, new) = dst
                    .insert_hashed(src.row(id), src.hashes()[id])
                    .expect("merge");
                let s = &src_st[id * STATE..(id + 1) * STATE];
                if new {
                    dst_st.extend_from_slice(s);
                } else {
                    let at = gid as usize * STATE;
                    merge(&mut dst_st[at..at + STATE], s);
                }
            }
        }
        let mut c = Check::default();
        dst_st.chunks(STATE).for_each(|s| c.add(s));
        c
    };
    let check = if partitioned {
        let work: Vec<Claim<Vec<KeyedStates>>> =
            by_part.into_iter().map(|v| Mutex::new(Some(v))).collect();
        let next = AtomicUsize::new(0);
        par(x.threads, |_| {
            let mut c = Check::default();
            loop {
                let p = next.fetch_add(1, Ordering::Relaxed);
                let Some(w) = work.get(p) else { break };
                let ts = w.lock().unwrap().take().unwrap();
                c = c.plus(merge_part(ts));
            }
            c
        })
        .into_iter()
        .fold(Check::default(), Check::plus)
    } else {
        // Today's in-memory merge runs on one thread.
        merge_part(by_part.pop().unwrap())
    };
    Timing {
        local,
        fin: t1.elapsed().as_secs_f64(),
        check,
    }
}

/// `b`, `c`, `d`, `e`: row-layout keys with inline states.
fn run_rows(c: Cand, x: &Ctx<'_>) -> Timing {
    let nk = x.kt.types().len();
    let layout = RowLayout::new(&x.kt.types(), &vec![false; nk]).expect("layout");
    let cursor = AtomicUsize::new(0);
    let threads = x.threads;
    let t0 = Instant::now();
    let locals: Vec<LocalRows> = par(threads, |_| {
        let mut lr = LocalRows {
            stores: (0..parts()).map(|_| Store::new(&layout)).collect(),
            hll: vec![Hll::new(); parts()],
        };
        let mut index = Index::new();
        let policy = if c == Cand::B {
            Policy::Flat
        } else {
            Policy::Adaptive
        };
        let mut hot = match c {
            Cand::C(n) => Some(Hot::new(n)),
            _ => None,
        };
        // `e`: DuckDB-style capacity (32,768 slots above two threads,
        // 2^18 at most two) at load 2/3, adapted every 2^20 rows.
        let mut cap = if threads <= 2 { 1 << 18 } else { 1 << 15 };
        let mut limit = cap * 2 / 3;
        let mut skip = false;
        let mut seen_hll = Hll::new();
        let (mut rows_seen, mut pushed, mut next_check) = (0usize, 0usize, 1usize << 20);
        if c == Cand::E {
            index.reserve(limit, Policy::Flat, &lr.stores);
        }
        let mut ids: Vec<u32> = Vec::with_capacity(BATCH);
        let mut gs: Vec<u64> = Vec::with_capacity(BATCH);
        loop {
            let i = cursor.fetch_add(1, Ordering::Relaxed);
            let Some(b) = x.data.batches.get(i) else {
                break;
            };
            let keys = layout.encode(&key_cols(b, nk), 0).expect("encode");
            let (v, d) = vals(b, nk);
            let n = keys.len();
            let hs = keys.hashes();
            gs.clear();
            gs.extend(hs.iter().map(|&h| fin(h)));
            ids.clear();
            let LocalRows { stores, hll } = &mut lr;
            let mut push = |stores: &mut [Store], k: usize| -> u32 {
                let h = hs[k];
                let p = part_of(h);
                hll[p].add(h);
                ((p as u32) << LOCAL_BITS) | stores[p].push(&keys, k)
            };
            match c {
                Cand::B | Cand::D => {
                    index.reserve_with(n, policy, stores, true);
                    for (start, end) in (0..n).step_by(CHUNK).map(|s| (s, (s + CHUNK).min(n))) {
                        index.touch(&gs[start..end], stores);
                        for (k, &g) in gs.iter().enumerate().take(end).skip(start) {
                            let (id, _) = index.find_or_insert(
                                g,
                                &layout,
                                keys.row(k),
                                keys.heap(),
                                stores,
                                |st| push(st, k),
                            );
                            ids.push(id);
                        }
                    }
                }
                Cand::C(_) => {
                    let hot = hot.as_mut().unwrap();
                    for k in 0..n {
                        let g = gs[k];
                        let next = gs.get(k + 1).copied().unwrap_or(!g);
                        let r = {
                            let st = &*stores;
                            let key = keys.row(k);
                            hot.lookup(g, next, |id| {
                                let (r, h) = row_of(st, id);
                                layout.eq(key, keys.heap(), r, h)
                            })
                        };
                        let id = match r {
                            HotHit::Hit(id) => id,
                            HotHit::Put(at) => {
                                let id = push(stores, k);
                                hot.put(at, g, id);
                                id
                            }
                            HotHit::Cold => push(stores, k),
                        };
                        ids.push(id);
                    }
                }
                Cand::E => {
                    // A 1/8 hash sample (scaled back below) keeps the sketch
                    // off the per-row path.
                    for &h in hs {
                        if h & 7 == 0 {
                            seen_hll.add(h);
                        }
                    }
                    rows_seen += n;
                    if skip {
                        ids.extend((0..n).map(|k| push(stores, k)));
                    } else {
                        for (k, &g) in gs.iter().enumerate() {
                            if index.len() >= limit {
                                index.clear();
                            }
                            let (id, new) = index.find_or_insert(
                                g,
                                &layout,
                                keys.row(k),
                                keys.heap(),
                                stores,
                                |st| push(st, k),
                            );
                            pushed += usize::from(new);
                            ids.push(id);
                        }
                    }
                    if !skip && rows_seen >= next_check {
                        next_check += 1 << 20;
                        let est = (seen_hll.estimate() * 8.0).max(1.0);
                        if est / rows_seen as f64 > 0.95 {
                            skip = true;
                        } else if rows_seen as f64 / est
                            >= 2.0 * rows_seen as f64 / pushed.max(1) as f64
                            && cap < 1 << 22
                        {
                            cap *= 4;
                            limit = cap * 2 / 3;
                            index = Index::new();
                            index.reserve(limit, Policy::Flat, stores);
                        }
                    }
                }
                _ => unreachable!(),
            }
            for (k, &id) in ids.iter().enumerate() {
                let s = &mut stores[(id >> LOCAL_BITS) as usize];
                update(s.state_mut(id & LOCAL_MASK), v[k], d[k]);
            }
        }
        lr
    });
    let local = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let check = finalize_rows(c, locals, &layout, threads);
    Timing {
        local,
        fin: t1.elapsed().as_secs_f64(),
        check,
    }
}

/// Per radix partition (claimed from a counter): one index over the
/// partition's pages from every thread, presized from the merged sketch;
/// duplicates merge in place into the first row.
fn finalize_rows(c: Cand, locals: Vec<LocalRows>, layout: &RowLayout, threads: usize) -> Check {
    let mut work: Vec<(Vec<Store>, Hll)> = (0..parts()).map(|_| (Vec::new(), Hll::new())).collect();
    for lr in locals {
        for (p, (s, h)) in lr.stores.into_iter().zip(lr.hll).enumerate() {
            work[p].0.push(s);
            work[p].1.merge(&h);
        }
    }
    let work: Vec<Claim<(Vec<Store>, Hll)>> =
        work.into_iter().map(|w| Mutex::new(Some(w))).collect();
    let next = AtomicUsize::new(0);
    let policy = if c == Cand::B {
        Policy::Flat
    } else {
        Policy::Adaptive
    };
    par(threads, |_| {
        let mut check = Check::default();
        loop {
            let p = next.fetch_add(1, Ordering::Relaxed);
            let Some(w) = work.get(p) else { break };
            let (mut stores, hll) = w.lock().unwrap().take().unwrap();
            let total: usize = stores.iter().map(|s| s.len).sum();
            let est = (hll.estimate() * 1.05) as usize + 16;
            let mut index = Index::new();
            let presize = est.min(total);
            index.reserve(presize, policy, &stores);
            let mut groups: Vec<u32> = Vec::new();
            let mut state = [0u8; STATE];
            for t in 0..stores.len() {
                for local in 0..stores[t].len as u32 {
                    let id = ((t as u32) << LOCAL_BITS) | local;
                    let (gid, new) = {
                        let st = &stores[t];
                        let row = st.row(local);
                        state.copy_from_slice(&row[st.key_stride..]);
                        if index.len() >= presize {
                            index.reserve(1, policy, &stores);
                        }
                        index.find_or_insert_ro(
                            fin(rd64(row, 0)),
                            layout,
                            row,
                            &st.heap,
                            &stores,
                            id,
                        )
                    };
                    if new {
                        groups.push(gid);
                    } else {
                        let s = &mut stores[(gid >> LOCAL_BITS) as usize];
                        merge(s.state_mut(gid & LOCAL_MASK), &state);
                    }
                }
            }
            for &gid in &groups {
                let s = &stores[(gid >> LOCAL_BITS) as usize];
                check.add(&s.row(gid & LOCAL_MASK)[s.key_stride..]);
            }
        }
        check
    })
    .into_iter()
    .fold(Check::default(), Check::plus)
}

/// `enc`: the floor of the row candidates: key encoding, `h` and `g`, and
/// reading the values, with no table.
fn run_enc(x: &Ctx<'_>) -> Timing {
    let nk = x.kt.types().len();
    let layout = RowLayout::new(&x.kt.types(), &vec![false; nk]).expect("layout");
    let cursor = AtomicUsize::new(0);
    let t0 = Instant::now();
    let sums = par(x.threads, |_| {
        let mut acc = 0u64;
        loop {
            let i = cursor.fetch_add(1, Ordering::Relaxed);
            let Some(b) = x.data.batches.get(i) else {
                break;
            };
            let keys = layout.encode(&key_cols(b, nk), 0).expect("encode");
            let (v, d) = vals(b, nk);
            for (k, &h) in keys.hashes().iter().enumerate() {
                acc = acc.wrapping_add(fin(h) ^ v[k] as u64 ^ d[k] as u64);
            }
        }
        acc
    });
    black_box(sums);
    Timing {
        local: t0.elapsed().as_secs_f64(),
        fin: 0.0,
        check: Check::default(),
    }
}

/// `op`: the real operator through the query runner.
fn run_op(x: &Ctx<'_>) -> Timing {
    let nk = x.kt.types().len();
    let s = Arc::clone(&x.data.schema);
    let spec = AggregateSpec {
        input: Arc::clone(&s),
        group_by: (0..nk).collect(),
        aggregates: vec![
            AggregateExpr::new(AggregateFunction::Sum, nk, DataType::Int64, "s"),
            AggregateExpr::count_star("c"),
            AggregateExpr::new(AggregateFunction::Avg, nk + 1, DEC, "a"),
        ],
        output: AggregateOutput::Final,
        ansi: false,
        spill: None,
        require_unit_weights: false,
    };
    let batches: Vec<Batch> = x.data.batches.iter().cloned().map(Batch::new).collect();
    let scan = Arc::new(Scan(Batches::new(s, batches)));
    let plan = AggregateNode::new(scan, HashAggregate::new(spec).expect("spec"));
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let runner = {
        let _g = rt.enter();
        QueryRunner::new(
            TokioExecutor::new(&ExecutorConfig {
                compute_threads: x.threads,
                io_threads: 1,
                blocking_threads: x.threads,
            })
            .expect("executor"),
        )
    };
    let root = TreePool::new(&MemoryConfig::default());
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig {
            seed: Some(1),
            ..ExecConfig::default()
        },
        root.child("agg", Some(8 << 30)),
    );
    let t0 = Instant::now();
    let groups = rt.block_on(async {
        let mut stream = runner.execute(plan, Arc::clone(&q)).await.expect("plan");
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.expect("batch").num_rows();
        }
        q.finished().await;
        rows
    });
    Timing {
        local: t0.elapsed().as_secs_f64(),
        fin: 0.0,
        check: Check {
            groups: groups as u64,
            ..Check::default()
        },
    }
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let groups: Vec<usize> = env_list("AGG_GROUPS", "1,64,10000,100000,1000000,10000000")
        .iter()
        .map(|s| s.parse().expect("AGG_GROUPS: positive integers"))
        .collect();
    assert!(groups.iter().all(|&g| g > 0), "AGG_GROUPS must be positive");
    let keys: Vec<KeyType> = env_list("AGG_KEYS", "i64,i64x2,dec,str")
        .iter()
        .map(|s| KeyType::parse(s))
        .collect();
    let dists = env_list("AGG_DISTS", "uniform,zipf");
    let threads: Vec<usize> = env_list("AGG_THREADS", "1,12")
        .iter()
        .map(|s| s.parse().expect("AGG_THREADS: positive integers"))
        .collect();
    assert!(
        threads.iter().all(|&t| (1..64).contains(&t)),
        "AGG_THREADS in 1..64"
    );
    let cands: Vec<Cand> = env_list("AGG_CANDS", "a,ap,b,c,d,e,op")
        .iter()
        .map(|s| Cand::parse(s))
        .collect();
    let rbits: u32 =
        std::env::var("AGG_RBITS").map_or(5, |s| s.parse().expect("AGG_RBITS: an integer"));
    assert!((1..=MAX_R_BITS).contains(&rbits), "AGG_RBITS in 1..=8");
    R_BITS.store(rbits, Ordering::Relaxed);
    let reps: usize =
        std::env::var("AGG_REPS").map_or(6, |s| s.parse().expect("AGG_REPS: a positive integer"));
    assert!(reps > 0, "AGG_REPS must be positive");
    // ABBA needs as many reversed passes as forward ones, so each
    // candidate's position is balanced by its mirrored position: round odd
    // counts up.
    let reps = reps
        .checked_next_multiple_of(2)
        .expect("AGG_REPS too large");
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "rbits\tkey\tdist\tgroups\trows\tthreads\tcand\tns_row_med\tns_row_min\tns_row_max\tlocal_ms\tfinal_ms"
    )
    .unwrap();
    for &kt in &keys {
        for dist in &dists {
            for &g in &groups {
                let d = data(kt, g, dist);
                for &t in &threads {
                    let x = Ctx {
                        data: &d,
                        kt,
                        threads: t,
                    };
                    let mut samples: Vec<Vec<(f64, f64)>> = vec![Vec::new(); cands.len()];
                    let mut reference: Option<Check> = None;
                    for rep in 0..reps {
                        // ABBA: forward on even repetitions, reversed on odd.
                        let order: Vec<usize> = if rep % 2 == 0 {
                            (0..cands.len()).collect()
                        } else {
                            (0..cands.len()).rev().collect()
                        };
                        for i in order {
                            let r = run(cands[i], &x);
                            if cands[i] == Cand::Enc {
                                samples[i].push((r.local, r.fin));
                                continue;
                            }
                            assert_eq!(
                                r.check.groups as usize, d.distinct,
                                "{:?} groups",
                                cands[i]
                            );
                            if cands[i] != Cand::Op {
                                assert_eq!(r.check.cnt as usize, d.rows, "{:?} count", cands[i]);
                                match reference {
                                    None => reference = Some(r.check),
                                    Some(c) => assert_eq!(c, r.check, "{:?} result", cands[i]),
                                }
                            }
                            samples[i].push((r.local, r.fin));
                        }
                    }
                    for (i, s) in samples.iter().enumerate() {
                        let mut tot: Vec<f64> = s
                            .iter()
                            .map(|(a, b)| (a + b) * 1e9 / d.rows as f64)
                            .collect();
                        let mut loc: Vec<f64> = s.iter().map(|(a, _)| a * 1e3).collect();
                        let mut fin: Vec<f64> = s.iter().map(|(_, b)| b * 1e3).collect();
                        let med = median(&mut tot);
                        writeln!(
                            out,
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.1}\t{:.1}",
                            rbits,
                            kt.name(),
                            dist,
                            g,
                            d.rows,
                            t,
                            cands[i].name(),
                            med,
                            tot[0],
                            tot[tot.len() - 1],
                            median(&mut loc),
                            median(&mut fin),
                        )
                        .unwrap();
                    }
                    out.flush().unwrap();
                }
            }
        }
    }
}
