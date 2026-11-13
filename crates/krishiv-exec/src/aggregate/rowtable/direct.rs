// SPDX-License-Identifier: Apache-2.0
//! Direct-address batch ids for a tiny key domain (study 32 D1,
//! `krishiv.exec.agg.directKeys`).
//!
//! A batch whose key columns are each proven to hold one byte per row (string
//! or binary views of length 1, or integers in a byte's range, and no NULLs;
//! at most two columns) packs each row's key into 16 bits. A row's batch-local
//! id is then `lut[packed]`: no encoded key row, no hash and no compare per
//! input row. Only each id's first row is encoded and looked up in the table
//! (in first-occurrence order), as the batch-local map of E2 does; every row
//! takes its first row's table row. A batch that fails the proof, holds more
//! than [`DIRECT_MAX`] distinct keys, or whose rows are mostly distinct keys takes
//! the ordinary path; after the latter two the next [`DIRECT_BACKOFF`]
//! batches skip the attempt. Every column is proven before any is packed.
//!
//! The rows a key's first row is encoded from are the batch's own values, so
//! the table sees the same key bytes and hashes as the per-row encode gives:
//! the groups and results are the same either way.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type};
use arrow_array::{Array, ArrayRef, UInt32Array};
use arrow_schema::DataType;
use krishiv_types::batch::Batch;

use super::{Mode, PARTS, RowPrepared, RowTable, fin, part_of};
use crate::error::{Result, internal};
use crate::hash::ProbeCount;
use crate::rows::EncodedKeys;

/// Most distinct keys a batch may hold on the direct path.
pub(super) const DIRECT_MAX: usize = 1024;
/// Batches that skip the direct attempt after one with more distinct keys
/// than it may hold (past [`DIRECT_MAX`], or past half its rows above
/// [`DIRECT_FLOOR`]).
const DIRECT_BACKOFF: u8 = 16;
/// Distinct keys a batch may always hold; past it, at most half its rows.
const DIRECT_FLOOR: usize = 64;
/// Packed keys are at most this many bits (one byte per key column).
const PACKED_BITS: u32 = 16;
/// Bytes of a table's direct scratch once allocated: the packed-key map and
/// the list of keys a batch set (at most [`DIRECT_MAX`]).
const SCRATCH_BYTES: usize = ((1 << PACKED_BITS) + DIRECT_MAX) * size_of::<u16>();

/// A table's scratch for the direct path: the packed-key to id map, kept
/// between batches and cleared entry by entry.
#[derive(Debug)]
pub(super) struct Direct {
    /// A `Mutex` only because [`RowTable::prepare`] takes `&self` (and
    /// keeps `RowTable: Sync`); one driver owns the table, so it is never
    /// contended, and a poisoned lock takes the ordinary path.
    scratch: Mutex<Scratch>,
    /// Whether `scratch` holds its map ([`SCRATCH_BYTES`]); it is allocated
    /// by the first batch that passes the proof and kept for the table's life.
    allocated: AtomicBool,
}

#[derive(Debug, Default)]
struct Scratch {
    /// `id + 1` of each packed key seen in the current batch, else 0.
    lut: Vec<u16>,
    /// The packed keys the current batch set in `lut`.
    set: Vec<u16>,
    /// Batches left to skip after one with more distinct keys than it may
    /// hold ([`DIRECT_BACKOFF`]).
    skip: u8,
}

/// A batch's rows by batch-local id.
pub(super) struct DirectIds {
    /// Each row's id: the position of its key's first row in `firsts`.
    pub(super) ids: Vec<u16>,
    /// Each id's first row, in first-occurrence order.
    pub(super) firsts: UInt32Array,
}

impl Direct {
    pub(super) fn new() -> Self {
        Self {
            scratch: Mutex::new(Scratch::default()),
            allocated: AtomicBool::new(false),
        }
    }

    /// Bytes the direct scratch holds: 0 until a batch first passes the
    /// proof, then [`SCRATCH_BYTES`] (counted in the table's size).
    pub(super) fn bytes(&self) -> usize {
        if self.allocated.load(Ordering::Relaxed) {
            SCRATCH_BYTES
        } else {
            0
        }
    }

    /// The batch-local ids of `keys` (`n` rows), or `None` when the batch
    /// is not provably tiny, holds more distinct keys than it may (more than
    /// [`DIRECT_MAX`], or more than half its rows above [`DIRECT_FLOOR`]), or
    /// falls in a back-off.
    pub(super) fn ids(&self, keys: &[ArrayRef], n: usize) -> Option<DirectIds> {
        if keys.is_empty() || keys.len() * 8 > PACKED_BITS as usize || n == 0 {
            return None;
        }
        // Row positions are `u32` first rows.
        u32::try_from(n).ok()?;
        let mut s = self.scratch.lock().ok()?;
        if s.skip > 0 {
            s.skip -= 1;
            return None;
        }
        if !keys
            .iter()
            .all(|k| k.len() == n && k.null_count() == 0 && one_byte(k.as_ref()))
        {
            return None;
        }
        let mut packed: Vec<u16> = Vec::with_capacity(n);
        for (c, k) in keys.iter().enumerate() {
            pack(&mut packed, k.as_ref(), c == 0, 8 * u32::try_from(c).ok()?);
        }
        // `pack` writes every row of a column `one_byte` accepted.
        if packed.len() != n {
            return None;
        }
        let Scratch { lut, set, skip } = &mut *s;
        if lut.is_empty() {
            lut.resize(1 << PACKED_BITS, 0);
            set.reserve_exact(DIRECT_MAX);
            self.allocated.store(true, Ordering::Relaxed);
        }
        // A batch whose rows are mostly distinct keys gains nothing (a small one
        // costs little either way).
        let most = DIRECT_MAX.min((n / 2).max(DIRECT_FLOOR));
        let mut ids = Vec::with_capacity(n);
        let mut firsts: Vec<u32> = Vec::new();
        let mut ok = true;
        for (k, &p) in packed.iter().enumerate() {
            let slot = &mut lut[usize::from(p)];
            if *slot == 0 {
                let (Ok(row), Ok(id)) = (u32::try_from(k), u16::try_from(firsts.len() + 1)) else {
                    ok = false;
                    break;
                };
                if firsts.len() == most {
                    ok = false;
                    break;
                }
                firsts.push(row);
                set.push(p);
                *slot = id;
            }
            ids.push(*slot - 1);
        }
        for &p in set.iter() {
            lut[usize::from(p)] = 0;
        }
        set.clear();
        if !ok {
            *skip = DIRECT_BACKOFF;
        }
        ok.then(|| DirectIds {
            ids,
            firsts: UInt32Array::from(firsts),
        })
    }
}

/// Whether every row of `a` holds a one-byte value (string or binary views
/// of length 1, integers in a byte's range); NULLs are checked by the caller.
fn one_byte(a: &dyn Array) -> bool {
    // A view's length is its low 32 bits.
    let views = |v: &[u128]| v.iter().all(|&w| (w & 0xffff_ffff) == 1);
    macro_rules! ints {
        ($t:ty) => {
            a.as_primitive::<$t>()
                .values()
                .iter()
                .all(|&x| i8::try_from(x).is_ok())
        };
    }
    match a.data_type() {
        DataType::Utf8View => views(a.as_string_view().views()),
        DataType::BinaryView => views(a.as_binary_view().views()),
        DataType::UInt8 | DataType::Int8 => true,
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        _ => false,
    }
}

/// Writes (`first`) or ORs in column `a`'s one-byte code of each row,
/// shifted left by `shift`. `a` passed [`one_byte`].
fn pack(packed: &mut Vec<u16>, a: &dyn Array, first: bool, shift: u32) {
    fn put(packed: &mut Vec<u16>, first: bool, shift: u32, bytes: impl Iterator<Item = u8>) {
        if first {
            packed.extend(bytes.map(u16::from));
        } else {
            for (p, b) in packed.iter_mut().zip(bytes) {
                *p |= u16::from(b) << shift;
            }
        }
    }
    // A value of up to 12 bytes is inline from byte 4 of its view.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the view's first data byte"
    )]
    let first_byte = |w: &u128| (*w >> 32) as u8;
    macro_rules! ints {
        ($t:ty) => {
            put(
                packed,
                first,
                shift,
                a.as_primitive::<$t>()
                    .values()
                    .iter()
                    .map(|&x| i8::try_from(x).map_or(0, i8::cast_unsigned)),
            )
        };
    }
    match a.data_type() {
        DataType::Utf8View => put(
            packed,
            first,
            shift,
            a.as_string_view().views().iter().map(first_byte),
        ),
        DataType::BinaryView => put(
            packed,
            first,
            shift,
            a.as_binary_view().views().iter().map(first_byte),
        ),
        DataType::UInt8 => put(
            packed,
            first,
            shift,
            a.as_primitive::<UInt8Type>().values().iter().copied(),
        ),
        DataType::Int8 => ints!(Int8Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        _ => {}
    }
}

impl RowTable {
    /// Finds a batch's groups by direct addressing when its keys are tiny
    /// (`krishiv.exec.agg.directKeys`).
    pub(crate) fn with_direct(mut self, on: bool) -> Self {
        self.direct = on.then(Direct::new);
        self
    }

    /// [`prepare`](RowTable::prepare) on the direct path: only the first
    /// row of each id is encoded; the partition counts are its rows'.
    pub(super) fn prepare_direct(
        &self,
        b: &Batch,
        keys: &[ArrayRef],
        d: DirectIds,
        states: bool,
    ) -> Result<RowPrepared> {
        let firsts: Vec<ArrayRef> = keys
            .iter()
            .map(|k| arrow_select::take::take(k.as_ref(), &d.firsts, None))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| internal(e.to_string()))?;
        let encoded = super::layout(&self.core)?.encode(&firsts)?;
        let n = d.ids.len();
        let mut counts = [0; PARTS];
        if self.bits == 0 {
            counts[0] = n;
        } else {
            let mut per = vec![0usize; encoded.len()];
            for &i in &d.ids {
                per[usize::from(i)] += 1;
            }
            for (&h, c) in encoded.hashes().iter().zip(per) {
                counts[part_of(h, self.bits)] += c;
            }
        }
        // One-byte values are inline: no heap bytes.
        let heap_counts = if self.views.is_some() {
            vec![0; PARTS]
        } else {
            Vec::new()
        };
        let cols = b.data().columns();
        Ok(RowPrepared {
            keys: encoded,
            local: Some(d.ids.into_boxed_slice()),
            counts,
            heap_counts,
            cols: cols.to_vec(),
            weights: b.weights().clone(),
            states,
        })
    }
}

impl RowTable {
    /// [`lookup`](RowTable::lookup) on the direct path: `keys` holds each
    /// id's first row, `ids` each input row's id. Every first row is looked
    /// up (or, appending, every row pushed) as the per-row path would.
    pub(super) fn lookup_direct(
        &mut self,
        keys: &EncodedKeys,
        ids: &[u16],
        refs: &mut Vec<u32>,
    ) -> Result<()> {
        let (hashes, heap) = (keys.hashes(), keys.heap());
        let limit = match self.mode {
            Mode::Append => {
                self.unique &= ids.is_empty();
                for &i in ids {
                    let i = usize::from(i);
                    let h = *hashes
                        .get(i)
                        .ok_or_else(|| internal("direct id out of range".to_owned()))?;
                    refs.push(self.push(keys.row(i), heap, h)?);
                }
                return Ok(());
            }
            Mode::Fixed { limit } => limit,
            Mode::Grow => usize::MAX,
        };
        let mut count = ProbeCount::default();
        let mut found = Vec::with_capacity(keys.len());
        for (k, &h) in hashes.iter().enumerate() {
            let hg = (h, fin(h, self.gseed));
            let row = keys.row(k);
            found.push(match (self.views.is_some(), self.probes.is_some()) {
                (true, false) => {
                    self.find_or_insert::<true, false>(row, heap, hg, limit, &mut count)?
                }
                (false, false) => {
                    self.find_or_insert::<false, false>(row, heap, hg, limit, &mut count)?
                }
                (true, true) => {
                    self.find_or_insert::<true, true>(row, heap, hg, limit, &mut count)?
                }
                (false, true) => {
                    self.find_or_insert::<false, true>(row, heap, hg, limit, &mut count)?
                }
            });
        }
        if let Some(totals) = &self.probes
            && count.lookups > 0
        {
            totals.add(&count);
        }
        refs.reserve(ids.len());
        for &i in ids {
            refs.push(
                *found
                    .get(usize::from(i))
                    .ok_or_else(|| internal("direct id out of range".to_owned()))?,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        BinaryViewArray, Int8Array, Int16Array, Int32Array, Int64Array, RecordBatch,
        StringViewArray, UInt8Array,
    };
    use arrow_schema::{Field, Schema};

    use super::super::RADIX_BITS;
    use super::*;
    use crate::aggregate::table::AggCore;
    use crate::aggregate::{AggregateExpr, AggregateFunction, AggregateOutput, AggregateSpec};
    use crate::hash::HashSeed;
    use krishiv_types::batch::Weights;

    fn core() -> Arc<AggCore> {
        core_of(DataType::Utf8View, DataType::Utf8View)
    }

    fn core_of(a: DataType, b: DataType) -> Arc<AggCore> {
        let input = Arc::new(Schema::new(vec![
            Field::new("a", a, true),
            Field::new("b", b, true),
            Field::new("v", DataType::Int64, true),
        ]));
        let spec = AggregateSpec {
            input,
            group_by: vec![0, 1],
            aggregates: vec![
                AggregateExpr::count_star("c"),
                AggregateExpr::new(AggregateFunction::Sum, 2, DataType::Int64, "s"),
            ],
            output: AggregateOutput::Final,
            ansi: true,
            spill: None,
            require_unit_weights: false,
            legacy_statistical: false,
            udafs: None,
        };
        Arc::new(AggCore::new(spec).unwrap())
    }

    fn batch(core: &AggCore, a: Vec<&str>, b: Vec<&str>) -> Batch {
        let v = Int64Array::from((0..a.len() as i64).collect::<Vec<_>>());
        let cols: Vec<ArrayRef> = vec![
            Arc::new(StringViewArray::from(a)),
            Arc::new(StringViewArray::from(b)),
            Arc::new(v),
        ];
        Batch::new(RecordBatch::try_new(Arc::clone(&core.spec.input), cols).unwrap())
    }

    /// `n` rows of keys from `alphabet`, by a fixed sequence.
    fn tiny(core: &AggCore, n: usize, start: u64, alphabet: &[&'static str]) -> Batch {
        let mut x = start.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut pick = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            alphabet[usize::try_from(x % alphabet.len() as u64).unwrap()]
        };
        let (a, b): (Vec<_>, Vec<_>) = (0..n).map(|_| (pick(), pick())).unzip();
        batch(core, a, b)
    }

    fn rows(out: &RecordBatch) -> Vec<(String, String, i64, i64)> {
        let s = |i: usize| out.column(i).as_string_view().clone();
        let n = |i: usize| out.column(i).as_primitive::<Int64Type>().clone();
        let (a, b, c, v) = (s(0), s(1), n(2), n(3));
        let mut r: Vec<_> = (0..out.num_rows())
            .map(|i| {
                (
                    a.value(i).to_owned(),
                    b.value(i).to_owned(),
                    c.value(i),
                    v.value(i),
                )
            })
            .collect();
        r.sort();
        r
    }

    #[test]
    fn ids_follow_first_occurrence_and_the_map_is_cleared_between_batches() {
        let core = core();
        let d = Direct::new();
        let b = batch(&core, vec!["x", "y", "x", "z"], vec!["1", "1", "1", "1"]);
        let cols = b.data().columns();
        let got = d.ids(&cols[..2], 4).unwrap();
        assert_eq!(got.ids, [0, 1, 0, 2]);
        assert_eq!(got.firsts.values().to_vec(), [0, 1, 3]);
        let b = batch(&core, vec!["z", "x"], vec!["1", "2"]);
        let got = d.ids(&b.data().columns()[..2], 2).unwrap();
        assert_eq!(got.ids, [0, 1]);
    }

    #[test]
    fn only_one_byte_values_without_nulls_qualify() {
        let core = core();
        let d = Direct::new();
        let two = batch(&core, vec!["x", "yy"], vec!["1", "1"]);
        assert!(d.ids(&two.data().columns()[..2], 2).is_none());
        let empty = batch(&core, vec!["x", ""], vec!["1", "1"]);
        assert!(d.ids(&empty.data().columns()[..2], 2).is_none());
        let nulls: ArrayRef = Arc::new(StringViewArray::from(vec![Some("x"), None]));
        assert!(d.ids(&[nulls], 2).is_none());
        let three: Vec<ArrayRef> = (0..3)
            .map(|_| Arc::new(StringViewArray::from(vec!["x"])) as ArrayRef)
            .collect();
        assert!(d.ids(&three, 1).is_none());
        let small: ArrayRef = Arc::new(Int64Array::from(vec![-128, 127, 5]));
        assert_eq!(d.ids(&[small], 3).unwrap().ids, [0, 1, 2]);
        let wide: ArrayRef = Arc::new(Int64Array::from(vec![1, 128]));
        assert!(d.ids(&[wide], 2).is_none());
        // -1 and 255 are different keys of an Int64 column: 255 does not fit.
        let neg: ArrayRef = Arc::new(Int64Array::from(vec![-1, 255]));
        assert!(d.ids(&[neg], 2).is_none());
    }

    #[test]
    fn too_many_distinct_keys_fall_back_and_leave_the_map_clean() {
        let d = Direct::new();
        let i = |v: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(v)) };
        let a: Vec<i64> = (0..2048).map(|k| k % 64).collect();
        let b: Vec<i64> = (0..2048).map(|k| k / 64).collect();
        assert!(d.ids(&[i(a), i(b)], 2048).is_none());
        // The next DIRECT_BACKOFF batches skip the attempt, then it resumes
        // with a clean map.
        let small = || [i(vec![3, 3]), i(vec![0, 0])];
        for _ in 0..DIRECT_BACKOFF {
            assert!(d.ids(&small(), 2).is_none());
        }
        let got = d.ids(&small(), 2).unwrap();
        assert_eq!(got.ids, [0, 0]);
        assert_eq!(got.firsts.values().to_vec(), [0]);
    }

    #[test]
    fn a_batch_of_mostly_distinct_keys_falls_back() {
        let d = Direct::new();
        let i = |v: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(v)) };
        // 200 rows, 100 keys: half the rows, allowed.
        let half: Vec<i64> = (0..200).map(|k| k % 100).collect();
        let b: Vec<i64> = (0..200).map(|k| (k % 100) / 50).collect();
        let half_a: Vec<i64> = half.iter().map(|k| k % 50).collect();
        let got = d.ids(&[i(half_a), i(b.clone())], 200).unwrap();
        assert_eq!(got.firsts.len(), 100);
        // 200 rows, 101 keys: more than half, the ordinary path; then back-off.
        let mut a: Vec<i64> = (0..200).map(|k| (k % 100) % 50).collect();
        let mut b2 = b;
        a[199] = 60;
        b2[199] = 0;
        assert!(d.ids(&[i(a), i(b2)], 200).is_none());
        assert!(d.ids(&[i(vec![1]), i(vec![1])], 1).is_none());
        // A small batch may be all new keys (up to DIRECT_FLOOR).
        let fresh = Direct::new();
        let all: Vec<i64> = (0..64).collect();
        assert_eq!(fresh.ids(&[i(all)], 64).unwrap().firsts.len(), 64);
        let over: Vec<i64> = (0..65).collect();
        assert!(Direct::new().ids(&[i(over)], 65).is_none());
    }

    /// The direct path gives the groups, counts and sums of the per-row path
    /// in every index mode and width, also when batches that do not qualify
    /// are mixed in.
    #[test]
    fn direct_tables_equal_per_row_tables() {
        let core = core();
        let seed = HashSeed::new(Some(3));
        let alphabet = ["A", "N", "R", "O", "F", "x"];
        for mode in [Mode::Grow, Mode::Fixed { limit: 3 }, Mode::Append] {
            for bits in [0, RADIX_BITS] {
                let out = |direct: bool| {
                    let mut t = RowTable::new(&core, &seed, mode, true, bits)
                        .unwrap()
                        .with_direct(direct);
                    for s in 0..8 {
                        let b = if s == 5 {
                            batch(&core, vec!["AA", "A", "N"], vec!["F", "F", "O"])
                        } else {
                            tiny(&core, 4096, s, &alphabet)
                        };
                        let p = t.prepare(&b, false).unwrap();
                        assert_eq!(p.local.is_some(), direct && s != 5);
                        let want = t.projected(&p);
                        t.apply(&p).unwrap();
                        assert!(t.memory_size() <= want, "{} > {want}", t.memory_size());
                    }
                    let mut m = RowTable::new(&core, &seed, Mode::Grow, false, bits).unwrap();
                    for p in 0..t.part_count() {
                        let c = t.merge_chunk(p, 0..t.part_len(p)).unwrap();
                        m.apply_merge(&t, &c).unwrap();
                    }
                    rows(&m.output_batch(&m.ids(0..m.len())).unwrap())
                };
                let (on, off) = (out(true), out(false));
                assert_eq!(on.len(), 37, "{mode:?} {bits}");
                assert_eq!(on, off, "{mode:?} {bits}");
            }
        }
    }

    /// A key column of `t` with `n` rows over a five-key domain, cut from a
    /// longer array at offset 3 (a sliced column).
    fn typed(t: &DataType, n: usize, start: usize) -> ArrayRef {
        let k: Vec<u8> = (0..n + 10)
            .map(|i| u8::try_from((i + start) * 7 % 5).unwrap())
            .collect();
        let it = || k.iter().copied();
        let a: ArrayRef = match t {
            DataType::Utf8View => Arc::new(StringViewArray::from_iter_values(
                it().map(|x| ["a", "b", "c", "d", "e"][usize::from(x)]),
            )),
            DataType::BinaryView => {
                Arc::new(BinaryViewArray::from_iter_values(it().map(|x| [b'a' + x])))
            }
            DataType::UInt8 => Arc::new(UInt8Array::from_iter_values(it().map(|x| 250 + x))),
            DataType::Int8 => Arc::new(Int8Array::from_iter_values(
                it().map(|x| i8::MIN + i8::try_from(x).unwrap()),
            )),
            DataType::Int16 => Arc::new(Int16Array::from_iter_values(
                it().map(|x| i16::from(x) * 31 - 64),
            )),
            DataType::Int32 => Arc::new(Int32Array::from_iter_values(
                it().map(|x| 127 - i32::from(x)),
            )),
            other => panic!("no test column of {other}"),
        };
        a.slice(3, n)
    }

    /// Every key type the proof takes, in mixed pairs and sliced, gives the
    /// groups, counts and sums of the per-row path.
    #[test]
    fn typed_and_sliced_keys_equal_per_row_tables() {
        use DataType::{BinaryView, Int8, Int16, Int32, UInt8, Utf8View};
        let seed = HashSeed::new(Some(5));
        for (ta, tb) in [
            (Utf8View, Int32),
            (BinaryView, UInt8),
            (Int8, Int16),
            (UInt8, Utf8View),
        ] {
            let core = core_of(ta.clone(), tb.clone());
            let out = |direct: bool| {
                let mut t = RowTable::new(&core, &seed, Mode::Grow, true, RADIX_BITS)
                    .unwrap()
                    .with_direct(direct);
                for start in [0, 1, 4] {
                    let n = 3000;
                    let v = Int64Array::from((0..n as i64).collect::<Vec<_>>());
                    let cols = vec![typed(&ta, n, start), typed(&tb, n, 2 * start), Arc::new(v)];
                    let b = Batch::new(
                        RecordBatch::try_new(Arc::clone(&core.spec.input), cols).unwrap(),
                    );
                    let p = t.prepare(&b, false).unwrap();
                    assert_eq!(p.local.is_some(), direct, "{ta} {tb}");
                    t.apply(&p).unwrap();
                }
                let mut m = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
                for p in 0..t.part_count() {
                    let c = t.merge_chunk(p, 0..t.part_len(p)).unwrap();
                    m.apply_merge(&t, &c).unwrap();
                }
                let out = m.output_batch(&m.ids(0..m.len())).unwrap();
                let mut r: Vec<String> = (0..out.num_rows())
                    .map(|i| {
                        let cells: Vec<_> = out.columns().iter().map(|c| c.slice(i, 1)).collect();
                        format!("{cells:?}")
                    })
                    .collect();
                r.sort();
                r
            };
            let (on, off) = (out(true), out(false));
            assert!(on.len() >= 5, "{ta} {tb}");
            assert_eq!(on, off, "{ta} {tb}");
        }
    }

    /// The direct map is part of the table's size from the first batch that
    /// takes the path, and `projected` covers it.
    #[test]
    fn the_map_is_counted_in_the_table_size_once_allocated() {
        let core = core();
        let seed = HashSeed::new(Some(3));
        let mut t = RowTable::new(&core, &seed, Mode::Grow, true, 0)
            .unwrap()
            .with_direct(true);
        let wide = t
            .prepare(&batch(&core, vec!["AA"], vec!["B"]), false)
            .unwrap();
        assert!(wide.local.is_none());
        t.apply(&wide).unwrap();
        assert_eq!(t.direct.as_ref().map(Direct::bytes), Some(0));
        let before = t.memory_size();
        let p = t
            .prepare(&batch(&core, vec!["A"], vec!["B"]), false)
            .unwrap();
        assert!(p.local.is_some());
        let want = t.projected(&p);
        t.apply(&p).unwrap();
        assert!(t.memory_size() >= before + SCRATCH_BYTES);
        assert!(t.memory_size() <= want, "{} > {want}", t.memory_size());
    }

    /// One batch of the property test: `n` rows of keys drawn from `domain`
    /// values per column (`nulls`: row 0's first key is NULL; `wide`: row 0's
    /// second key does not fit a byte), with per-row weights when `weighted`.
    #[derive(Debug, Clone)]
    struct Shape {
        n: usize,
        domain: u8,
        seed: u64,
        nulls: bool,
        wide: bool,
        weighted: bool,
    }

    fn shapes() -> impl proptest::strategy::Strategy<Value = Vec<Shape>> {
        use proptest::prelude::*;
        proptest::collection::vec(
            (
                // Mostly small batches; some past 2 * DIRECT_MAX rows over a
                // wide domain cross DIRECT_MAX.
                prop_oneof![4 => 1usize..400, 1 => 2100usize..2600],
                1u8..48,
                any::<u64>(),
                any::<bool>(),
                any::<bool>(),
                any::<bool>(),
            )
                .prop_map(|(n, domain, seed, nulls, wide, weighted)| Shape {
                    n,
                    domain,
                    seed,
                    nulls: nulls && seed % 4 == 0,
                    wide: wide && seed % 5 == 0,
                    weighted,
                }),
            // Past DIRECT_BACKOFF batches: direct, bailed, skipped and
            // resumed batches meet in one table.
            1..24,
        )
    }

    /// A key column of type `t` for `s` (column `c`).
    fn column(t: &DataType, s: &Shape, c: u64) -> ArrayRef {
        let mut x = (s.seed ^ c.wrapping_mul(0x9e37_79b9)) | 1;
        let mut keys: Vec<Option<i64>> = (0..s.n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                Some(i64::try_from(x % u64::from(s.domain)).unwrap())
            })
            .collect();
        if s.nulls && c == 0 {
            keys[0] = None;
        }
        let wide = s.wide && c == 1;
        let chars: Vec<char> = ('A'..='Z').chain('a'..='z').collect();
        match t {
            DataType::Utf8View => Arc::new(StringViewArray::from_iter(
                keys.iter().enumerate().map(|(i, k)| {
                    k.map(|k| {
                        let ch = chars[usize::try_from(k).unwrap()].to_string();
                        if wide && i == 0 {
                            format!("{ch}{ch}")
                        } else {
                            ch
                        }
                    })
                }),
            )),
            DataType::Int32 => Arc::new(Int32Array::from_iter(keys.iter().enumerate().map(
                |(i, k)| {
                    k.map(|k| {
                        if wide && i == 0 {
                            1000
                        } else {
                            i32::try_from(k).unwrap() - 20
                        }
                    })
                },
            ))),
            DataType::UInt8 => Arc::new(UInt8Array::from_iter(
                keys.iter()
                    .map(|k| k.map(|k| u8::try_from(k).unwrap() + 200)),
            )),
            other => panic!("no property column of {other}"),
        }
    }

    /// The final rows of `t`, merged into one table, as sorted strings.
    fn final_rows(core: &Arc<AggCore>, seed: &HashSeed, t: &RowTable) -> Vec<String> {
        let m = merged(core, seed, t);
        let out = m.output_batch(&m.ids(0..m.len())).unwrap();
        let mut r: Vec<String> = (0..out.num_rows())
            .map(|i| {
                let cells: Vec<_> = out.columns().iter().map(|c| c.slice(i, 1)).collect();
                format!("{cells:?}")
            })
            .collect();
        r.sort();
        r
    }

    fn merged(core: &Arc<AggCore>, seed: &HashSeed, t: &RowTable) -> RowTable {
        let mut m = RowTable::new(core, seed, Mode::Grow, false, RADIX_BITS).unwrap();
        for p in 0..t.part_count() {
            let c = t.merge_chunk(p, 0..t.part_len(p)).unwrap();
            m.apply_merge(t, &c).unwrap();
        }
        m
    }

    /// A weighted batch and a batch of states take the direct path, and give
    /// the per-row path's results.
    #[test]
    fn weights_and_states_take_the_direct_path() {
        let core = core_of(DataType::Utf8View, DataType::Int32);
        let seed = HashSeed::new(Some(13));
        let shape = Shape {
            n: 600,
            domain: 6,
            seed: 7,
            nulls: false,
            wide: false,
            weighted: true,
        };
        let run = |direct: bool| {
            let mut t = RowTable::new(&core, &seed, Mode::Grow, true, RADIX_BITS)
                .unwrap()
                .with_direct(direct);
            let v = Int64Array::from((0..600).collect::<Vec<i64>>());
            let cols = vec![
                column(&DataType::Utf8View, &shape, 0),
                column(&DataType::Int32, &shape, 1),
                Arc::new(v) as ArrayRef,
            ];
            let data = RecordBatch::try_new(Arc::clone(&core.spec.input), cols).unwrap();
            let w = Int64Array::from((0..600).map(|i| i % 3 - 1).collect::<Vec<i64>>());
            let p = t
                .prepare(
                    &Batch::with_weights(data, Weights::Column(w)).unwrap(),
                    false,
                )
                .unwrap();
            assert_eq!(p.local.is_some(), direct);
            t.apply(&p).unwrap();
            let m = merged(&core, &seed, &t);
            let states = Batch::new(m.state_batch(&m.ids(0..m.len())).unwrap());
            let mut again = RowTable::new(&core, &seed, Mode::Grow, true, RADIX_BITS)
                .unwrap()
                .with_direct(direct);
            // One row per group, at most 36: within the floor.
            let p = again.prepare(&states, true).unwrap();
            assert!(states.num_rows() <= DIRECT_FLOOR);
            assert_eq!(p.local.is_some(), direct);
            again.apply(&p).unwrap();
            (
                final_rows(&core, &seed, &t),
                final_rows(&core, &seed, &again),
            )
        };
        let (on, off) = (run(true), run(false));
        assert_eq!(on, off);
        assert_eq!(on.0, on.1);
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(32))]
        /// The flag on gives the groups, counts and sums of the flag off for
        /// any mix of key types, domains, NULLs, too-wide values, weights and
        /// batch sizes, over input rows and over the states they make.
        #[test]
        fn direct_keys_on_equal_off(
            types in proptest::sample::select(vec![
                (DataType::Utf8View, DataType::Int32),
                (DataType::UInt8, DataType::Utf8View),
                (DataType::Int32, DataType::UInt8),
            ]),
            batches in shapes(),
            mode in proptest::sample::select(vec![Mode::Grow, Mode::Fixed { limit: 7 }, Mode::Append]),
        ) {
            let (ta, tb) = types;
            let core = core_of(ta.clone(), tb.clone());
            let seed = HashSeed::new(Some(11));
            let run = |direct: bool| {
                let mut t = RowTable::new(&core, &seed, mode, true, RADIX_BITS)
                    .unwrap()
                    .with_direct(direct);
                for s in &batches {
                    let v = Int64Array::from((0..s.n as i64).collect::<Vec<_>>());
                    let cols = vec![column(&ta, s, 0), column(&tb, s, 1), Arc::new(v) as ArrayRef];
                    let data = RecordBatch::try_new(Arc::clone(&core.spec.input), cols).unwrap();
                    let b = if s.weighted {
                        let w = Int64Array::from(
                            (0..s.n as i64).map(|i| i % 3 - 1).collect::<Vec<_>>(),
                        );
                        Batch::with_weights(data, Weights::Column(w)).unwrap()
                    } else {
                        Batch::new(data)
                    };
                    let p = t.prepare(&b, false).unwrap();
                    let want = t.projected(&p);
                    t.apply(&p).unwrap();
                    assert!(t.memory_size() <= want, "{} > {want}", t.memory_size());
                }
                // The merged states, fed again as a state batch.
                let m = merged(&core, &seed, &t);
                let states = Batch::new(m.state_batch(&m.ids(0..m.len())).unwrap());
                let mut again = RowTable::new(&core, &seed, Mode::Grow, true, RADIX_BITS)
                    .unwrap()
                    .with_direct(direct);
                let p = again.prepare(&states, true).unwrap();
                again.apply(&p).unwrap();
                (final_rows(&core, &seed, &t), final_rows(&core, &seed, &again))
            };
            let (on, off) = (run(true), run(false));
            proptest::prop_assert_eq!(&on.0, &off.0);
            proptest::prop_assert_eq!(&on.1, &off.1);
            proptest::prop_assert_eq!(&on.0, &on.1);
        }
    }
}
