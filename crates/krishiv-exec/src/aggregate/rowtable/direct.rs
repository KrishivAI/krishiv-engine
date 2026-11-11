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
//! takes its first row's table row. A batch that fails the proof, or holds more
//! than [`DIRECT_MAX`] distinct keys, takes the ordinary path.
//!
//! The rows a key's first row is encoded from are the batch's own values, so
//! the table sees the same key bytes and hashes as the per-row encode gives:
//! the groups and results are the same either way.

use std::sync::Mutex;

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
/// Packed keys are at most this many bits (one byte per key column).
const PACKED_BITS: u32 = 16;

/// A table's scratch for the direct path: the packed-key to id map, kept
/// between batches and cleared entry by entry.
#[derive(Debug)]
pub(super) struct Direct {
    scratch: Mutex<Scratch>,
}

#[derive(Debug, Default)]
struct Scratch {
    /// `id + 1` of each packed key seen in the current batch, else 0.
    lut: Vec<u16>,
    /// The packed keys the current batch set in `lut`.
    set: Vec<u16>,
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
        }
    }

    /// The batch-local ids of `keys` (`n` rows), or `None` when the batch
    /// is not provably tiny or holds more than [`DIRECT_MAX`] keys.
    pub(super) fn ids(&self, keys: &[ArrayRef], n: usize) -> Option<DirectIds> {
        if keys.is_empty() || keys.len() * 8 > PACKED_BITS as usize || n == 0 {
            return None;
        }
        let mut packed: Vec<u16> = Vec::new();
        for (c, k) in keys.iter().enumerate() {
            if k.len() != n || k.null_count() > 0 {
                return None;
            }
            add_bytes(&mut packed, k.as_ref(), c == 0, 8 * u32::try_from(c).ok()?)?;
        }
        let mut s = self.scratch.lock().ok()?;
        let Scratch { lut, set } = &mut *s;
        if lut.is_empty() {
            lut.resize(1 << PACKED_BITS, 0);
        }
        let mut ids = Vec::with_capacity(n);
        let mut firsts: Vec<u32> = Vec::new();
        let mut ok = true;
        for (k, &p) in packed.iter().enumerate() {
            let slot = &mut lut[usize::from(p)];
            if *slot == 0 {
                if firsts.len() == DIRECT_MAX {
                    ok = false;
                    break;
                }
                // `k` < `n`, a batch's row count.
                firsts.push(u32::try_from(k).unwrap_or(u32::MAX));
                set.push(p);
                // At most DIRECT_MAX (< u16::MAX) ids.
                *slot = u16::try_from(firsts.len()).unwrap_or(u16::MAX);
            }
            ids.push(*slot - 1);
        }
        for &p in set.iter() {
            lut[usize::from(p)] = 0;
        }
        set.clear();
        ok.then(|| DirectIds {
            ids,
            firsts: UInt32Array::from(firsts),
        })
    }
}

/// Writes (`first`) or ORs in column `a`'s one-byte code of each row,
/// shifted left by `shift`; `None` when a row's value is not one byte.
fn add_bytes(packed: &mut Vec<u16>, a: &dyn Array, first: bool, shift: u32) -> Option<()> {
    fn put(packed: &mut Vec<u16>, first: bool, shift: u32, bytes: impl Iterator<Item = u8>) {
        if first {
            packed.extend(bytes.map(u16::from));
        } else {
            for (p, b) in packed.iter_mut().zip(bytes) {
                *p |= u16::from(b) << shift;
            }
        }
    }
    // A view's length is its low 32 bits; a value of up to 12 bytes is
    // inline from byte 4.
    let mut views = |v: &[u128]| -> Option<()> {
        if !v.iter().all(|&w| (w & 0xffff_ffff) == 1) {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the view's first data byte"
        )]
        put(packed, first, shift, v.iter().map(|&w| (w >> 32) as u8));
        Some(())
    };
    macro_rules! ints {
        ($t:ty) => {{
            let v = a.as_primitive::<$t>().values();
            if !v.iter().all(|&x| i8::try_from(x).is_ok()) {
                return None;
            }
            put(
                packed,
                first,
                shift,
                v.iter()
                    .map(|&x| i8::try_from(x).map_or(0, i8::cast_unsigned)),
            );
            Some(())
        }};
    }
    match a.data_type() {
        DataType::Utf8View => views(a.as_string_view().views()),
        DataType::BinaryView => views(a.as_binary_view().views()),
        DataType::UInt8 => {
            put(
                packed,
                first,
                shift,
                a.as_primitive::<UInt8Type>().values().iter().copied(),
            );
            Some(())
        }
        DataType::Int8 => ints!(Int8Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        _ => None,
    }
}

impl RowTable {
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

    use arrow_array::{Int64Array, RecordBatch, StringViewArray};
    use arrow_schema::{Field, Schema};

    use super::super::RADIX_BITS;
    use super::*;
    use crate::aggregate::table::AggCore;
    use crate::aggregate::{AggregateExpr, AggregateFunction, AggregateOutput, AggregateSpec};
    use crate::hash::HashSeed;

    fn core() -> Arc<AggCore> {
        let input = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8View, true),
            Field::new("b", DataType::Utf8View, true),
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
        let got = d.ids(&[i(vec![3, 3]), i(vec![0, 0])], 2).unwrap();
        assert_eq!(got.ids, [0, 0]);
        assert_eq!(got.firsts.values().to_vec(), [0]);
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
}
