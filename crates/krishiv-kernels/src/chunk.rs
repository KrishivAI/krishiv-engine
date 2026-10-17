// SPDX-License-Identifier: Apache-2.0
//! The generic 64-row chunk engine behind every elementwise binary kernel.
//!
//! [`compute`] inspects the operands once, picks the
//! `<NULLS, L_SCALAR, R_SCALAR>` monomorph and the CPU target, and runs the
//! loop. Each chunk is computed branch-free; a policy that checks overflow
//! ORs a cheap "may be bad" flag over the chunk and, only if it is set,
//! re-scans the chunk with the exact verdict to build a 64-bit word of bad
//! rows, which is then masked with the validity word (a NULL row never
//! raises).

use std::marker::PhantomData;

use arrow_array::{Array, ArrowPrimitiveType};
use arrow_buffer::bit_chunk_iterator::BitChunks;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};

use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::ops::Operand;
use crate::policy::{Overflow, OverflowMode};
use crate::unsafe_dispatch::{Kernel, run};

/// Rows per chunk: one validity word.
pub(crate) const CHUNK: usize = 64;

/// An elementwise operation on two lanes.
pub(crate) trait LaneOp<T>: Copy {
    /// The unchecked result.
    fn wrapping(self, a: T, b: T) -> T;
    /// The result and a cheap flag that is set for every bad row (it may
    /// also be set for a good one; [`LaneOp::is_bad`] decides).
    fn checked(self, a: T, b: T) -> (T, bool);
    /// The exact verdict for a row whose flag was set (slow path). For a good
    /// row, the value [`LaneOp::checked`] returned must be the exact result.
    fn is_bad(self, a: T, b: T) -> bool;
}

/// One operand reduced to its values.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Side<'a, T> {
    Array(&'a [T]),
    Scalar(T),
}

/// Operands resolved once per batch.
#[derive(Debug)]
pub(crate) enum Resolved<'a, T> {
    /// A NULL scalar: every row is NULL and nothing is computed.
    AllNull(usize),
    /// Values to compute, with the union of the input validities (only
    /// present when some row is NULL).
    Lanes {
        l: Side<'a, T>,
        r: Side<'a, T>,
        nulls: Option<NullBuffer>,
        len: usize,
    },
}

/// Resolves the operands of a binary kernel. Errors (engine bugs): two
/// scalars, or arrays of different lengths.
pub(crate) fn resolve<'a, P: ArrowPrimitiveType>(
    l: Operand<'a, P>,
    r: Operand<'a, P>,
) -> Result<Resolved<'a, P::Native>, KernelError> {
    let len = match (l, r) {
        (Operand::Array(a), Operand::Array(b)) if a.len() != b.len() => {
            return Err(KernelError::internal(format!(
                "binary kernel operands differ in length: {} vs {}",
                a.len(),
                b.len()
            )));
        }
        (Operand::Array(a), _) | (_, Operand::Array(a)) => a.len(),
        (Operand::Scalar(_), Operand::Scalar(_)) => {
            return Err(KernelError::internal(
                "binary kernel needs at least one array operand",
            ));
        }
    };
    let side = |o: Operand<'a, P>| match o {
        Operand::Array(a) => Some((
            Side::Array(a.values().as_ref()),
            a.nulls().filter(|n| n.null_count() > 0),
        )),
        Operand::Scalar(Some(v)) => Some((Side::Scalar(v), None)),
        Operand::Scalar(None) => None,
    };
    let (Some((l, ln)), Some((r, rn))) = (side(l), side(r)) else {
        return Ok(Resolved::AllNull(len));
    };
    Ok(Resolved::Lanes {
        l,
        r,
        nulls: NullBuffer::union(ln, rn),
        len,
    })
}

/// The values of a chunk computation and, for [`OverflowMode::NullOnOverflow`],
/// the output validity (input validity minus bad rows).
#[derive(Debug)]
pub(crate) struct Computed<T> {
    pub(crate) values: Vec<T>,
    pub(crate) nulls: Option<NullBuffer>,
}

/// Runs `op` over two resolved sides into `out` (one slot per row; `nulls`
/// is the input validity). `Err(row)`: under [`OverflowMode::AnsiError`],
/// the first bad non-NULL row.
pub(crate) fn compute<T, L, O>(
    ctx: &KernelCtx,
    l: Side<'_, T>,
    r: Side<'_, T>,
    nulls: Option<NullBuffer>,
    mut out: Vec<T>,
    op: L,
) -> Result<Computed<T>, usize>
where
    T: Copy + Default,
    L: LaneOp<T>,
    O: Overflow,
{
    let len = out.len();
    let n = nulls.as_ref();
    macro_rules! go {
        ($nulls:literal, $ls:literal, $rs:literal) => {
            run(
                ctx,
                Chunks::<'_, T, L, O, $nulls, $ls, $rs> {
                    l,
                    r,
                    validity: n,
                    out: &mut out,
                    op,
                    policy: PhantomData,
                },
            )
        };
    }
    let bad = match (n.is_some(), l, r) {
        (false, Side::Array(_), Side::Array(_)) => go!(false, false, false),
        (false, Side::Array(_), Side::Scalar(_)) => go!(false, false, true),
        (false, Side::Scalar(_), _) => go!(false, true, false),
        (true, Side::Array(_), Side::Array(_)) => go!(true, false, false),
        (true, Side::Array(_), Side::Scalar(_)) => go!(true, false, true),
        (true, Side::Scalar(_), _) => go!(true, true, false),
    }?;
    let nulls = match bad {
        None => nulls,
        Some(bad) => Some(clear_bits(nulls.as_ref(), &bad, len)),
    };
    Ok(Computed { values: out, nulls })
}

/// `validity AND NOT bad` as a new null buffer (all valid when `validity` is
/// `None`). `bad` has one word per chunk.
fn clear_bits(validity: Option<&NullBuffer>, bad: &[u64], len: usize) -> NullBuffer {
    let mut bytes = Vec::with_capacity(bad.len().saturating_mul(8));
    let chunks = validity.map(|v| v.inner().bit_chunks());
    let mut valid = chunks.as_ref().map(BitChunks::iter_padded);
    for &b in bad {
        let v = valid.as_mut().map_or(u64::MAX, |it| it.next().unwrap_or(0));
        bytes.extend_from_slice(&(v & !b).to_le_bytes());
    }
    NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(bytes), 0, len))
}

/// Mask of the low `n` bits (`n <= 64`).
fn low_bits(n: usize) -> u64 {
    u32::try_from(n)
        .ok()
        .and_then(|n| 1_u64.checked_shl(n))
        .map_or(u64::MAX, |b| b.wrapping_sub(1))
}

/// The chunk loop for one monomorph; `run` compiles it per CPU target.
struct Chunks<'a, T, L, O, const NULLS: bool, const L_SCALAR: bool, const R_SCALAR: bool> {
    l: Side<'a, T>,
    r: Side<'a, T>,
    validity: Option<&'a NullBuffer>,
    out: &'a mut [T],
    op: L,
    policy: PhantomData<O>,
}

/// Splits a side into full chunks, the remainder and the scalar value.
pub(crate) fn split<T: Copy + Default>(s: Side<'_, T>) -> (&[[T; CHUNK]], &[T], T) {
    match s {
        Side::Array(a) => {
            let (c, t) = a.as_chunks::<CHUNK>();
            (c, t, T::default())
        }
        Side::Scalar(v) => (&[], &[], v),
    }
}

impl<T, L, O, const NULLS: bool, const L_SCALAR: bool, const R_SCALAR: bool> Kernel
    for Chunks<'_, T, L, O, NULLS, L_SCALAR, R_SCALAR>
where
    T: Copy + Default,
    L: LaneOp<T>,
    O: Overflow,
{
    /// `Ok(Some(words))`: bad-row words for `NullOnOverflow`; `Err(row)`:
    /// the first bad non-NULL row for `AnsiError`.
    type Output = Result<Option<Vec<u64>>, usize>;

    #[inline(always)]
    fn run(self) -> Self::Output {
        let (lc, lt, ls) = split(self.l);
        let (rc, rt, rs) = split(self.r);
        let zero = [T::default(); CHUNK];
        let (out_chunks, out_tail) = self.out.as_chunks_mut::<CHUNK>();
        let n_words = out_chunks
            .len()
            .saturating_add(usize::from(!out_tail.is_empty()));
        let chunks = self
            .validity
            .filter(|_| NULLS)
            .map(|v| v.inner().bit_chunks());
        let mut valid = chunks.as_ref().map(BitChunks::iter_padded);
        let mut next_valid = || valid.as_mut().map_or(u64::MAX, |it| it.next().unwrap_or(0));
        let mut bad_words: Option<Vec<u64>> = None;
        let mut on_bad = |word: usize, bad: u64| -> Result<(), usize> {
            if bad == 0 {
                return Ok(());
            }
            match O::MODE {
                OverflowMode::AnsiError => Err(word
                    .wrapping_mul(CHUNK)
                    .wrapping_add(bad.trailing_zeros() as usize)),
                _ => {
                    if let Some(w) = bad_words
                        .get_or_insert_with(|| vec![0; n_words])
                        .get_mut(word)
                    {
                        *w = bad;
                    }
                    Ok(())
                }
            }
        };
        for (i, o) in out_chunks.iter_mut().enumerate() {
            let a = lc.get(i).unwrap_or(&zero);
            let b = rc.get(i).unwrap_or(&zero);
            let bad = chunk::<T, L, O, L_SCALAR, R_SCALAR>(a, ls, b, rs, o, self.op);
            if O::CHECKS {
                on_bad(i, bad & next_valid())?;
            }
        }
        let rem = out_tail.len();
        if rem > 0 {
            let (mut a, mut b, mut o) = (zero, zero, zero);
            copy_prefix(&mut a, lt);
            copy_prefix(&mut b, rt);
            let bad = chunk::<T, L, O, L_SCALAR, R_SCALAR>(&a, ls, &b, rs, &mut o, self.op);
            for (d, s) in out_tail.iter_mut().zip(&o) {
                *d = *s;
            }
            if O::CHECKS {
                on_bad(out_chunks.len(), bad & next_valid() & low_bits(rem))?;
            }
        }
        Ok(bad_words)
    }
}

/// Copies `src` (shorter than a chunk) into the front of `dst`.
fn copy_prefix<T: Copy>(dst: &mut [T; CHUNK], src: &[T]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = *s;
    }
}

/// One chunk: writes 64 results and returns the exact bad-row word (always
/// 0 when the policy does not check).
#[inline(always)]
fn chunk<T, L, O, const L_SCALAR: bool, const R_SCALAR: bool>(
    a: &[T; CHUNK],
    ls: T,
    b: &[T; CHUNK],
    rs: T,
    out: &mut [T; CHUNK],
    op: L,
) -> u64
where
    T: Copy,
    L: LaneOp<T>,
    O: Overflow,
{
    let pick = |x: T, y: T| (if L_SCALAR { ls } else { x }, if R_SCALAR { rs } else { y });
    if !O::CHECKS {
        for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
            let (x, y) = pick(x, y);
            *o = op.wrapping(x, y);
        }
        return 0;
    }
    let mut any = false;
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        let (x, y) = pick(x, y);
        let (v, flag) = op.checked(x, y);
        *o = v;
        any |= flag;
    }
    if !any {
        return 0;
    }
    let mut word = 0_u64;
    for ((&x, &y), bit) in a.iter().zip(b).zip(0_u32..) {
        let (x, y) = pick(x, y);
        if op.is_bad(x, y) {
            word |= 1_u64.wrapping_shl(bit);
        }
    }
    word
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_bits_edges() {
        assert_eq!(low_bits(0), 0);
        assert_eq!(low_bits(1), 1);
        assert_eq!(low_bits(63), u64::MAX >> 1);
        assert_eq!(low_bits(64), u64::MAX);
    }

    #[test]
    fn clear_bits_without_validity() {
        let n = clear_bits(None, &[0b101], 3);
        assert_eq!(n.iter().collect::<Vec<_>>(), [false, true, false]);
    }
}
