// SPDX-License-Identifier: Apache-2.0
//! The generic 64-row chunk engine behind every elementwise binary kernel.
//!
//! [`compute`] inspects the operands once, picks the `<L_SCALAR, R_SCALAR>`
//! monomorph and the CPU target, and runs the loop. There is no
//! has-nulls axis here: values are computed through NULL slots, so the loop
//! is the same with or without NULLs (measured: ~1.0x on the same data), and
//! the validity is read only when a checked chunk faults. Kernels whose work
//! does depend on NULLs opt into [`crate::specialize`]. Each chunk is computed branch-free straight into the output (no
//! pre-zeroing); a checking policy ORs a cheap "may fault" flag over the
//! chunk and, only if it is set, re-scans the chunk with the exact
//! [`LaneOp::fault`] verdict. Each fault kind maps to an [`Action`] of the
//! policy: wrap, NULL (a word of rows cleared from the validity) or raise
//! (the first such row whose validity bit is set — a NULL row never raises).

use std::marker::PhantomData;

use arrow_array::{Array, ArrowPrimitiveType};
use arrow_buffer::bit_chunk_iterator::BitChunks;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};

use crate::dispatch::KernelCtx;
use crate::error::{KernelError, with_capacity};
use crate::ops::Operand;
use crate::policy::{Action, Fault, Overflow};
use crate::unsafe_dispatch::{Kernel, run};

/// Rows per chunk: one validity word.
pub(crate) const CHUNK: usize = 64;

/// An elementwise operation on two lanes.
pub(crate) trait LaneOp<T>: Copy {
    /// Whether the operation can fault with [`Fault::DivideByZero`].
    const MAY_DIVIDE_BY_ZERO: bool = false;
    /// The result without checks. Total: never panics.
    fn wrapping(self, a: T, b: T) -> T;
    /// The result and a cheap flag that is set for every faulting row (it
    /// may also be set for a good one; [`LaneOp::fault`] decides).
    fn checked(self, a: T, b: T) -> (T, bool);
    /// The exact verdict for a row whose flag was set (slow path). For a row
    /// without a fault, the value [`LaneOp::checked`] returned must be exact.
    fn fault(self, a: T, b: T) -> Option<Fault>;
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

/// The values of a chunk computation and the output validity (the input
/// validity minus rows a policy turned NULL).
#[derive(Debug)]
pub(crate) struct Computed<T> {
    pub(crate) values: Vec<T>,
    pub(crate) nulls: Option<NullBuffer>,
}

/// Whether policy `O` must check lane op `L` at all.
fn checks<T, L: LaneOp<T>, O: Overflow>() -> bool {
    O::MODE.action(Fault::Overflow) != Action::Wrap || L::MAY_DIVIDE_BY_ZERO
}

/// Runs `op` over two resolved sides (`len` rows; `nulls` is the input
/// validity). A raised fault becomes the error `raise(row, fault)`.
pub(crate) fn compute<T, L, O>(
    ctx: &KernelCtx,
    l: Side<'_, T>,
    r: Side<'_, T>,
    nulls: Option<NullBuffer>,
    len: usize,
    op: L,
    raise: impl FnOnce(usize, Fault) -> KernelError,
) -> Result<Computed<T>, KernelError>
where
    T: Copy + Default,
    L: LaneOp<T>,
    O: Overflow,
{
    let mut out = with_capacity::<T>(len)?;
    let n_words = len.div_ceil(CHUNK);
    // Space for one NULL-row word per chunk, reserved up front so the loop
    // never allocates. Only policies that can turn a row NULL need it.
    let may_null = checks::<T, L, O>()
        && (O::MODE.action(Fault::Overflow) == Action::Null
            || (L::MAY_DIVIDE_BY_ZERO && O::MODE.action(Fault::DivideByZero) == Action::Null));
    let mut null_words = with_capacity::<u64>(if may_null { n_words } else { 0 })?;
    let n = nulls.as_ref();
    macro_rules! go {
        ($ls:literal, $rs:literal) => {
            run(
                ctx,
                Chunks::<'_, T, L, O, $ls, $rs> {
                    l,
                    r,
                    validity: n,
                    out: &mut out,
                    null_words: &mut null_words,
                    len,
                    op,
                    policy: PhantomData,
                },
            )
        };
    }
    let raised = match (l, r) {
        (Side::Array(_), Side::Array(_)) => go!(false, false),
        (Side::Array(_), Side::Scalar(_)) => go!(false, true),
        (Side::Scalar(_), _) => go!(true, false),
    };
    if let Some((row, fault)) = raised {
        return Err(raise(row, fault));
    }
    let nulls = if null_words.iter().any(|&w| w != 0) {
        Some(clear_bits(nulls.as_ref(), null_words, len))
    } else {
        nulls
    };
    Ok(Computed { values: out, nulls })
}

/// `validity AND NOT words` as a new null buffer (all valid when `validity`
/// is `None`), built in place in `words` (one word per chunk).
fn clear_bits(validity: Option<&NullBuffer>, mut words: Vec<u64>, len: usize) -> NullBuffer {
    let chunks = validity.map(|v| v.inner().bit_chunks());
    let mut valid = chunks.as_ref().map(BitChunks::iter_padded);
    for w in &mut words {
        let v = valid.as_mut().map_or(u64::MAX, |it| it.next().unwrap_or(0));
        // Arrow bitmaps are little-endian bytes.
        *w = (v & !*w).to_le();
    }
    NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(words), 0, len))
}

/// Mask of the low `n` bits (`n <= 64`).
fn low_bits(n: usize) -> u64 {
    u32::try_from(n)
        .ok()
        .and_then(|n| 1_u64.checked_shl(n))
        .map_or(u64::MAX, |b| b.wrapping_sub(1))
}

/// The chunk loop for one monomorph; `run` compiles it per CPU target.
struct Chunks<'a, T, L, O, const L_SCALAR: bool, const R_SCALAR: bool> {
    l: Side<'a, T>,
    r: Side<'a, T>,
    validity: Option<&'a NullBuffer>,
    /// Empty, with capacity for `len` values.
    out: &'a mut Vec<T>,
    /// Empty, with capacity for one word per chunk when the policy can
    /// turn rows NULL.
    null_words: &'a mut Vec<u64>,
    len: usize,
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

/// What the slow scan of one chunk found: rows to turn NULL, and the first
/// non-NULL row to raise at (bit index and fault).
#[derive(Debug, Default, Clone, Copy)]
struct ChunkFaults {
    null: u64,
    raise: Option<(u32, Fault)>,
}

impl<T, L, O, const L_SCALAR: bool, const R_SCALAR: bool> Kernel
    for Chunks<'_, T, L, O, L_SCALAR, R_SCALAR>
where
    T: Copy + Default,
    L: LaneOp<T>,
    O: Overflow,
{
    /// The first raised `(row, fault)`, if any.
    type Output = Option<(usize, Fault)>;

    #[inline(always)]
    fn run(self) -> Self::Output {
        let checks = checks::<T, L, O>();
        let (lc, lt, ls) = split(self.l);
        let (rc, rt, rs) = split(self.r);
        let zero = [T::default(); CHUNK];
        let full = self.len / CHUNK;
        let chunks = self.validity.map(|v| v.inner().bit_chunks());
        let mut valid = chunks.as_ref().map(BitChunks::iter_padded);
        let mut next_valid = || valid.as_mut().map_or(u64::MAX, |it| it.next().unwrap_or(0));
        let null_words = self.null_words;
        let mut record = |word: usize, f: ChunkFaults| -> Option<(usize, Fault)> {
            if null_words.capacity() > 0 {
                null_words.push(f.null);
            }
            f.raise
                .map(|(bit, fault)| (word.wrapping_mul(CHUNK).wrapping_add(bit as usize), fault))
        };
        for i in 0..full {
            let a = lc.get(i).unwrap_or(&zero);
            let b = rc.get(i).unwrap_or(&zero);
            let flags =
                chunk::<T, L, L_SCALAR, R_SCALAR>(a, ls, b, rs, self.out, CHUNK, self.op, checks);
            if !checks {
                continue;
            }
            // Only a flagged non-NULL row needs the slow scan: garbage under
            // a NULL slot (often a zeroed divisor) costs nothing.
            let valid = next_valid();
            let f = if flags & valid == 0 {
                ChunkFaults::default()
            } else {
                faults::<T, L, O, L_SCALAR, R_SCALAR>(a, ls, b, rs, self.op, valid)
            };
            if let Some(raised) = record(i, f) {
                return Some(raised);
            }
        }
        let rem = self.len.wrapping_sub(full.wrapping_mul(CHUNK));
        if rem > 0 {
            let (mut a, mut b) = (zero, zero);
            copy_prefix(&mut a, lt);
            copy_prefix(&mut b, rt);
            let flags =
                chunk::<T, L, L_SCALAR, R_SCALAR>(&a, ls, &b, rs, self.out, rem, self.op, checks);
            if checks {
                let valid = next_valid() & low_bits(rem);
                let f = if flags & valid == 0 {
                    ChunkFaults::default()
                } else {
                    faults::<T, L, O, L_SCALAR, R_SCALAR>(&a, ls, &b, rs, self.op, valid)
                };
                return record(full, f);
            }
        }
        None
    }
}

/// Copies `src` (shorter than a chunk) into the front of `dst`.
fn copy_prefix<T: Copy>(dst: &mut [T; CHUNK], src: &[T]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = *s;
    }
}

/// One chunk: appends the first `take` results to `out` (whose capacity
/// was reserved) and returns the "may fault" flags: one bit per row for
/// dividing lanes, all bits for any flagged row otherwise, and zero when the
/// policy does not check.
#[expect(
    clippy::too_many_arguments,
    reason = "the hot loop takes its operands unpacked so it inlines into each clone"
)]
#[inline(always)]
fn chunk<T, L, const L_SCALAR: bool, const R_SCALAR: bool>(
    a: &[T; CHUNK],
    ls: T,
    b: &[T; CHUNK],
    rs: T,
    out: &mut Vec<T>,
    take: usize,
    op: L,
    checks: bool,
) -> u64
where
    T: Copy,
    L: LaneOp<T>,
{
    let pick = move |x: T, y: T| (if L_SCALAR { ls } else { x }, if R_SCALAR { rs } else { y });
    let lanes = a.iter().zip(b).take(take);
    if !checks {
        out.extend(lanes.map(|(&x, &y)| {
            let (x, y) = pick(x, y);
            op.wrapping(x, y)
        }));
        return 0;
    }
    if L::MAY_DIVIDE_BY_ZERO {
        // Dividing lanes never vectorize, and a zeroed divisor under a NULL
        // slot is common: keep one flag bit per row so the caller can mask
        // NULL rows out before deciding on the slow scan.
        let mut flags = 0_u64;
        out.extend(lanes.zip(0_u32..).map(|((&x, &y), bit)| {
            let (x, y) = pick(x, y);
            let (v, flag) = op.checked(x, y);
            flags |= u64::from(flag).wrapping_shl(bit);
            v
        }));
        return flags;
    }
    // Vectorizable lanes: one OR-reduced flag (a per-row word measured
    // ~2x slower on `i32 +`); it marks the whole chunk.
    let mut any = false;
    out.extend(lanes.map(|(&x, &y)| {
        let (x, y) = pick(x, y);
        let (v, flag) = op.checked(x, y);
        any |= flag;
        v
    }));
    if any { u64::MAX } else { 0 }
}

/// The slow scan of a flagged chunk: classifies each row with the exact
/// verdict and applies the policy. `valid` has the row's validity bits (and
/// zeros past the end of the data), so NULL and padding rows never raise.
#[cold]
#[inline(never)]
fn faults<T, L, O, const L_SCALAR: bool, const R_SCALAR: bool>(
    a: &[T; CHUNK],
    ls: T,
    b: &[T; CHUNK],
    rs: T,
    op: L,
    valid: u64,
) -> ChunkFaults
where
    T: Copy,
    L: LaneOp<T>,
    O: Overflow,
{
    let mut f = ChunkFaults::default();
    for ((&x, &y), bit) in a.iter().zip(b).zip(0_u32..) {
        let x = if L_SCALAR { ls } else { x };
        let y = if R_SCALAR { rs } else { y };
        let Some(fault) = op.fault(x, y) else {
            continue;
        };
        let mask = 1_u64.wrapping_shl(bit);
        match O::MODE.action(fault) {
            Action::Null => f.null |= mask,
            Action::Raise if valid & mask != 0 => {
                f.raise = Some((bit, fault));
                break;
            }
            // Wrap keeps the value; a NULL row never raises.
            Action::Wrap | Action::Raise => {}
        }
    }
    f.null &= valid;
    f
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
        let n = clear_bits(None, vec![0b101], 3);
        assert_eq!(n.iter().collect::<Vec<_>>(), [false, true, false]);
    }

    /// A division lane, the shape T021a's `div`/`%` take: two fault kinds,
    /// with legacy mode mixing them (x/0 is NULL, MIN/-1 wraps).
    #[derive(Clone, Copy)]
    struct Div;

    impl LaneOp<i32> for Div {
        const MAY_DIVIDE_BY_ZERO: bool = true;
        fn wrapping(self, a: i32, b: i32) -> i32 {
            a.checked_div(b).unwrap_or(0)
        }
        fn checked(self, a: i32, b: i32) -> (i32, bool) {
            let bad = (b == 0) | ((a == i32::MIN) & (b == -1));
            (
                a.checked_div(b)
                    .unwrap_or(if b == 0 { 0 } else { i32::MIN }),
                bad,
            )
        }
        fn fault(self, a: i32, b: i32) -> Option<Fault> {
            match (a, b) {
                (_, 0) => Some(Fault::DivideByZero),
                (i32::MIN, -1) => Some(Fault::Overflow),
                _ => None,
            }
        }
    }

    fn div<O: Overflow>(
        a: &[i32],
        b: &[i32],
        nulls: Option<NullBuffer>,
    ) -> Result<Computed<i32>, KernelError> {
        compute::<i32, Div, O>(
            &KernelCtx::new(),
            Side::Array(a),
            Side::Array(b),
            nulls,
            a.len(),
            Div,
            |row, f| {
                KernelError::for_fault(row, f, |r| {
                    KernelError::arithmetic_overflow(r, "overflow", None)
                })
            },
        )
    }

    #[test]
    fn fault_kinds_map_per_policy() {
        use crate::policy::{AnsiError, JavaWrap, Unchecked};
        use krishiv_common::error::ErrorClass;
        let mut a = vec![6; 70];
        let mut b = vec![3; 70];
        (a[1], b[1]) = (i32::MIN, -1);
        b[66] = 0;
        // Legacy: x/0 is NULL, MIN / -1 wraps.
        let c = div::<JavaWrap>(&a, &b, None).unwrap();
        assert_eq!((c.values[0], c.values[1]), (2, i32::MIN));
        let n = c.nulls.unwrap();
        assert_eq!((n.null_count(), n.is_null(66)), (1, true));
        // ANSI raises the first fault, with its kind.
        let e = div::<AnsiError>(&a, &b, None).unwrap_err();
        assert_eq!(
            (e.row(), e.error_class()),
            (Some(1), ErrorClass::ArithmeticOverflow)
        );
        // A NULL row never raises: with row 1 NULL, the next fault is x/0.
        let mut valid = vec![true; 70];
        valid[1] = false;
        let e = div::<AnsiError>(&a, &b, Some(NullBuffer::from(valid))).unwrap_err();
        assert_eq!(
            (e.row(), e.error_class()),
            (Some(66), ErrorClass::DivideByZero)
        );
        // The lane's unchecked form is total.
        assert_eq!((Div.wrapping(7, 0), Div.wrapping(7, 2)), (0, 3));
        // Unchecked proves no overflow, not a non-zero divisor.
        let e = div::<Unchecked>(&a[2..], &b[2..], None).unwrap_err();
        assert_eq!(e.row(), Some(64));
    }
}
