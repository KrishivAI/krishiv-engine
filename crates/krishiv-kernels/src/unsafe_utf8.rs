// SPDX-License-Identifier: Apache-2.0
//! String arrays over bytes checked once: the decode of row-format strings
//! (group keys, join keys, sort rows) builds one value buffer per column and
//! hands it here. Arrow's `try_new` checks it with `std::str::from_utf8`
//! and then every offset; `std` takes its slow path on multi-byte text
//! (ClickBench q14: 7.7% of samples). [`string_array`] checks the whole
//! buffer with `simdutf8` (SIMD, chosen at run time) and every offset, then
//! builds the array with `new_unchecked`, so nothing is checked twice.
//! The last 128 bytes go to `std` instead ([`is_utf8`]: simdutf8's AVX2
//! prefetch pointer would otherwise leave the allocation).
//!
//! Its checks are a superset of `try_new`'s in arrow-array 60.0 through
//! 60.2 (the manifest asks for `"60"`): 60.0 validates the whole buffer,
//! 60.1 and 60.2 only the bytes between the first and the last offset. A
//! buffer whose bytes outside the offsets are not UTF-8, or whose values are
//! all empty at an offset inside a character, is refused here and accepted
//! by 60.1 / 60.2; the decode never builds one (its buffer is exactly the
//! values).
//!
//! The only `unsafe` here is that one constructor call; every invariant it
//! needs is checked first, so the function is safe for any input.

use arrow_array::{GenericStringArray, OffsetSizeTrait};
use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer};

/// Why [`string_array`] refused its input. Arrow's `try_new` refuses each
/// case too, except over bytes outside the offsets (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StringArrayError {
    /// The last offset is past the end of the values.
    #[error("the last offset is past the end of the values")]
    OffsetsPastValues,
    /// The null buffer's length is not the number of values.
    #[error("the null buffer's length is not the number of values")]
    NullsLength,
    /// The value buffer is not UTF-8.
    #[error("the value bytes are not UTF-8")]
    NotUtf8,
    /// An offset falls inside a multi-byte character.
    #[error("an offset splits a UTF-8 character (the values are not UTF-8)")]
    SplitCharacter,
}

/// A string array of `values` cut at `offsets`, with `nulls`.
///
/// # Errors
/// [`StringArrayError`] when an invariant of a string array does not hold,
/// or when the bytes outside the offsets are not UTF-8.
pub fn string_array<O: OffsetSizeTrait>(
    offsets: OffsetBuffer<O>,
    values: Buffer,
    nulls: Option<NullBuffer>,
) -> Result<GenericStringArray<O>, StringArrayError> {
    // `OffsetBuffer` already guarantees a non-empty, non-negative,
    // non-decreasing sequence.
    if offsets.last().as_usize() > values.len() {
        return Err(StringArrayError::OffsetsPastValues);
    }
    if nulls
        .as_ref()
        .is_some_and(|n| n.len() != offsets.len().saturating_sub(1))
    {
        return Err(StringArrayError::NullsLength);
    }
    // The whole buffer, a superset of what any arrow 60 checks.
    let bytes = values.as_slice();
    if !is_utf8(bytes) {
        return Err(StringArrayError::NotUtf8);
    }
    // Every offset on a character boundary of the whole buffer: in valid
    // UTF-8, the end or a byte that is not a continuation byte.
    if !offsets
        .iter()
        .all(|o| bytes.get(o.as_usize()).is_none_or(|&b| !is_continuation(b)))
    {
        return Err(StringArrayError::SplitCharacter);
    }
    debug_assert!(
        GenericStringArray::<O>::try_new(offsets.clone(), values.clone(), nulls.clone()).is_ok(),
        "string_array accepted what try_new refuses"
    );
    // SAFETY: `GenericStringArray::new_unchecked` (arrow-array 60,
    // `GenericByteArray::new_unchecked`) is safe when `try_new` would not
    // fail. `try_new` checks: offsets within `values` (monotonic and
    // non-negative by `OffsetBuffer`'s own invariant, so the last one at most
    // `values.len()` suffices, checked above); a null buffer of one bit per
    // value (checked above); and `GenericStringType::validate`, which in
    // 60.0 requires the whole of `values` to be UTF-8 with every offset a
    // character boundary of it, and in 60.1 / 60.2 requires
    // `values[first..last]` to be UTF-8 with every offset, less `first`, a
    // boundary of that slice. The whole buffer is UTF-8 and every offset a
    // boundary of it (checked above), so `first` and `last` are boundaries,
    // `values[first..last]` is UTF-8, and a boundary of the whole buffer
    // inside it is a boundary of the slice: both forms hold. The fuzz target
    // `fuzz/fuzz_targets/string_array.rs` checks that whatever this accepts,
    // `try_new` accepts and builds into an equal array.
    let array = unsafe { GenericStringArray::new_unchecked(offsets, values, nulls) };
    Ok(array)
}

/// How far ahead of its position simdutf8 0.1.5's AVX2 loop forms a
/// prefetch pointer (`algorithm.rs`: `input.as_ptr().add(idx +
/// SIMD_CHUNK_SIZE * 2)`, 64-byte chunks).
const PREFETCH_AHEAD: usize = 128;

/// A UTF-8 continuation byte (`0b10xx_xxxx`).
const fn is_continuation(b: u8) -> bool {
    b & 0xc0 == 0x80
}

/// Whether `bytes` is UTF-8: all but the last [`PREFETCH_AHEAD`] bytes
/// with `simdutf8`, the rest with `std`.
///
/// simdutf8 0.1.5's AVX2 loop computes its prefetch pointer 128 bytes past
/// the chunk it checks, so over the last two chunks of an allocation the
/// pointer lies outside it: `pointer::add` out of bounds, undefined
/// behaviour that Miri reports even though nothing is read through it. Its
/// input here therefore ends at least 128 bytes before the buffer does, cut
/// at a character's first byte (found within the 3 bytes before the cut in
/// valid UTF-8; if there is none the buffer is not UTF-8, and either half
/// then fails or the whole would be valid). Both halves valid is exactly
/// the whole valid when the cut is at a character's start, so the answer is
/// `from_utf8`'s for any input.
fn is_utf8(bytes: &[u8]) -> bool {
    let mut cut = bytes.len().saturating_sub(PREFETCH_AHEAD);
    let floor = cut.saturating_sub(3);
    while cut > floor && bytes.get(cut).is_some_and(|&b| is_continuation(b)) {
        cut = cut.saturating_sub(1);
    }
    let (head, tail) = bytes.split_at(cut);
    simdutf8::basic::from_utf8(head).is_ok() && std::str::from_utf8(tail).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use arrow_buffer::ScalarBuffer;

    fn offsets(o: &[i64]) -> OffsetBuffer<i64> {
        OffsetBuffer::new(ScalarBuffer::from(o.to_vec()))
    }

    #[test]
    fn valid_values_build_the_array_arrow_builds() {
        let bytes = "aжbЖ€c".as_bytes().to_vec();
        let o = offsets(&[0, 1, 3, 4, 6, 9, 10]);
        let got = string_array(o.clone(), Buffer::from_vec(bytes.clone()), None).unwrap();
        let want =
            arrow_array::LargeStringArray::try_new(o, Buffer::from_vec(bytes), None).unwrap();
        assert_eq!(got, want);
        let nulls = NullBuffer::from(vec![true, false]);
        let got = string_array(
            offsets(&[0, 1, 1]),
            Buffer::from_vec(b"a".to_vec()),
            Some(nulls),
        )
        .unwrap();
        assert!(got.is_null(1));
        assert_eq!(got.value(0), "a");
    }

    #[test]
    fn every_broken_invariant_is_refused() {
        let b = || Buffer::from_vec("aж".as_bytes().to_vec());
        assert_eq!(
            string_array(offsets(&[0, 4]), b(), None).unwrap_err(),
            StringArrayError::OffsetsPastValues
        );
        // The buffer is valid; the offset 2 cuts `ж` in two.
        assert_eq!(
            string_array(offsets(&[0, 2]), b(), None).unwrap_err(),
            StringArrayError::SplitCharacter
        );
        // The whole range is valid, but an offset splits `ж`.
        assert_eq!(
            string_array(offsets(&[0, 2, 3]), b(), None).unwrap_err(),
            StringArrayError::SplitCharacter
        );
        assert_eq!(
            string_array(offsets(&[0, 1]), b(), Some(NullBuffer::new_null(2))).unwrap_err(),
            StringArrayError::NullsLength
        );
        let invalid = Buffer::from_vec(vec![b'a', 0xff]);
        assert_eq!(
            string_array(offsets(&[0, 1, 2]), invalid, None).unwrap_err(),
            StringArrayError::NotUtf8
        );
    }

    #[test]
    fn offsets_need_not_start_at_zero() {
        let o = offsets(&[1, 3]);
        let got = string_array(o, Buffer::from_vec("aжb".as_bytes().to_vec()), None).unwrap();
        assert_eq!(got.value(0), "ж");
        // Bytes outside the offsets are checked too (stricter than
        // arrow-array 60.1 / 60.2, which accept this buffer).
        let o = offsets(&[0, 0]);
        let bad_tail = Buffer::from_vec(vec![0, 0xf1]);
        assert_eq!(
            string_array(o, bad_tail, None).unwrap_err(),
            StringArrayError::NotUtf8
        );
    }

    /// Multi-byte text of 275 bytes (past simdutf8's 64-byte SIMD chunk and
    /// not a multiple of it) and its character boundaries.
    fn long_text() -> (Vec<u8>, Vec<i64>) {
        let text: String = "aж€😀b".chars().cycle().take(125).collect();
        let mut cuts: Vec<i64> = text
            .char_indices()
            .map(|(i, _)| i64::try_from(i).unwrap())
            .step_by(3)
            .collect();
        cuts.push(i64::try_from(text.len()).unwrap());
        assert!(text.len() > 64 * 4 && !text.len().is_multiple_of(64));
        (text.into_bytes(), cuts)
    }

    // simdutf8 sends inputs under 64 bytes to its scalar fallback; these
    // reach the vector path (AVX2 / SSE4.2 chosen at run time on x86-64,
    // NEON on aarch64). Miri sees no target feature that is not enabled at
    // compile time, so `miri-kernels` runs them a second time with
    // `RUSTFLAGS=-Ctarget-feature=+avx2,+sse4.2`, which makes simdutf8 call
    // its AVX2 code: with `PREFETCH_AHEAD` at 0 that run reports simdutf8's
    // out-of-bounds pointer (`algorithm.rs:234`) here.
    #[test]
    fn long_valid_text_builds_the_array_arrow_builds() {
        let (bytes, cuts) = long_text();
        let o = offsets(&cuts);
        let got = string_array(o.clone(), Buffer::from_vec(bytes.clone()), None).unwrap();
        let want =
            arrow_array::LargeStringArray::try_new(o, Buffer::from_vec(bytes), None).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn long_text_with_a_late_invalid_byte_is_refused() {
        let (bytes, cuts) = long_text();
        let o = offsets(&cuts);
        // An ASCII `a` in the third 64-byte chunk, and one in the tail past
        // the last full chunk, each replaced by a byte never valid in UTF-8.
        let late = 130 + bytes[130..].iter().position(|&b| b == b'a').unwrap();
        let tail = 256 + bytes[256..].iter().position(|&b| b == b'a').unwrap();
        for at in [late, tail] {
            let mut bad = bytes.clone();
            bad[at] = 0xff;
            assert_eq!(
                string_array(o.clone(), Buffer::from_vec(bad), None).unwrap_err(),
                StringArrayError::NotUtf8,
                "byte {at}"
            );
        }
    }

    #[test]
    fn long_text_split_past_the_first_chunk_is_refused() {
        let (bytes, mut cuts) = long_text();
        // The first byte of a multi-byte character past byte 64, plus one.
        let at = (65..bytes.len()).find(|&i| bytes[i] >= 0xc0).unwrap();
        let split = i64::try_from(at + 1).unwrap();
        let pos = cuts.partition_point(|&c| c < split);
        cuts.insert(pos, split);
        assert_eq!(
            string_array(offsets(&cuts), Buffer::from_vec(bytes), None).unwrap_err(),
            StringArrayError::SplitCharacter
        );
    }

    #[test]
    fn is_utf8_answers_as_std_around_the_prefetch_cut() {
        let (bytes, _) = long_text();
        // Prefixes that end inside a character or not, near both cuts.
        for len in (120..=140).chain(250..=bytes.len()) {
            let b = &bytes[..len];
            assert_eq!(is_utf8(b), std::str::from_utf8(b).is_ok(), "len {len}");
        }
        // A bad byte at each place near the cut.
        for at in bytes.len() - 136..bytes.len() {
            for bad in [0xff, 0x80, 0xc0] {
                let mut b = bytes.clone();
                b[at] = bad;
                assert_eq!(is_utf8(&b), std::str::from_utf8(&b).is_ok(), "{at} {bad}");
            }
        }
    }
}
