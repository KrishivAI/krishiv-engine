// SPDX-License-Identifier: Apache-2.0
//! String arrays over bytes checked once: the decode of row-format strings
//! (group keys, join keys, sort rows) builds one value buffer per column and
//! hands it here. Arrow's `try_new` checks it with `std::str::from_utf8`
//! and then every offset; `std` takes its slow path on multi-byte text
//! (ClickBench q14: 7.7% of samples). [`string_array`] checks the same two
//! things, the buffer with `simdutf8` (SIMD, chosen at run time), then builds
//! the array with `new_unchecked`, so nothing is checked twice.
//!
//! The only `unsafe` here is that one constructor call; every invariant it
//! needs is checked first, so the function is safe for any input.

use arrow_array::{GenericStringArray, OffsetSizeTrait};
use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer};

/// Why [`string_array`] refused its input (each is a case arrow's
/// `try_new` refuses too).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringArrayError {
    /// The last offset is past the end of the values.
    OffsetsPastValues,
    /// The null buffer's length is not the number of values.
    NullsLength,
    /// The value buffer is not UTF-8.
    NotUtf8,
    /// An offset falls inside a multi-byte character.
    SplitCharacter,
}

/// A string array of `values` cut at `offsets`, with `nulls`.
///
/// # Errors
/// [`StringArrayError`] when an invariant of a string array does not hold.
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
    // The whole buffer, as arrow's `try_new` checks it (bytes outside the
    // offsets too: the fuzz target `string_array` found the difference).
    let text =
        simdutf8::basic::from_utf8(values.as_slice()).map_err(|_| StringArrayError::NotUtf8)?;
    // Every offset on a character boundary: with the whole buffer valid,
    // each value between two offsets is then valid too.
    if !offsets.iter().all(|o| text.is_char_boundary(o.as_usize())) {
        return Err(StringArrayError::SplitCharacter);
    }
    // SAFETY: `GenericStringArray::new_unchecked` (arrow-array 60,
    // `GenericByteArray::new_unchecked`) is safe when `try_new` would not
    // fail, and `try_new` checks exactly what is checked above: offsets
    // within `values` (monotonic by `OffsetBuffer`'s own invariant, the last
    // one at most `values.len()`), the whole of `values` valid UTF-8 and
    // every offset a character boundary of it (`GenericStringType::validate`),
    // and a null buffer of one bit per value. The fuzz target
    // `fuzz/fuzz_targets/string_array.rs` compares the two over any input.
    let array = unsafe { GenericStringArray::new_unchecked(offsets, values, nulls) };
    debug_assert!(
        array
            .iter()
            .flatten()
            .all(|s| std::str::from_utf8(s.as_bytes()).is_ok()),
        "a string array built from unchecked bytes"
    );
    Ok(array)
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
        // Bytes outside the offsets are checked too, as arrow checks them.
        let o = offsets(&[0, 0]);
        let bad_tail = Buffer::from_vec(vec![0, 0xf1]);
        assert_eq!(
            string_array(o, bad_tail, None).unwrap_err(),
            StringArrayError::NotUtf8
        );
    }
}
