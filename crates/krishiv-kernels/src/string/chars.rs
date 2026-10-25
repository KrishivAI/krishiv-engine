// SPDX-License-Identifier: Apache-2.0
//! The character-level string functions: `ascii`, `chr`, `right`, `repeat`,
//! `space`, `reverse`, `initcap`, `translate`, `locate`, `substring_index`,
//! `split_part`, `find_in_set`, `overlay`, `levenshtein`, `soundex`,
//! `luhn_check`, `quote`, `elt` and `concat_ws`.
//!
//! A "character" is a Unicode scalar value, as in the rest of the module.
//! Work happens in one scratch buffer per kernel call; only rows whose
//! operands are all valid are evaluated (a NULL operand makes the row NULL
//! unless the function says otherwise: `locate`, `levenshtein`'s threshold,
//! `concat_ws`, `elt`).

use std::collections::HashMap;

use arrow_array::types::Int32Type;
use arrow_array::{
    Array, BooleanArray, GenericBinaryArray, GenericStringArray, Int32Array, Int64Array,
    OffsetSizeTrait,
};
use krishiv_common::error::{InvalidArrayIndex, InvalidIndexOfZero};

use super::{
    BinOperand, BinSink, Slot, StrOperand, StrSink, all_valid, bool_result, check_offset, int_at,
    reserve, row_count, substring_range,
};
use crate::cast::text::java_digit;
use crate::error::KernelError;
use crate::ops::Operand;

/// An `int` result column: `value(i)` for rows whose operands are all valid.
fn int_rows(
    slots: &[&dyn Slot],
    mut value: impl FnMut(usize) -> i32,
) -> Result<Int32Array, KernelError> {
    let rows = row_count(slots)?;
    Ok((0..rows)
        .map(|i| all_valid(slots, i).then(|| value(i)))
        .collect())
}

/// A string result column: `fill` writes the row's value into the cleared
/// scratch buffer for every row whose operands are all valid.
fn str_rows<O: OffsetSizeTrait>(
    slots: &[&dyn Slot],
    mut fill: impl FnMut(usize, &mut String) -> Result<(), KernelError>,
) -> Result<GenericStringArray<O>, KernelError> {
    let rows = row_count(slots)?;
    let mut sink = StrSink::<O>::new(rows, 0);
    let mut scratch = String::new();
    for i in 0..rows {
        if !all_valid(slots, i) {
            sink.push_null();
            continue;
        }
        scratch.clear();
        fill(i, &mut scratch)?;
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// Reserves `need` bytes of scratch, reporting failure as
/// `UNABLE_TO_ACQUIRE_MEMORY` instead of aborting.
fn reserve_str(scratch: &mut String, need: usize) -> Result<(), KernelError> {
    scratch
        .try_reserve(need)
        .map_err(|_| KernelError::out_of_memory(need))
}

/// The number of characters of `s`.
fn chars_of(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// Byte offset of character `n` of `s` (`s.len()` past the end).
fn byte_of_char(s: &str, n: usize) -> usize {
    if s.is_ascii() {
        n.min(s.len())
    } else {
        s.char_indices().nth(n).map_or(s.len(), |(at, _)| at)
    }
}

// ---- ascii, chr, space ----

/// `ascii(str)`: the code point of the first character, 0 for the empty
/// string.
#[must_use]
pub fn ascii<O: OffsetSizeTrait>(a: &GenericStringArray<O>) -> Int32Array {
    (0..a.len())
        .map(|i| {
            a.is_valid(i).then(|| {
                a.value(i)
                    .chars()
                    .next()
                    .map_or(0, |c| i32::try_from(u32::from(c)).unwrap_or(0))
            })
        })
        .collect()
}

/// `chr(n)` / `char(n)`: the character of code `n & 255` (Latin-1); NUL
/// when that is 0, the empty string for a negative `n`.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn chr<O: OffsetSizeTrait>(a: &Int64Array) -> Result<GenericStringArray<O>, KernelError> {
    let mut sink = StrSink::<O>::new(a.len(), a.len());
    for i in 0..a.len() {
        if a.is_null(i) {
            sink.push_null();
            continue;
        }
        let n = a.value(i);
        let mut buf = [0_u8; 4];
        let out: &str = if n < 0 {
            ""
        } else {
            let code = u8::try_from(n & 0xFF).unwrap_or(0);
            char::from(code).encode_utf8(&mut buf)
        };
        sink.push(out)?;
    }
    Ok(sink.finish())
}

/// `space(n)`: `n` spaces (none for `n <= 0`).
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit memory or the
/// offset type.
pub fn space<O: OffsetSizeTrait>(
    n: Operand<'_, Int32Type>,
) -> Result<GenericStringArray<O>, KernelError> {
    let rows = row_count(&[&n])?;
    let mut sink = StrSink::<O>::new(rows, 0);
    let mut scratch = String::new();
    for i in 0..rows {
        if !n.is_valid(i) {
            sink.push_null();
            continue;
        }
        let count = usize::try_from(int_at(n, i)).unwrap_or(0);
        check_offset::<O>(sink.bytes.saturating_add(count))?;
        scratch.clear();
        scratch
            .try_reserve(count)
            .map_err(|_| KernelError::out_of_memory(count))?;
        scratch.extend(std::iter::repeat_n(' ', count));
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

// ---- right, repeat, reverse ----

/// `right(str, len)`: the last `len` characters (the empty string for
/// `len <= 0`).
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn right<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    len: Operand<'_, Int32Type>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 2] = [&input, &len];
    str_rows(&slots, |i, out| {
        let s = input.at(i);
        reserve_str(out, s.len())?;
        let n = usize::try_from(int_at(len, i)).unwrap_or(0);
        if n > 0 {
            let skip = chars_of(s).saturating_sub(n);
            out.push_str(s.get(byte_of_char(s, skip)..).unwrap_or_default());
        }
        Ok(())
    })
}

/// `repeat(str, n)`: `str` `n` times (empty for `n <= 0`).
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit memory or the
/// offset type.
pub fn repeat<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    times: Operand<'_, Int32Type>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 2] = [&input, &times];
    let rows = row_count(&slots)?;
    let mut sink = StrSink::<O>::new(rows, 0);
    let mut scratch = String::new();
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let s = input.at(i);
        let n = usize::try_from(int_at(times, i)).unwrap_or(0);
        let total = s.len().saturating_mul(n);
        check_offset::<O>(sink.bytes.saturating_add(total))?;
        scratch.clear();
        scratch
            .try_reserve(total)
            .map_err(|_| KernelError::out_of_memory(total))?;
        for _ in 0..n {
            scratch.push_str(s);
        }
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// `reverse(str)`: the characters in reverse order.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn reverse<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let input = StrOperand::Array(a);
    str_rows(&[&input], |i, out| {
        let s = a.value(i);
        reserve_str(out, s.len())?;
        out.extend(s.chars().rev());
        Ok(())
    })
}

// ---- initcap ----

/// The title case of `c` for the start of a word: the full upper case with
/// the characters after the first lowered (`ß` is `Ss`), except where the
/// title case differs from that: the Latin digraphs and the Georgian
/// letters, which keep their form.
fn push_title_case(c: char, out: &mut String) {
    let title = match u32::from(c) {
        0x01C4..=0x01C6 => Some('\u{01C5}'),
        0x01C7..=0x01C9 => Some('\u{01C8}'),
        0x01CA..=0x01CC => Some('\u{01CB}'),
        0x01F1..=0x01F3 => Some('\u{01F2}'),
        0x10D0..=0x10FA | 0x10FD..=0x10FF => Some(c),
        _ => None,
    };
    if let Some(t) = title {
        out.push(t);
        return;
    }
    let mut upper = c.to_uppercase();
    if let Some(first) = upper.next() {
        out.push(first);
    }
    for rest in upper {
        out.extend(rest.to_lowercase());
    }
}

/// `initcap(str)`: the first letter of each word title-cased and the rest
/// lower-cased; a word starts at the beginning of the string or after an
/// ASCII space (no other character separates words).
///
/// Non-ASCII words are lower-cased with the full mapping of the toolchain
/// (final sigma included). The first character of a word takes its full upper
/// case with the later characters lowered (`ß` is `Ss`), except the Latin
/// digraphs and Georgian letters, which have their own title case. That
/// differs from ICU's for characters whose title case is neither (Greek
/// letters with an iota subscript).
///
/// This is a known approximation: the case mapping is the toolchain's Unicode
/// data, not a version-keyed table (see `super::case_table` for `upper`/
/// `lower`). The outputs for the divergent code points are pinned by
/// `toolchain_unicode_is_pinned`, so a toolchain upgrade fails the test
/// instead of silently changing the result.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit memory or the
/// offset type.
pub fn initcap<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let input = StrOperand::Array(a);
    str_rows(&[&input], |i, out| {
        let s = a.value(i);
        reserve_str(out, s.len().saturating_mul(4).saturating_add(4))?;
        if s.is_ascii() {
            let mut word_start = true;
            for b in s.bytes() {
                out.push(char::from(if word_start {
                    b.to_ascii_uppercase()
                } else {
                    b.to_ascii_lowercase()
                }));
                word_start = b == b' ';
            }
            return Ok(());
        }
        for (n, word) in s.split(' ').enumerate() {
            if n > 0 {
                out.push(' ');
            }
            let Some(first) = word.chars().next() else {
                continue;
            };
            // The lower case of the whole word (final sigma included), whose
            // first character is then replaced by its title case.
            let need = word.len().saturating_mul(4).saturating_add(4);
            let mut lowered = String::new();
            reserve_str(&mut lowered, need)?;
            push_lower(word, &mut lowered)?;
            push_title_case(first, out);
            let lowered_first: usize = first.to_lowercase().map(char::len_utf8).sum();
            out.push_str(lowered.get(lowered_first..).unwrap_or_default());
        }
        Ok(())
    })
}

/// Appends the lower case of `word` (final sigma included, as
/// `str::to_lowercase` does) without the intermediate allocation that
/// cannot fail gracefully.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the working row does not fit memory.
fn push_lower(word: &str, out: &mut String) -> Result<(), KernelError> {
    let mut chars = Vec::new();
    chars
        .try_reserve(word.chars().count())
        .map_err(|_| KernelError::out_of_memory(word.len()))?;
    chars.extend(word.chars());
    let cased = |c: &char| c.is_uppercase() || c.is_lowercase();
    for (i, c) in chars.iter().enumerate() {
        if *c == '\u{03A3}' {
            // Final sigma: a preceding cased letter and no cased letter
            // after (ignoring nothing; combinables are rare in `initcap`
            // words and `str::to_lowercase` agrees otherwise).
            let prev_cased = chars.get(i.wrapping_sub(1)).is_some_and(cased);
            // `i == 0` underflows to `usize::MAX`; `get` is `None` there.
            let next_cased = chars.get(i.saturating_add(1)).is_some_and(cased);
            out.push(if prev_cased && !next_cased {
                '\u{03C2}'
            } else {
                '\u{03C3}'
            });
            continue;
        }
        out.extend(c.to_lowercase());
    }
    Ok(())
}

// ---- translate ----

/// The mapping `translate(_, from, to)` applies: each character of `from`
/// (the first occurrence wins) to the character of `to` at the same index,
/// or to nothing when `to` is shorter or has a NUL there.
fn translation(from: &str, to: &str) -> HashMap<char, Option<char>> {
    let mut to = to.chars();
    let mut dict = HashMap::new();
    for f in from.chars() {
        let t = to.next().filter(|c| *c != '\0');
        dict.entry(f).or_insert(t);
    }
    dict
}

/// `translate(str, from, to)`.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn translate<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    from: StrOperand<'_, O>,
    to: StrOperand<'_, O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&input, &from, &to];
    let mut cached: Option<(String, String, HashMap<char, Option<char>>)> = None;
    str_rows(&slots, |i, out| {
        let s = input.at(i);
        // Each character maps to at most one replacement (4 bytes worst
        // case for a 1-byte input), so `4 * len` bounds the output.
        reserve_str(out, s.len().saturating_mul(4))?;
        let (f, t) = (from.at(i), to.at(i));
        let stale = cached.as_ref().is_none_or(|(cf, ct, _)| cf != f || ct != t);
        if stale {
            cached = Some((f.to_owned(), t.to_owned(), translation(f, t)));
        }
        let Some((_, _, dict)) = cached.as_ref() else {
            return Ok(());
        };
        for c in s.chars() {
            match dict.get(&c) {
                None => out.push(c),
                Some(Some(r)) => out.push(*r),
                Some(None) => {}
            }
        }
        Ok(())
    })
}

// ---- locate, substring_index, split_part, find_in_set ----

/// `locate(substr, str, start)` / `position` / `instr`: the 1-based
/// character position of the first `substr` at or after character `start`
/// (1-based), 0 when there is none or `start < 1`. A NULL `start` gives 0
/// whatever the other operands are; an empty `substr` is at position 1
/// whatever `start` is (as Spark does).
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths.
pub fn locate<O: OffsetSizeTrait>(
    substr: StrOperand<'_, O>,
    input: StrOperand<'_, O>,
    start: Operand<'_, Int32Type>,
) -> Result<Int32Array, KernelError> {
    let slots: [&dyn Slot; 3] = [&substr, &input, &start];
    let rows = row_count(&slots)?;
    Ok((0..rows)
        .map(|i| {
            if !start.is_valid(i) {
                return Some(0);
            }
            if !(substr.is_valid(i) && input.is_valid(i)) {
                return None;
            }
            let begin = int_at(start, i);
            if begin < 1 {
                return Some(0);
            }
            let (sub, s) = (substr.at(i), input.at(i));
            if sub.is_empty() {
                return Some(1);
            }
            let from = byte_of_char(s, usize::try_from(begin.saturating_sub(1)).unwrap_or(0));
            let found = s
                .get(from..)
                .and_then(|rest| rest.find(sub))
                .map(|at| chars_of(s.get(..from.saturating_add(at)).unwrap_or_default()));
            Some(found.map_or(0, |c| {
                i32::try_from(c.saturating_add(1)).unwrap_or(i32::MAX)
            }))
        })
        .collect())
}

/// The part of `s` `substring_index` selects: everything before the
/// `count`-th `delim` from the left (`count > 0`) or after the
/// `|count|`-th from the right; all of `s` when there are fewer. Matches may
/// overlap, as in Spark.
///
/// With `count = Int.MIN` (`-2^31`) there are never that many delimiters, so
/// the whole of `s` is returned (when `delim` is non-empty). Spark's JVM
/// raises an unclassified exception here; Krishiv returns what Spark's
/// algorithm would.
fn substring_index_of<'a>(s: &'a str, delim: &str, count: i32) -> &'a str {
    if delim.is_empty() || count == 0 {
        return "";
    }
    let (bytes, d) = (s.as_bytes(), delim.as_bytes());
    // No string holds more than `len + 1` matches; capping the loop bounds
    // the `Int.MIN` (`2^31`) range without changing the result.
    let cap = bytes.len().saturating_add(1);
    let at = |from: usize| {
        bytes
            .get(from..)
            .and_then(|r| r.windows(d.len()).position(|w| w == d))
            .map(|p| from.saturating_add(p))
    };
    if count > 0 {
        let mut idx = None;
        let want = usize::try_from(count).unwrap_or(0).min(cap);
        for _ in 0..want {
            match at(idx.map_or(0, |i: usize| i.saturating_add(1))) {
                Some(found) => idx = Some(found),
                None => return s,
            }
        }
        // A `count` beyond the cap would need more matches than the string
        // can hold, so reaching the cap means matches ran out.
        if usize::try_from(count).unwrap_or(usize::MAX) > cap {
            return s;
        }
        return idx.and_then(|i| s.get(..i)).unwrap_or_default();
    }
    let mut idx = bytes.len().saturating_sub(d.len()).saturating_add(1);
    let want = usize::try_from(count.unsigned_abs())
        .unwrap_or(usize::MAX)
        .min(cap);
    for _ in 0..want {
        let Some(prev) = idx.checked_sub(1) else {
            return s;
        };
        let found = (0..=prev)
            .rev()
            .find(|&p| bytes.get(p..p.saturating_add(d.len())) == Some(d));
        match found {
            Some(p) => idx = p,
            None => return s,
        }
    }
    if usize::try_from(count.unsigned_abs()).unwrap_or(usize::MAX) > cap {
        return s;
    }
    s.get(idx.saturating_add(d.len())..).unwrap_or_default()
}

/// `substring_index(str, delim, count)`.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn substring_index<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    delim: StrOperand<'_, O>,
    count: Operand<'_, Int32Type>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&input, &delim, &count];
    str_rows(&slots, |i, out| {
        let s = input.at(i);
        // The result is a slice of the input.
        reserve_str(out, s.len())?;
        out.push_str(substring_index_of(s, delim.at(i), int_at(count, i)));
        Ok(())
    })
}

/// `split_part(str, delim, n)`: the `n`-th piece of `str` split at the
/// literal `delim` (an empty `delim` splits nothing); negative `n` counts
/// from the end; the empty string when there is no such piece.
///
/// With `n = Int.MIN` (`-2^31`) the piece is past the start, so the result
/// is always the empty string. Spark's JVM raises an unclassified exception
/// here; Krishiv returns what Spark's algorithm would.
///
/// # Errors
/// `INVALID_INDEX_OF_ZERO` for `n = 0` on a row whose operands are valid;
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn split_part<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    delim: StrOperand<'_, O>,
    part: Operand<'_, Int32Type>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&input, &delim, &part];
    str_rows(&slots, |i, out| {
        let (s, d) = (input.at(i), delim.at(i));
        // The result is a slice of the input.
        reserve_str(out, s.len())?;
        let n = int_at(part, i);
        if n == 0 {
            return Err(KernelError::with_row(Some(i), InvalidIndexOfZero {}.into()));
        }
        // Pieces are cut left to right, also when counted from the end.
        let pieces = || -> Box<dyn Iterator<Item = &str> + '_> {
            if d.is_empty() {
                Box::new(std::iter::once(s))
            } else {
                Box::new(s.split(d))
            }
        };
        let at = if n > 0 {
            usize::try_from(n.saturating_sub(1)).ok()
        } else {
            let from_end = usize::try_from(n.unsigned_abs()).unwrap_or(usize::MAX);
            pieces().count().checked_sub(from_end)
        };
        if let Some(piece) = at.and_then(|k| pieces().nth(k)) {
            out.push_str(piece);
        }
        Ok(())
    })
}

/// `find_in_set(word, set)`: the 1-based index of `word` in the
/// comma-separated `set`; 0 when absent or when `word` contains a comma.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths.
pub fn find_in_set<O: OffsetSizeTrait>(
    word: StrOperand<'_, O>,
    set: StrOperand<'_, O>,
) -> Result<Int32Array, KernelError> {
    let slots: [&dyn Slot; 2] = [&word, &set];
    int_rows(&slots, |i| {
        let w = word.at(i);
        if w.contains(',') {
            return 0;
        }
        set.at(i)
            .split(',')
            .position(|piece| piece == w)
            .map_or(0, |p| {
                i32::try_from(p.saturating_add(1)).unwrap_or(i32::MAX)
            })
    })
}

// ---- overlay ----

/// The part of `n` units Spark's `overlay` keeps before `pos` and after the
/// replaced `len` units (`Int` arithmetic, wrapping as the JVM's).
fn overlay_ranges(n: usize, pos: i32, len: i32, replace_len: usize) -> [Option<(usize, usize)>; 2] {
    let length = if len >= 0 {
        len
    } else {
        i32::try_from(replace_len).unwrap_or(i32::MAX)
    };
    [
        substring_range(n, 1, pos.wrapping_sub(1)),
        substring_range(n, pos.wrapping_add(length), i32::MAX),
    ]
}

/// `overlay(input PLACING replace FROM pos [FOR len])` on strings; a
/// negative or absent `len` is the length of `replace`.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn overlay<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    replace: StrOperand<'_, O>,
    pos: Operand<'_, Int32Type>,
    len: Option<Operand<'_, Int32Type>>,
) -> Result<GenericStringArray<O>, KernelError> {
    let len = len.unwrap_or(Operand::Scalar(Some(-1)));
    let slots: [&dyn Slot; 4] = [&input, &replace, &pos, &len];
    str_rows(&slots, |i, out| {
        let (s, r) = (input.at(i), replace.at(i));
        reserve_str(out, s.len().saturating_add(r.len()))?;
        let [before, after] =
            overlay_ranges(chars_of(s), int_at(pos, i), int_at(len, i), chars_of(r));
        let piece = |range: Option<(usize, usize)>| {
            range
                .map(|(a, b)| {
                    let (from, to) = (byte_of_char(s, a), byte_of_char(s, b));
                    s.get(from..to).unwrap_or_default()
                })
                .unwrap_or_default()
        };
        out.push_str(piece(before));
        out.push_str(r);
        out.push_str(piece(after));
        Ok(())
    })
}

/// `overlay` on binary: bytes instead of characters.
///
/// # Errors
/// As [`overlay`].
pub fn overlay_binary<O: OffsetSizeTrait>(
    input: BinOperand<'_, O>,
    replace: BinOperand<'_, O>,
    pos: Operand<'_, Int32Type>,
    len: Option<Operand<'_, Int32Type>>,
) -> Result<GenericBinaryArray<O>, KernelError> {
    let len = len.unwrap_or(Operand::Scalar(Some(-1)));
    let slots: [&dyn Slot; 4] = [&input, &replace, &pos, &len];
    let rows = row_count(&slots)?;
    let mut sink = BinSink::<O>::new(rows, 0);
    let mut scratch = Vec::new();
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let (s, r) = (input.at(i), replace.at(i));
        let [before, after] = overlay_ranges(s.len(), int_at(pos, i), int_at(len, i), r.len());
        scratch.clear();
        reserve(&mut scratch, s.len().saturating_add(r.len()))?;
        scratch.extend_from_slice(before.and_then(|(a, b)| s.get(a..b)).unwrap_or_default());
        scratch.extend_from_slice(r);
        scratch.extend_from_slice(after.and_then(|(a, b)| s.get(a..b)).unwrap_or_default());
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

// ---- levenshtein, soundex, luhn_check ----

/// The edit distance between `a` and `b` in characters, two rows of work.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the working rows do not fit memory.
fn edit_distance(a: &str, b: &str) -> Result<usize, KernelError> {
    let mut av = Vec::new();
    av.try_reserve(a.chars().count())
        .map_err(|_| KernelError::out_of_memory(a.len()))?;
    av.extend(a.chars());
    let mut bv = Vec::new();
    bv.try_reserve(b.chars().count())
        .map_err(|_| KernelError::out_of_memory(b.len()))?;
    bv.extend(b.chars());
    let (short, long) = if av.len() <= bv.len() {
        (&av, &bv)
    } else {
        (&bv, &av)
    };
    let mut prev = Vec::new();
    prev.try_reserve(short.len().saturating_add(1))
        .map_err(|_| KernelError::out_of_memory(short.len().saturating_add(1)))?;
    prev.extend(0..=short.len());
    let mut cur = Vec::new();
    cur.try_reserve(short.len().saturating_add(1))
        .map_err(|_| KernelError::out_of_memory(short.len().saturating_add(1)))?;
    cur.resize(short.len().saturating_add(1), 0);
    for (j, lc) in long.iter().enumerate() {
        if let Some(first) = cur.first_mut() {
            *first = j.saturating_add(1);
        }
        for (i, sc) in short.iter().enumerate() {
            let diag = prev.get(i).copied().unwrap_or(0);
            let up = prev.get(i.saturating_add(1)).copied().unwrap_or(0);
            let left = cur.get(i).copied().unwrap_or(0);
            let cost = usize::from(sc != lc);
            if let Some(slot) = cur.get_mut(i.saturating_add(1)) {
                *slot = diag
                    .saturating_add(cost)
                    .min(up.saturating_add(1))
                    .min(left.saturating_add(1));
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok(prev.get(short.len()).copied().unwrap_or(0))
}

/// The edit distance between `a` and `b` in characters capped at
/// `threshold`, or `threshold + 1` when the distance exceeds it: Ukkonen's
/// banded dynamic program, exact for distances within the threshold.
///
/// `a` and `b` are the rows' characters; the caller guarantees their length
/// gap does not exceed `threshold`.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the working rows do not fit memory.
fn edit_distance_capped(a: &[char], b: &[char], threshold: usize) -> Result<usize, KernelError> {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let (m, n) = (short.len(), long.len());
    // The caller checks the length gap, so this is a defensive early exit.
    if n.saturating_sub(m) > threshold {
        return Ok(threshold.saturating_add(1));
    }
    if m == 0 {
        return Ok(n);
    }
    let cap = threshold.saturating_add(1);
    let mut prev = Vec::new();
    prev.try_reserve(m.saturating_add(1))
        .map_err(|_| KernelError::out_of_memory(m.saturating_add(1)))?;
    prev.resize(m.saturating_add(1), cap);
    for (j, slot) in prev.iter_mut().enumerate().take(m.saturating_add(1)) {
        if j <= threshold {
            *slot = j;
        }
    }
    let mut cur = Vec::new();
    cur.try_reserve(m.saturating_add(1))
        .map_err(|_| KernelError::out_of_memory(m.saturating_add(1)))?;
    cur.resize(m.saturating_add(1), cap);
    for (i, lc) in long.iter().enumerate() {
        let row = i.saturating_add(1);
        for slot in cur.iter_mut() {
            *slot = cap;
        }
        if row <= threshold {
            cur[0] = row;
        }
        let lo = row.saturating_sub(threshold).max(1);
        let hi = (row.saturating_add(threshold)).min(m);
        if lo > hi {
            return Ok(cap);
        }
        let mut row_min = cap;
        for j in lo..=hi {
            let cost = usize::from(short.get(j.saturating_sub(1)) != Some(lc));
            let best = prev
                .get(j.saturating_sub(1))
                .copied()
                .unwrap_or(cap)
                .saturating_add(cost)
                .min(prev.get(j).copied().unwrap_or(cap).saturating_add(1))
                .min(
                    cur.get(j.saturating_sub(1))
                        .copied()
                        .unwrap_or(cap)
                        .saturating_add(1),
                )
                .min(cap);
            if let Some(slot) = cur.get_mut(j) {
                *slot = best;
            }
            row_min = row_min.min(best);
        }
        if row_min > threshold {
            return Ok(cap);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok(prev.get(m).copied().unwrap_or(cap).min(cap))
}

/// `levenshtein(a, b[, threshold])`: the edit distance in characters; with
/// a `threshold`, -1 when the distance exceeds it (always -1 for a negative
/// threshold; a NULL threshold counts as 0, as in Spark).
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the working rows do not fit memory.
pub fn levenshtein<O: OffsetSizeTrait>(
    left: StrOperand<'_, O>,
    right: StrOperand<'_, O>,
    threshold: Option<Operand<'_, Int32Type>>,
) -> Result<Int32Array, KernelError> {
    let slots: [&dyn Slot; 2] = [&left, &right];
    let rows = row_count(&slots)?;
    if let Some(Operand::Array(t)) = threshold
        && t.len() != rows
    {
        return Err(KernelError::internal(format!(
            "string operands differ in length: {rows} vs {}",
            t.len()
        )));
    }
    let mut out = Vec::new();
    out.try_reserve(rows)
        .map_err(|_| KernelError::out_of_memory(rows))?;
    for i in 0..rows {
        if !all_valid(&slots, i) {
            out.push(None);
            continue;
        }
        // A NULL threshold reads as 0 (Scala's `null.asInstanceOf[Int]`).
        let Some(limit) = threshold.map(|t| if t.is_valid(i) { int_at(t, i) } else { 0 }) else {
            let d = i32::try_from(edit_distance(left.at(i), right.at(i))?).unwrap_or(i32::MAX);
            out.push(Some(d));
            continue;
        };
        if limit < 0 {
            out.push(Some(-1));
            continue;
        }
        let bound = usize::try_from(limit).unwrap_or(0);
        let (a, b) = (left.at(i), right.at(i));
        let (an, bn) = (a.chars().count(), b.chars().count());
        if an.saturating_sub(bn).max(bn.saturating_sub(an)) > bound {
            out.push(Some(-1));
            continue;
        }
        let mut av = Vec::new();
        av.try_reserve(an)
            .map_err(|_| KernelError::out_of_memory(a.len()))?;
        av.extend(a.chars());
        let mut bv = Vec::new();
        bv.try_reserve(bn)
            .map_err(|_| KernelError::out_of_memory(b.len()))?;
        bv.extend(b.chars());
        let d = edit_distance_capped(&av, &bv, bound)?;
        out.push(Some(if d > bound {
            -1
        } else {
            i32::try_from(d).unwrap_or(i32::MAX)
        }));
    }
    Ok(out.into_iter().collect())
}

/// The Soundex digit of an upper-case ASCII letter; `'7'` for `H` and `W`
/// (skipped without resetting the previous digit).
fn soundex_code(upper: u8) -> u8 {
    const CODES: &[u8; 26] = b"01230127022455012623017202";
    upper
        .checked_sub(b'A')
        .and_then(|n| CODES.get(usize::from(n)))
        .copied()
        .unwrap_or(b'0')
}

/// `soundex(str)`: the four-character code (a first character that is not a
/// letter returns the string unchanged; the empty string stays empty).
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn soundex<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let input = StrOperand::Array(a);
    str_rows(&[&input], |i, out| {
        let s = a.value(i);
        // Four ASCII bytes, or the whole string when it is returned unchanged.
        reserve_str(out, s.len().max(4))?;
        let mut bytes = s.bytes();
        let Some(first) = bytes.next() else {
            return Ok(());
        };
        if !first.is_ascii_alphabetic() {
            out.push_str(s);
            return Ok(());
        }
        let first = first.to_ascii_uppercase();
        out.push(char::from(first));
        let mut last = soundex_code(first);
        for b in bytes {
            if !b.is_ascii_alphabetic() {
                last = b'0';
                continue;
            }
            let code = soundex_code(b.to_ascii_uppercase());
            if code == b'7' {
                continue;
            }
            if code != b'0' && code != last {
                out.push(char::from(code));
                if out.len() > 3 {
                    break;
                }
            }
            last = code;
        }
        while out.len() < 4 {
            out.push('0');
        }
        Ok(())
    })
}

/// `luhn_check(str)`: whether the digits (any Unicode decimal digit in the
/// Basic Multilingual Plane; no other character) pass the Luhn checksum. The
/// empty string is not a number.
#[must_use]
pub fn luhn_check<O: OffsetSizeTrait>(a: &GenericStringArray<O>) -> BooleanArray {
    let input = StrOperand::Array(a);
    bool_result(a.len(), &[&input], |i| {
        let s = a.value(i);
        if s.is_empty() {
            return false;
        }
        let mut sum = 0_u32;
        for (n, c) in s.chars().rev().enumerate() {
            let digit = match java_digit(c) {
                Some(d) if u32::from(c) <= 0xFFFF => d,
                _ => return false,
            };
            let d = u32::from(digit).saturating_mul(if n % 2 == 1 { 2 } else { 1 });
            sum = sum.saturating_add((d % 10).saturating_add(d / 10));
        }
        sum.is_multiple_of(10)
    })
}

// ---- quote ----

/// `quote(str)`: the string in single quotes, each embedded quote preceded
/// by a backslash.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit memory or the
/// offset type.
pub fn quote<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let input = StrOperand::Array(a);
    str_rows(&[&input], |i, out| {
        let s = a.value(i);
        reserve_str(out, s.len().saturating_mul(2).saturating_add(2))?;
        out.push('\'');
        for c in s.chars() {
            if c == '\'' {
                out.push('\\');
            }
            out.push(c);
        }
        out.push('\'');
        Ok(())
    })
}

// ---- elt, concat_ws ----

/// The 0-based argument `elt` picks per row: `None` for a NULL index and,
/// without `ansi`, an index outside `1..=count`; with `ansi` that index is
/// `INVALID_ARRAY_INDEX`.
fn elt_picks(
    index: Operand<'_, Int32Type>,
    count: usize,
    rows: usize,
    ansi: bool,
) -> Result<Vec<Option<usize>>, KernelError> {
    let mut picks = Vec::new();
    picks
        .try_reserve(rows)
        .map_err(|_| KernelError::out_of_memory(rows))?;
    for i in 0..rows {
        if !index.is_valid(i) {
            picks.push(None);
            continue;
        }
        let n = int_at(index, i);
        let pick = usize::try_from(n)
            .ok()
            .filter(|n| (1..=count).contains(n))
            .map(|n| n.saturating_sub(1));
        if pick.is_none() && ansi {
            return Err(KernelError::with_row(
                Some(i),
                InvalidArrayIndex {
                    index_value: n.to_string(),
                    array_size: count.to_string(),
                }
                .into(),
            ));
        }
        picks.push(pick);
    }
    Ok(picks)
}

/// `elt(n, str, ...)`: the `n`-th of the strings; NULL for a NULL `n` or a
/// NULL chosen string, and for an `n` outside `1..=count` unless `ansi`,
/// where that is `INVALID_ARRAY_INDEX`.
///
/// # Errors
/// `INVALID_ARRAY_INDEX` as above; `INTERNAL_ERROR` for no column operand or
/// operands of different lengths; `UNABLE_TO_ACQUIRE_MEMORY` if the result
/// does not fit the offset type.
pub fn elt<O: OffsetSizeTrait>(
    index: Operand<'_, Int32Type>,
    parts: &[StrOperand<'_, O>],
    ansi: bool,
) -> Result<GenericStringArray<O>, KernelError> {
    let mut slots: Vec<&dyn Slot> = parts.iter().map(|p| p as &dyn Slot).collect();
    slots.push(&index);
    let rows = row_count(&slots)?;
    let picks = elt_picks(index, parts.len(), rows, ansi)?;
    let mut sink = StrSink::<O>::new(rows, 0);
    for (i, pick) in picks.iter().enumerate() {
        match pick.and_then(|p| parts.get(p)).filter(|p| p.is_valid(i)) {
            Some(p) => sink.push(p.at(i))?,
            None => sink.push_null(),
        }
    }
    Ok(sink.finish())
}

/// `elt` on binary arguments.
///
/// # Errors
/// As [`elt()`].
pub fn elt_binary<O: OffsetSizeTrait>(
    index: Operand<'_, Int32Type>,
    parts: &[BinOperand<'_, O>],
    ansi: bool,
) -> Result<GenericBinaryArray<O>, KernelError> {
    let mut slots: Vec<&dyn Slot> = parts.iter().map(|p| p as &dyn Slot).collect();
    slots.push(&index);
    let rows = row_count(&slots)?;
    let picks = elt_picks(index, parts.len(), rows, ansi)?;
    let mut sink = BinSink::<O>::new(rows, 0);
    for (i, pick) in picks.iter().enumerate() {
        match pick.and_then(|p| parts.get(p)).filter(|p| p.is_valid(i)) {
            Some(p) => sink.push(p.at(i))?,
            None => sink.push_null(),
        }
    }
    Ok(sink.finish())
}

/// `concat_ws(sep, str, ...)`: the non-NULL strings joined by `sep`; NULL
/// only when `sep` is NULL (no strings give the empty string).
///
/// # Errors
/// `INTERNAL_ERROR` for no column operand or operands of different
/// lengths; `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit memory or
/// the offset type.
pub fn concat_ws<O: OffsetSizeTrait>(
    sep: StrOperand<'_, O>,
    parts: &[StrOperand<'_, O>],
) -> Result<GenericStringArray<O>, KernelError> {
    let mut slots: Vec<&dyn Slot> = parts.iter().map(|p| p as &dyn Slot).collect();
    slots.push(&sep);
    let rows = row_count(&slots)?;
    let mut sink = StrSink::<O>::new(rows, 0);
    let mut scratch = String::new();
    for i in 0..rows {
        if !sep.is_valid(i) {
            sink.push_null();
            continue;
        }
        let need: usize = parts
            .iter()
            .filter(|p| p.is_valid(i))
            .map(|p| p.at(i).len().saturating_add(sep.at(i).len()))
            .fold(0, usize::saturating_add);
        check_offset::<O>(sink.bytes.saturating_add(need))?;
        scratch.clear();
        reserve_str(&mut scratch, need)?;
        for (n, p) in parts.iter().filter(|p| p.is_valid(i)).enumerate() {
            if n > 0 {
                scratch.push_str(sep.at(i));
            }
            scratch.push_str(p.at(i));
        }
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

// ---- lengths ----

/// `bit_length(str | binary)`: eight times the byte length, wrapping like the
/// JVM's `int` product.
#[must_use]
pub fn bit_length<T: arrow_array::types::ByteArrayType>(
    a: &arrow_array::GenericByteArray<T>,
) -> Int32Array {
    super::byte_length(a).unary(|n| n.wrapping_mul(8))
}

/// `reverse(binary)`: the bytes in reverse order.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn reverse_binary<O: OffsetSizeTrait>(
    a: &GenericBinaryArray<O>,
) -> Result<GenericBinaryArray<O>, KernelError> {
    let mut sink = BinSink::<O>::new(a.len(), a.values().len());
    let mut scratch = Vec::new();
    for i in 0..a.len() {
        if a.is_null(i) {
            sink.push_null();
            continue;
        }
        scratch.clear();
        scratch.extend(a.value(i).iter().rev());
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}
