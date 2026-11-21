// SPDX-License-Identifier: Apache-2.0
//! The string functions whose result depends on the collation, over `UTF8_LCASE` and the
//! `_RTRIM` collations (collations unit 5c): the `trim` family, `replace`, `translate`,
//! `split_part` and `substring_index`.
//!
//! Matching is on lower-cased code points as in [`super::chars::locate_lcase`] (`İ` lower-cases
//! to `i` + U+0307, final sigma to `σ`; a match may not start or end between the two). Every
//! behaviour below was read off a Spark 4.2.0 run (2026-11-10), not derived from the source.

use std::collections::{HashMap, HashSet};

use arrow_array::types::Int32Type;
use arrow_array::{GenericStringArray, OffsetSizeTrait};
use krishiv_common::error::InvalidIndexOfZero;
use krishiv_types::Collation;

use super::chars::{lcase_match_from, lcase_points, reserve_str, str_rows};
use super::collation_key::icu_case_map;
use super::{Slot, StrOperand, TrimSide, int_at};
use crate::error::KernelError;
use crate::ops::Operand;

/// The `mode` operand of [`trim_collated`] for `side`, a lower-case comparison (`UTF8_LCASE`)
/// and the `_RTRIM` rule.
#[must_use]
pub fn trim_mode(side: TrimSide, lcase: bool, rtrim: bool) -> i32 {
    let side = match side {
        TrimSide::Both => 0,
        TrimSide::Leading => 1,
        TrimSide::Trailing => 2,
    };
    side | (i32::from(lcase) << 2) | (i32::from(rtrim) << 3)
}

/// One character in Java's per-character case folding, as `Pattern.CASE_INSENSITIVE | UNICODE_CASE`
/// compares: the simple lower case of the simple upper case (a mapping to more than one character
/// leaves the character as it is; `İ` lower-cases to `i`, as `Character.toLowerCase` does).
fn java_fold(c: char) -> char {
    let mut up = c.to_uppercase();
    let upper = match (up.next(), up.next()) {
        (Some(u), None) => u,
        _ => c,
    };
    if upper == '\u{130}' {
        return 'i';
    }
    let mut low = upper.to_lowercase();
    match (low.next(), low.next()) {
        (Some(l), None) => l,
        _ => upper,
    }
}

/// The text `LIKE` and `ILIKE` compare over a `UTF8_LCASE` string (SQL-40). Spark matches each
/// character by Java's case folding, so both operands go through `java_fold`. With `ilike` the
/// operand is first lower-cased in full (`İ` becomes `i` and U+0307, which is why `ILIKE` and `LIKE`
/// differ for `İ`). One key per character cannot express Java's one-directional matches: the pattern
/// `ß` matches only `ß` while the pattern `ẞ` matches both, so value `ẞ` against pattern `ß` is true here
/// and false in Spark (likewise the titlecase letters `U+1F88`..`U+1FAF`, `U+1FBC`, `U+1FCC`, `U+1FFC`
/// against their lower-case partners). Oracle: a Spark 4.3.0 run, 2026-11-21 (`tests/fixtures/spark-4.3.0/like-lcase.json`).
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn like_fold<O: OffsetSizeTrait>(
    ilike: Operand<'_, Int32Type>,
    input: StrOperand<'_, O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 2] = [&ilike, &input];
    str_rows(&slots, |i, out| {
        let s = input.at(i);
        reserve_str(out, s.len().saturating_mul(2))?;
        let full = int_at(ilike, i) != 0;
        for c in s.chars() {
            if full {
                for l in c.to_lowercase() {
                    out.push(java_fold(l));
                }
            } else {
                out.push(java_fold(c));
            }
        }
        Ok(())
    })
}

/// [`like_fold`] for a `LIKE ... ESCAPE` pattern: Spark reads the escape character off the pattern as
/// written (`ILIKE`: off the lower-cased pattern) and only then compiles the literal characters
/// case-insensitively, so the escape matches itself exactly, never its other cases (SQL-40, review
/// of PR 949). A character that folds to the escape is written escaped, so it stays a literal.
///
/// An `escape` that is not one character leaves the pattern folded as [`like_fold`] does (the
/// `like` call then refuses the escape).
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn like_fold_pattern<O: OffsetSizeTrait>(
    ilike: Operand<'_, Int32Type>,
    pattern: StrOperand<'_, O>,
    escape: StrOperand<'_, O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&ilike, &pattern, &escape];
    str_rows(&slots, |i, out| {
        let s = pattern.at(i);
        reserve_str(out, s.len().saturating_mul(3))?;
        let full = int_at(ilike, i) != 0;
        let mut esc_chars = escape.at(i).chars();
        let esc = match (esc_chars.next(), esc_chars.next()) {
            (Some(e), None) => Some(e),
            _ => None,
        };
        let mut pieces: Vec<char> = Vec::with_capacity(s.len());
        for c in s.chars() {
            if full {
                pieces.extend(c.to_lowercase());
            } else {
                pieces.push(c);
            }
        }
        let mut it = pieces.into_iter();
        while let Some(p) = it.next() {
            if Some(p) == esc {
                out.push(p);
                if let Some(next) = it.next() {
                    out.push(java_fold(next));
                }
                continue;
            }
            let f = java_fold(p);
            if Some(f) == esc {
                out.push(f);
            }
            out.push(f);
        }
        Ok(())
    })
}

/// The lower-cased code points of `window` (or the characters themselves when not `lcase`).
fn key(window: &[char], lcase: bool, buf: &mut Vec<char>) {
    buf.clear();
    for c in window {
        if lcase {
            lcase_points(*c, buf);
        } else {
            buf.push(*c);
        }
    }
}

/// One row of [`trim_collated`].
fn trim_row(src: &str, chars: &str, mode: i32, out: &mut String) {
    let (lcase, rtrim) = (mode & 4 != 0, mode & 8 != 0);
    let (left, right) = match mode & 3 {
        1 => (true, false),
        2 => (false, true),
        _ => (true, true),
    };
    let text: Vec<char> = src.chars().collect();
    let mut buf = Vec::with_capacity(4);
    // The trim set: each character's lower-cased form, as a string (`İ` is the pair `i` U+0307
    // and is trimmed as the pair only, never as its halves).
    let mut set: HashSet<Vec<char>> = HashSet::new();
    for c in chars.chars() {
        key(&[c], lcase, &mut buf);
        set.insert(buf.clone());
    }
    // The widest window that can match: a character that lower-cases to two code points.
    let widest: usize = if lcase { 2 } else { 1 };
    let (mut start, mut end) = (0_usize, text.len());
    if left {
        'scan: while start < end {
            for w in (1..=widest).rev() {
                let Some(window) = text.get(start..start.saturating_add(w).min(end)) else {
                    continue;
                };
                if window.len() == w && {
                    key(window, lcase, &mut buf);
                    set.contains(buf.as_slice())
                } {
                    start = start.saturating_add(w);
                    continue 'scan;
                }
            }
            break;
        }
    }
    // A `_RTRIM` string ignores its trailing spaces while the end is trimmed (unless the
    // characters to trim hold a space), and keeps them.
    let mut spaces = 0_usize;
    if right && rtrim && !set.contains(&vec![' ']) {
        while end > start && text.get(end.saturating_sub(1)) == Some(&' ') {
            end = end.saturating_sub(1);
            spaces = spaces.saturating_add(1);
        }
    }
    if right {
        'scan: while end > start {
            for w in (1..=widest).rev() {
                if end.saturating_sub(start) < w {
                    continue;
                }
                let Some(window) = text.get(end.saturating_sub(w)..end) else {
                    continue;
                };
                key(window, lcase, &mut buf);
                if set.contains(buf.as_slice()) {
                    end = end.saturating_sub(w);
                    continue 'scan;
                }
            }
            break;
        }
    }
    out.extend(text.get(start..end).unwrap_or_default());
    out.extend(std::iter::repeat_n(' ', spaces));
}

/// `trim` / `ltrim` / `rtrim` / `btrim` of `input` by the characters `chars` over `UTF8_LCASE`
/// (characters match by their lower-cased form, the pair `İ` lower-cases to as a pair) or
/// `_RTRIM` strings (trailing spaces of `input` are kept and not trimmed unless `chars` holds a
/// space). `mode` is [`trim_mode`], per row.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn trim_collated<O: OffsetSizeTrait>(
    mode: Operand<'_, Int32Type>,
    input: StrOperand<'_, O>,
    chars: StrOperand<'_, O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&mode, &input, &chars];
    str_rows(&slots, |i, out| {
        let s = input.at(i);
        // The result is a slice of the input, plus spaces it had.
        reserve_str(out, s.len())?;
        trim_row(s, chars.at(i), int_at(mode, i), out);
        Ok(())
    })
}

/// The characters of `s` as a vector.
fn chars_of(s: &str) -> Vec<char> {
    s.chars().collect()
}

/// The lower-cased code points of the whole of `s`.
fn lowered(s: &str) -> Vec<char> {
    let mut out = Vec::new();
    for c in s.chars() {
        lcase_points(c, &mut out);
    }
    out
}

/// `replace(str, search, replacement)` over `UTF8_LCASE`: every non-overlapping match of the
/// lower-cased `search`, left to right; an empty `search` replaces nothing.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn replace_lcase<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    search: StrOperand<'_, O>,
    replacement: StrOperand<'_, O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&input, &search, &replacement];
    str_rows(&slots, |i, out| {
        let (s, to) = (input.at(i), replacement.at(i));
        if search.at(i).is_empty() {
            reserve_str(out, s.len())?;
            out.push_str(s);
            return Ok(());
        }
        let text = chars_of(s);
        let pattern = lowered(search.at(i));
        let mut at = 0_usize;
        while at < text.len() {
            match lcase_match_from(&text, at, &pattern) {
                Some(n) if n > 0 => {
                    out.push_str(to);
                    at = at.saturating_add(n);
                }
                _ => {
                    out.extend(text.get(at));
                    at = at.saturating_add(1);
                }
            }
        }
        Ok(())
    })
}

/// A translation keyed by lower-cased characters.
type Dict = HashMap<Vec<char>, Option<char>>;

/// The translation of `from` to `to` keyed by lower-cased characters: the first occurrence of a
/// character wins, and a character without a counterpart (or one that is NUL) is deleted.
fn translation_lcase(from: &str, to: &str) -> Dict {
    let mut to = to.chars();
    let mut dict = HashMap::new();
    for f in from.chars() {
        let t = to.next().filter(|c| *c != '\0');
        let mut k = Vec::with_capacity(2);
        lcase_points(f, &mut k);
        dict.entry(k).or_insert(t);
    }
    dict
}

/// `translate(str, from, to)` over `UTF8_LCASE`: a character of `str` is replaced when its
/// lower-cased form (the pair `i` U+0307 as one for `İ`) is that of a character of `from`.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn translate_lcase<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    from: StrOperand<'_, O>,
    to: StrOperand<'_, O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&input, &from, &to];
    let mut cached: Option<(String, String, Dict)> = None;
    str_rows(&slots, |i, out| {
        let s = input.at(i);
        reserve_str(out, s.len().saturating_mul(4))?;
        let (f, t) = (from.at(i), to.at(i));
        if cached.as_ref().is_none_or(|(cf, ct, _)| cf != f || ct != t) {
            cached = Some((f.to_owned(), t.to_owned(), translation_lcase(f, t)));
        }
        let Some((_, _, dict)) = cached.as_ref() else {
            return Ok(());
        };
        let text = chars_of(s);
        let mut buf = Vec::with_capacity(4);
        let mut at = 0_usize;
        'chars: while at < text.len() {
            // The pair `İ` lower-cases to may also be written as two characters.
            for w in [2_usize, 1] {
                let Some(window) = text.get(at..at.saturating_add(w)) else {
                    continue;
                };
                // Spark joins only a written lower-case `i` with the dot (a capital `I`
                // followed by U+0307 stays two characters).
                if w == 2 && window.first() != Some(&'i') {
                    continue;
                }
                buf.clear();
                for c in window {
                    lcase_points(*c, &mut buf);
                }
                if let Some(r) = dict.get(buf.as_slice()) {
                    out.extend(*r);
                    at = at.saturating_add(w);
                    continue 'chars;
                }
            }
            out.extend(text.get(at));
            at = at.saturating_add(1);
        }
        Ok(())
    })
}

/// The pieces of `text` split at the matches of the lower-cased `pattern`, left to right and
/// without overlap.
fn pieces_lcase(text: &[char], pattern: &[char]) -> Vec<(usize, usize)> {
    let (mut out, mut from, mut at) = (Vec::new(), 0_usize, 0_usize);
    while at < text.len() {
        match lcase_match_from(text, at, pattern) {
            Some(n) if n > 0 => {
                out.push((from, at));
                at = at.saturating_add(n);
                from = at;
            }
            _ => at = at.saturating_add(1),
        }
    }
    out.push((from, text.len()));
    out
}

/// `split_part(str, delim, n)` over `UTF8_LCASE`: the `n`-th piece of `str` split at the
/// matches of the lower-cased `delim` (an empty `delim` splits nothing); negative `n` counts
/// from the end; the empty string when there is no such piece.
///
/// # Errors
/// `INVALID_INDEX_OF_ZERO` for `n = 0` on a row whose operands are valid;
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn split_part_lcase<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    delim: StrOperand<'_, O>,
    part: Operand<'_, Int32Type>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&input, &delim, &part];
    str_rows(&slots, |i, out| {
        let s = input.at(i);
        reserve_str(out, s.len())?;
        let n = int_at(part, i);
        if n == 0 {
            return Err(KernelError::with_row(Some(i), InvalidIndexOfZero {}.into()));
        }
        let text = chars_of(s);
        let pieces = if delim.at(i).is_empty() {
            vec![(0, text.len())]
        } else {
            pieces_lcase(&text, &lowered(delim.at(i)))
        };
        let at = if n > 0 {
            usize::try_from(n.saturating_sub(1)).ok()
        } else {
            pieces
                .len()
                .checked_sub(usize::try_from(n.unsigned_abs()).unwrap_or(usize::MAX))
        };
        if let Some((a, b)) = at.and_then(|k| pieces.get(k)) {
            out.extend(text.get(*a..*b).unwrap_or_default());
        }
        Ok(())
    })
}

/// `substring_index(str, delim, count)` over `UTF8_LCASE`: everything before the `count`-th match
/// of the lower-cased `delim` from the left (`count > 0`) or after the `|count|`-th from the
/// right; all of `str` when there are fewer. Matches may overlap. With `count = Int.MIN` the
/// answer is the empty string, as in Spark.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn substring_index_lcase<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    delim: StrOperand<'_, O>,
    count: Operand<'_, Int32Type>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&input, &delim, &count];
    str_rows(&slots, |i, out| {
        let s = input.at(i);
        reserve_str(out, s.len())?;
        let n = int_at(count, i);
        if delim.at(i).is_empty() || n == 0 || n == i32::MIN {
            return Ok(());
        }
        let text = chars_of(s);
        let pattern = lowered(delim.at(i));
        let want = usize::try_from(n.unsigned_abs()).unwrap_or(usize::MAX);
        let range = if n > 0 {
            // The start of the `want`-th match, each search beginning one character after the
            // previous match's start.
            let mut from = 0_usize;
            let mut found = None;
            for _ in 0..want {
                let hit =
                    (from..text.len()).find(|a| lcase_match_from(&text, *a, &pattern).is_some());
                let Some(a) = hit else {
                    found = None;
                    break;
                };
                found = Some(a);
                from = a.saturating_add(1);
            }
            found.map(|a| 0..a)
        } else {
            // The `want`-th match from the right: each search ends before the previous match's start.
            let mut below = text.len();
            let mut found = None;
            for _ in 0..want {
                let hit = (0..below)
                    .rev()
                    .find_map(|a| lcase_match_from(&text, a, &pattern).map(|len| (a, len)));
                let Some((a, len)) = hit else {
                    found = None;
                    break;
                };
                found = Some((a, len));
                below = a;
            }
            found.map(|(a, len)| a.saturating_add(len)..text.len())
        };
        out.extend(range.map_or(text.as_slice(), |r| text.get(r).unwrap_or_default()));
        Ok(())
    })
}

/// `upper(input)` (`upper` nonzero) or `lower(input)` under the ICU collation whose Spark id is
/// `collation`: ICU's full case mapping in the collation's language (collations unit 5, design
/// `P2-collations.md` 3.3; values read off Spark 4.2.0).
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths, an id that is no ICU collation, or a
/// build without `collations-icu`; `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the
/// offset type.
pub fn icu_case<O: OffsetSizeTrait>(
    upper: Operand<'_, Int32Type>,
    collation: Operand<'_, Int32Type>,
    input: StrOperand<'_, O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 3] = [&upper, &collation, &input];
    str_rows(&slots, |i, out| {
        let id = u32::try_from(int_at(collation, i))
            .map_err(|_| KernelError::internal("collation id".to_owned()))?;
        let mapped = icu_case_map(Collation::from_id(id), int_at(upper, i) != 0, input.at(i))?;
        reserve_str(out, mapped.len())?;
        out.push_str(&mapped);
        Ok(())
    })
}

#[cfg(test)]
mod like_fold_tests {
    use super::java_fold;

    #[test]
    fn java_fold_is_lower_of_upper_per_character() {
        assert_eq!(java_fold('A'), 'a');
        assert_eq!(java_fold('a'), 'a');
        // Long s and Kelvin sign fold to their ASCII letters through the upper case.
        assert_eq!(java_fold('\u{17f}'), 's');
        assert_eq!(java_fold('\u{212a}'), 'k');
        // `İ` folds to `i`; `ß` upper-cases to two characters, so it stays.
        assert_eq!(java_fold('\u{130}'), 'i');
        assert_eq!(java_fold('\u{df}'), '\u{df}');
        // Final and non-final sigma meet at the same character.
        assert_eq!(java_fold('\u{3c2}'), java_fold('\u{3a3}'));
        assert_eq!(java_fold('1'), '1');
    }

    #[test]
    fn pattern_fold_reads_the_escape_before_folding() {
        use arrow_array::StringArray;
        let fold = |ilike: i32, pat: &str, esc: &str| {
            let out = super::like_fold_pattern::<i32>(
                crate::ops::Operand::Scalar(Some(ilike)),
                super::StrOperand::Array(&StringArray::from(vec![pat])),
                super::StrOperand::Scalar(Some(esc)),
            )
            .unwrap();
            let a: &StringArray = &out;
            a.value(0).to_owned()
        };
        // `e` is not the escape `E`: it folds as a literal and `%` stays a wildcard.
        assert_eq!(fold(0, "E%", "E"), "E%");
        assert_eq!(fold(0, "e%", "E"), "e%");
        assert_eq!(fold(0, "AE%", "E"), "aE%");
        // A character folding to the escape is written escaped, so it stays literal.
        assert_eq!(fold(0, "E%", "e"), "ee%");
        // `ILIKE` lower-cases first, so an upper-case escape never occurs in the pattern.
        assert_eq!(fold(1, "E%", "E"), "e%");
    }

    #[test]
    fn java_fold_is_idempotent_over_the_bmp() {
        for c in ('\0'..='\u{ffff}').filter_map(|c| {
            (c as u32 >= 0xd800 && c as u32 <= 0xdfff)
                .then_some(c)
                .map_or(Some(c), |_| None)
        }) {
            assert_eq!(java_fold(java_fold(c)), java_fold(c), "{c:?}");
        }
    }
}
